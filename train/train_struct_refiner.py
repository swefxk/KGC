import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.struct_backbone_factory import load_struct_backbone, resolve_struct_ckpt
from models.struct_refiner import StructRefiner
from tools.run_meta import write_run_metadata


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def build_to_skip(processor: KGProcessor, include_test: bool = False):
    """
    filtered evaluation 的 filter map。
    - valid 上：include_test=False（train+valid）
    - test 上：include_test=True（train+valid+test）
    """
    parts = [processor.train_triplets, processor.valid_triplets]
    if include_test:
        parts.append(processor.test_triplets)

    all_triplets = torch.cat(parts, dim=0).tolist()

    to_skip = {'rhs': {}, 'lhs': {}}
    for h, r, t in all_triplets:
        h, r, t = int(h), int(r), int(t)
        to_skip['rhs'].setdefault((h, r), set()).add(t)   # (h,r)->{t}
        to_skip['lhs'].setdefault((t, r), set()).add(h)   # (t,r)->{h}
    return to_skip


def build_true_head_tail_from_train(train_triplets: torch.LongTensor):
    """
    只用 train 构建真值集合（防泄漏）：
    true_tail[(h,r)] = {t}
    true_head[(t,r)] = {h}   # key 约定与 eval_full_entity_filtered 的 lhs 一致：(t,r)->{h}
    """
    true_tail = {}
    true_head = {}
    for h, r, t in train_triplets.tolist():
        h, r, t = int(h), int(r), int(t)
        true_tail.setdefault((h, r), set()).add(t)
        true_head.setdefault((t, r), set()).add(h)
    return true_head, true_tail


def _sample_one_filtered(num_entities: int, num_neg: int, target: int, forbidden_set, oversample: int):
    """
    采样一个样本的 filtered negatives：
    - 不采 target
    - 不采 forbidden_set 里的真值
    """
    need = num_neg
    negs = []

    forb_arr = None
    if forbidden_set:
        forb_arr = np.fromiter(forbidden_set, dtype=np.int64)

    while need > 0:
        cand = np.random.randint(num_entities, size=need * oversample, dtype=np.int64)

        # 去掉 target
        cand = cand[cand != target]
        if cand.size == 0:
            continue

        # 去掉 forbidden 真值
        if forb_arr is not None and forb_arr.size > 0:
            mask = ~np.isin(cand, forb_arr, assume_unique=False)
            cand = cand[mask]
            if cand.size == 0:
                continue

        take = cand[:need]
        negs.append(take)
        need -= take.size

    out = np.concatenate(negs, axis=0)[:num_neg]
    return out


def sample_from_topk(cand_2d: torch.Tensor, k_sample: int, prefix_k: int):
    """
    Sample k_sample negatives from cand_2d [B,K_full], taking prefix_k from the beginning
    and the rest randomly from the remainder.
    Returns sampled candidates and their indices in the full array.
    """
    B, K_full = cand_2d.shape
    if k_sample <= 0:
        empty = torch.empty((B, 0), dtype=cand_2d.dtype, device=cand_2d.device)
        empty_idx = torch.empty((B, 0), dtype=torch.long, device=cand_2d.device)
        return empty, empty_idx

    if k_sample >= K_full:
        idx = torch.arange(K_full, device=cand_2d.device).unsqueeze(0).expand(B, -1)
        return cand_2d, idx

    if prefix_k >= k_sample:
        idx = torch.arange(k_sample, device=cand_2d.device).unsqueeze(0).expand(B, -1)
        return cand_2d[:, :k_sample], idx

    prefix_idx = torch.arange(prefix_k, device=cand_2d.device).unsqueeze(0).expand(B, -1)
    prefix_cands = cand_2d[:, :prefix_k]

    remaining_k = k_sample - prefix_k
    remaining_pool_size = K_full - prefix_k
    rand_offset = torch.randint(0, remaining_pool_size, (B, remaining_k), device=cand_2d.device)
    rand_idx = prefix_k + rand_offset
    rand_cands = cand_2d.gather(1, rand_idx)

    combined_cands = torch.cat([prefix_cands, rand_cands], dim=1)
    combined_idx = torch.cat([prefix_idx, rand_idx], dim=1)
    return combined_cands, combined_idx


class IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def sample_filtered_negatives_batch(
    anchor_ids: torch.LongTensor,
    rel_ids: torch.LongTensor,
    target_ids: torch.LongTensor,
    conj_flag: torch.BoolTensor,
    num_entities: int,
    num_neg: int,
    true_head_train: dict,
    true_tail_train: dict,
    device: torch.device,
    oversample: int = 4,
):
    """
    批量 filtered negatives（train-only truths）：
    - conj=False: RHS, forbidden = true_tail[(h,r)]
    - conj=True : LHS, forbidden = true_head[(t,r)]，此时 anchor 就是 t
    """
    a_cpu = anchor_ids.detach().cpu().numpy()
    r_cpu = rel_ids.detach().cpu().numpy()
    tgt_cpu = target_ids.detach().cpu().numpy()
    conj_cpu = conj_flag.detach().cpu().numpy()

    neg_np = np.empty((a_cpu.shape[0], num_neg), dtype=np.int64)

    for i in range(a_cpu.shape[0]):
        a = int(a_cpu[i])
        r = int(r_cpu[i])
        tgt = int(tgt_cpu[i])
        conj = bool(conj_cpu[i])

        if not conj:
            forbidden = true_tail_train.get((a, r), set())
        else:
            forbidden = true_head_train.get((a, r), set())

        neg_np[i] = _sample_one_filtered(
            num_entities=num_entities,
            num_neg=num_neg,
            target=tgt,
            forbidden_set=forbidden,
            oversample=oversample
        )

    return torch.from_numpy(neg_np).to(device=device, dtype=torch.long)


@torch.no_grad()
def eval_refiner_struct_only(
    processor,
    rotate_model,
    refiner,
    ent_ids_all,
    to_skip,
    split_triplets,
    device,
    batch_size=8,
    chunk_size=2048,
    num_workers=2
):
    """
    这个 eval 返回的是 RHS/LHS 合在一起的 overall（每个 triple 计两次：rhs+lhs）
    用于训练中 early stopping/select best。
    """
    rotate_model.eval()
    refiner.eval()

    loader = DataLoader(
        TrainDataset(split_triplets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    mrr_sum = 0.0
    hits1 = 0
    hits10 = 0
    total = 0
    N = processor.num_entities

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h = batch[:, 0]
        r = batch[:, 1]
        t = batch[:, 2]
        B = h.size(0)

        # --- RHS ---
        h_ref = refiner.refine_anchor(
            h, rotate_model,
            processor.nbr_ent, processor.nbr_rel, processor.nbr_dir, processor.nbr_mask, processor.freq
        )

        scores_rhs = torch.empty((B, N), device=device)
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            cand = ent_ids_all[s:e]
            scores_rhs[:, s:e] = rotate_model.score_from_head_emb(h_ref, r, cand, conj=False)

        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            score = scores_rhs[i]
            target_score = score[ti].item()

            filt = to_skip['rhs'].get((hi, ri), set())
            if filt:
                idx = torch.tensor(list(filt), device=device)
                mask = idx != ti
                score[idx[mask]] = -1e9

            rank = (score > target_score).sum().item() + 1
            mrr_sum += 1.0 / rank
            hits1 += int(rank <= 1)
            hits10 += int(rank <= 10)
            total += 1

        # --- LHS (score(t, conj(r), h)) ---
        t_ref = refiner.refine_anchor(
            t, rotate_model,
            processor.nbr_ent, processor.nbr_rel, processor.nbr_dir, processor.nbr_mask, processor.freq
        )

        scores_lhs = torch.empty((B, N), device=device)
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            cand = ent_ids_all[s:e]
            scores_lhs[:, s:e] = rotate_model.score_from_head_emb(t_ref, r, cand, conj=True)

        for i in range(B):
            hi, ri, ti = h[i].item(), r[i].item(), t[i].item()
            score = scores_lhs[i]
            target_score = score[hi].item()

            filt = to_skip['lhs'].get((ti, ri), set())
            if filt:
                idx = torch.tensor(list(filt), device=device)
                mask = idx != hi
                score[idx[mask]] = -1e9

            rank = (score > target_score).sum().item() + 1
            mrr_sum += 1.0 / rank
            hits1 += int(rank <= 1)
            hits10 += int(rank <= 10)
            total += 1

    return {
        "MRR": mrr_sum / max(total, 1),
        "Hits@1": hits1 / max(total, 1),
        "Hits@10": hits10 / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/fb15k_custom")
    parser.add_argument("--struct_type", type=str, default="rotate", choices=["rotate", "complex"])
    parser.add_argument("--pretrained_struct", type=str, default=None,
                        help="struct backbone checkpoint (overrides --pretrained_rotate)")
    parser.add_argument("--pretrained_rotate", type=str, default=None,
                        help="legacy RotatE checkpoint (struct_type=rotate)")
    parser.add_argument("--save_dir", type=str, default="checkpoints/struct_refiner")

    parser.add_argument("--emb_dim", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=9.0, help="RotatE margin (ignored by ComplEx)")

    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--num_neg", type=int, default=64, help="used only for random filtered negatives")
    parser.add_argument("--train_cache_rhs", type=str, default=None, help="RHS hard-neg cache (neg_t)")
    parser.add_argument("--train_cache_lhs", type=str, default=None, help="LHS hard-neg cache (neg_h)")
    parser.add_argument("--hard_k", type=int, default=128)
    parser.add_argument("--sample_prefix_k", type=int, default=64)
    parser.add_argument("--rand_neg", type=int, default=8)
    parser.add_argument("--neg_oversample", type=int, default=4, help="oversampling factor for filtered negatives")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_chunk_size", type=int, default=2048)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=2)

    # ⭐ Gate regularization（防止 g 崩到接近 0）
    parser.add_argument("--gate_reg_lambda", type=float, default=1e-3,
                        help="L2 regularize gate params (w,b) to prevent gate collapse; set 0 to disable")
    parser.add_argument("--gate_w_target", type=float, default=1.0,
                        help="target for w = softplus(w_raw)")
    parser.add_argument("--gate_b_target", type=float, default=0.0,
                        help="target for b")

    # 可选：限制 eta 无限制变大（默认不加）
    parser.add_argument("--eta_reg_lambda", type=float, default=0.0,
                        help="L2 regularize eta to prevent overly strong refine; set 0 to disable")
    parser.add_argument("--anchor_reg_lambda", type=float, default=0.0,
                        help="L2 regularize anchor drift: ||e_ref - e||^2, weighted by freq")
    parser.add_argument("--eta_floor", type=float, default=None,
                        help="training-only eta floor to avoid vanishing gradients (e.g., 0.1)")
    parser.add_argument("--pair_lambda", type=float, default=0.0,
                        help="pairwise violator loss weight (topK rank-aligned)")
    parser.add_argument("--logit_temp", type=float, default=1.0)
    parser.add_argument("--rhs_only", action="store_true", help="train only RHS branch")
    parser.add_argument("--train_gamma", type=float, default=1.0, help="delta scale during training")
    parser.add_argument("--viol_tau", type=float, default=0.2, help="temperature for violator loss")
    parser.add_argument("--lambda_safe", type=float, default=1.0, help="non-violator safety loss weight")
    parser.add_argument("--lambda_l2", type=float, default=1e-4, help="delta L2 weight")

    # 诊断打印频率
    parser.add_argument("--diag_every", type=int, default=1, help="print eta/g stats every N epochs")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    write_run_metadata(args.save_dir, args)

    processor = KGProcessor(args.data_path, max_neighbors=args.K)
    processor.load_files()

    processor.nbr_ent = processor.nbr_ent.to(device)
    processor.nbr_rel = processor.nbr_rel.to(device)
    processor.nbr_dir = processor.nbr_dir.to(device)
    processor.nbr_mask = processor.nbr_mask.to(device)
    processor.freq = processor.freq.to(device)

    # train-only 真值集合（用于 random filtered negatives）
    true_head_train, true_tail_train = build_true_head_tail_from_train(processor.train_triplets)

    struct_ckpt = resolve_struct_ckpt(args)
    if struct_ckpt is None:
        raise ValueError("Missing struct checkpoint: provide --pretrained_struct or --pretrained_rotate.")
    print(f"[Struct] type={args.struct_type} ckpt={struct_ckpt}")
    rotate_model = load_struct_backbone(
        struct_type=args.struct_type,
        num_entities=processor.num_entities,
        num_relations=processor.num_relations,
        emb_dim=args.emb_dim,
        margin=args.margin,
        ckpt_path=struct_ckpt,
        device=device,
    )
    for p in rotate_model.parameters():
        p.requires_grad = False

    print("Initializing StructRefiner...")
    refiner = StructRefiner(
        emb_dim=args.emb_dim,
        K=args.K,
        num_relations=processor.num_relations,
    ).to(device)
    optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(args.seed)

    if args.train_cache_rhs and args.train_cache_lhs:
        rhs_cache = torch.load(args.train_cache_rhs, map_location="cpu")
        lhs_cache = torch.load(args.train_cache_lhs, map_location="cpu")
        neg_t_full = (rhs_cache["neg_t"] if isinstance(rhs_cache, dict) else rhs_cache).to(torch.long)
        neg_h_full = (lhs_cache["neg_h"] if isinstance(lhs_cache, dict) else lhs_cache).to(torch.long)
        if neg_t_full.size(0) != processor.train_triplets.size(0) or neg_h_full.size(0) != processor.train_triplets.size(0):
            raise ValueError("Cache rows do not match train size.")
    else:
        neg_t_full = None
        neg_h_full = None

    train_loader = DataLoader(
        IndexDataset(processor.train_triplets.size(0)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0)
    )

    ent_ids_all = torch.arange(processor.num_entities, device=device)

    # ✅ valid 的 filter：train+valid（无 test）
    to_skip_valid = build_to_skip(processor, include_test=False)

    best_mrr = -1.0
    save_path = os.path.join(args.save_dir, "refiner_best.pth")

    print("Start Training Refiner (filtered negatives, train-only truths)...")
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        total_loss = 0.0
        p_pos_gt_maxneg = 0.0
        margin_sum = 0.0
        delta_std_sum = 0.0
        flip_sum = 0.0
        viol_rate_sum = 0.0
        fix_rate_sum = 0.0
        break_rate_sum = 0.0
        n_steps = 0

        for row_idx in train_loader:
            row_idx_cpu = row_idx
            batch = processor.train_triplets[row_idx_cpu].to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            B = h.size(0)

            # Hard negatives from cache (topK) + optional random filtered negatives
            if neg_t_full is not None and neg_h_full is not None:
                rhs_cand = neg_t_full[row_idx_cpu].to(device)
                lhs_cand = neg_h_full[row_idx_cpu].to(device)
                rhs_hard, _ = sample_from_topk(rhs_cand, args.hard_k, args.sample_prefix_k)
                lhs_hard, _ = sample_from_topk(lhs_cand, args.hard_k, args.sample_prefix_k)
            else:
                rhs_hard = torch.empty((B, 0), dtype=torch.long, device=device)
                lhs_hard = torch.empty((B, 0), dtype=torch.long, device=device)

            if args.rand_neg > 0:
                # random filtered negatives (train-only truths)
                rhs_rand = sample_filtered_negatives_batch(
                    anchor_ids=h,
                    rel_ids=r,
                    target_ids=t,
                    conj_flag=torch.zeros(B, dtype=torch.bool, device=device),
                    num_entities=processor.num_entities,
                    num_neg=args.rand_neg,
                    true_head_train=true_head_train,
                    true_tail_train=true_tail_train,
                    device=device,
                    oversample=args.neg_oversample
                )
                lhs_rand = sample_filtered_negatives_batch(
                    anchor_ids=t,
                    rel_ids=r,
                    target_ids=h,
                    conj_flag=torch.ones(B, dtype=torch.bool, device=device),
                    num_entities=processor.num_entities,
                    num_neg=args.rand_neg,
                    true_head_train=true_head_train,
                    true_tail_train=true_tail_train,
                    device=device,
                    oversample=args.neg_oversample
                )
            else:
                rhs_rand = torch.empty((B, 0), dtype=torch.long, device=device)
                lhs_rand = torch.empty((B, 0), dtype=torch.long, device=device)

            rhs_neg = torch.cat([rhs_hard, rhs_rand], dim=1)
            lhs_neg = torch.cat([lhs_hard, lhs_rand], dim=1)

            if args.rhs_only:
                anchor_ids = h
                rel_ids = r
                target_ids = t
                neg_ids = rhs_neg
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
            else:
                # 2B 混合：RHS + LHS
                anchor_ids = torch.cat([h, t], dim=0)  # [2B]
                rel_ids = torch.cat([r, r], dim=0)     # [2B]
                target_ids = torch.cat([t, h], dim=0)  # [2B]
                neg_ids = torch.cat([rhs_neg, lhs_neg], dim=0)  # [2B, Neg]
                conj_flag = torch.cat([
                    torch.zeros(B, dtype=torch.bool, device=device),  # RHS
                    torch.ones(B, dtype=torch.bool, device=device)    # LHS
                ], dim=0)
            # base structural scores (struct backbone)
            base_emb = rotate_model.entity_embedding[anchor_ids]
            s_neg_base = rotate_model.score_from_head_emb(base_emb, rel_ids, neg_ids, conj=conj_flag)
            s_pos_base = rotate_model.score_from_head_emb(
                base_emb, rel_ids, target_ids.unsqueeze(1), conj=conj_flag
            ).squeeze(1)

            # candidate-aware delta (refiner reranking)
            cand_ids = torch.cat([target_ids.unsqueeze(1), neg_ids], dim=1)  # [2B,1+Neg]
            delta_ref = refiner.score_delta_topk(
                anchor_ids=anchor_ids,
                rel_ids=rel_ids,
                cand_ids=cand_ids,
                dir_ids=conj_flag.long(),
                rotate_model=rotate_model,
                nbr_ent=processor.nbr_ent,
                nbr_rel=processor.nbr_rel,
                nbr_dir=processor.nbr_dir,
                nbr_mask=processor.nbr_mask,
                freq=processor.freq,
            )

            if args.train_gamma != 1.0:
                delta_ref = delta_ref * float(args.train_gamma)

            s_pos = s_pos_base + delta_ref[:, 0]
            s_neg = s_neg_base + delta_ref[:, 1:]

            viol = (s_neg_base > s_pos_base.unsqueeze(1))
            nonviol = ~viol

            tau = float(args.viol_tau)
            fix_loss = torch.nn.functional.softplus((s_neg - s_pos.unsqueeze(1)) / max(tau, 1e-6))
            if viol.any():
                L_fix = fix_loss[viol].mean()
            else:
                L_fix = s_pos.sum() * 0.0

            L_safe = torch.relu(s_neg - s_pos.unsqueeze(1))
            if nonviol.any():
                L_safe = L_safe[nonviol].mean()
            else:
                L_safe = s_pos.sum() * 0.0

            L_l2 = (delta_ref[:, 1:] ** 2).mean()

            loss = L_fix + float(args.lambda_safe) * L_safe + float(args.lambda_l2) * L_l2

            # pairwise violator loss (aligned with filtered rank)
            if args.pair_lambda and args.pair_lambda > 0:
                viol_w = (s_neg_base > s_pos_base.unsqueeze(1)).float()
                diff = s_neg - s_pos.unsqueeze(1)
                loss_pair = (viol_w * torch.nn.functional.softplus(diff)).sum() / (viol_w.sum() + 1e-6)
                loss = loss + args.pair_lambda * loss_pair

            # Anchor drift regularization (head stronger, tail weaker)
            if args.anchor_reg_lambda and args.anchor_reg_lambda > 0:
                anchor_ref = refiner.refine_anchor(
                    anchor_ids, rotate_model,
                    processor.nbr_ent, processor.nbr_rel, processor.nbr_dir, processor.nbr_mask, processor.freq,
                    eta_floor=args.eta_floor
                )
                with torch.no_grad():
                    f = processor.freq[anchor_ids].to(device)
                    logf = torch.log1p(f)
                    w = (logf - logf.min()) / (logf.max() - logf.min() + 1e-6)
                reg = ((anchor_ref - base_emb) ** 2).sum(dim=1)
                loss = loss + args.anchor_reg_lambda * (w * reg).mean()

            # ⭐ Gate regularization: keep (w,b) near init to prevent gate collapse
            if args.gate_reg_lambda and args.gate_reg_lambda > 0:
                w = F.softplus(refiner.w_raw)
                b = refiner.b
                gate_reg = (w - args.gate_w_target).pow(2) + (b - args.gate_b_target).pow(2)
                loss = loss + args.gate_reg_lambda * gate_reg

            # 可选：eta regularization（防止 eta 变得过大）
            if args.eta_reg_lambda and args.eta_reg_lambda > 0:
                eta = refiner.eta()
                loss = loss + args.eta_reg_lambda * (eta * eta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                max_neg = s_neg.max(dim=1).values
                p_pos_gt_maxneg += float((s_pos > max_neg).float().mean().item())
                margin_sum += float((s_pos - max_neg).mean().item())

                # delta variance + flip diagnostics (topK aligned)
                delta_std_sum += float(delta_ref[:, 1:].std(dim=1).mean().item())
                before = (s_neg_base > s_pos_base.unsqueeze(1))
                after = (s_neg > s_pos.unsqueeze(1))
                flip = ((before != after) & before).float().mean().item()
                flip_sum += float(flip)
                viol_rate_sum += float(before.any(dim=1).float().mean().item())
                fix = (before & (~after)).float().sum().item()
                brk = ((~before) & after).float().sum().item()
                fix_rate_sum += float(fix / (before.float().sum().item() + 1e-6))
                break_rate_sum += float(brk / ((~before).float().sum().item() + 1e-6))
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        p_pos = p_pos_gt_maxneg / max(n_steps, 1)
        margin = margin_sum / max(n_steps, 1)
        delta_std = delta_std_sum / max(n_steps, 1)
        flip_rate = flip_sum / max(n_steps, 1)
        viol_rate = viol_rate_sum / max(n_steps, 1)
        fix_rate = fix_rate_sum / max(n_steps, 1)
        break_rate = break_rate_sum / max(n_steps, 1)
        print(f"Epoch {epoch} | AvgLoss: {avg_loss:.6f} | p(pos>maxneg)={p_pos:.4f} | margin={margin:.4f}")
        print(f"[Diag] delta_std={delta_std:.4f} | flip_rate={flip_rate:.4f} | viol_rate={viol_rate:.4f} | "
              f"fix_rate={fix_rate:.4f} | break_rate={break_rate:.4f}")

        # 诊断：打印 eta / w / b / g 分布
        if args.diag_every and (epoch % args.diag_every == 0):
            with torch.no_grad():
                eta_val = refiner.eta().item()
                w_val = F.softplus(refiner.w_raw).item()
                b_val = refiner.b.item()
                print(f"[Refiner] eta={eta_val:.6f} | w={w_val:.6f} | b={b_val:.6f}")

                sample_ids = torch.randint(0, processor.num_entities, (512,), device=device)
                sample_freq = processor.freq[sample_ids]

                x = torch.log1p(sample_freq)
                w = F.softplus(refiner.w_raw)
                g_sample = torch.sigmoid(w * x + refiner.b)

                print(f"[Refiner] g: mean={g_sample.mean().item():.4f} "
                      f"min={g_sample.min().item():.4f} max={g_sample.max().item():.4f}")

                tail_mask = sample_freq <= sample_freq.median()
                if tail_mask.any():
                    g_tail = g_sample[tail_mask]
                    print(f"[Refiner] g_tail: mean={g_tail.mean().item():.4f} "
                          f"min={g_tail.min().item():.4f} max={g_tail.max().item():.4f}")

        if args.eval_every and (epoch % args.eval_every == 0):
            print("Evaluating on VALID (filtered train+valid, no test leakage)...")
            metrics = eval_refiner_struct_only(
                processor, rotate_model, refiner, ent_ids_all,
                to_skip_valid, processor.valid_triplets, device,
                batch_size=args.eval_batch_size,
                chunk_size=args.eval_chunk_size,
                num_workers=args.eval_num_workers
            )
            print(f"Valid MRR: {metrics['MRR']:.4f} | H@1: {metrics['Hits@1']:.4f} | H@10: {metrics['Hits@10']:.4f}")

            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                torch.save(refiner.state_dict(), save_path)
                print(f"Saved best checkpoint to {save_path} (Valid MRR={best_mrr:.4f})")

    print(f"Done. Best VALID MRR={best_mrr:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
