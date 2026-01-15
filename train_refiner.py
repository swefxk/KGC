import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from models.struct_refiner import StructRefiner


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
    true_head[(t,r)] = {h}   # key 约定与 test_semres 的 lhs 一致：(t,r)->{h}
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
    parser.add_argument("--pretrained_rotate", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/struct_refiner")

    parser.add_argument("--emb_dim", type=int, default=500)
    parser.add_argument("--margin", type=float, default=9.0)

    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--num_neg", type=int, default=64)
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

    # 诊断打印频率
    parser.add_argument("--diag_every", type=int, default=1, help="print eta/g stats every N epochs")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    processor = KGProcessor(args.data_path, max_neighbors=args.K)
    processor.load_files()

    processor.nbr_ent = processor.nbr_ent.to(device)
    processor.nbr_rel = processor.nbr_rel.to(device)
    processor.nbr_dir = processor.nbr_dir.to(device)
    processor.nbr_mask = processor.nbr_mask.to(device)
    processor.freq = processor.freq.to(device)

    # train-only 真值集合（用于 filtered negative sampling）
    true_head_train, true_tail_train = build_true_head_tail_from_train(processor.train_triplets)

    print(f"Loading frozen RotatE from {args.pretrained_rotate}")
    rotate_model = RotatEModel(processor.num_entities, processor.num_relations, args.emb_dim, args.margin).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    for p in rotate_model.parameters():
        p.requires_grad = False
    rotate_model.eval()

    print("Initializing StructRefiner...")
    refiner = StructRefiner(emb_dim=args.emb_dim, K=args.K).to(device)
    optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        TrainDataset(processor.train_triplets),
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
        n_steps = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            B = h.size(0)

            # 2B 混合：RHS + LHS
            anchor_ids = torch.cat([h, t], dim=0)  # [2B]
            rel_ids = torch.cat([r, r], dim=0)     # [2B]
            target_ids = torch.cat([t, h], dim=0)  # [2B]

            conj_flag = torch.cat([
                torch.zeros(B, dtype=torch.bool, device=device),  # RHS
                torch.ones(B, dtype=torch.bool, device=device)    # LHS
            ], dim=0)

            anchor_ref = refiner.refine_anchor(
                anchor_ids, rotate_model,
                processor.nbr_ent, processor.nbr_rel, processor.nbr_dir, processor.nbr_mask, processor.freq
            )

            # ✅ filtered negatives（train-only）
            neg_ids = sample_filtered_negatives_batch(
                anchor_ids=anchor_ids,
                rel_ids=rel_ids,
                target_ids=target_ids,
                conj_flag=conj_flag,
                num_entities=processor.num_entities,
                num_neg=args.num_neg,
                true_head_train=true_head_train,
                true_tail_train=true_tail_train,
                device=device,
                oversample=args.neg_oversample
            )

            s_pos = rotate_model.score_from_head_emb(
                anchor_ref, rel_ids, target_ids.unsqueeze(1), conj=conj_flag
            ).squeeze(1)
            s_neg = rotate_model.score_from_head_emb(anchor_ref, rel_ids, neg_ids, conj=conj_flag)

            logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)  # [2B, 1+Neg]
            labels = torch.zeros(2 * B, dtype=torch.long, device=device)

            loss = ce_loss(logits, labels)

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
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch} | AvgLoss: {avg_loss:.6f}")

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
