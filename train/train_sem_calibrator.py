import argparse
import json
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.experimental.semantic_confidence import SemanticConfidenceNet
from models.struct_backbone_base import StructBackboneBase
from models.struct_backbone_factory import load_struct_backbone, resolve_struct_ckpt
from eval.eval_full_entity_filtered import load_embeddings
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


class IndexedTripletDataset(Dataset):
    def __init__(self, triplets: torch.LongTensor):
        self.triplets = triplets

    def __len__(self):
        return self.triplets.size(0)

    def __getitem__(self, idx: int):
        return self.triplets[idx], idx


def indexed_collate_fn(batch):
    triples = torch.stack([x[0] for x in batch], dim=0)
    idxs = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return triples, idxs


def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved args to {path}")


@torch.no_grad()
def sem_encode_query(sem_model, ent_embs, rel_embs, anchor_ids, rel_ids, dir_ids):
    a_txt = ent_embs[anchor_ids]
    r_txt = rel_embs[rel_ids]
    return sem_model.encode_query(a_txt, r_txt, dir_ids=dir_ids)


@torch.no_grad()
def sem_score_with_q0(sem_model, ent_embs, q0, cand_ids):
    B, K = cand_ids.shape
    v = sem_model.encode_entity(ent_embs[cand_ids.reshape(-1)]).view(B, K, -1)
    s = torch.einsum("bd,bkd->bk", q0, v)
    return s


def sample_from_cache(cache_cpu: torch.Tensor, row_idx_cpu: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.empty((row_idx_cpu.size(0), 0), dtype=torch.long)
    if k >= cache_cpu.size(1):
        return cache_cpu[row_idx_cpu]
    return cache_cpu[row_idx_cpu, :k]


@torch.no_grad()
def build_struct_scores(rotate_model, anchor_ids, rel_ids, cand_ids, conj_flag):
    anchor_emb = rotate_model.entity_embedding[anchor_ids]
    return rotate_model.score_from_head_emb(anchor_emb, rel_ids, cand_ids, conj=conj_flag)


def make_union_topk(
    rotate_model,
    anchor_ids,
    rel_ids,
    gold_ids,
    cand_ids,
    conj_flag,
):
    s_struct = build_struct_scores(rotate_model, anchor_ids, rel_ids, cand_ids, conj_flag)
    s_gold = build_struct_scores(
        rotate_model, anchor_ids, rel_ids, gold_ids.unsqueeze(1), conj_flag
    ).squeeze(1)

    cand = cand_ids.clone()
    scores = s_struct.clone()
    in_topk = (cand == gold_ids.unsqueeze(1)).any(dim=1)
    if (~in_topk).any():
        cand[~in_topk, -1] = gold_ids[~in_topk]
        scores[~in_topk, -1] = s_gold[~in_topk]

    scores_sorted, idx = torch.topk(scores, k=cand.size(1), dim=1)
    cand_sorted = torch.gather(cand, 1, idx)
    gold_index = (cand_sorted == gold_ids.unsqueeze(1)).long().argmax(dim=1)
    return cand_sorted, scores_sorted, gold_index


def rank_in_topk(scores, gold_index, eps: float):
    s_gold = scores.gather(1, gold_index.unsqueeze(1)).squeeze(1)
    return 1 + (scores > (s_gold.unsqueeze(1) + eps)).sum(dim=1)


def compute_dir_loss(
    scn: SemanticConfidenceNet,
    sem_model: SemanticBiEncoderScorer,
    rotate_model: StructBackboneBase,
    ent_embs: torch.Tensor,
    rel_embs: torch.Tensor,
    anchor_ids: torch.Tensor,
    rel_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    cand_ids: torch.Tensor,
    dir_ids: torch.Tensor,
    b_scale: float,
    struct_weight: float,
    delta_rank_margin: int,
    agree_topm: int,
    sem_temp: float,
    eps: float,
    r_reg_target: float,
    r_reg_lambda: float,
) -> tuple[torch.Tensor | None, dict]:
    device = anchor_ids.device
    conj_flag = dir_ids.bool()

    cand_ids, s_struct, gold_index = make_union_topk(
        rotate_model, anchor_ids, rel_ids, gold_ids, cand_ids, conj_flag
    )
    q0 = sem_encode_query(sem_model, ent_embs, rel_embs, anchor_ids, rel_ids, dir_ids)
    s_sem = sem_score_with_q0(sem_model, ent_embs, q0, cand_ids)

    r = scn(s_sem, s_struct, rel_ids, dir_ids, topm=agree_topm, temp=sem_temp)

    score_struct = float(struct_weight) * s_struct
    score_sem = score_struct + float(b_scale) * r.unsqueeze(1) * s_sem

    rank_struct = rank_in_topk(score_struct, gold_index, eps=eps)
    rank_sem = rank_in_topk(score_sem, gold_index, eps=eps)
    delta_rank = rank_struct - rank_sem

    pos = delta_rank >= delta_rank_margin
    neg = delta_rank <= -delta_rank_margin
    mask = pos | neg
    stats = {
        "pos": int(pos.sum().item()),
        "neg": int(neg.sum().item()),
        "ignore": int((~mask).sum().item()),
    }
    if not mask.any():
        return None, stats

    y = pos[mask].float()
    r_sel = r[mask]
    pos_count = int(pos[mask].sum().item())
    neg_count = int(neg[mask].sum().item())
    pos_weight = neg_count / max(pos_count, 1)
    w = torch.ones_like(y, device=device)
    w[y > 0] = pos_weight
    loss = F.binary_cross_entropy(r_sel, y, weight=w)

    if r_reg_lambda > 0:
        reg = (r.mean() - float(r_reg_target)) ** 2
        loss = loss + float(r_reg_lambda) * reg

    return loss, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--struct_type", type=str, default="rotate", choices=["rotate", "complex"])
    ap.add_argument("--pretrained_struct", type=str, default=None,
                    help="struct backbone checkpoint (overrides --pretrained_rotate)")
    ap.add_argument("--pretrained_rotate", type=str, default=None,
                    help="legacy RotatE checkpoint (struct_type=rotate)")
    ap.add_argument("--pretrained_sem", type=str, required=True)
    ap.add_argument("--train_cache_rhs", type=str, required=True)
    ap.add_argument("--train_cache_lhs", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--emb_dim", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=9.0, help="RotatE margin (ignored by ComplEx)")
    ap.add_argument("--topk", type=int, default=200)

    ap.add_argument("--b_rhs", type=float, default=2.0)
    ap.add_argument("--b_lhs", type=float, default=2.5)
    ap.add_argument("--struct_w_rhs", type=float, default=1.0)
    ap.add_argument("--struct_w_lhs", type=float, default=1.0)
    ap.add_argument("--delta_rank_margin", type=int, default=1)
    ap.add_argument("--agree_topm", type=int, default=10)
    ap.add_argument("--sem_temp", type=float, default=1.0)
    ap.add_argument("--score_eps", type=float, default=0.0)

    ap.add_argument("--rel_emb_dim", type=int, default=32)
    ap.add_argument("--dir_emb_dim", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--r_min", type=float, default=0.05)
    ap.add_argument("--r_max", type=float, default=0.95)

    ap.add_argument("--r_reg_target", type=float, default=0.5)
    ap.add_argument("--r_reg_lambda", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--rhs_only", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    write_run_metadata(args.save_dir, args)
    save_args(args, args.save_dir)

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rhs_cache = torch.load(args.train_cache_rhs, map_location="cpu")
    lhs_cache = torch.load(args.train_cache_lhs, map_location="cpu")
    cache_neg_t = (rhs_cache["neg_t"] if isinstance(rhs_cache, dict) else rhs_cache).to(torch.long)
    cache_neg_h = (lhs_cache["neg_h"] if isinstance(lhs_cache, dict) else lhs_cache).to(torch.long)
    if args.topk > cache_neg_t.size(1) or args.topk > cache_neg_h.size(1):
        raise ValueError("--topk exceeds cache K")

    struct_ckpt = resolve_struct_ckpt(args)
    if struct_ckpt is None:
        raise ValueError("Missing struct checkpoint: provide --pretrained_struct or --pretrained_rotate.")
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

    sem_ckpt = torch.load(args.pretrained_sem, map_location="cpu")
    if not isinstance(sem_ckpt, dict) or sem_ckpt.get("model_type", "") != "biencoder":
        raise ValueError("Expected a biencoder checkpoint with model_type='biencoder'.")
    cfg = sem_ckpt["model_args"]
    sem_model = SemanticBiEncoderScorer(
        text_dim=cfg["text_dim"],
        num_relations=cfg["num_relations"],
        proj_dim=cfg.get("proj_dim", 256),
        dropout=cfg.get("dropout", 0.1),
        text_norm=cfg.get("text_norm", True),
    ).to(device)
    sem_model.load_state_dict(sem_ckpt["state_dict"], strict=False)
    sem_model.eval()
    for p in sem_model.parameters():
        p.requires_grad = False

    ent_embs, rel_embs = load_embeddings(processor, args, device=torch.device("cpu"))
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    scn = SemanticConfidenceNet(
        num_relations=processor.num_relations,
        rel_emb_dim=args.rel_emb_dim,
        dir_emb_dim=args.dir_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        r_min=args.r_min,
        r_max=args.r_max,
    ).to(device)

    opt = optim.AdamW(scn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        IndexedTripletDataset(processor.train_triplets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        collate_fn=indexed_collate_fn,
    )

    best_loss = float("inf")
    best_path = os.path.join(args.save_dir, "scn_best.pth")
    last_path = os.path.join(args.save_dir, "scn_last.pth")

    for epoch in range(1, args.epochs + 1):
        scn.train()
        total_loss = 0.0
        n_steps = 0
        pos_sum = neg_sum = ign_sum = 0

        for triples, row_idx in train_loader:
            triples = triples.to(device, non_blocking=True)
            row_idx_cpu = row_idx.cpu()
            h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

            rhs_ids = sample_from_cache(cache_neg_t, row_idx_cpu, args.topk).to(device, non_blocking=True)
            lhs_ids = sample_from_cache(cache_neg_h, row_idx_cpu, args.topk).to(device, non_blocking=True)

            loss_rhs, stats_rhs = compute_dir_loss(
                scn=scn,
                sem_model=sem_model,
                rotate_model=rotate_model,
                ent_embs=ent_embs,
                rel_embs=rel_embs,
                anchor_ids=h,
                rel_ids=r,
                gold_ids=t,
                cand_ids=rhs_ids,
                dir_ids=torch.zeros_like(h),
                b_scale=args.b_rhs,
                struct_weight=args.struct_w_rhs,
                delta_rank_margin=args.delta_rank_margin,
                agree_topm=args.agree_topm,
                sem_temp=args.sem_temp,
                eps=args.score_eps,
                r_reg_target=args.r_reg_target,
                r_reg_lambda=args.r_reg_lambda,
            )
            pos_sum += stats_rhs["pos"]
            neg_sum += stats_rhs["neg"]
            ign_sum += stats_rhs["ignore"]

            if args.rhs_only:
                loss = loss_rhs
            else:
                loss_lhs, stats_lhs = compute_dir_loss(
                    scn=scn,
                    sem_model=sem_model,
                    rotate_model=rotate_model,
                    ent_embs=ent_embs,
                    rel_embs=rel_embs,
                    anchor_ids=t,
                    rel_ids=r,
                    gold_ids=h,
                    cand_ids=lhs_ids,
                    dir_ids=torch.ones_like(h),
                    b_scale=args.b_lhs,
                    struct_weight=args.struct_w_lhs,
                    delta_rank_margin=args.delta_rank_margin,
                    agree_topm=args.agree_topm,
                    sem_temp=args.sem_temp,
                    eps=args.score_eps,
                    r_reg_target=args.r_reg_target,
                    r_reg_lambda=args.r_reg_lambda,
                )
                pos_sum += stats_lhs["pos"]
                neg_sum += stats_lhs["neg"]
                ign_sum += stats_lhs["ignore"]
                if loss_rhs is None:
                    loss = loss_lhs
                elif loss_lhs is None:
                    loss = loss_rhs
                else:
                    loss = 0.5 * (loss_rhs + loss_lhs)

            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(scn.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.item())
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        total = max(pos_sum + neg_sum + ign_sum, 1)
        print(f"Epoch {epoch} | AvgLoss: {avg_loss:.6f} | pos={pos_sum/total:.3f} neg={neg_sum/total:.3f} ignore={ign_sum/total:.3f}")

        ckpt = {
            "model_type": "scn",
            "model_args": {
                "num_relations": int(processor.num_relations),
                "rel_emb_dim": int(args.rel_emb_dim),
                "dir_emb_dim": int(args.dir_emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "r_min": float(args.r_min),
                "r_max": float(args.r_max),
            },
            "state_dict": scn.state_dict(),
            "train_args": vars(args),
            "best_train_loss": float(avg_loss),
        }
        torch.save(ckpt, last_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, best_path)
            print(f"Saved best to {best_path} (best_loss={best_loss:.6f})")

    print(f"Done. Best train loss: {best_loss:.6f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
