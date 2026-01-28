import argparse
import os
import random
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor
from models.struct_backbone_factory import load_struct_backbone, resolve_struct_ckpt
from models.struct_refiner_rank import RankStructRefiner
from tools.graph_stats import load_graph_stats
from tools.refiner_rank_features import (
    build_neighbor_features,
    build_query_stats,
    build_rank_features,
    compute_scale,
)
from tools.repro import setup_reproducibility
from tools.run_meta import write_run_metadata


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx


def _dedup_and_union(
    top_ids_raw: torch.Tensor,
    gold_id: int,
    cand_ids_full: torch.Tensor,
    cand_scores_full: torch.Tensor,
    topk: int,
    num_entities: int,
    rng: torch.Generator,
) -> List[int]:
    ids_raw = top_ids_raw.tolist()

    if gold_id not in ids_raw:
        ids_raw = ids_raw[:-1] + [gold_id]

    seen = set()
    out: List[int] = []
    for cid in ids_raw:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        if len(out) >= topk:
            return out

    # fill from full candidate list by score order
    order = cand_scores_full.argsort(descending=True).tolist()
    for idx in order:
        cid = int(cand_ids_full[idx].item())
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        if len(out) >= topk:
            return out

    # deterministic random fill (rare)
    while len(out) < topk:
        rid = int(torch.randint(0, num_entities, (1,), generator=rng).item())
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--struct_type", type=str, default="rotate", choices=["rotate", "complex"])
    ap.add_argument("--pretrained_struct", type=str, default=None,
                    help="struct backbone checkpoint (overrides --pretrained_rotate)")
    ap.add_argument("--pretrained_rotate", type=str, default=None,
                    help="legacy RotatE checkpoint (struct_type=rotate)")
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--emb_dim", type=int, default=1000)
    ap.add_argument("--margin", type=float, default=9.0, help="RotatE margin (ignored by ComplEx)")

    ap.add_argument("--train_cache_rhs", type=str, required=True)
    ap.add_argument("--train_cache_lhs", type=str, default=None)
    ap.add_argument("--train_sides", type=str, default="both",
                    choices=["rhs", "lhs", "both"],
                    help="train which sides (requires corresponding cache)")
    ap.add_argument("--cand_k", type=int, default=500, help="number of cached negatives to use per query")
    ap.add_argument("--topk", type=int, default=200, help="list size for Δ-X")
    ap.add_argument("--graph_stats_path", type=str, default=None)

    ap.add_argument("--rank_feat_dim", type=int, default=9)
    ap.add_argument("--nbr_feat_dim", type=int, default=9)
    ap.add_argument("--q_stat_dim", type=int, default=6)
    ap.add_argument("--rel_emb_dim", type=int, default=64)
    ap.add_argument("--dir_emb_dim", type=int, default=16)
    ap.add_argument("--model_dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--delta_cap", type=float, default=4.0)
    ap.add_argument("--max_pos", type=int, default=500)
    ap.add_argument("--use_hist", action="store_true", help="use rel-hist embeddings")
    ap.add_argument("--no_hist", dest="use_hist", action="store_false", help="disable rel-hist embeddings")
    ap.set_defaults(use_hist=True)

    ap.add_argument("--train_gamma", type=float, default=1.0,
                    help="deprecated; used when train_gamma_rhs/lhs not set")
    ap.add_argument("--train_gamma_rhs", type=float, default=None)
    ap.add_argument("--train_gamma_lhs", type=float, default=None)
    ap.add_argument("--w_lhs", type=float, default=1.0, help="lhs loss weight")
    ap.add_argument("--viol_tau", type=float, default=1.0)
    ap.add_argument("--lambda_safe", type=float, default=0.5)
    ap.add_argument("--lambda_l2", type=float, default=1e-2)
    ap.add_argument("--lambda_tv", type=float, default=0.02)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--disable_tf32", action="store_true")
    ap.add_argument("--matmul_precision", type=str, default="highest",
                    choices=["highest", "high", "medium"])
    args = ap.parse_args()

    setup_reproducibility(
        deterministic=args.deterministic,
        disable_tf32=args.disable_tf32,
        matmul_precision=args.matmul_precision,
        seed=args.seed,
        out_dir=args.save_dir,
        verbose=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    write_run_metadata(args.save_dir, args)

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    struct_ckpt = resolve_struct_ckpt(args)
    if struct_ckpt is None:
        raise ValueError("Missing struct checkpoint: provide --pretrained_struct or --pretrained_rotate.")
    print(f"[Struct] type={args.struct_type} ckpt={struct_ckpt}")
    struct_model = load_struct_backbone(
        struct_type=args.struct_type,
        num_entities=processor.num_entities,
        num_relations=processor.num_relations,
        emb_dim=args.emb_dim,
        margin=args.margin,
        ckpt_path=struct_ckpt,
        device=device,
    )
    for p in struct_model.parameters():
        p.requires_grad = False
    struct_model.eval()

    if args.graph_stats_path is None:
        args.graph_stats_path = os.path.join(args.data_path, "graph_stats_1hop.pt")
    if not os.path.exists(args.graph_stats_path):
        raise FileNotFoundError(
            f"graph_stats not found: {args.graph_stats_path}. Run build/build_graph_stats_1hop.py first."
        )
    graph_stats = load_graph_stats(args.graph_stats_path, device)
    dir_rhs = int(graph_stats.get("dir_rhs", 0))
    dir_lhs = int(graph_stats.get("dir_lhs", 1))

    if args.train_sides in ["rhs", "both"]:
        if args.train_cache_rhs is None:
            raise ValueError("train_cache_rhs required for rhs training.")
        rhs_cache = torch.load(args.train_cache_rhs, map_location="cpu")
        neg_t_full = (rhs_cache["neg_t"] if isinstance(rhs_cache, dict) else rhs_cache).to(torch.long)
        if neg_t_full.size(0) != processor.train_triplets.size(0):
            raise ValueError("RHS cache rows do not match train size.")
    else:
        neg_t_full = None

    if args.train_sides in ["lhs", "both"]:
        if args.train_cache_lhs is None:
            raise ValueError("train_cache_lhs required for lhs training.")
        lhs_cache = torch.load(args.train_cache_lhs, map_location="cpu")
        neg_h_full = (lhs_cache["neg_h"] if isinstance(lhs_cache, dict) else lhs_cache).to(torch.long)
        if neg_h_full.size(0) != processor.train_triplets.size(0):
            raise ValueError("LHS cache rows do not match train size.")
    else:
        neg_h_full = None

    train_loader = DataLoader(
        IndexDataset(processor.train_triplets.size(0)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=(args.num_workers > 0),
    )

    print("Initializing RankStructRefiner...")
    refiner = RankStructRefiner(
        num_relations=processor.num_relations,
        rank_feat_dim=args.rank_feat_dim,
        nbr_feat_dim=args.nbr_feat_dim,
        q_stat_dim=args.q_stat_dim,
        rel_emb_dim=args.rel_emb_dim,
        dir_emb_dim=args.dir_emb_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        delta_cap=args.delta_cap,
        max_pos=args.max_pos,
        use_hist=args.use_hist,
    ).to(device)

    optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, "refiner_rank_best.pth")

    train_gamma_rhs = float(args.train_gamma_rhs) if args.train_gamma_rhs is not None else float(args.train_gamma)
    train_gamma_lhs = float(args.train_gamma_lhs) if args.train_gamma_lhs is not None else float(args.train_gamma)
    print(f"[Train] sides={args.train_sides} gamma_rhs={train_gamma_rhs} gamma_lhs={train_gamma_lhs} w_lhs={args.w_lhs}")
    print("Start Training RankStructRefiner (Δ-X)...")
    for epoch in range(1, args.epochs + 1):
        refiner.train()
        total_loss = 0.0
        n_steps = 0

        for row_idx in train_loader:
            row_idx_cpu = row_idx
            batch = processor.train_triplets[row_idx_cpu].to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            B = h.size(0)

            def side_loss(anchor_ids, rel_ids, target_ids, neg_cache, score_fn, dir_val, train_gamma):
                cand_cache = neg_cache[row_idx_cpu][:, :args.cand_k].to(device)
                cand_ids = torch.cat([target_ids.unsqueeze(1), cand_cache], dim=1)
                if cand_ids.size(1) < args.topk:
                    raise ValueError(
                        f"cand_k too small: got {cand_ids.size(1)} < topk={args.topk}. Increase --cand_k."
                    )

                with torch.no_grad():
                    s_cand = score_fn(anchor_ids, rel_ids, cand_ids)
                    s_pos_base = s_cand[:, 0]

                    top_scores_raw, top_idx = torch.topk(s_cand, k=args.topk, dim=1)
                    top_ids_raw = cand_ids.gather(1, top_idx)

                    top_ids_union_list = []
                    for i in range(B):
                        ids = _dedup_and_union(
                            top_ids_raw[i].detach().cpu(),
                            int(target_ids[i].item()),
                            cand_ids[i].detach().cpu(),
                            s_cand[i].detach().cpu(),
                            args.topk,
                            processor.num_entities,
                            rng,
                        )
                        top_ids_union_list.append(ids)

                    top_ids_union = torch.tensor(top_ids_union_list, device=device, dtype=torch.long)
                    top_scores_union = score_fn(anchor_ids, rel_ids, top_ids_union)
                    top_scores_union, idx_union = torch.topk(top_scores_union, k=top_ids_union.size(1), dim=1)
                    top_ids_union = top_ids_union.gather(1, idx_union)

                scale = compute_scale(top_scores_raw)
                rank_feats = build_rank_features(top_scores_union, top_scores_raw)
                q_stats = build_query_stats(top_scores_union)
                nbr_feats, anchor_hist_ids, anchor_hist_counts, cand_hist_ids, cand_hist_counts = build_neighbor_features(
                    anchor_ids=anchor_ids,
                    cand_ids=top_ids_union,
                    rel_ids=rel_ids,
                    dir_val=dir_val,
                    graph_stats=graph_stats,
                )
                if rank_feats.size(-1) != args.rank_feat_dim:
                    raise ValueError(f"rank_feat_dim mismatch: got {rank_feats.size(-1)} != {args.rank_feat_dim}")
                if nbr_feats.size(-1) != args.nbr_feat_dim:
                    raise ValueError(f"nbr_feat_dim mismatch: got {nbr_feats.size(-1)} != {args.nbr_feat_dim}")
                if q_stats.size(-1) != args.q_stat_dim:
                    raise ValueError(f"q_stat_dim mismatch: got {q_stats.size(-1)} != {args.q_stat_dim}")

                dir_ids = torch.full((B,), dir_val, device=device, dtype=torch.long)
                delta_topk = refiner.score_delta_topk_rank(
                    rank_feats=rank_feats,
                    nbr_feats=nbr_feats,
                    q_stats=q_stats,
                    rel_ids=rel_ids,
                    dir_ids=dir_ids,
                    scale=scale,
                    anchor_hist_ids=anchor_hist_ids if args.use_hist else None,
                    anchor_hist_counts=anchor_hist_counts if args.use_hist else None,
                    cand_hist_ids=cand_hist_ids if args.use_hist else None,
                    cand_hist_counts=cand_hist_counts if args.use_hist else None,
                )

                pos_idx = (top_ids_union == target_ids.unsqueeze(1)).long().argmax(dim=1)
                delta_pos = delta_topk.gather(1, pos_idx.unsqueeze(1)).squeeze(1)

                mask = torch.ones_like(top_ids_union, dtype=torch.bool)
                mask.scatter_(1, pos_idx.unsqueeze(1), False)
                s_neg_base = top_scores_union[mask].view(B, -1)
                delta_neg = delta_topk[mask].view(B, -1)

                s_pos = s_pos_base + train_gamma * delta_pos
                s_neg = s_neg_base + train_gamma * delta_neg

                viol = (s_neg_base > s_pos_base.unsqueeze(1))
                nonviol = ~viol
                tau = float(args.viol_tau)
                fix_loss = torch.nn.functional.softplus((s_neg - s_pos.unsqueeze(1)) / max(tau, 1e-6))
                L_fix = fix_loss[viol].mean() if viol.any() else s_pos.sum() * 0.0

                L_safe = torch.relu(s_neg - s_pos.unsqueeze(1))
                L_safe = L_safe[nonviol].mean() if nonviol.any() else s_pos.sum() * 0.0

                L_l2 = (delta_neg ** 2).mean()
                L_tv = (delta_topk[:, 1:] - delta_topk[:, :-1]).abs().mean()

                return L_fix, L_safe, L_l2, L_tv

            loss = 0.0
            if args.train_sides in ["rhs", "both"]:
                L_fix, L_safe, L_l2, L_tv = side_loss(
                    anchor_ids=h,
                    rel_ids=r,
                    target_ids=t,
                    neg_cache=neg_t_full,
                    score_fn=struct_model.score_rhs_cands,
                    dir_val=dir_rhs,
                    train_gamma=train_gamma_rhs,
                )
                loss = loss + L_fix + float(args.lambda_safe) * L_safe + float(args.lambda_l2) * L_l2 + float(args.lambda_tv) * L_tv

            if args.train_sides in ["lhs", "both"]:
                L_fix, L_safe, L_l2, L_tv = side_loss(
                    anchor_ids=t,
                    rel_ids=r,
                    target_ids=h,
                    neg_cache=neg_h_full,
                    score_fn=struct_model.score_lhs_cands,
                    dir_val=dir_lhs,
                    train_gamma=train_gamma_lhs,
                )
                loss_lhs = L_fix + float(args.lambda_safe) * L_safe + float(args.lambda_l2) * L_l2 + float(args.lambda_tv) * L_tv
                loss = loss + float(args.w_lhs) * loss_lhs

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            n_steps += 1

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Epoch {epoch} | loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "state_dict": refiner.state_dict(),
                    "config": {
                        "num_relations": processor.num_relations,
                        "rank_feat_dim": args.rank_feat_dim,
                        "nbr_feat_dim": args.nbr_feat_dim,
                        "q_stat_dim": args.q_stat_dim,
                        "rel_emb_dim": args.rel_emb_dim,
                        "dir_emb_dim": args.dir_emb_dim,
                        "model_dim": args.model_dim,
                        "n_layers": args.n_layers,
                        "n_heads": args.n_heads,
                        "dropout": args.dropout,
                        "delta_cap": args.delta_cap,
                        "max_pos": args.max_pos,
                        "use_hist": args.use_hist,
                    },
                },
                save_path,
            )
            print(f"[IO] Saved best to {save_path} (loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
