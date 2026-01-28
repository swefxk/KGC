import argparse
import json
import os
import sys
import bisect
from typing import Dict, List, Tuple

import torch

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor
from models.struct_backbone_factory import load_struct_backbone
from models.gate_injector import ConfidenceGate
from models.struct_refiner_rank import RankStructRefiner
from tools.graph_stats import load_graph_stats
from tools.refiner_rank_features import (
    build_rank_features,
    build_query_stats,
    build_neighbor_features,
    compute_scale,
)
from tools.repro import setup_reproducibility
from eval.eval_full_entity_filtered import build_to_skip, load_embeddings
from eval.eval_topk_inject import (
    sem_encode_query_rhs,
    sem_encode_query_lhs,
    sem_score_with_q0,
    load_sem_model,
)


def _sorted_filters(
    to_skip_map: Dict[Tuple[int, int], set],
    keys: List[Tuple[int, int]],
) -> List[List[int]]:
    out = []
    for key in keys:
        filt = to_skip_map.get(key, set())
        out.append(sorted(list(filt)))
    return out


def _mask_filtered_chunk(
    filt_sorted: List[List[int]],
    start: int,
    end: int,
    gold_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    rows, cols = [], []
    for i, filt in enumerate(filt_sorted):
        if not filt:
            continue
        low = bisect.bisect_left(filt, start)
        high = bisect.bisect_left(filt, end)
        gold_val = gold_ids[i]
        for idx in range(low, high):
            fid = filt[idx]
            if fid == gold_val:
                continue
            rows.append(i)
            cols.append(fid - start)
    if rows:
        mask = torch.zeros((len(filt_sorted), end - start), device=device, dtype=torch.bool)
        mask[rows, cols] = True
        return mask
    return torch.zeros((len(filt_sorted), end - start), device=device, dtype=torch.bool)


@torch.no_grad()
def _check_side(
    side: str,
    batch: torch.Tensor,
    processor: KGProcessor,
    struct_model,
    sem_model,
    ent_embs: torch.Tensor,
    rel_embs: torch.Tensor,
    gate_model,
    refiner_rank: RankStructRefiner,
    graph_stats: Dict,
    to_skip: Dict,
    all_ent_ids: torch.Tensor,
    topk: int,
    chunk_size: int,
    b_scale: float,
    refiner_gamma: float,
    struct_weight: float,
    gate_ent_temp: float,
    score_eps: float,
    max_abs_score_diff: float,
    mismatch_examples: List[Dict],
    mismatch_limit: int,
) -> Tuple[Dict, float, List[Dict]]:
    device = batch.device
    h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
    B = h.size(0)

    if side == "rhs":
        anchor = h
        gold = t
        conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
        to_skip_map = to_skip["rhs"]
        dir_val = int(graph_stats.get("dir_rhs", 0))
        dir_ids = torch.zeros_like(h)
        q0 = sem_encode_query_rhs(sem_model, ent_embs, rel_embs, h, r)
    else:
        anchor = t
        gold = h
        conj_flag = torch.ones(B, dtype=torch.bool, device=device)
        to_skip_map = to_skip["lhs"]
        dir_val = int(graph_stats.get("dir_lhs", 1))
        dir_ids = torch.ones_like(t)
        q0 = sem_encode_query_lhs(sem_model, ent_embs, rel_embs, t, r)

    h_cpu = h.tolist()
    t_cpu = t.tolist()
    r_cpu = r.tolist()
    gold_cpu = gold.tolist()
    key_list = [(h_cpu[i], r_cpu[i]) if side == "rhs" else (t_cpu[i], r_cpu[i]) for i in range(B)]
    filt_sorted = _sorted_filters(to_skip_map, key_list)

    anchor_emb = struct_model.entity_embedding[anchor]
    s_gold_struct = struct_model.score_from_head_emb(anchor_emb, r, gold.unsqueeze(1), conj=conj_flag).squeeze(1)
    s_gold_sem = sem_score_with_q0(sem_model, ent_embs, q0, gold.unsqueeze(1)).squeeze(1)

    # topK selection (struct only, filtered + gold excluded)
    top_scores = torch.full((B, topk), -1e9, device=device)
    top_ids = torch.full((B, topk), -1, device=device, dtype=torch.long)
    for start in range(0, processor.num_entities, chunk_size):
        end = min(start + chunk_size, processor.num_entities)
        cand_1d = all_ent_ids[start:end]
        s_chunk = struct_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
        s_for_topk = s_chunk.clone()
        mask = _mask_filtered_chunk(filt_sorted, start, end, gold_cpu, device)
        if mask.any():
            s_for_topk = s_for_topk.masked_fill(mask, -1e9)
        # mask gold
        for i in range(B):
            gt = gold_cpu[i]
            if start <= gt < end:
                s_for_topk[i, gt - start] = -1e9
        merged_scores = torch.cat([top_scores, s_for_topk], dim=1)
        merged_ids = torch.cat([top_ids, cand_1d.unsqueeze(0).expand(B, -1)], dim=1)
        top_scores, idx = torch.topk(merged_scores, k=topk, dim=1)
        top_ids = torch.gather(merged_ids, 1, idx)

    # check top_scores are pure struct
    struct_topk_check = struct_model.score_from_head_emb(anchor_emb, r, top_ids, conj=conj_flag)
    max_abs_score_diff = max(max_abs_score_diff, float((struct_topk_check - top_scores).abs().max().item()))

    # sem + gate
    sem_topk = sem_score_with_q0(sem_model, ent_embs, q0, top_ids)
    g = gate_model(top_scores, r, dir_ids, ent_temp=gate_ent_temp).to(s_gold_struct.dtype)
    b = torch.full((B,), float(b_scale), device=device, dtype=s_gold_struct.dtype)
    sem_scale_total = b * g
    s_thresh = struct_weight * s_gold_struct + sem_scale_total * s_gold_sem

    # refiner_rank delta
    rank_feats = build_rank_features(top_scores, top_scores)
    q_stats = build_query_stats(top_scores)
    nbr_feats, anchor_hist_ids, anchor_hist_counts, cand_hist_ids, cand_hist_counts = build_neighbor_features(
        anchor_ids=anchor,
        cand_ids=top_ids,
        rel_ids=r,
        dir_val=dir_val,
        graph_stats=graph_stats,
    )
    scale = compute_scale(top_scores)
    delta_ref_topk = refiner_rank.score_delta_topk_rank(
        rank_feats=rank_feats,
        nbr_feats=nbr_feats,
        q_stats=q_stats,
        rel_ids=r,
        dir_ids=torch.full((B,), dir_val, device=device, dtype=torch.long),
        scale=scale,
        anchor_hist_ids=anchor_hist_ids,
        anchor_hist_counts=anchor_hist_counts,
        cand_hist_ids=cand_hist_ids,
        cand_hist_counts=cand_hist_counts,
    )
    delta_ref_topk = delta_ref_topk * float(refiner_gamma)

    total_topk = struct_weight * top_scores + delta_ref_topk + sem_scale_total.unsqueeze(1) * sem_topk

    # filtered entities inside topK?
    filt_in_topk_max = 0
    for i in range(B):
        filt = to_skip_map.get(key_list[i], set())
        bad = [eid for eid in top_ids[i].tolist() if (eid in filt and eid != gold_cpu[i])]
        filt_in_topk_max = max(filt_in_topk_max, len(bad))

    # pass2: greater counts (struct vs total)
    greater_struct = torch.zeros(B, device=device, dtype=torch.long)
    greater_brute = torch.zeros(B, device=device, dtype=torch.long)
    for start in range(0, processor.num_entities, chunk_size):
        end = min(start + chunk_size, processor.num_entities)
        cand_1d = all_ent_ids[start:end]
        s_chunk_struct = struct_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
        s_chunk_struct = struct_weight * s_chunk_struct

        mask = _mask_filtered_chunk(filt_sorted, start, end, gold_cpu, device)
        comp_struct = s_chunk_struct > (s_thresh.unsqueeze(1) + score_eps)
        greater_struct += (comp_struct & (~mask)).sum(dim=1)

        # brute: replace topK scores inside this chunk
        s_chunk_total = s_chunk_struct.clone()
        in_chunk = (top_ids >= start) & (top_ids < end)
        if in_chunk.any():
            row_idx, col_idx = torch.where(in_chunk)
            ent_idx = top_ids[row_idx, col_idx] - start
            s_chunk_total[row_idx, ent_idx] = total_topk[row_idx, col_idx]
        comp_total = s_chunk_total > (s_thresh.unsqueeze(1) + score_eps)
        greater_brute += (comp_total & (~mask)).sum(dim=1)

    greater_struct_topk = (struct_weight * top_scores > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)
    greater_total_topk = (total_topk > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)
    rank_script = greater_struct - greater_struct_topk + greater_total_topk + 1
    rank_brute = greater_brute + 1

    diff = (rank_script - rank_brute).abs()
    mismatch_mask = diff != 0
    mismatch_count = int(mismatch_mask.sum().item())
    max_rank_diff = int(diff.max().item()) if diff.numel() > 0 else 0

    if mismatch_count > 0 and len(mismatch_examples) < mismatch_limit:
        idxs = torch.where(mismatch_mask)[0].tolist()
        for i in idxs:
            if len(mismatch_examples) >= mismatch_limit:
                break
            mismatch_examples.append(
                {
                    "h": int(h[i].item()),
                    "r": int(r[i].item()),
                    "t": int(t[i].item()),
                    "rank_script": int(rank_script[i].item()),
                    "rank_brute": int(rank_brute[i].item()),
                }
            )

    stats = {
        "mismatch_count": mismatch_count,
        "max_rank_diff": max_rank_diff,
        "max_abs_score_diff": max_abs_score_diff,
        "filt_in_topk_max": filt_in_topk_max,
    }
    return stats, max_abs_score_diff, mismatch_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    ap.add_argument("--struct_type", type=str, default="rotate", choices=["rotate", "complex"])
    ap.add_argument("--pretrained_struct", type=str, required=True)
    ap.add_argument("--emb_dim", type=int, required=True)
    ap.add_argument("--pretrained_sem", type=str, required=True)
    ap.add_argument("--pretrained_gate", type=str, required=True)
    ap.add_argument("--pretrained_refiner_rank", type=str, required=True)
    ap.add_argument("--graph_stats_path", type=str, default=None)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--b_rhs", type=float, default=2.0)
    ap.add_argument("--b_lhs", type=float, default=2.5)
    ap.add_argument("--refiner_gamma_rhs", type=float, default=2.0)
    ap.add_argument("--refiner_gamma_lhs", type=float, default=1.0)
    ap.add_argument("--struct_weight_rhs", type=float, default=1.0)
    ap.add_argument("--struct_weight_lhs", type=float, default=1.0)
    ap.add_argument("--gate_rel_dim", type=int, default=16)
    ap.add_argument("--gate_dir_dim", type=int, default=8)
    ap.add_argument("--gate_hidden_dim", type=int, default=64)
    ap.add_argument("--gate_g_min", type=float, default=0.0)
    ap.add_argument("--gate_g_max", type=float, default=2.0)
    ap.add_argument("--gate_init_bias", type=float, default=0.5413)
    ap.add_argument("--gate_ent_temp", type=float, default=1.0)
    ap.add_argument("--score_eps", type=float, default=0.0)
    ap.add_argument("--mismatch_limit", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--disable_tf32", action="store_true")
    ap.add_argument("--matmul_precision", type=str, default="highest",
                    choices=["highest", "high", "medium"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    setup_reproducibility(
        deterministic=args.deterministic,
        disable_tf32=args.disable_tf32,
        matmul_precision=args.matmul_precision,
        seed=args.seed,
        out_dir=None,
        verbose=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    struct_model = load_struct_backbone(
        struct_type=args.struct_type,
        num_entities=processor.num_entities,
        num_relations=processor.num_relations,
        emb_dim=args.emb_dim,
        margin=9.0,
        ckpt_path=args.pretrained_struct,
        device=device,
    )
    struct_model.eval()

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    sem_model = load_sem_model(args.pretrained_sem, ent_embs.size(1), processor.num_relations, device)
    sem_model.eval()
    ent_embs = ent_embs.to(device)
    rel_embs = rel_embs.to(device)

    gate_model = ConfidenceGate(
        num_relations=processor.num_relations,
        rel_emb_dim=args.gate_rel_dim,
        dir_emb_dim=args.gate_dir_dim,
        hidden_dim=args.gate_hidden_dim,
        g_min=args.gate_g_min,
        g_max=args.gate_g_max,
        init_bias=args.gate_init_bias,
    ).to(device)
    gate_ckpt = torch.load(args.pretrained_gate, map_location="cpu")
    if isinstance(gate_ckpt, dict) and "state_dict" in gate_ckpt:
        gate_ckpt = gate_ckpt["state_dict"]
    gate_model.load_state_dict(gate_ckpt, strict=True)
    gate_model.eval()

    rank_ckpt = torch.load(args.pretrained_refiner_rank, map_location="cpu")
    refiner_cfg = rank_ckpt["config"]
    refiner_rank = RankStructRefiner(**refiner_cfg).to(device)
    refiner_rank.load_state_dict(rank_ckpt["state_dict"], strict=True)
    refiner_rank.eval()

    if args.graph_stats_path is None:
        args.graph_stats_path = os.path.join(args.data_path, "graph_stats_1hop.pt")
    graph_stats = load_graph_stats(args.graph_stats_path, device)

    split_triplets = processor.test_triplets if args.eval_split == "test" else processor.valid_triplets
    to_skip = build_to_skip(processor, split=args.eval_split)

    all_ent_ids = torch.arange(processor.num_entities, device=device, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        split_triplets,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    for rep in range(1, args.repeat + 1):
        rhs_summary = {"mismatch_count": 0, "max_rank_diff": 0, "max_abs_score_diff": 0.0, "filt_in_topk_max": 0}
        lhs_summary = {"mismatch_count": 0, "max_rank_diff": 0, "max_abs_score_diff": 0.0, "filt_in_topk_max": 0}
        rhs_examples: List[Dict] = []
        lhs_examples: List[Dict] = []

        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            rhs_stats, rhs_max_abs, rhs_examples = _check_side(
                side="rhs",
                batch=batch,
                processor=processor,
                struct_model=struct_model,
                sem_model=sem_model,
                ent_embs=ent_embs,
                rel_embs=rel_embs,
                gate_model=gate_model,
                refiner_rank=refiner_rank,
                graph_stats=graph_stats,
                to_skip=to_skip,
                all_ent_ids=all_ent_ids,
                topk=args.topk,
                chunk_size=args.chunk_size,
                b_scale=args.b_rhs,
                refiner_gamma=args.refiner_gamma_rhs,
                struct_weight=args.struct_weight_rhs,
                gate_ent_temp=args.gate_ent_temp,
                score_eps=args.score_eps,
                max_abs_score_diff=rhs_summary["max_abs_score_diff"],
                mismatch_examples=rhs_examples,
                mismatch_limit=args.mismatch_limit,
            )
            lhs_stats, lhs_max_abs, lhs_examples = _check_side(
                side="lhs",
                batch=batch,
                processor=processor,
                struct_model=struct_model,
                sem_model=sem_model,
                ent_embs=ent_embs,
                rel_embs=rel_embs,
                gate_model=gate_model,
                refiner_rank=refiner_rank,
                graph_stats=graph_stats,
                to_skip=to_skip,
                all_ent_ids=all_ent_ids,
                topk=args.topk,
                chunk_size=args.chunk_size,
                b_scale=args.b_lhs,
                refiner_gamma=args.refiner_gamma_lhs,
                struct_weight=args.struct_weight_lhs,
                gate_ent_temp=args.gate_ent_temp,
                score_eps=args.score_eps,
                max_abs_score_diff=lhs_summary["max_abs_score_diff"],
                mismatch_examples=lhs_examples,
                mismatch_limit=args.mismatch_limit,
            )
            # aggregate
            for k in ["mismatch_count", "filt_in_topk_max"]:
                rhs_summary[k] += rhs_stats[k]
                lhs_summary[k] += lhs_stats[k]
            rhs_summary["max_rank_diff"] = max(rhs_summary["max_rank_diff"], rhs_stats["max_rank_diff"])
            lhs_summary["max_rank_diff"] = max(lhs_summary["max_rank_diff"], lhs_stats["max_rank_diff"])
            rhs_summary["max_abs_score_diff"] = max(rhs_summary["max_abs_score_diff"], rhs_stats["max_abs_score_diff"])
            lhs_summary["max_abs_score_diff"] = max(lhs_summary["max_abs_score_diff"], lhs_stats["max_abs_score_diff"])

        print(f"\\n=== Repeat {rep}/{args.repeat} ===")
        print(f"[RHS] {rhs_summary}")
        if rhs_examples:
            print(f"[RHS] mismatch_examples (first {len(rhs_examples)}): {rhs_examples}")
        print(f"[LHS] {lhs_summary}")
        if lhs_examples:
            print(f"[LHS] mismatch_examples (first {len(lhs_examples)}): {lhs_examples}")


if __name__ == "__main__":
    main()
