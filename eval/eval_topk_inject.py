import argparse
import json
import os
from datetime import datetime
import time
import os
import sys
import bisect
import torch
from torch.utils.data import DataLoader
#2026-01-17 
sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from eval.eval_full_entity_filtered import (
    build_to_skip,
    load_embeddings,
    get_freq_buckets,
    eval_chunked_bidirectional,
)

# biencoder
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.struct_refiner import StructRefiner
from models.gate_injector import ConfidenceGate
from models.gate_injector import gate_features_from_top_scores
from models.semantic_confidence import SemanticConfidenceNet
from tools.run_meta import write_run_metadata
from tools.repro import setup_reproducibility


def get_relation_buckets(processor, device):
    counts = torch.zeros(processor.num_relations, device=device, dtype=torch.long)
    if processor.train_triplets is not None and len(processor.train_triplets) > 0:
        r_ids = processor.train_triplets[:, 1].to(device)
        counts.scatter_add_(0, r_ids, torch.ones_like(r_ids, dtype=torch.long))
    q1 = torch.quantile(counts.float(), 1 / 3).item()
    q2 = torch.quantile(counts.float(), 2 / 3).item()
    bucket = torch.zeros_like(counts, dtype=torch.long)
    bucket[counts > q1] = 1
    bucket[counts > q2] = 2
    return bucket


def load_calibrator(calib_path, num_relations, device, calib_b_max=None):
    obj = torch.load(calib_path, map_location="cpu")
    per_relation = bool(obj["per_relation"])
    b_max = float(obj["b_max"])
    if calib_b_max is not None:
        b_max = float(calib_b_max)
    sd = obj["state_dict"]

    if per_relation:
        w = sd["w.weight"].to(device)  # [R,1]

        def get_b(r_ids):
            x = w[r_ids].squeeze(-1)
            return b_max * torch.sigmoid(x)
    else:
        w = sd["w"].to(device)  # scalar

        def get_b(r_ids):
            return (b_max * torch.sigmoid(w)).expand_as(r_ids)

    return get_b, b_max, per_relation


@torch.no_grad()
def sem_encode_query_rhs(sem_model, ent_embs, rel_embs, h, r):
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    return sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))


def sem_encode_query_lhs(sem_model, ent_embs, rel_embs, t, r):
    t_txt = ent_embs[t]
    r_txt = rel_embs[r]
    return sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))


def sem_score_with_q0(sem_model, ent_embs, q0, cand_ids):
    B, K = cand_ids.shape
    v = sem_model.encode_entity(ent_embs[cand_ids.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q0, v)


def sem_score_rhs_biencoder(sem_model, ent_embs, rel_embs, h, r, cand_t_2d):
    """
    h,r: [B]
    cand_t_2d: [B,K]
    return sem scores [B,K]
    """
    B, K = cand_t_2d.shape
    q0 = sem_encode_query_rhs(sem_model, ent_embs, rel_embs, h, r)
    return sem_score_with_q0(sem_model, ent_embs, q0, cand_t_2d)


@torch.no_grad()
def sem_score_rhs_biencoder_pos(sem_model, ent_embs, rel_embs, h, r, t):
    q0 = sem_encode_query_rhs(sem_model, ent_embs, rel_embs, h, r)
    return sem_score_with_q0(sem_model, ent_embs, q0, t.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def sem_score_lhs_biencoder(sem_model, ent_embs, rel_embs, t, r, cand_h_2d):
    """
    t,r: [B]
    cand_h_2d: [B,K]
    return sem scores [B,K]
    """
    B, K = cand_h_2d.shape
    q0 = sem_encode_query_lhs(sem_model, ent_embs, rel_embs, t, r)
    return sem_score_with_q0(sem_model, ent_embs, q0, cand_h_2d)


@torch.no_grad()
def sem_score_lhs_biencoder_pos(sem_model, ent_embs, rel_embs, t, r, h):
    q0 = sem_encode_query_lhs(sem_model, ent_embs, rel_embs, t, r)
    return sem_score_with_q0(sem_model, ent_embs, q0, h.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def eval_rhs_topk_inject(
    processor,
    rotate_model,
    sem_model,
    ent_text_embs,
    rel_text_embs,
    refiner,
    nbr_ent,
    nbr_rel,
    nbr_dir,
    nbr_mask,
    freq,
    gate_model,
    gate_ent_temp,
    loader,
    to_skip,
    split,
    device,
    topk=500,
    chunk_size=2048,
    b_scale=0.0,
    get_b_fn=None,
    refiner_gamma=1.0,
    gamma_by_rel=None,
    gold_struct_threshold=False,
    gold_struct_threshold_no_sem=False,
    delta_gate_m12=None,
    delta_gate_ent=None,
    delta_gate_ent_q=None,
    refiner_viol_only=False,
    refiner_topm=0,
    score_eps=0.0,
    collect_ranks=False,
    collect_r=False,
    profile_time=False,
    refiner_diag=False,
    struct_weight=1.0,
    scn_model=None,
    scn_topm=10,
    scn_temp=1.0,
    scn_in_thresh=True,
    scn_force_r=None,
):
    rotate_model.eval()
    use_sem = sem_model is not None
    if use_sem:
        sem_model.eval()
        ent_text_embs = ent_text_embs.to(device, non_blocking=True)
        rel_text_embs = rel_text_embs.to(device, non_blocking=True)

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device, dtype=torch.long)

    bucket_map = get_freq_buckets(processor, device)
    bucket_names = {0: "Tail", 1: "Torso", 2: "Head"}

    stats = {k: {"mrr": 0.0, "h1": 0, "h3": 0, "h10": 0, "n": 0} for k in ["total", 0, 1, 2]}
    diag = {"delta_rank_sum": 0.0, "p_improve": 0.0, "p_hurt": 0.0,
            "fix_num": 0.0, "fix_den": 0.0, "break_num": 0.0, "break_den": 0.0,
            "delta_gold_sum": 0.0, "gate_on_sum": 0.0, "gate_on_den": 0.0,
            "rec_num": 0.0, "rec_den": 0.0,
            "rec_num_struct": 0.0, "rec_den_struct": 0.0, "n": 0,
            "flip_num": 0.0, "flip_den": 0.0,
            "p_up_num": 0.0, "p_up_den": 0.0,
            "p_pair_sum": 0.0, "p_pair_den": 0.0}
    ranks_all = [] if collect_ranks else None
    r_all = [] if collect_r else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        use_refiner_rerank = refiner is not None and hasattr(refiner, "score_delta_topk")

        struct_w = float(struct_weight)
        # gold total (align scoring path with full-entity eval)
        conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
        if refiner is None or use_refiner_rerank:
            anchor_emb = rotate_model.entity_embedding[h]
        else:
            anchor_emb = refiner.refine_anchor(
                h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
        s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, t.unsqueeze(1), conj=conj_flag).squeeze(1)
        q0 = None
        if use_sem:
            if profile_time:
                t0 = time.time()
            q0 = sem_encode_query_rhs(sem_model, ent_text_embs, rel_text_embs, h, r)
            s_gold_sem = sem_score_with_q0(sem_model, ent_text_embs, q0, t.unsqueeze(1)).squeeze(1)  # [B]
            if profile_time:
                time_stats["sem"] += time.time() - t0
        else:
            s_gold_sem = torch.zeros_like(s_gold_struct)
        if get_b_fn is None:
            b = torch.full((B,), float(b_scale), device=device, dtype=s_gold_struct.dtype)
        else:
            b = get_b_fn(r).to(device, non_blocking=True).to(s_gold_struct.dtype)

        # precompute sorted filters
        rhs_filters_sorted = []
        for i in range(B):
            filt = to_skip["rhs"].get((h_cpu[i], r_cpu[i]), set())
            rhs_filters_sorted.append(sorted(list(filt)))

        # maintain topK by struct (excluding gold + filtered)
        top_scores = torch.full((B, topk), -1e9, device=device)
        top_ids = torch.full((B, topk), -1, device=device, dtype=torch.long)

        # pass 1: build topK by struct only
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            if refiner is None or use_refiner_rerank:
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            # --- topK maintenance: exclude filtered + exclude gold ---
            s_for_topk = s_chunk.clone()

            # mask filtered (including gold for safety)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = rhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low = bisect.bisect_left(filt_sorted, start)
                high = bisect.bisect_left(filt_sorted, end)
                gold_val = t_cpu[i]
                for idx in range(low, high):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask = torch.zeros((B, cand_1d.size(0)), device=device, dtype=torch.bool)
                mask[rows, cols] = True
                s_for_topk = s_for_topk.masked_fill(mask, -1e9)

            # mask gold itself
            # gold in this chunk?
            for i in range(B):
                gt = t_cpu[i]
                if start <= gt < end:
                    s_for_topk[i, gt - start] = -1e9

            # merge
            cand_2d = cand_1d.unsqueeze(0).expand(B, -1)
            merged_scores = torch.cat([top_scores, s_for_topk], dim=1)
            merged_ids = torch.cat([top_ids, cand_2d], dim=1)
            top_scores, idx = torch.topk(merged_scores, k=topk, dim=1)
            top_ids = torch.gather(merged_ids, 1, idx)
        if profile_time:
            time_stats["struct_topk"] += time.time() - t0

        # recall@K (gold in candidate set)
        rec_hit = (s_gold_struct >= top_scores[:, -1]).float()
        diag["rec_num"] += float(rec_hit.sum().item())
        diag["rec_den"] += float(B)

        # gate and gold total (consistent with topK)
        dir_ids = torch.zeros_like(h)
        if gate_model is None:
            g = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        else:
            g = gate_model(top_scores, r, dir_ids, ent_temp=gate_ent_temp).to(s_gold_struct.dtype)

        if use_refiner_rerank:
            if profile_time:
                t0 = time.time()
            delta_ref_topk = refiner.score_delta_topk(
                anchor_ids=h,
                rel_ids=r,
                cand_ids=top_ids,
                dir_ids=dir_ids,
                rotate_model=rotate_model,
                nbr_ent=nbr_ent, nbr_rel=nbr_rel, nbr_dir=nbr_dir, nbr_mask=nbr_mask, freq=freq,
            )
            delta_ref_gold = refiner.score_delta_topk(
                anchor_ids=h,
                rel_ids=r,
                cand_ids=t.unsqueeze(1),
                dir_ids=dir_ids,
                rotate_model=rotate_model,
                nbr_ent=nbr_ent, nbr_rel=nbr_rel, nbr_dir=nbr_dir, nbr_mask=nbr_mask, freq=freq,
            ).squeeze(1)
            if profile_time:
                time_stats["refiner"] += time.time() - t0
            if gamma_by_rel is not None:
                gamma_eff = gamma_by_rel[r].to(delta_ref_topk.dtype)
            else:
                gamma_eff = torch.full((B,), float(refiner_gamma), device=device, dtype=delta_ref_topk.dtype)
            delta_ref_topk = delta_ref_topk * gamma_eff.unsqueeze(1)
            if torch.is_tensor(delta_ref_gold):
                delta_ref_gold = delta_ref_gold * gamma_eff
        else:
            delta_ref_topk = 0.0
            delta_ref_gold = 0.0

        if refiner_diag and use_refiner_rerank and torch.is_tensor(delta_ref_topk):
            if torch.is_tensor(delta_ref_gold):
                diag["p_up_num"] += float((delta_ref_gold > 0).float().sum().item())
                diag["p_up_den"] += float(B)
                s_gold_ref = s_gold_struct + delta_ref_gold
            else:
                s_gold_ref = s_gold_struct
            s_neg_ref = top_scores + delta_ref_topk
            s_neg_struct = top_scores
            margin_ref = s_gold_ref.unsqueeze(1) - s_neg_ref
            margin_base = s_gold_struct.unsqueeze(1) - s_neg_struct
            diag["p_pair_sum"] += float((margin_ref > margin_base).float().mean(dim=1).sum().item())
            diag["p_pair_den"] += float(B)
            new_idx = (s_neg_ref).argmax(dim=1)
            diag["flip_num"] += float((new_idx != 0).float().sum().item())
            diag["flip_den"] += float(B)

        if refiner_diag and use_refiner_rerank and torch.is_tensor(delta_ref_topk):
            if torch.is_tensor(delta_ref_gold):
                diag["p_up_num"] += float((delta_ref_gold > 0).float().sum().item())
                diag["p_up_den"] += float(B)
                s_gold_ref = s_gold_struct + delta_ref_gold
            else:
                s_gold_ref = s_gold_struct
            s_neg_ref = top_scores + delta_ref_topk
            s_neg_struct = top_scores
            margin_ref = s_gold_ref.unsqueeze(1) - s_neg_ref
            margin_base = s_gold_struct.unsqueeze(1) - s_neg_struct
            diag["p_pair_sum"] += float((margin_ref > margin_base).float().mean(dim=1).sum().item())
            diag["p_pair_den"] += float(B)
            new_idx = (s_neg_ref).argmax(dim=1)
            diag["flip_num"] += float((new_idx != 0).float().sum().item())
            diag["flip_den"] += float(B)

        # optional masks for candidate-aware delta
        if use_refiner_rerank and refiner_viol_only:
            viol_mask = (struct_w * top_scores > (struct_w * s_gold_struct).unsqueeze(1))
            delta_ref_topk = delta_ref_topk * viol_mask.to(delta_ref_topk.dtype)
        if use_refiner_rerank and refiner_topm and refiner_topm > 0:
            m = min(refiner_topm, top_scores.size(1))
            keep = torch.arange(top_scores.size(1), device=device).unsqueeze(0) < m
            delta_ref_topk = delta_ref_topk * keep.to(delta_ref_topk.dtype)

        # optional delta gate by confidence stats (m12/entropy)
        if use_refiner_rerank and (delta_gate_m12 is not None or delta_gate_ent is not None or delta_gate_ent_q is not None):
            feats = gate_features_from_top_scores(top_scores, ent_temp=gate_ent_temp)
            m12 = feats[:, 0]
            ent = feats[:, 4]
            keep = torch.ones(B, device=device, dtype=torch.bool)
            if delta_gate_m12 is not None:
                keep = keep & (m12 <= float(delta_gate_m12))
            if delta_gate_ent_q is not None:
                thr = torch.quantile(ent, float(delta_gate_ent_q))
                keep = keep & (ent >= thr)
            elif delta_gate_ent is not None:
                keep = keep & (ent >= float(delta_gate_ent))
            delta_ref_topk = delta_ref_topk * keep.unsqueeze(1).to(delta_ref_topk.dtype)
            if torch.is_tensor(delta_ref_gold):
                delta_ref_gold = delta_ref_gold * keep.to(delta_ref_gold.dtype)
            diag["gate_on_sum"] += float(keep.float().sum().item())
            diag["gate_on_den"] += float(B)

        # optional delta gate by confidence stats (m12/entropy)
        if use_refiner_rerank and (delta_gate_m12 is not None or delta_gate_ent is not None or delta_gate_ent_q is not None):
            feats = gate_features_from_top_scores(top_scores, ent_temp=gate_ent_temp)
            m12 = feats[:, 0]
            ent = feats[:, 4]
            keep = torch.ones(B, device=device, dtype=torch.bool)
            if delta_gate_m12 is not None:
                keep = keep & (m12 <= float(delta_gate_m12))
            if delta_gate_ent_q is not None:
                thr = torch.quantile(ent, float(delta_gate_ent_q))
                keep = keep & (ent >= thr)
            elif delta_gate_ent is not None:
                keep = keep & (ent >= float(delta_gate_ent))
            delta_ref_topk = delta_ref_topk * keep.unsqueeze(1).to(delta_ref_topk.dtype)
            if torch.is_tensor(delta_ref_gold):
                delta_ref_gold = delta_ref_gold * keep.to(delta_ref_gold.dtype)
            diag["gate_on_sum"] += float(keep.float().sum().item())
            diag["gate_on_den"] += float(B)

        # compute sem_topk + SCN r before threshold so s_thresh stays consistent
        if use_sem:
            if profile_time:
                t0 = time.time()
            if q0 is None:
                q0 = sem_encode_query_rhs(sem_model, ent_text_embs, rel_text_embs, h, r)
            sem_topk = sem_score_with_q0(sem_model, ent_text_embs, q0, top_ids)
            if scn_model is None:
                r_sem = torch.ones(B, device=device, dtype=sem_topk.dtype)
            else:
                r_sem = scn_model(sem_topk, top_scores, r, torch.zeros_like(h), topm=scn_topm, temp=scn_temp)
            if scn_force_r is not None:
                r_sem = torch.full_like(r_sem, float(scn_force_r))
            if profile_time:
                time_stats["sem"] += time.time() - t0
        else:
            sem_topk = None
            r_sem = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        if collect_r:
            r_all.append(r_sem.detach().cpu())

        # ---- threshold (must exclude delta, must keep sem) ----
        sem_scale_total = b * g * r_sem
        sem_scale_thresh = b * g * (r_sem if scn_in_thresh else 1.0)
        s_thresh = struct_w * s_gold_struct + sem_scale_thresh * s_gold_sem
        if gold_struct_threshold:
            if gold_struct_threshold_no_sem:
                s_thresh = struct_w * s_gold_struct
            else:
                s_thresh = struct_w * s_gold_struct + sem_scale_thresh * s_gold_sem

        # pass 2: count greater over all entities with threshold s_thresh
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None or use_refiner_rerank:
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
            s_chunk = struct_w * s_chunk_struct

            comp = s_chunk > (s_thresh.unsqueeze(1) + score_eps)
            mask = torch.zeros_like(comp, dtype=torch.bool)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = rhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low = bisect.bisect_left(filt_sorted, start)
                high = bisect.bisect_left(filt_sorted, end)
                gold_val = t_cpu[i]
                for idx in range(low, high):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask[rows, cols] = True
            greater_struct += (comp & (~mask)).sum(dim=1)
        if profile_time:
            time_stats["pass2"] += time.time() - t0

        # correction inside topK: replace struct comparison with total comparison
        if use_sem:
            total_topk = struct_w * top_scores + delta_ref_topk + sem_scale_total.unsqueeze(1) * sem_topk
        else:
            total_topk = struct_w * top_scores + delta_ref_topk

        greater_struct_topk = (struct_w * top_scores > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)
        greater_total_topk = (total_topk > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)

        # diagnostics: topK-only effects
        delta_rank_topk = (greater_total_topk - greater_struct_topk).float()
        diag["delta_rank_sum"] += float(delta_rank_topk.mean().item()) * B
        diag["p_improve"] += float((delta_rank_topk < 0).float().mean().item()) * B
        diag["p_hurt"] += float((delta_rank_topk > 0).float().mean().item()) * B

        viol = top_scores > (s_thresh.unsqueeze(1) + score_eps)
        fixed = viol & (total_topk <= (s_thresh.unsqueeze(1) + score_eps))
        broken = (~viol) & (total_topk > (s_thresh.unsqueeze(1) + score_eps))
        diag["fix_num"] += float(fixed.sum().item())
        diag["fix_den"] += float(viol.sum().item())
        diag["break_num"] += float(broken.sum().item())
        diag["break_den"] += float((~viol).sum().item())
        if torch.is_tensor(delta_ref_gold):
            diag["delta_gold_sum"] += float(delta_ref_gold.mean().item()) * B
        diag["n"] += int(B)

        greater = greater_struct - greater_struct_topk + greater_total_topk
        rank = greater + 1  # [B]
        if collect_ranks:
            ranks_all.append(rank.detach().cpu())

        mrr = (1.0 / rank.float())
        h1 = (rank <= 1)
        h3 = (rank <= 3)
        h10 = (rank <= 10)

        stats["total"]["mrr"] += mrr.sum().item()
        stats["total"]["h1"] += h1.sum().item()
        stats["total"]["h3"] += h3.sum().item()
        stats["total"]["h10"] += h10.sum().item()
        stats["total"]["n"] += int(B)

        buckets = bucket_map[t]
        for bid in [0, 1, 2]:
            m = (buckets == bid)
            if m.any():
                stats[bid]["mrr"] += mrr[m].sum().item()
                stats[bid]["h1"] += h1[m].sum().item()
                stats[bid]["h3"] += h3[m].sum().item()
                stats[bid]["h10"] += h10[m].sum().item()
                stats[bid]["n"] += int(m.sum().item())

    # report
    def safe_div(a, b): return a / max(1, b)
    print("\n==================== RHS TopK-Injection (Exact Rank) ====================")
    for key in [0, 1, 2]:
        n = stats[key]["n"]
        name = bucket_names[key]
        print(f"RHS {name:<5} | MRR={safe_div(stats[key]['mrr'], n):.4f} | "
              f"H@1={safe_div(stats[key]['h1'], n):.4f} | "
              f"H@3={safe_div(stats[key]['h3'], n):.4f} | "
              f"H@10={safe_div(stats[key]['h10'], n):.4f} | n={n}")
    n = stats["total"]["n"]
    print(f"RHS TOTAL | MRR={safe_div(stats['total']['mrr'], n):.4f} | "
          f"H@1={safe_div(stats['total']['h1'], n):.4f} | "
          f"H@3={safe_div(stats['total']['h3'], n):.4f} | "
          f"H@10={safe_div(stats['total']['h10'], n):.4f} | n={n}")
    print("==========================================================================\n")

    if profile_time and time_stats:
        total = sum(time_stats.values())
        if total > 0:
            print(f"[Time][RHS] total={total:.2f}s "
                  f"struct_topk={time_stats['struct_topk']:.2f}s "
                  f"sem={time_stats['sem']:.2f}s "
                  f"refiner={time_stats['refiner']:.2f}s "
                  f"pass2={time_stats['pass2']:.2f}s "
                  f"sem_ratio={time_stats['sem']/total:.2%}")

    if collect_ranks:
        r_values = torch.cat(r_all, dim=0) if collect_r else None
        return stats, torch.cat(ranks_all, dim=0), diag, r_values
    if diag["rec_den"] > 0:
        print(f"[Recall@K][RHS] recall@{topk}={diag['rec_num'] / max(diag['rec_den'], 1e-6):.4f}")
    if diag["gate_on_den"] > 0:
        print(f"[DeltaGate][RHS] on_rate={diag['gate_on_sum'] / max(diag['gate_on_den'], 1e-6):.4f}")
    if refiner_diag and diag["flip_den"] > 0:
        print(f"[RefinerDiag][RHS] flip@K={diag['flip_num'] / max(diag['flip_den'], 1e-6):.4f} "
              f"p_up={diag['p_up_num'] / max(diag['p_up_den'], 1e-6):.4f} "
              f"p_pair={diag['p_pair_sum'] / max(diag['p_pair_den'], 1e-6):.4f}")
    r_values = torch.cat(r_all, dim=0) if collect_r else None
    return stats, None, diag, r_values


@torch.no_grad()
def eval_lhs_topk_inject(
    processor,
    rotate_model,
    sem_model,
    ent_text_embs,
    rel_text_embs,
    refiner,
    nbr_ent,
    nbr_rel,
    nbr_dir,
    nbr_mask,
    freq,
    gate_model,
    gate_ent_temp,
    loader,
    to_skip,
    split,
    device,
    topk=500,
    chunk_size=2048,
    b_scale=0.0,
    get_b_fn=None,
    refiner_gamma=1.0,
    gamma_by_rel=None,
    gold_struct_threshold=False,
    gold_struct_threshold_no_sem=False,
    delta_gate_m12=None,
    delta_gate_ent=None,
    delta_gate_ent_q=None,
    refiner_viol_only=False,
    refiner_topm=0,
    score_eps=0.0,
    collect_ranks=False,
    collect_r=False,
    profile_time=False,
    refiner_diag=False,
    sem_union_topk=0,
    struct_weight=1.0,
    scn_model=None,
    scn_topm=10,
    scn_temp=1.0,
    scn_in_thresh=True,
    scn_force_r=None,
):
    rotate_model.eval()
    use_sem = sem_model is not None
    if sem_union_topk and sem_union_topk > 0 and not use_sem:
        raise ValueError("sem_union_topk requires sem_model")
    if use_sem:
        sem_model.eval()
        ent_text_embs = ent_text_embs.to(device, non_blocking=True)
        rel_text_embs = rel_text_embs.to(device, non_blocking=True)

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device, dtype=torch.long)

    bucket_map = get_freq_buckets(processor, device)
    bucket_names = {0: "Tail", 1: "Torso", 2: "Head"}

    stats = {k: {"mrr": 0.0, "h1": 0, "h3": 0, "h10": 0, "n": 0} for k in ["total", 0, 1, 2]}
    diag = {"delta_rank_sum": 0.0, "p_improve": 0.0, "p_hurt": 0.0,
            "fix_num": 0.0, "fix_den": 0.0, "break_num": 0.0, "break_den": 0.0,
            "delta_gold_sum": 0.0, "gate_on_sum": 0.0, "gate_on_den": 0.0,
            "rec_num": 0.0, "rec_den": 0.0,
            "rec_num_struct": 0.0, "rec_den_struct": 0.0, "n": 0,
            "flip_num": 0.0, "flip_den": 0.0,
            "p_up_num": 0.0, "p_up_den": 0.0,
            "p_pair_sum": 0.0, "p_pair_den": 0.0}
    ranks_all = [] if collect_ranks else None
    r_all = [] if collect_r else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        use_refiner_rerank = refiner is not None and hasattr(refiner, "score_delta_topk")

        struct_w = float(struct_weight)
        # gold total (align scoring path with full-entity eval)
        conj_flag = torch.ones(B, dtype=torch.bool, device=device)
        if refiner is None or use_refiner_rerank:
            anchor_emb = rotate_model.entity_embedding[t]
        else:
            anchor_emb = refiner.refine_anchor(
                t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
        s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, h.unsqueeze(1), conj=conj_flag).squeeze(1)

        q0 = None
        if use_sem:
            if profile_time:
                t0 = time.time()
            q0 = sem_encode_query_lhs(sem_model, ent_text_embs, rel_text_embs, t, r)
            s_gold_sem = sem_score_with_q0(sem_model, ent_text_embs, q0, h.unsqueeze(1)).squeeze(1)  # [B]
            if profile_time:
                time_stats["sem"] += time.time() - t0
        else:
            s_gold_sem = torch.zeros_like(s_gold_struct)
        if get_b_fn is None:
            b = torch.full((B,), float(b_scale), device=device, dtype=s_gold_struct.dtype)
        else:
            b = get_b_fn(r).to(device, non_blocking=True).to(s_gold_struct.dtype)

        # precompute sorted filters
        lhs_filters_sorted = []
        for i in range(B):
            filt = to_skip["lhs"].get((t_cpu[i], r_cpu[i]), set())
            lhs_filters_sorted.append(sorted(list(filt)))

        # maintain topK by struct (excluding gold + filtered)
        top_scores = torch.full((B, topk), -1e9, device=device)
        top_ids = torch.full((B, topk), -1, device=device, dtype=torch.long)
        sem_top_scores = None
        sem_top_ids = None
        q_sem = None
        if sem_union_topk and sem_union_topk > 0:
            sem_top_scores = torch.full((B, sem_union_topk), -1e9, device=device)
            sem_top_ids = torch.full((B, sem_union_topk), -1, device=device, dtype=torch.long)
            t_txt = ent_text_embs[t]
            r_txt = rel_text_embs[r]
            q_sem = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))

        # pass 1: build topK by struct only (and optional sem topK for union)
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            if refiner is None or use_refiner_rerank:
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            # --- topK maintenance: exclude filtered + exclude gold ---
            s_for_topk = s_chunk_struct.clone()

            # mask filtered (including gold for safety)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = lhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low = bisect.bisect_left(filt_sorted, start)
                high = bisect.bisect_left(filt_sorted, end)
                gold_val = h_cpu[i]
                for idx in range(low, high):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask = torch.zeros((B, cand_1d.size(0)), device=device, dtype=torch.bool)
                mask[rows, cols] = True
                s_for_topk = s_for_topk.masked_fill(mask, -1e9)

            # mask gold itself
            for i in range(B):
                gt = h_cpu[i]
                if start <= gt < end:
                    s_for_topk[i, gt - start] = -1e9

            # merge
            cand_2d = cand_1d.unsqueeze(0).expand(B, -1)
            merged_scores = torch.cat([top_scores, s_for_topk], dim=1)
            merged_ids = torch.cat([top_ids, cand_2d], dim=1)
            top_scores, idx = torch.topk(merged_scores, k=topk, dim=1)
            top_ids = torch.gather(merged_ids, 1, idx)

            if sem_union_topk and sem_union_topk > 0:
                if profile_time:
                    t_sem = time.time()
                cand_txt = ent_text_embs[cand_1d]
                cand_vec = sem_model.encode_entity(cand_txt)
                sem_chunk = torch.matmul(q_sem, cand_vec.t())
                if profile_time:
                    time_stats["sem"] += time.time() - t_sem

                sem_for_topk = sem_chunk.clone()
                # mask filtered + exclude gold
                rows, cols = [], []
                for i in range(B):
                    filt_sorted = lhs_filters_sorted[i]
                    if not filt_sorted:
                        continue
                    low = bisect.bisect_left(filt_sorted, start)
                    high = bisect.bisect_left(filt_sorted, end)
                    gold_val = h_cpu[i]
                    for idx in range(low, high):
                        fid = filt_sorted[idx]
                        if fid == gold_val:
                            continue
                        rows.append(i)
                        cols.append(fid - start)
                if rows:
                    mask = torch.zeros((B, cand_1d.size(0)), device=device, dtype=torch.bool)
                    mask[rows, cols] = True
                    sem_for_topk = sem_for_topk.masked_fill(mask, -1e9)
                for i in range(B):
                    gt = h_cpu[i]
                    if start <= gt < end:
                        sem_for_topk[i, gt - start] = -1e9

                merged_scores = torch.cat([sem_top_scores, sem_for_topk], dim=1)
                merged_ids = torch.cat([sem_top_ids, cand_2d], dim=1)
                sem_top_scores, idx = torch.topk(merged_scores, k=sem_union_topk, dim=1)
                sem_top_ids = torch.gather(merged_ids, 1, idx)
        if profile_time:
            time_stats["struct_topk"] += time.time() - t0

        if sem_union_topk and sem_union_topk > 0:
            # union candidates: rotate topK ∪ sem topK
            union_k = topk + sem_union_topk
            union_ids = torch.full((B, union_k), -1, device=device, dtype=torch.long)
            for i in range(B):
                seen = set()
                out = []
                for cid in top_ids[i].tolist() + sem_top_ids[i].tolist():
                    if cid < 0 or cid in seen:
                        continue
                    seen.add(cid)
                    out.append(cid)
                    if len(out) >= union_k:
                        break
                if out:
                    union_ids[i, :len(out)] = torch.tensor(out, device=device, dtype=torch.long)

            # recompute struct scores on union candidates
            struct_union = torch.full((B, union_k), -1e9, device=device)
            for i in range(B):
                ids = union_ids[i]
                valid = ids >= 0
                if not valid.any():
                    continue
                cand = ids[valid]
                if refiner is None or use_refiner_rerank:
                    s = rotate_model.score_head_batch(r[i:i+1], t[i:i+1], cand).squeeze(0)
                else:
                    anchor_emb = refiner.refine_anchor(
                        t[i:i+1], rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                    )
                    conj_flag = torch.ones(1, dtype=torch.bool, device=device)
                    s = rotate_model.score_from_head_emb(anchor_emb, r[i:i+1], cand, conj=conj_flag).squeeze(0)
                struct_union[i, valid] = s

            top_scores = struct_union
            top_ids = union_ids

        # recall@K (gold in candidate set)
        if sem_union_topk and sem_union_topk > 0:
            rec_hit_struct = (top_ids[:, :topk] == h.unsqueeze(1)).any(dim=1).float()
            rec_hit_union = (top_ids == h.unsqueeze(1)).any(dim=1).float()
            diag["rec_num_struct"] += float(rec_hit_struct.sum().item())
            diag["rec_den_struct"] += float(B)
            diag["rec_num"] += float(rec_hit_union.sum().item())
            diag["rec_den"] += float(B)
        else:
            rec_hit = (s_gold_struct >= top_scores[:, -1]).float()
            diag["rec_num"] += float(rec_hit.sum().item())
            diag["rec_den"] += float(B)

        # gate and gold total (consistent with topK)
        dir_ids = torch.ones_like(t)
        if gate_model is None:
            g = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        else:
            g = gate_model(top_scores, r, dir_ids, ent_temp=gate_ent_temp).to(s_gold_struct.dtype)

        if use_refiner_rerank:
            if profile_time:
                t0 = time.time()
            delta_ref_topk = refiner.score_delta_topk(
                anchor_ids=t,
                rel_ids=r,
                cand_ids=top_ids,
                dir_ids=dir_ids,
                rotate_model=rotate_model,
                nbr_ent=nbr_ent, nbr_rel=nbr_rel, nbr_dir=nbr_dir, nbr_mask=nbr_mask, freq=freq,
            )
            delta_ref_gold = refiner.score_delta_topk(
                anchor_ids=t,
                rel_ids=r,
                cand_ids=h.unsqueeze(1),
                dir_ids=dir_ids,
                rotate_model=rotate_model,
                nbr_ent=nbr_ent, nbr_rel=nbr_rel, nbr_dir=nbr_dir, nbr_mask=nbr_mask, freq=freq,
            ).squeeze(1)
            if profile_time:
                time_stats["refiner"] += time.time() - t0
            if gamma_by_rel is not None:
                gamma_eff = gamma_by_rel[r].to(delta_ref_topk.dtype)
            else:
                gamma_eff = torch.full((B,), float(refiner_gamma), device=device, dtype=delta_ref_topk.dtype)
            delta_ref_topk = delta_ref_topk * gamma_eff.unsqueeze(1)
            if torch.is_tensor(delta_ref_gold):
                delta_ref_gold = delta_ref_gold * gamma_eff
        else:
            delta_ref_topk = 0.0
            delta_ref_gold = 0.0

        if use_refiner_rerank and refiner_viol_only:
            viol_mask = (struct_w * top_scores > (struct_w * s_gold_struct).unsqueeze(1))
            delta_ref_topk = delta_ref_topk * viol_mask.to(delta_ref_topk.dtype)
        if use_refiner_rerank and refiner_topm and refiner_topm > 0:
            m = min(refiner_topm, top_scores.size(1))
            keep = torch.arange(top_scores.size(1), device=device).unsqueeze(0) < m
            delta_ref_topk = delta_ref_topk * keep.to(delta_ref_topk.dtype)

        # compute sem_topk + SCN r before threshold so s_thresh stays consistent
        if use_sem:
            if profile_time:
                t0 = time.time()
            if q0 is None:
                q0 = sem_encode_query_lhs(sem_model, ent_text_embs, rel_text_embs, t, r)
            sem_topk = sem_score_with_q0(sem_model, ent_text_embs, q0, top_ids)
            if scn_model is None:
                r_sem = torch.ones(B, device=device, dtype=sem_topk.dtype)
            else:
                r_sem = scn_model(sem_topk, top_scores, r, torch.ones_like(t), topm=scn_topm, temp=scn_temp)
            if scn_force_r is not None:
                r_sem = torch.full_like(r_sem, float(scn_force_r))
            if profile_time:
                time_stats["sem"] += time.time() - t0
        else:
            sem_topk = None
            r_sem = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        if collect_r:
            r_all.append(r_sem.detach().cpu())

        # ---- threshold (must exclude delta, must keep sem) ----
        sem_scale_total = b * g * r_sem
        sem_scale_thresh = b * g * (r_sem if scn_in_thresh else 1.0)
        s_thresh = struct_w * s_gold_struct + sem_scale_thresh * s_gold_sem
        if gold_struct_threshold:
            if gold_struct_threshold_no_sem:
                s_thresh = struct_w * s_gold_struct
            else:
                s_thresh = struct_w * s_gold_struct + sem_scale_thresh * s_gold_sem

        # pass 2: count greater over all entities with threshold s_thresh
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None or use_refiner_rerank:
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            comp = (struct_w * s_chunk_struct) > (s_thresh.unsqueeze(1) + score_eps)
            mask = torch.zeros_like(comp, dtype=torch.bool)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = lhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low = bisect.bisect_left(filt_sorted, start)
                high = bisect.bisect_left(filt_sorted, end)
                gold_val = h_cpu[i]
                for idx in range(low, high):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask[rows, cols] = True
            greater_struct += (comp & (~mask)).sum(dim=1)
        if profile_time:
            time_stats["pass2"] += time.time() - t0

        # correction inside topK: replace struct comparison with total comparison
        if use_sem:
            total_topk = struct_w * top_scores + delta_ref_topk + sem_scale_total.unsqueeze(1) * sem_topk
        else:
            total_topk = struct_w * top_scores + delta_ref_topk

        greater_struct_topk = (struct_w * top_scores > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)
        greater_total_topk = (total_topk > (s_thresh.unsqueeze(1) + score_eps)).sum(dim=1)

        # diagnostics: topK-only effects
        delta_rank_topk = (greater_total_topk - greater_struct_topk).float()
        diag["delta_rank_sum"] += float(delta_rank_topk.mean().item()) * B
        diag["p_improve"] += float((delta_rank_topk < 0).float().mean().item()) * B
        diag["p_hurt"] += float((delta_rank_topk > 0).float().mean().item()) * B

        viol = (struct_w * top_scores) > (s_thresh.unsqueeze(1) + score_eps)
        fixed = viol & (total_topk <= (s_thresh.unsqueeze(1) + score_eps))
        broken = (~viol) & (total_topk > (s_thresh.unsqueeze(1) + score_eps))
        diag["fix_num"] += float(fixed.sum().item())
        diag["fix_den"] += float(viol.sum().item())
        diag["break_num"] += float(broken.sum().item())
        diag["break_den"] += float((~viol).sum().item())
        if torch.is_tensor(delta_ref_gold):
            diag["delta_gold_sum"] += float(delta_ref_gold.mean().item()) * B
        diag["n"] += int(B)

        greater = greater_struct - greater_struct_topk + greater_total_topk
        rank = greater + 1  # [B]
        if collect_ranks:
            ranks_all.append(rank.detach().cpu())

        mrr = (1.0 / rank.float())
        h1 = (rank <= 1)
        h3 = (rank <= 3)
        h10 = (rank <= 10)

        stats["total"]["mrr"] += mrr.sum().item()
        stats["total"]["h1"] += h1.sum().item()
        stats["total"]["h3"] += h3.sum().item()
        stats["total"]["h10"] += h10.sum().item()
        stats["total"]["n"] += int(B)

        buckets = bucket_map[h]
        for bid in [0, 1, 2]:
            m = (buckets == bid)
            if m.any():
                stats[bid]["mrr"] += mrr[m].sum().item()
                stats[bid]["h1"] += h1[m].sum().item()
                stats[bid]["h3"] += h3[m].sum().item()
                stats[bid]["h10"] += h10[m].sum().item()
                stats[bid]["n"] += int(m.sum().item())

    # report
    def safe_div(a, b): return a / max(1, b)
    print("\n==================== LHS TopK-Injection (Exact Rank) ====================")
    for key in [0, 1, 2]:
        n = stats[key]["n"]
        name = bucket_names[key]
        print(f"LHS {name:<5} | MRR={safe_div(stats[key]['mrr'], n):.4f} | "
              f"H@1={safe_div(stats[key]['h1'], n):.4f} | "
              f"H@3={safe_div(stats[key]['h3'], n):.4f} | "
              f"H@10={safe_div(stats[key]['h10'], n):.4f} | n={n}")
    n = stats["total"]["n"]
    print(f"LHS TOTAL | MRR={safe_div(stats['total']['mrr'], n):.4f} | "
          f"H@1={safe_div(stats['total']['h1'], n):.4f} | "
          f"H@3={safe_div(stats['total']['h3'], n):.4f} | "
          f"H@10={safe_div(stats['total']['h10'], n):.4f} | n={n}")
    print("==========================================================================\n")

    if profile_time and time_stats:
        total = sum(time_stats.values())
        if total > 0:
            print(f"[Time][LHS] total={total:.2f}s "
                  f"struct_topk={time_stats['struct_topk']:.2f}s "
                  f"sem={time_stats['sem']:.2f}s "
                  f"refiner={time_stats['refiner']:.2f}s "
                  f"pass2={time_stats['pass2']:.2f}s "
                  f"sem_ratio={time_stats['sem']/total:.2%}")

    if collect_ranks:
        r_values = torch.cat(r_all, dim=0) if collect_r else None
        return stats, torch.cat(ranks_all, dim=0), diag, r_values
    if diag["rec_den"] > 0:
        if sem_union_topk and sem_union_topk > 0:
            base_rec = diag["rec_num_struct"] / max(diag["rec_den_struct"], 1e-6)
            union_k = topk + sem_union_topk
            union_rec = diag["rec_num"] / max(diag["rec_den"], 1e-6)
            print(f"[Recall@K][LHS] rotate@{topk}={base_rec:.4f} | union@{union_k}={union_rec:.4f}")
        else:
            print(f"[Recall@K][LHS] recall@{topk}={diag['rec_num'] / max(diag['rec_den'], 1e-6):.4f}")
    if diag["gate_on_den"] > 0:
        print(f"[DeltaGate][LHS] on_rate={diag['gate_on_sum'] / max(diag['gate_on_den'], 1e-6):.4f}")
    if refiner_diag and diag["flip_den"] > 0:
        print(f"[RefinerDiag][LHS] flip@K={diag['flip_num'] / max(diag['flip_den'], 1e-6):.4f} "
              f"p_up={diag['p_up_num'] / max(diag['p_up_den'], 1e-6):.4f} "
              f"p_pair={diag['p_pair_sum'] / max(diag['p_pair_den'], 1e-6):.4f}")
    r_values = torch.cat(r_all, dim=0) if collect_r else None
    return stats, None, diag, r_values


def load_sem_model(ckpt_path, text_dim, num_relations, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("model_type", "") == "biencoder":
        cfg = ckpt["model_args"]
        model = SemanticBiEncoderScorer(
            text_dim=cfg["text_dim"],
            num_relations=cfg["num_relations"],
            proj_dim=cfg.get("proj_dim", 256),
            dropout=cfg.get("dropout", 0.1),
            text_norm=cfg.get("text_norm", True),
        ).to(device)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        except RuntimeError:
            incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if missing.issubset({"dir_emb.weight"}) and not unexpected:
                print("[WARN] Sem ckpt missing dir_emb.weight; using default init for dir_emb.")
            else:
                raise
        return model
    raise ValueError("This eval script expects a biencoder checkpoint with model_type='biencoder'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--eval_split", type=str, default="valid", choices=["valid", "test"])

    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--pretrained_sem", type=str, default=None)
    ap.add_argument("--pretrained_scn", type=str, default=None)
    ap.add_argument("--pretrained_refiner", type=str, default=None)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--eval_sides", type=str, default="rhs", choices=["rhs", "lhs", "both"])
    ap.add_argument("--pretrained_gate", type=str, default=None)
    ap.add_argument("--gate_rel_dim", type=int, default=16)
    ap.add_argument("--gate_dir_dim", type=int, default=8)
    ap.add_argument("--gate_hidden_dim", type=int, default=64)
    ap.add_argument("--gate_g_min", type=float, default=0.0)
    ap.add_argument("--gate_g_max", type=float, default=2.0)
    ap.add_argument("--gate_init_bias", type=float, default=0.5413)
    ap.add_argument("--gate_ent_temp", type=float, default=1.0)

    ap.add_argument("--scn_rel_dim", type=int, default=32)
    ap.add_argument("--scn_dir_dim", type=int, default=8)
    ap.add_argument("--scn_hidden_dim", type=int, default=64)
    ap.add_argument("--scn_dropout", type=float, default=0.1)
    ap.add_argument("--scn_r_min", type=float, default=0.05)
    ap.add_argument("--scn_r_max", type=float, default=0.95)
    ap.add_argument("--scn_topm", type=int, default=10)
    ap.add_argument("--scn_temp", type=float, default=1.0)
    ap.add_argument("--scn_force_r", type=float, default=None,
                    help="sanity: override r(q) with a constant value (e.g., 1.0 or 0.0)")
    ap.add_argument(
        "--scn_in_thresh",
        dest="scn_in_thresh",
        action="store_true",
        help="include r in threshold (default)",
    )
    ap.add_argument(
        "--no_scn_in_thresh",
        dest="scn_in_thresh",
        action="store_false",
        help="exclude r from threshold (A variant)",
    )
    ap.set_defaults(scn_in_thresh=True)

    ap.add_argument("--calib_path", type=str, default=None)
    ap.add_argument("--calib_b_max", type=float, default=None)

    ap.add_argument("--b_scale", type=float, default=0.0)
    ap.add_argument("--b_rhs", type=float, default=None)
    ap.add_argument("--b_lhs", type=float, default=None)
    ap.add_argument("--struct_weight_rhs", type=float, default=1.0,
                    help="weight for structural score on RHS (set 0 for sem-only rerank)")
    ap.add_argument("--struct_weight_lhs", type=float, default=1.0,
                    help="weight for structural score on LHS (set 0 for sem-only rerank)")
    ap.add_argument("--refiner_gamma_rhs", type=float, default=1.0)
    ap.add_argument("--refiner_gamma_lhs", type=float, default=1.0)
    ap.add_argument("--gold_struct_threshold", action="store_true",
                    help="use s_gold_struct as threshold for comparisons (rerank-only)")
    ap.add_argument("--gold_struct_threshold_no_sem", action="store_true",
                    help="ablation: when gold_struct_threshold, exclude sem from threshold (incorrect)")
    ap.add_argument("--refiner_viol_only", action="store_true",
                    help="apply delta only to topK violators (diagnostic, uses s_gold_struct)")
    ap.add_argument("--refiner_topm", type=int, default=0,
                    help="apply delta only to topM candidates (prefix of topK)")
    ap.add_argument("--delta_gate_m12", type=float, default=None,
                    help="gate delta by margin m12 <= thr (confidence filter)")
    ap.add_argument("--delta_gate_ent", type=float, default=None,
                    help="gate delta by entropy >= thr (uncertainty filter)")
    ap.add_argument("--delta_gate_ent_q", type=float, default=None,
                    help="gate delta by entropy quantile q (e.g., 0.8 keeps top 20%)")
    ap.add_argument("--profile_time", action="store_true",
                    help="profile time breakdown and sem ratio")
    ap.add_argument("--strict_r0", action="store_true",
                    help="force strict R0 exact-rank via full-entity eval (no sem/gate/refiner)")
    ap.add_argument("--refiner_diag", action="store_true",
                    help="print refiner diagnostics (flip@K / p_up / p_pair)")
    ap.add_argument("--score_eps", type=float, default=0.0,
                    help="epsilon added to s_thresh for stable comparisons (batch-size invariant)")
    ap.add_argument("--deterministic", action="store_true",
                    help="enable deterministic algorithms (may reduce speed)")
    ap.add_argument("--disable_tf32", action="store_true",
                    help="disable TF32 for matmul/convolution")
    ap.add_argument("--matmul_precision", type=str, default="highest",
                    choices=["highest", "high", "medium"],
                    help="torch float32 matmul precision")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for evaluation")
    ap.add_argument("--rel_gamma_mode", type=str, default="none",
                    choices=["none", "bucket"],
                    help="apply relation-wise gamma (bucketed by frequency)")
    ap.add_argument("--rel_gamma_head", type=float, default=1.0)
    ap.add_argument("--rel_gamma_torso", type=float, default=1.0)
    ap.add_argument("--rel_gamma_tail", type=float, default=1.0)
    ap.add_argument("--save_ranks_path", type=str, default=None,
                    help="optional path to save per-query RHS/LHS ranks for paired bootstrap")
    ap.add_argument("--save_r_path", type=str, default=None,
                    help="optional path to save per-query SCN r values (rhs/lhs)")
    ap.add_argument("--paired_bootstrap", action="store_true",
                    help="compute paired bootstrap ΔMRR using baseline ranks")
    ap.add_argument("--paired_baseline_ranks", type=str, default=None,
                    help="path to baseline ranks .pt (from --save_ranks_path)")
    ap.add_argument("--paired_baseline_out_dir", type=str, default=None,
                    help="baseline out_dir that contains ranks.pt (from --save_ranks_path)")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--lhs_union_sem_topk", type=int, default=0,
                    help="if >0, use RotatE topK ∪ Sem topK for LHS candidates")
    ap.add_argument("--emb_dim", type=int, default=1000, help="RotatE embedding dim")
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bootstrap_samples", type=int, default=0)
    ap.add_argument("--bootstrap_ci", type=float, default=0.95)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    args = ap.parse_args()
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("artifacts", f"eval_topk_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)
    write_run_metadata(args.out_dir, args)
    setup_reproducibility(
        deterministic=args.deterministic,
        disable_tf32=args.disable_tf32,
        matmul_precision=args.matmul_precision,
        seed=args.seed,
        out_dir=args.out_dir,
        verbose=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate = RotatEModel(
        processor.num_entities,
        processor.num_relations,
        emb_dim=args.emb_dim,
        margin=9.0,
    ).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()

    refiner = None
    nbr_ent = nbr_rel = nbr_dir = nbr_mask = freq = None
    if args.pretrained_refiner:
        print(f"Loading StructRefiner: {args.pretrained_refiner}")
        refiner = StructRefiner(
            emb_dim=args.emb_dim,
            K=args.K,
            num_relations=processor.num_relations,
        ).to(device)
        ref_ckpt = torch.load(args.pretrained_refiner, map_location=device)
        if isinstance(ref_ckpt, dict) and "state_dict" in ref_ckpt:
            ref_ckpt = ref_ckpt["state_dict"]
        missing, unexpected = refiner.load_state_dict(ref_ckpt, strict=False)
        if missing:
            print(f"[Refiner] missing_keys={len(missing)} (e.g., {missing[:5]})")
        if unexpected:
            print(f"[Refiner] unexpected_keys={len(unexpected)} (e.g., {unexpected[:5]})")
        refiner.eval()
        nbr_ent = processor.nbr_ent.to(device, non_blocking=True)
        nbr_rel = processor.nbr_rel.to(device, non_blocking=True)
        nbr_dir = processor.nbr_dir.to(device, non_blocking=True)
        nbr_mask = processor.nbr_mask.to(device, non_blocking=True)
        freq = processor.freq.to(device, non_blocking=True)

        # sanity: anchor perturbation magnitude
        with torch.no_grad():
            sample_n = min(512, processor.num_entities)
            sample_ids = torch.randperm(processor.num_entities, device=device)[:sample_n]
            anchor_ref = refiner.refine_anchor(
                sample_ids, rotate, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
            base_emb = rotate.entity_embedding[sample_ids]
            delta_norm = (anchor_ref - base_emb).norm(dim=1).mean().item()
            print(f"[Refiner] anchor_delta_norm_mean={delta_norm:.6f}")

    if args.pretrained_sem:
        ent_embs, rel_embs = load_embeddings(processor, args, device)
        sem = load_sem_model(args.pretrained_sem, ent_embs.size(1), processor.num_relations, device)
    else:
        ent_embs = rel_embs = None
        sem = None

    scn_model = None
    if args.pretrained_scn:
        if sem is None:
            raise ValueError("--pretrained_scn requires --pretrained_sem")
        scn_ckpt = torch.load(args.pretrained_scn, map_location="cpu")
        if not isinstance(scn_ckpt, dict) or scn_ckpt.get("model_type", "") != "scn":
            raise ValueError("Expected SCN checkpoint with model_type='scn'.")
        cfg = scn_ckpt["model_args"]
        scn_model = SemanticConfidenceNet(**cfg).to(device)
        scn_model.load_state_dict(scn_ckpt["state_dict"], strict=True)
        scn_model.eval()
        print(f"[SCN] Loaded: {args.pretrained_scn}")

    gate_model = None
    if args.pretrained_gate:
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

    split_triplets = processor.valid_triplets if args.eval_split == "valid" else processor.test_triplets
    to_skip = build_to_skip(processor, split=args.eval_split)

    loader = DataLoader(
        TrainDataset(split_triplets),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    get_b_fn = None
    if args.calib_path is not None:
        get_b_fn, b_max, per_rel = load_calibrator(
            args.calib_path,
            processor.num_relations,
            device,
            calib_b_max=args.calib_b_max,
        )
        print(f"[Calib] loaded {args.calib_path} per_relation={per_rel} b_max={b_max}")

    gamma_by_rel = None
    if args.rel_gamma_mode == "bucket":
        rel_bucket = get_relation_buckets(processor, device)
        gamma_by_rel = torch.empty(processor.num_relations, device=device, dtype=torch.float32)
        gamma_by_rel[rel_bucket == 0] = float(args.rel_gamma_tail)
        gamma_by_rel[rel_bucket == 1] = float(args.rel_gamma_torso)
        gamma_by_rel[rel_bucket == 2] = float(args.rel_gamma_head)
        print(f"[RelGamma] bucket tail/torso/head = {args.rel_gamma_tail}/{args.rel_gamma_torso}/{args.rel_gamma_head}")

    # default to gold-struct threshold only when candidate-aware delta is active
    use_delta = (args.refiner_gamma_rhs > 0.0) or (args.refiner_gamma_lhs > 0.0)
    if refiner is not None and hasattr(refiner, "score_delta_topk") and use_delta and not args.gold_struct_threshold:
        args.gold_struct_threshold = True
        print("[Refiner] gold_struct_threshold auto-enabled (gamma>0)")
    if args.gold_struct_threshold and not use_delta:
        raise ValueError("gold_struct_threshold=True but gamma_rhs/gamma_lhs are 0; this will break Sem+Gate baseline.")

    using_refiner = refiner is not None
    using_sem = sem is not None
    using_gate = gate_model is not None
    using_scn = scn_model is not None
    gold_mode = "STRUCT" if args.gold_struct_threshold else "TOTAL"
    print(f"[Mode] using_refiner={using_refiner} refiner_ckpt={args.pretrained_refiner}")
    print(f"[Mode] using_sem={using_sem} using_gate={using_gate} using_scn={using_scn}")
    print(f"[Mode] gamma_rhs={args.refiner_gamma_rhs} gamma_lhs={args.refiner_gamma_lhs} gold_threshold_mode={gold_mode}")

    rhs_stats = lhs_stats = None
    rhs_ranks = lhs_ranks = None
    rhs_diag = lhs_diag = None
    rhs_r_values = lhs_r_values = None

    b_rhs = float(args.b_scale) if args.b_rhs is None else float(args.b_rhs)
    b_lhs = float(args.b_scale) if args.b_lhs is None else float(args.b_lhs)
    score_eps = float(args.score_eps)
    if sem is None:
        if (abs(b_rhs) > 1e-12) or (abs(b_lhs) > 1e-12) or (args.lhs_union_sem_topk > 0):
            raise ValueError("Sem model required for nonzero b or union candidates. Provide --pretrained_sem.")

    # strict R0: bypass topK-inject logic and use full-entity exact rank
    is_strict_r0 = (
        sem is None and gate_model is None and refiner is None and
        abs(b_rhs) <= 1e-12 and abs(b_lhs) <= 1e-12 and
        abs(args.refiner_gamma_rhs) <= 1e-12 and abs(args.refiner_gamma_lhs) <= 1e-12 and
        abs(args.struct_weight_rhs - 1.0) <= 1e-12 and abs(args.struct_weight_lhs - 1.0) <= 1e-12 and
        not args.gold_struct_threshold and args.lhs_union_sem_topk <= 0 and
        args.refiner_topm == 0 and not args.refiner_viol_only and
        args.delta_gate_m12 is None and args.delta_gate_ent is None and args.delta_gate_ent_q is None and
        args.calib_path is None and args.rel_gamma_mode == "none"
    )
    if args.strict_r0:
        if not is_strict_r0:
            raise ValueError("--strict_r0 requires pure RotatE with no sem/gate/refiner/delta and zero weights.")
        if args.bootstrap_samples > 0 or args.paired_bootstrap:
            raise ValueError("--strict_r0 does not support bootstrap; use topK-inject path instead.")

        split_triplets = processor.valid_triplets if args.eval_split == "valid" else processor.test_triplets
        strict_loader = DataLoader(
            TrainDataset(split_triplets),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        collect_ranks = args.save_ranks_path is not None
        if collect_ranks:
            results, rhs_ranks, lhs_ranks = eval_chunked_bidirectional(
                processor=processor,
                rotate_model=rotate,
                test_loader=strict_loader,
                device=device,
                to_skip=to_skip,
                eval_split=args.eval_split,
                refiner=None,
                semres_model=None,
                ent_text_embs=None,
                rel_text_embs=None,
                chunk_size=args.chunk_size,
                sem_subchunk=256,
                disable_refiner=True,
                disable_semres=True,
                sem_rhs_only=False,
                sem_lhs_only=False,
                refiner_topk_only=False,
                refiner_topk=args.topk,
                print_sem_stats=False,
                refiner_diag=False,
                recall_k=args.topk,
                rel_bucket_map=None,
                collect_ranks=True,
                score_eps=score_eps,
            )
        else:
            results = eval_chunked_bidirectional(
                processor=processor,
                rotate_model=rotate,
                test_loader=strict_loader,
                device=device,
                to_skip=to_skip,
                eval_split=args.eval_split,
                refiner=None,
                semres_model=None,
                ent_text_embs=None,
                rel_text_embs=None,
                chunk_size=args.chunk_size,
                sem_subchunk=256,
                disable_refiner=True,
                disable_semres=True,
                sem_rhs_only=False,
                sem_lhs_only=False,
                refiner_topk_only=False,
                refiner_topk=args.topk,
                print_sem_stats=False,
                refiner_diag=False,
                recall_k=args.topk,
                rel_bucket_map=None,
                score_eps=score_eps,
            )

        metrics = {
            "split": args.eval_split,
            "topk": args.topk,
            "recall_k": args.topk,
            "rhs": results["rhs"],
            "lhs": results["lhs"],
            "avg": {
                "MRR": 0.5 * (results["rhs"]["total"]["MRR"] + results["lhs"]["total"]["MRR"]),
                "H1": 0.5 * (results["rhs"]["total"]["H1"] + results["lhs"]["total"]["H1"]),
                "H3": 0.5 * (results["rhs"]["total"]["H3"] + results["lhs"]["total"]["H3"]),
                "H10": 0.5 * (results["rhs"]["total"]["H10"] + results["lhs"]["total"]["H10"]),
                "RecK": 0.5 * (results["rhs"]["total"]["RecK"] + results["lhs"]["total"]["RecK"]),
            },
        }
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[Metrics] saved to {metrics_path}")
        if args.save_ranks_path is not None:
            os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
            torch.save({"rhs_ranks": rhs_ranks, "lhs_ranks": lhs_ranks}, args.save_ranks_path)
            print(f"[Ranks] saved to {args.save_ranks_path}")
        return

    collect_ranks = (args.bootstrap_samples > 0) or args.paired_bootstrap or (args.save_ranks_path is not None)
    collect_r = args.save_r_path is not None
    if args.paired_bootstrap and args.bootstrap_samples <= 0:
        raise ValueError("--paired_bootstrap requires --bootstrap_samples > 0")

    if args.eval_sides in ["rhs", "both"]:
        rhs_stats, rhs_ranks, rhs_diag, rhs_r_values = eval_rhs_topk_inject(
            processor=processor,
            rotate_model=rotate,
            sem_model=sem,
            ent_text_embs=ent_embs,
            rel_text_embs=rel_embs,
            refiner=refiner,
            nbr_ent=nbr_ent,
            nbr_rel=nbr_rel,
            nbr_dir=nbr_dir,
            nbr_mask=nbr_mask,
            freq=freq,
            gate_model=gate_model,
            gate_ent_temp=args.gate_ent_temp,
            loader=loader,
            to_skip=to_skip,
            split=args.eval_split,
            device=device,
            topk=args.topk,
            chunk_size=args.chunk_size,
            b_scale=b_rhs,
            get_b_fn=get_b_fn,
            collect_ranks=collect_ranks,
            collect_r=collect_r,
            refiner_gamma=args.refiner_gamma_rhs,
            gamma_by_rel=gamma_by_rel,
            gold_struct_threshold=args.gold_struct_threshold,
            gold_struct_threshold_no_sem=args.gold_struct_threshold_no_sem,
            delta_gate_m12=args.delta_gate_m12,
            delta_gate_ent=args.delta_gate_ent,
            delta_gate_ent_q=args.delta_gate_ent_q,
            refiner_viol_only=args.refiner_viol_only,
            refiner_topm=args.refiner_topm,
            profile_time=args.profile_time,
            refiner_diag=args.refiner_diag,
            struct_weight=args.struct_weight_rhs,
            score_eps=score_eps,
            scn_model=scn_model,
            scn_topm=args.scn_topm,
            scn_temp=args.scn_temp,
            scn_in_thresh=args.scn_in_thresh,
            scn_force_r=args.scn_force_r,
        )

    if args.eval_sides in ["lhs", "both"]:
        lhs_stats, lhs_ranks, lhs_diag, lhs_r_values = eval_lhs_topk_inject(
            processor=processor,
            rotate_model=rotate,
            sem_model=sem,
            ent_text_embs=ent_embs,
            rel_text_embs=rel_embs,
            refiner=refiner,
            nbr_ent=nbr_ent,
            nbr_rel=nbr_rel,
            nbr_dir=nbr_dir,
            nbr_mask=nbr_mask,
            freq=freq,
            gate_model=gate_model,
            gate_ent_temp=args.gate_ent_temp,
            loader=loader,
            to_skip=to_skip,
            split=args.eval_split,
            device=device,
            topk=args.topk,
            chunk_size=args.chunk_size,
            b_scale=b_lhs,
            get_b_fn=get_b_fn,
            collect_ranks=collect_ranks,
            collect_r=collect_r,
            refiner_gamma=args.refiner_gamma_lhs,
            gamma_by_rel=gamma_by_rel,
            gold_struct_threshold=args.gold_struct_threshold,
            gold_struct_threshold_no_sem=args.gold_struct_threshold_no_sem,
            delta_gate_m12=args.delta_gate_m12,
            delta_gate_ent=args.delta_gate_ent,
            delta_gate_ent_q=args.delta_gate_ent_q,
            refiner_viol_only=args.refiner_viol_only,
            refiner_topm=args.refiner_topm,
            profile_time=args.profile_time,
            refiner_diag=args.refiner_diag,
            struct_weight=args.struct_weight_lhs,
            score_eps=score_eps,
            scn_model=scn_model,
            scn_topm=args.scn_topm,
            scn_temp=args.scn_temp,
            scn_in_thresh=args.scn_in_thresh,
            scn_force_r=args.scn_force_r,
        )

    if args.eval_sides == "both" and rhs_stats and lhs_stats:
        def safe_div(a, b): return a / max(1, b)
        print("\n==================== AVG (RHS+LHS) ====================")
        for key in [0, 1, 2, "total"]:
            n = rhs_stats[key]["n"]
            mrr = 0.5 * (safe_div(rhs_stats[key]["mrr"], n) + safe_div(lhs_stats[key]["mrr"], n))
            h1 = 0.5 * (safe_div(rhs_stats[key]["h1"], n) + safe_div(lhs_stats[key]["h1"], n))
            h3 = 0.5 * (safe_div(rhs_stats[key]["h3"], n) + safe_div(lhs_stats[key]["h3"], n))
            h10 = 0.5 * (safe_div(rhs_stats[key]["h10"], n) + safe_div(lhs_stats[key]["h10"], n))
            name = "TOTAL" if key == "total" else {0: "Tail", 1: "Torso", 2: "Head"}[key]
            print(f"AVG {name:<5} | MRR={mrr:.4f} | H@1={h1:.4f} | H@3={h3:.4f} | H@10={h10:.4f} | n={n}")
        print("========================================================\n")

        # diagnostics
        if rhs_diag and rhs_diag["n"] > 0:
            p_improve = rhs_diag["p_improve"] / rhs_diag["n"]
            p_hurt = rhs_diag["p_hurt"] / rhs_diag["n"]
            mean_delta_rank = rhs_diag["delta_rank_sum"] / rhs_diag["n"]
            fix_rate = rhs_diag["fix_num"] / max(rhs_diag["fix_den"], 1e-6)
            break_rate = rhs_diag["break_num"] / max(rhs_diag["break_den"], 1e-6)
            mean_delta_gold = rhs_diag["delta_gold_sum"] / rhs_diag["n"]
            print(f"[Diag RHS] mean(delta_rank_topk)={mean_delta_rank:.4f} p_improve={p_improve:.4f} p_hurt={p_hurt:.4f}")
            print(f"[Diag RHS] fix_rate={fix_rate:.4f} break_rate={break_rate:.4f}")
            print(f"[Diag RHS] mean(delta_gold)={mean_delta_gold:.4f}")
        if lhs_diag and lhs_diag["n"] > 0:
            p_improve = lhs_diag["p_improve"] / lhs_diag["n"]
            p_hurt = lhs_diag["p_hurt"] / lhs_diag["n"]
            mean_delta_rank = lhs_diag["delta_rank_sum"] / lhs_diag["n"]
            fix_rate = lhs_diag["fix_num"] / max(lhs_diag["fix_den"], 1e-6)
            break_rate = lhs_diag["break_num"] / max(lhs_diag["break_den"], 1e-6)
            mean_delta_gold = lhs_diag["delta_gold_sum"] / lhs_diag["n"]
            print(f"[Diag LHS] mean(delta_rank_topk)={mean_delta_rank:.4f} p_improve={p_improve:.4f} p_hurt={p_hurt:.4f}")
            print(f"[Diag LHS] fix_rate={fix_rate:.4f} break_rate={break_rate:.4f}")
            print(f"[Diag LHS] mean(delta_gold)={mean_delta_gold:.4f}")

    # save metrics.json
    def _pack_stats(side_stats, side_diag):
        def safe_div(a, b): return a / max(1, b)
        out = {}
        for key in [0, 1, 2, "total"]:
            name = "total" if key == "total" else {0: "Tail", 1: "Torso", 2: "Head"}[key]
            n = side_stats[key]["n"]
            out[name] = {
                "MRR": safe_div(side_stats[key]["mrr"], n),
                "H1": safe_div(side_stats[key]["h1"], n),
                "H3": safe_div(side_stats[key]["h3"], n),
                "H10": safe_div(side_stats[key]["h10"], n),
                "n": int(n),
            }
        if side_diag and side_diag.get("rec_den", 0) > 0:
            out["total"]["RecK"] = side_diag["rec_num"] / max(side_diag["rec_den"], 1e-6)
        else:
            out["total"]["RecK"] = None
        return out

    metrics = {
        "split": args.eval_split,
        "topk": args.topk,
        "recall_k": args.topk,
    }
    if rhs_stats:
        metrics["rhs"] = _pack_stats(rhs_stats, rhs_diag)
    if lhs_stats:
        metrics["lhs"] = _pack_stats(lhs_stats, lhs_diag)
    if rhs_stats and lhs_stats:
        metrics["avg"] = {
            "MRR": 0.5 * (metrics["rhs"]["total"]["MRR"] + metrics["lhs"]["total"]["MRR"]),
            "H1": 0.5 * (metrics["rhs"]["total"]["H1"] + metrics["lhs"]["total"]["H1"]),
            "H3": 0.5 * (metrics["rhs"]["total"]["H3"] + metrics["lhs"]["total"]["H3"]),
            "H10": 0.5 * (metrics["rhs"]["total"]["H10"] + metrics["lhs"]["total"]["H10"]),
            "RecK": None,
        }
        if metrics["rhs"]["total"]["RecK"] is not None and metrics["lhs"]["total"]["RecK"] is not None:
            metrics["avg"]["RecK"] = 0.5 * (metrics["rhs"]["total"]["RecK"] + metrics["lhs"]["total"]["RecK"])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Metrics] saved to {metrics_path}")

    if args.bootstrap_samples > 0:
        torch.manual_seed(args.bootstrap_seed)
        alpha = (1.0 - float(args.bootstrap_ci)) / 2.0
        if args.eval_sides == "both" and rhs_ranks is not None and lhs_ranks is not None:
            rhs_r = rhs_ranks.float()
            lhs_r = lhs_ranks.float()
            n = rhs_r.numel()
            mrr_base = 0.5 * ((1.0 / rhs_r).mean().item() + (1.0 / lhs_r).mean().item())
            mrr_samples = []
            for _ in range(args.bootstrap_samples):
                idx = torch.randint(0, n, (n,))
                mrr_samples.append(0.5 * ((1.0 / rhs_r[idx]).mean().item() + (1.0 / lhs_r[idx]).mean().item()))
            mrr_samples = torch.tensor(mrr_samples)
            lo = torch.quantile(mrr_samples, alpha).item()
            hi = torch.quantile(mrr_samples, 1.0 - alpha).item()
            print(f"[Bootstrap][AVG] MRR={mrr_base:.4f} CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] "
                  f"samples={args.bootstrap_samples}")
        elif args.eval_sides == "rhs" and rhs_ranks is not None:
            ranks = rhs_ranks.float()
            n = ranks.numel()
            mrr_base = (1.0 / ranks).mean().item()
            mrr_samples = []
            for _ in range(args.bootstrap_samples):
                idx = torch.randint(0, n, (n,))
                mrr_samples.append((1.0 / ranks[idx]).mean().item())
            mrr_samples = torch.tensor(mrr_samples)
            lo = torch.quantile(mrr_samples, alpha).item()
            hi = torch.quantile(mrr_samples, 1.0 - alpha).item()
            print(f"[Bootstrap] MRR={mrr_base:.4f} CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] "
                  f"samples={args.bootstrap_samples}")
        elif args.eval_sides == "lhs" and lhs_ranks is not None:
            ranks = lhs_ranks.float()
            n = ranks.numel()
            mrr_base = (1.0 / ranks).mean().item()
            mrr_samples = []
            for _ in range(args.bootstrap_samples):
                idx = torch.randint(0, n, (n,))
                mrr_samples.append((1.0 / ranks[idx]).mean().item())
            mrr_samples = torch.tensor(mrr_samples)
            lo = torch.quantile(mrr_samples, alpha).item()
            hi = torch.quantile(mrr_samples, 1.0 - alpha).item()
            print(f"[Bootstrap] MRR={mrr_base:.4f} CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] "
                  f"samples={args.bootstrap_samples}")

    # optional save ranks for paired bootstrap
    if args.save_ranks_path and args.eval_sides == "both" and rhs_ranks is not None and lhs_ranks is not None:
        torch.save({"rhs_ranks": rhs_ranks.cpu(), "lhs_ranks": lhs_ranks.cpu()}, args.save_ranks_path)
        print(f"[Ranks] saved to {args.save_ranks_path}")

    # optional save SCN r values for sanity checks
    if args.save_r_path and (rhs_r_values is not None or lhs_r_values is not None):
        os.makedirs(os.path.dirname(args.save_r_path), exist_ok=True)
        obj = {}
        if rhs_r_values is not None:
            obj["rhs_r"] = rhs_r_values.cpu()
        if lhs_r_values is not None:
            obj["lhs_r"] = lhs_r_values.cpu()
        torch.save(obj, args.save_r_path)
        print(f"[SCN] r values saved to {args.save_r_path}")

    # paired bootstrap (ΔMRR)
    if args.paired_bootstrap and args.eval_sides == "both":
        base_path = args.paired_baseline_ranks
        if base_path is None and args.paired_baseline_out_dir is not None:
            base_path = os.path.join(args.paired_baseline_out_dir, "ranks.pt")
        if base_path is None:
            raise ValueError("paired bootstrap requires --paired_baseline_ranks or --paired_baseline_out_dir")
        base_obj = torch.load(base_path, map_location="cpu")
        base_rhs = base_obj["rhs_ranks"].float()
        base_lhs = base_obj["lhs_ranks"].float()
        if rhs_ranks is None or lhs_ranks is None:
            raise ValueError("paired bootstrap requires current run ranks; set --save_ranks_path or use --bootstrap_samples > 0")
        rhs_r = rhs_ranks.float().cpu()
        lhs_r = lhs_ranks.float().cpu()
        if base_rhs.numel() != rhs_r.numel() or base_lhs.numel() != lhs_r.numel():
            raise ValueError("paired baseline ranks size mismatch")

        n = rhs_r.numel()
        torch.manual_seed(args.bootstrap_seed)
        alpha = (1.0 - float(args.bootstrap_ci)) / 2.0

        # AVG ΔMRR
        base_avg = 0.5 * ((1.0 / base_rhs) + (1.0 / base_lhs))
        cur_avg = 0.5 * ((1.0 / rhs_r) + (1.0 / lhs_r))
        delta_avg = cur_avg - base_avg
        samples = []
        for _ in range(args.bootstrap_samples):
            idx = torch.randint(0, n, (n,))
            samples.append(delta_avg[idx].mean().item())
        samples = torch.tensor(samples)
        lo = torch.quantile(samples, alpha).item()
        hi = torch.quantile(samples, 1.0 - alpha).item()
        print(f"[PairedBootstrap][AVG] ΔMRR={delta_avg.mean().item():.4f} "
              f"CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] samples={args.bootstrap_samples}")

        # RHS ΔMRR
        delta_rhs = (1.0 / rhs_r) - (1.0 / base_rhs)
        samples = []
        for _ in range(args.bootstrap_samples):
            idx = torch.randint(0, n, (n,))
            samples.append(delta_rhs[idx].mean().item())
        samples = torch.tensor(samples)
        lo = torch.quantile(samples, alpha).item()
        hi = torch.quantile(samples, 1.0 - alpha).item()
        print(f"[PairedBootstrap][RHS] ΔMRR={delta_rhs.mean().item():.4f} "
              f"CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] samples={args.bootstrap_samples}")

        # LHS ΔMRR
        delta_lhs = (1.0 / lhs_r) - (1.0 / base_lhs)
        samples = []
        for _ in range(args.bootstrap_samples):
            idx = torch.randint(0, n, (n,))
            samples.append(delta_lhs[idx].mean().item())
        samples = torch.tensor(samples)
        lo = torch.quantile(samples, alpha).item()
        hi = torch.quantile(samples, 1.0 - alpha).item()
        print(f"[PairedBootstrap][LHS] ΔMRR={delta_lhs.mean().item():.4f} "
              f"CI{int(args.bootstrap_ci*100)}%=[{lo:.4f}, {hi:.4f}] samples={args.bootstrap_samples}")

if __name__ == "__main__":
    main()
