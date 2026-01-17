import argparse
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
from eval.eval_full_entity_filtered import build_to_skip, load_embeddings, get_freq_buckets

# biencoder
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.struct_refiner import StructRefiner
from models.gate_injector import ConfidenceGate
from models.gate_injector import gate_features_from_top_scores
from tools.run_meta import write_run_metadata


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
def sem_score_rhs_biencoder(sem_model, ent_embs, rel_embs, h, r, cand_t_2d):
    """
    h,r: [B]
    cand_t_2d: [B,K]
    return sem scores [B,K]
    """
    B, K = cand_t_2d.shape
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    q = sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))  # [B,d]
    v = sem_model.encode_entity(ent_embs[cand_t_2d.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q, v)


@torch.no_grad()
def sem_score_rhs_biencoder_pos(sem_model, ent_embs, rel_embs, h, r, t):
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    t_txt = ent_embs[t]
    q = sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))
    v = sem_model.encode_entity(t_txt)
    return (q * v).sum(dim=-1)  # [B]


@torch.no_grad()
def sem_score_lhs_biencoder(sem_model, ent_embs, rel_embs, t, r, cand_h_2d):
    """
    t,r: [B]
    cand_h_2d: [B,K]
    return sem scores [B,K]
    """
    B, K = cand_h_2d.shape
    t_txt = ent_embs[t]
    r_txt = rel_embs[r]
    q = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))  # [B,d]
    v = sem_model.encode_entity(ent_embs[cand_h_2d.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q, v)


@torch.no_grad()
def sem_score_lhs_biencoder_pos(sem_model, ent_embs, rel_embs, t, r, h):
    t_txt = ent_embs[t]
    r_txt = rel_embs[r]
    h_txt = ent_embs[h]
    q = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))
    v = sem_model.encode_entity(h_txt)
    return (q * v).sum(dim=-1)  # [B]


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
    collect_ranks=False,
    profile_time=False,
):
    rotate_model.eval()
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
            "rec_num": 0.0, "rec_den": 0.0, "n": 0}
    ranks_all = [] if collect_ranks else None
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

        # gold total
        if refiner is None or use_refiner_rerank:
            s_gold_struct = rotate_model(h, r, t, mode="single")  # [B]
        else:
            anchor_emb = refiner.refine_anchor(
                h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
            conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
            s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, t.unsqueeze(1), conj=conj_flag).squeeze(1)
        if profile_time:
            t0 = time.time()
        s_gold_sem = sem_score_rhs_biencoder_pos(sem_model, ent_text_embs, rel_text_embs, h, r, t)  # [B]
        if profile_time:
            time_stats["sem"] += time.time() - t0
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
                s_chunk = rotate_model(h, r, cand_1d, mode="batch_neg")
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
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

        # recall@K (gold in struct topK, filtered)
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

        # optional masks for candidate-aware delta
        if use_refiner_rerank and refiner_viol_only:
            viol_mask = (top_scores > s_gold_struct.unsqueeze(1))
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

        s_gold_total = s_gold_struct + delta_ref_gold + (b * g) * s_gold_sem
        # threshold excludes delta when gold_struct_threshold is on, but keeps sem if enabled
        if gold_struct_threshold:
            if gold_struct_threshold_no_sem:
                gold_thr = s_gold_struct
            else:
                gold_thr = s_gold_struct + (b * g) * s_gold_sem
        else:
            gold_thr = s_gold_total

        # pass 2: count greater over all entities with threshold s_thresh
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None or use_refiner_rerank:
                s_chunk = rotate_model(h, r, cand_1d, mode="batch_neg")
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            comp = s_chunk > gold_thr.unsqueeze(1)
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
        # compute sem on top_ids
        if profile_time:
            t0 = time.time()
        sem_topk = sem_score_rhs_biencoder(sem_model, ent_text_embs, rel_text_embs, h, r, top_ids)
        total_topk = top_scores + delta_ref_topk + (b * g).unsqueeze(1) * sem_topk
        if profile_time:
            time_stats["sem"] += time.time() - t0

        greater_struct_topk = (top_scores > gold_thr.unsqueeze(1)).sum(dim=1)
        greater_total_topk = (total_topk > gold_thr.unsqueeze(1)).sum(dim=1)

        # diagnostics: topK-only effects
        delta_rank_topk = (greater_total_topk - greater_struct_topk).float()
        diag["delta_rank_sum"] += float(delta_rank_topk.mean().item()) * B
        diag["p_improve"] += float((delta_rank_topk < 0).float().mean().item()) * B
        diag["p_hurt"] += float((delta_rank_topk > 0).float().mean().item()) * B

        viol = top_scores > gold_thr.unsqueeze(1)
        fixed = viol & (total_topk <= gold_thr.unsqueeze(1))
        broken = (~viol) & (total_topk > gold_thr.unsqueeze(1))
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
        return stats, torch.cat(ranks_all, dim=0), diag
    if diag["rec_den"] > 0:
        print(f"[Recall@K][RHS] recall@{topk}={diag['rec_num'] / max(diag['rec_den'], 1e-6):.4f}")
    if diag["gate_on_den"] > 0:
        print(f"[DeltaGate][RHS] on_rate={diag['gate_on_sum'] / max(diag['gate_on_den'], 1e-6):.4f}")
    return stats, None, diag


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
    collect_ranks=False,
    profile_time=False,
):
    rotate_model.eval()
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
            "rec_num": 0.0, "rec_den": 0.0, "n": 0}
    ranks_all = [] if collect_ranks else None
    time_stats = {"struct_topk": 0.0, "sem": 0.0, "refiner": 0.0, "pass2": 0.0} if profile_time else None

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        use_refiner_rerank = refiner is not None and hasattr(refiner, "score_delta_topk")

        # gold total
        if refiner is None or use_refiner_rerank:
            s_gold_struct = rotate_model.score_head_batch(r, t, h.unsqueeze(1)).squeeze(1)
        else:
            anchor_emb = refiner.refine_anchor(
                t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
            conj_flag = torch.ones(B, dtype=torch.bool, device=device)
            s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, h.unsqueeze(1), conj=conj_flag).squeeze(1)

        if profile_time:
            t0 = time.time()
        s_gold_sem = sem_score_lhs_biencoder_pos(sem_model, ent_text_embs, rel_text_embs, t, r, h)  # [B]
        if profile_time:
            time_stats["sem"] += time.time() - t0
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

        # pass 1: build topK by struct only
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            if refiner is None or use_refiner_rerank:
                s_chunk = rotate_model.score_head_batch(r, t, cand_1d)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.ones(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            # --- topK maintenance: exclude filtered + exclude gold ---
            s_for_topk = s_chunk.clone()

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
        if profile_time:
            time_stats["struct_topk"] += time.time() - t0

        # recall@K (gold in struct topK, filtered)
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
            viol_mask = (top_scores > s_gold_struct.unsqueeze(1))
            delta_ref_topk = delta_ref_topk * viol_mask.to(delta_ref_topk.dtype)
        if use_refiner_rerank and refiner_topm and refiner_topm > 0:
            m = min(refiner_topm, top_scores.size(1))
            keep = torch.arange(top_scores.size(1), device=device).unsqueeze(0) < m
            delta_ref_topk = delta_ref_topk * keep.to(delta_ref_topk.dtype)

        s_gold_total = s_gold_struct + delta_ref_gold + (b * g) * s_gold_sem
        # threshold excludes delta when gold_struct_threshold is on, but keeps sem if enabled
        if gold_struct_threshold:
            if gold_struct_threshold_no_sem:
                gold_thr = s_gold_struct
            else:
                gold_thr = s_gold_struct + (b * g) * s_gold_sem
        else:
            gold_thr = s_gold_total

        # pass 2: count greater over all entities with threshold s_thresh
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        if profile_time:
            t0 = time.time()
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None or use_refiner_rerank:
                s_chunk = rotate_model.score_head_batch(r, t, cand_1d)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.ones(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            comp = s_chunk > gold_thr.unsqueeze(1)
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
        if profile_time:
            t0 = time.time()
        sem_topk = sem_score_lhs_biencoder(sem_model, ent_text_embs, rel_text_embs, t, r, top_ids)
        total_topk = top_scores + delta_ref_topk + (b * g).unsqueeze(1) * sem_topk
        if profile_time:
            time_stats["sem"] += time.time() - t0

        greater_struct_topk = (top_scores > gold_thr.unsqueeze(1)).sum(dim=1)
        greater_total_topk = (total_topk > gold_thr.unsqueeze(1)).sum(dim=1)

        # diagnostics: topK-only effects
        delta_rank_topk = (greater_total_topk - greater_struct_topk).float()
        diag["delta_rank_sum"] += float(delta_rank_topk.mean().item()) * B
        diag["p_improve"] += float((delta_rank_topk < 0).float().mean().item()) * B
        diag["p_hurt"] += float((delta_rank_topk > 0).float().mean().item()) * B

        viol = top_scores > gold_thr.unsqueeze(1)
        fixed = viol & (total_topk <= gold_thr.unsqueeze(1))
        broken = (~viol) & (total_topk > gold_thr.unsqueeze(1))
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
        return stats, torch.cat(ranks_all, dim=0), diag
    if diag["rec_den"] > 0:
        print(f"[Recall@K][LHS] recall@{topk}={diag['rec_num'] / max(diag['rec_den'], 1e-6):.4f}")
    if diag["gate_on_den"] > 0:
        print(f"[DeltaGate][LHS] on_rate={diag['gate_on_sum'] / max(diag['gate_on_den'], 1e-6):.4f}")
    return stats, None, diag


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
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return model
    raise ValueError("This eval script expects a biencoder checkpoint with model_type='biencoder'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--eval_split", type=str, default="valid", choices=["valid", "test"])

    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--pretrained_sem", type=str, required=True)
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

    ap.add_argument("--calib_path", type=str, default=None)
    ap.add_argument("--calib_b_max", type=float, default=None)

    ap.add_argument("--b_scale", type=float, default=0.0)
    ap.add_argument("--b_rhs", type=float, default=None)
    ap.add_argument("--b_lhs", type=float, default=None)
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
    ap.add_argument("--rel_gamma_mode", type=str, default="none",
                    choices=["none", "bucket"],
                    help="apply relation-wise gamma (bucketed by frequency)")
    ap.add_argument("--rel_gamma_head", type=float, default=1.0)
    ap.add_argument("--rel_gamma_torso", type=float, default=1.0)
    ap.add_argument("--rel_gamma_tail", type=float, default=1.0)
    ap.add_argument("--save_ranks_path", type=str, default=None,
                    help="optional path to save per-query RHS/LHS ranks for paired bootstrap")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bootstrap_samples", type=int, default=0)
    ap.add_argument("--bootstrap_ci", type=float, default=0.95)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    args = ap.parse_args()
    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("artifacts", f"eval_topk_{ts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_run_metadata(args.out_dir, args)

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()

    refiner = None
    nbr_ent = nbr_rel = nbr_dir = nbr_mask = freq = None
    if args.pretrained_refiner:
        print(f"Loading StructRefiner: {args.pretrained_refiner}")
        refiner = StructRefiner(
            emb_dim=500,
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

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    sem = load_sem_model(args.pretrained_sem, ent_embs.size(1), processor.num_relations, device)

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
    gold_mode = "STRUCT" if args.gold_struct_threshold else "TOTAL"
    print(f"[Mode] using_refiner={using_refiner} refiner_ckpt={args.pretrained_refiner}")
    print(f"[Mode] using_sem={using_sem} using_gate={using_gate}")
    print(f"[Mode] gamma_rhs={args.refiner_gamma_rhs} gamma_lhs={args.refiner_gamma_lhs} gold_threshold_mode={gold_mode}")

    rhs_stats = lhs_stats = None
    rhs_ranks = lhs_ranks = None
    rhs_diag = lhs_diag = None

    b_rhs = float(args.b_scale) if args.b_rhs is None else float(args.b_rhs)
    b_lhs = float(args.b_scale) if args.b_lhs is None else float(args.b_lhs)

    if args.eval_sides in ["rhs", "both"]:
        rhs_stats, rhs_ranks, rhs_diag = eval_rhs_topk_inject(
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
            collect_ranks=args.bootstrap_samples > 0,
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
        )

    if args.eval_sides in ["lhs", "both"]:
        lhs_stats, lhs_ranks, lhs_diag = eval_lhs_topk_inject(
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
            collect_ranks=args.bootstrap_samples > 0,
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

if __name__ == "__main__":
    main()
