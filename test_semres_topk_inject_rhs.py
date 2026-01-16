import argparse
import os
import sys
import bisect
import torch
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from test_semres import build_to_skip, load_embeddings, get_freq_buckets

# biencoder
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.struct_refiner import StructRefiner
from models.gate_injector import ConfidenceGate


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
    collect_ranks=False,
):
    rotate_model.eval()
    sem_model.eval()

    ent_text_embs = ent_text_embs.to(device, non_blocking=True)
    rel_text_embs = rel_text_embs.to(device, non_blocking=True)

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device, dtype=torch.long)

    bucket_map = get_freq_buckets(processor, device)
    bucket_names = {0: "Tail", 1: "Torso", 2: "Head"}

    stats = {k: {"mrr": 0.0, "h1": 0, "h10": 0, "n": 0} for k in ["total", 0, 1, 2]}
    ranks_all = [] if collect_ranks else None

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        # gold total
        if refiner is None:
        s_gold_struct = rotate_model(h, r, t, mode="single")  # [B]
        else:
            anchor_emb = refiner.refine_anchor(
                h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
            conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
            s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, t.unsqueeze(1), conj=conj_flag).squeeze(1)
        s_gold_sem = sem_score_rhs_biencoder_pos(sem_model, ent_text_embs, rel_text_embs, h, r, t)  # [B]
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
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            if refiner is None:
            s_chunk = rotate_model(h, r, cand_1d, mode="batch_neg")
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            # --- filtered mask for rank counting (exclude gold) ---
            comp = s_chunk > s_gold_total.unsqueeze(1)
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

            # --- topK maintenance: exclude filtered + exclude gold ---
            s_for_topk = s_chunk.clone()

            # mask filtered (including gold for safety)
            if rows:
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

        # gate and gold total (consistent with topK)
        dir_ids = torch.zeros_like(h)
        if gate_model is None:
            g = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        else:
            g = gate_model(top_scores, r, dir_ids, ent_temp=gate_ent_temp).to(s_gold_struct.dtype)
        s_gold_total = s_gold_struct + (b * g) * s_gold_sem

        # pass 2: count greater over all entities with s_gold_total
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None:
                s_chunk = rotate_model(h, r, cand_1d, mode="batch_neg")
            else:
                anchor_emb = refiner.refine_anchor(
                    h, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            comp = s_chunk > s_gold_total.unsqueeze(1)
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

        # correction inside topK: replace struct comparison with total comparison
        # compute sem on top_ids
        sem_topk = sem_score_rhs_biencoder(sem_model, ent_text_embs, rel_text_embs, h, r, top_ids)
        total_topk = top_scores + (b * g).unsqueeze(1) * sem_topk

        greater_struct_topk = (top_scores > s_gold_total.unsqueeze(1)).sum(dim=1)
        greater_total_topk = (total_topk > s_gold_total.unsqueeze(1)).sum(dim=1)

        greater = greater_struct - greater_struct_topk + greater_total_topk
        rank = greater + 1  # [B]
        if collect_ranks:
            ranks_all.append(rank.detach().cpu())

        mrr = (1.0 / rank.float())
        h1 = (rank <= 1)
        h10 = (rank <= 10)

        stats["total"]["mrr"] += mrr.sum().item()
        stats["total"]["h1"] += h1.sum().item()
        stats["total"]["h10"] += h10.sum().item()
        stats["total"]["n"] += int(B)

        buckets = bucket_map[t]
        for bid in [0, 1, 2]:
            m = (buckets == bid)
            if m.any():
                stats[bid]["mrr"] += mrr[m].sum().item()
                stats[bid]["h1"] += h1[m].sum().item()
                stats[bid]["h10"] += h10[m].sum().item()
                stats[bid]["n"] += int(m.sum().item())

    # report
    def safe_div(a, b): return a / max(1, b)
    print("\n==================== RHS TopK-Injection (Exact Rank) ====================")
    for key in [0, 1, 2]:
        n = stats[key]["n"]
        name = bucket_names[key]
        print(f"RHS {name:<5} | MRR={safe_div(stats[key]['mrr'], n):.4f} | "
              f"H@1={safe_div(stats[key]['h1'], n):.4f} | H@10={safe_div(stats[key]['h10'], n):.4f} | n={n}")
    n = stats["total"]["n"]
    print(f"RHS TOTAL | MRR={safe_div(stats['total']['mrr'], n):.4f} | "
          f"H@1={safe_div(stats['total']['h1'], n):.4f} | H@10={safe_div(stats['total']['h10'], n):.4f} | n={n}")
    print("==========================================================================\n")

    if collect_ranks:
        return stats, torch.cat(ranks_all, dim=0)
    return stats, None


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
    collect_ranks=False,
):
    rotate_model.eval()
    sem_model.eval()

    ent_text_embs = ent_text_embs.to(device, non_blocking=True)
    rel_text_embs = rel_text_embs.to(device, non_blocking=True)

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device, dtype=torch.long)

    bucket_map = get_freq_buckets(processor, device)
    bucket_names = {0: "Tail", 1: "Torso", 2: "Head"}

    stats = {k: {"mrr": 0.0, "h1": 0, "h10": 0, "n": 0} for k in ["total", 0, 1, 2]}
    ranks_all = [] if collect_ranks else None

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        # gold total
        if refiner is None:
            s_gold_struct = rotate_model.score_head_batch(r, t, h.unsqueeze(1)).squeeze(1)
        else:
            anchor_emb = refiner.refine_anchor(
                t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
            )
            conj_flag = torch.ones(B, dtype=torch.bool, device=device)
            s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, h.unsqueeze(1), conj=conj_flag).squeeze(1)

        s_gold_sem = sem_score_lhs_biencoder_pos(sem_model, ent_text_embs, rel_text_embs, t, r, h)  # [B]
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
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            if refiner is None:
                s_chunk = rotate_model.score_head_batch(r, t, cand_1d)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.ones(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            # --- filtered mask for rank counting (exclude gold) ---
            comp = s_chunk > s_gold_total.unsqueeze(1)
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

            # --- topK maintenance: exclude filtered + exclude gold ---
            s_for_topk = s_chunk.clone()

            # mask filtered (including gold for safety)
            if rows:
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

        # gate and gold total (consistent with topK)
        dir_ids = torch.ones_like(t)
        if gate_model is None:
            g = torch.ones(B, device=device, dtype=s_gold_struct.dtype)
        else:
            g = gate_model(top_scores, r, dir_ids, ent_temp=gate_ent_temp).to(s_gold_struct.dtype)
        s_gold_total = s_gold_struct + (b * g) * s_gold_sem

        # pass 2: count greater over all entities with s_gold_total
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]
            if refiner is None:
                s_chunk = rotate_model.score_head_batch(r, t, cand_1d)
            else:
                anchor_emb = refiner.refine_anchor(
                    t, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq
                )
                conj_flag = torch.ones(B, dtype=torch.bool, device=device)
                s_chunk = rotate_model.score_from_head_emb(anchor_emb, r, cand_1d, conj=conj_flag)

            comp = s_chunk > s_gold_total.unsqueeze(1)
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

        # correction inside topK: replace struct comparison with total comparison
        sem_topk = sem_score_lhs_biencoder(sem_model, ent_text_embs, rel_text_embs, t, r, top_ids)
        total_topk = top_scores + (b * g).unsqueeze(1) * sem_topk

        greater_struct_topk = (top_scores > s_gold_total.unsqueeze(1)).sum(dim=1)
        greater_total_topk = (total_topk > s_gold_total.unsqueeze(1)).sum(dim=1)

        greater = greater_struct - greater_struct_topk + greater_total_topk
        rank = greater + 1  # [B]
        if collect_ranks:
            ranks_all.append(rank.detach().cpu())

        mrr = (1.0 / rank.float())
        h1 = (rank <= 1)
        h10 = (rank <= 10)

        stats["total"]["mrr"] += mrr.sum().item()
        stats["total"]["h1"] += h1.sum().item()
        stats["total"]["h10"] += h10.sum().item()
        stats["total"]["n"] += int(B)

        buckets = bucket_map[h]
        for bid in [0, 1, 2]:
            m = (buckets == bid)
            if m.any():
                stats[bid]["mrr"] += mrr[m].sum().item()
                stats[bid]["h1"] += h1[m].sum().item()
                stats[bid]["h10"] += h10[m].sum().item()
                stats[bid]["n"] += int(m.sum().item())

    # report
    def safe_div(a, b): return a / max(1, b)
    print("\n==================== LHS TopK-Injection (Exact Rank) ====================")
    for key in [0, 1, 2]:
        n = stats[key]["n"]
        name = bucket_names[key]
        print(f"LHS {name:<5} | MRR={safe_div(stats[key]['mrr'], n):.4f} | "
              f"H@1={safe_div(stats[key]['h1'], n):.4f} | H@10={safe_div(stats[key]['h10'], n):.4f} | n={n}")
    n = stats["total"]["n"]
    print(f"LHS TOTAL | MRR={safe_div(stats['total']['mrr'], n):.4f} | "
          f"H@1={safe_div(stats['total']['h1'], n):.4f} | H@10={safe_div(stats['total']['h10'], n):.4f} | n={n}")
    print("==========================================================================\n")

    if collect_ranks:
        return stats, torch.cat(ranks_all, dim=0)
    return stats, None


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
    ap.add_argument("--gate_g_max", type=float, default=1.0)
    ap.add_argument("--gate_ent_temp", type=float, default=1.0)

    ap.add_argument("--calib_path", type=str, default=None)
    ap.add_argument("--calib_b_max", type=float, default=None)

    ap.add_argument("--b_scale", type=float, default=0.0)
    ap.add_argument("--b_rhs", type=float, default=None)
    ap.add_argument("--b_lhs", type=float, default=None)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bootstrap_samples", type=int, default=0)
    ap.add_argument("--bootstrap_ci", type=float, default=0.95)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()

    refiner = None
    nbr_ent = nbr_rel = nbr_dir = nbr_mask = freq = None
    if args.pretrained_refiner:
        print(f"Loading StructRefiner: {args.pretrained_refiner}")
        refiner = StructRefiner(emb_dim=500, K=args.K).to(device)
        refiner.load_state_dict(torch.load(args.pretrained_refiner, map_location=device))
        refiner.eval()
        nbr_ent = processor.nbr_ent.to(device, non_blocking=True)
        nbr_rel = processor.nbr_rel.to(device, non_blocking=True)
        nbr_dir = processor.nbr_dir.to(device, non_blocking=True)
        nbr_mask = processor.nbr_mask.to(device, non_blocking=True)
        freq = processor.freq.to(device, non_blocking=True)

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    sem = load_sem_model(args.pretrained_sem, ent_embs.size(1), processor.num_relations, device)

    gate_model = None
    if args.pretrained_gate:
        gate_model = ConfidenceGate(
            num_relations=processor.num_relations,
            rel_emb_dim=args.gate_rel_dim,
            dir_emb_dim=args.gate_dir_dim,
            hidden_dim=args.gate_hidden_dim,
            g_max=args.gate_g_max,
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

    rhs_stats = lhs_stats = None
    rhs_ranks = lhs_ranks = None

    b_rhs = float(args.b_scale) if args.b_rhs is None else float(args.b_rhs)
    b_lhs = float(args.b_scale) if args.b_lhs is None else float(args.b_lhs)

    if args.eval_sides in ["rhs", "both"]:
        rhs_stats, rhs_ranks = eval_rhs_topk_inject(
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
        )

    if args.eval_sides in ["lhs", "both"]:
        lhs_stats, lhs_ranks = eval_lhs_topk_inject(
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
        )

    if args.eval_sides == "both" and rhs_stats and lhs_stats:
        def safe_div(a, b): return a / max(1, b)
        print("\n==================== AVG (RHS+LHS) ====================")
        for key in [0, 1, 2, "total"]:
            n = rhs_stats[key]["n"]
            mrr = 0.5 * (safe_div(rhs_stats[key]["mrr"], n) + safe_div(lhs_stats[key]["mrr"], n))
            h1 = 0.5 * (safe_div(rhs_stats[key]["h1"], n) + safe_div(lhs_stats[key]["h1"], n))
            h10 = 0.5 * (safe_div(rhs_stats[key]["h10"], n) + safe_div(lhs_stats[key]["h10"], n))
            name = "TOTAL" if key == "total" else {0: "Tail", 1: "Torso", 2: "Head"}[key]
            print(f"AVG {name:<5} | MRR={mrr:.4f} | H@1={h1:.4f} | H@10={h10:.4f} | n={n}")
        print("========================================================\n")

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


if __name__ == "__main__":
    main()
