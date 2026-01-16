import argparse
import torch
import os
import sys
import bisect
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from models.text_encoder import TextEncoder
from models.semantic_residual import SemanticResidualScorerV2
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.struct_refiner import StructRefiner


# =========================================================
# 1. Helpers
# =========================================================

@torch.no_grad()
def build_to_skip(processor: KGProcessor, split: str):
    """
    Filter sets for filtered ranking evaluation.

    - split=valid: filter uses train+valid truths
    - split=test : filter uses train+valid+test truths (standard filtered test)
    """
    if split == "valid":
        all_triplets = torch.cat([processor.train_triplets, processor.valid_triplets], dim=0)
    elif split == "test":
        all_triplets = torch.cat([processor.train_triplets, processor.valid_triplets, processor.test_triplets], dim=0)
    else:
        raise ValueError(f"Unknown split: {split}")

    all_list = all_triplets.tolist()
    to_skip = {"rhs": {}, "lhs": {}}
    for h, r, t in all_list:
        h, r, t = int(h), int(r), int(t)
        to_skip["rhs"].setdefault((h, r), set()).add(t)   # (h,r) -> {t}
        to_skip["lhs"].setdefault((t, r), set()).add(h)   # (t,r) -> {h}
    return to_skip


@torch.no_grad()
def load_embeddings(processor, args, device):
    ent_emb_path = os.path.join(args.data_path, "text_embs.pt")
    if os.path.exists(ent_emb_path):
        print(f"[Init] Loading cached entity embeddings: {ent_emb_path}")
        ent_embs = torch.load(ent_emb_path, map_location="cpu")
    else:
        print("[Init] Cache not found. Initializing BERT for entities...")
        encoder = TextEncoder(device=device)
        ent_texts = [""] * processor.num_entities
        text_file = os.path.join(args.data_path, "entity2text.txt")
        if os.path.exists(text_file):
            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 2:
                        key_str, content = parts[0], parts[1]
                        if key_str in processor.entity2id:
                            ent_texts[processor.entity2id[key_str]] = content
        ent_embs = encoder.encode_all_entities(ent_texts, save_path=ent_emb_path)

    rel_emb_path = os.path.join(args.data_path, "rel_text_embs.pt")
    if os.path.exists(rel_emb_path):
        print(f"[Init] Loading cached relation embeddings: {rel_emb_path}")
        rel_embs = torch.load(rel_emb_path, map_location="cpu")
    else:
        print("[Init] Cache not found. Initializing BERT for relations...")
        encoder = TextEncoder(device=device)
        rel_texts = [""] * processor.num_relations
        rel_file = os.path.join(args.data_path, "relation2text.txt")
        if os.path.exists(rel_file):
            with open(rel_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 2:
                        key_str, content = parts[0], parts[1]
                        if key_str in processor.relation2id:
                            rel_texts[processor.relation2id[key_str]] = content
        rel_embs = encoder.encode_all_relations(rel_texts, save_path=rel_emb_path)

    # keep embeddings on CPU first; move to GPU inside eval when needed
    return ent_embs, rel_embs


@torch.no_grad()
def get_freq_buckets(processor, device):
    """
    Bucket entities by train frequency:
    - Head: top 10%
    - Torso: next 40%
    - Tail: bottom 50%
    """
    triplets = processor.train_triplets
    all_nodes = torch.cat([triplets[:, 0], triplets[:, 2]])
    counts = torch.bincount(all_nodes, minlength=processor.num_entities).float()

    sorted_counts, sorted_indices = torch.sort(counts, descending=True)
    num_ent = processor.num_entities
    idx_10 = int(num_ent * 0.1)
    idx_50 = int(num_ent * 0.5)

    thresh_head = sorted_counts[idx_10]
    thresh_tail = sorted_counts[idx_50]

    bucket_map = torch.zeros(num_ent, dtype=torch.long, device=device)  # 0: Tail, 1: Torso, 2: Head
    bucket_map[sorted_indices[:idx_10]] = 2
    bucket_map[sorted_indices[idx_10:idx_50]] = 1

    print(f"[Stats] Head Threshold: >{thresh_head:.0f} (Top 10%)")
    print(f"[Stats] Tail Threshold: <={thresh_tail:.0f} (Bottom 50%)")
    return bucket_map


def _stat_line(x: torch.Tensor):
    x = x.detach()
    return (
        f"mean={x.mean().item():.6f} std={x.std(unbiased=False).item():.6f} "
        f"min={x.min().item():.6f} max={x.max().item():.6f} "
        f"range={(x.max()-x.min()).item():.6f} "
        f"abs_mean={x.abs().mean().item():.6f} abs_max={x.abs().max().item():.6f}"
    )


# =========================================================
# 2. Core evaluation (bidirectional, filtered, chunked)
# =========================================================

@torch.no_grad()
def eval_chunked_bidirectional(
        processor,
        rotate_model,
        test_loader,
        device,
        to_skip,
        eval_split: str,
        refiner=None,
        semres_model=None,
        ent_text_embs=None,
        rel_text_embs=None,
        chunk_size=2048,
        sem_subchunk=256,
        disable_refiner=False,
        disable_semres=False,
        sem_rhs_only=False,
        sem_lhs_only=False,
        print_sem_stats=False,
        verbose_every=50,
):
    rotate_model.eval()
    if refiner:
        refiner.eval()
    if semres_model:
        semres_model.eval()

    if disable_refiner:
        refiner = None
    if disable_semres:
        semres_model, ent_text_embs, rel_text_embs = None, None, None

    if sem_rhs_only and sem_lhs_only:
        raise ValueError("Cannot set both sem_rhs_only and sem_lhs_only.")

    apply_sem_rhs = (semres_model is not None) and (not sem_lhs_only)
    apply_sem_lhs = (semres_model is not None) and (not sem_rhs_only)

    # --- topology tensors for refiner ---
    if refiner is not None:
        nbr_ent = processor.nbr_ent.to(device, non_blocking=True)
        nbr_rel = processor.nbr_rel.to(device, non_blocking=True)
        nbr_dir = processor.nbr_dir.to(device, non_blocking=True)
        nbr_mask = processor.nbr_mask.to(device, non_blocking=True)
        freq = processor.freq.to(device, non_blocking=True)
    else:
        nbr_ent = nbr_rel = nbr_dir = nbr_mask = freq = None

    # --- move text embeddings to device (once) ---
    if semres_model is not None:
        ent_text_embs = ent_text_embs.to(device, non_blocking=True)
        rel_text_embs = rel_text_embs.to(device, non_blocking=True)

    print(f"Building Filter Sets... (split={eval_split})")
    print("Calculating Frequency Buckets...")
    bucket_map = get_freq_buckets(processor, device)
    bucket_names = {0: "Tail", 1: "Torso", 2: "Head"}

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device)

    keys = ["total", 0, 1, 2]
    stats = {
        "rhs": {k: {"mrr": 0.0, "h1": 0, "h10": 0, "n": 0} for k in keys},
        "lhs": {k: {"mrr": 0.0, "h1": 0, "h10": 0, "n": 0} for k in keys},
    }

    # --- helpers ---
    def get_anchor_emb(anchor_ids):
        if refiner is None:
            return rotate_model.entity_embedding[anchor_ids]
        return refiner.refine_anchor(anchor_ids, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq)

    def sem_forward(h_ids, r_ids, t_ids, rel_ids):
        # h_ids,r_ids,t_ids,rel_ids are 1D aligned tensors
        h_txt = ent_text_embs[h_ids]
        r_txt = rel_text_embs[r_ids]
        t_txt = ent_text_embs[t_ids]
        delta, lam = semres_model(h_txt, r_txt, t_txt, rel_ids)
        return delta, lam, lam * delta

    def sem_rhs_scores(h_ids, r_ids, cand_t_1d):
        # returns [B, C]
        B = h_ids.size(0)
        C = cand_t_1d.size(0)
        h_txt = ent_text_embs[h_ids]          # [B,D]
        r_txt = rel_text_embs[r_ids]          # [B,D]
        t_txt = ent_text_embs[cand_t_1d]      # [C,D]

        # fast path for biencoder
        if hasattr(semres_model, "encode_query") and hasattr(semres_model, "encode_entity"):
            q = semres_model.encode_query(h_txt, r_txt)      # [B,d]
            v = semres_model.encode_entity(t_txt)            # [C,d]
            return q @ v.t()                                 # [B,C]

        # fallback (old semres)
        out = torch.empty(B, C, device=device)
        t_txt_all = t_txt
        for s in range(0, C, sem_subchunk):
            e = min(s + sem_subchunk, C)
            t_sub = t_txt_all[s:e]
            c = e - s
            h_rep = h_txt.repeat_interleave(c, dim=0)
            r_rep = r_txt.repeat_interleave(c, dim=0)
            t_rep = t_sub.repeat(B, 1)
            rel_rep = r_ids.repeat_interleave(c, dim=0)
            delta, lam = semres_model(h_rep, r_rep, t_rep, rel_rep)
            out[:, s:e] = (lam * delta).view(B, c)
        return out

    def sem_lhs_scores(t_ids, r_ids, cand_h_1d):
        # returns [B, C]
        B = t_ids.size(0)
        C = cand_h_1d.size(0)
        t_txt = ent_text_embs[t_ids]
        r_txt = rel_text_embs[r_ids]
        h_txt = ent_text_embs[cand_h_1d]

        if hasattr(semres_model, "encode_query") and hasattr(semres_model, "encode_entity"):
            q = semres_model.encode_query(t_txt, r_txt)  # [B,d]
            v = semres_model.encode_entity(h_txt)        # [C,d]
            return q @ v.t()

        # fallback (old semres)
        out = torch.empty(B, C, device=device)
        h_txt_all = h_txt
        for s in range(0, C, sem_subchunk):
            e = min(s + sem_subchunk, C)
            h_sub = h_txt_all[s:e]
            c = e - s
            h_rep = h_sub.repeat(B, 1)
            r_rep = r_txt.repeat_interleave(c, dim=0)
            t_rep = t_txt.repeat_interleave(c, dim=0)
            rel_rep = r_ids.repeat_interleave(c, dim=0)
            delta, lam = semres_model(h_rep, r_rep, t_rep, rel_rep)
            out[:, s:e] = (lam * delta).view(B, c)
        return out

    def update_stats(side, rank, target_ids):
        mrr_val = (1.0 / rank.float())
        h1_val = (rank <= 1)
        h10_val = (rank <= 10)

        stats[side]["total"]["mrr"] += mrr_val.sum().item()
        stats[side]["total"]["h1"] += h1_val.sum().item()
        stats[side]["total"]["h10"] += h10_val.sum().item()
        stats[side]["total"]["n"] += int(target_ids.numel())

        buckets = bucket_map[target_ids]
        for bid in [0, 1, 2]:
            mask = (buckets == bid)
            if mask.any():
                stats[side][bid]["mrr"] += mrr_val[mask].sum().item()
                stats[side][bid]["h1"] += h1_val[mask].sum().item()
                stats[side][bid]["h10"] += h10_val[mask].sum().item()
                stats[side][bid]["n"] += int(mask.sum().item())

    print(f"Start Bidirectional Evaluation (split={eval_split}, Chunk={chunk_size}, SemSubChunk={sem_subchunk}, Bisect Optimized)...")
    for b_idx, batch in enumerate(test_loader):
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        # ======================
        # RHS: predict t
        # ======================
        anchor_emb = get_anchor_emb(h)
        conj_flag = torch.zeros(B, dtype=torch.bool, device=device)

        # gold
        s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, t.unsqueeze(1), conj=conj_flag).squeeze(1)
        s_gold = s_gold_struct

        if apply_sem_rhs:
            delta_g, lam_g, s_gold_sem = sem_forward(h, r, t, r)
            s_gold = s_gold + s_gold_sem
        else:
            delta_g = lam_g = s_gold_sem = None

        # precompute sorted filters for bisect
        rhs_filters_sorted = []
        for i in range(B):
            filt = to_skip["rhs"].get((h_cpu[i], r_cpu[i]), set())
            rhs_filters_sorted.append(sorted(list(filt)))

        greater = torch.zeros(B, dtype=torch.long, device=device)

        first_chunk_struct = None
        first_chunk_sem = None

        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand = all_ent_ids[start:end]  # [C]

            s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand, conj=conj_flag)
            s_chunk = s_chunk_struct

            if apply_sem_rhs:
                s_chunk_sem = sem_rhs_scores(h, r, cand)
                s_chunk = s_chunk + s_chunk_sem
            else:
                s_chunk_sem = None

            # optional stats dump: only first batch and first chunk
            if print_sem_stats and b_idx == 0 and start == 0:
                print("\n==================== DEBUG (first batch, first chunk) ====================\n")
                print("[RHS] Predict tail t")
                print(f"s_gold_struct:  {_stat_line(s_gold_struct)}")
                print(f"s_chunk_struct: {_stat_line(s_chunk_struct)}")
                if apply_sem_rhs:
                    print(f"delta_gold:     {_stat_line(delta_g)}")
                    print(f"lambda_gold:    {_stat_line(lam_g)}")
                    print(f"s_gold_sem:     {_stat_line(s_gold_sem)}")
                    print(f"s_chunk_sem:    {_stat_line(s_chunk_sem)}")
                    std_ratio = (s_chunk_sem.std(unbiased=False) / (s_chunk_struct.std(unbiased=False) + 1e-12)).item()
                    range_ratio = ((s_chunk_sem.max()-s_chunk_sem.min()) / ((s_chunk_struct.max()-s_chunk_struct.min()) + 1e-12)).item()
                    print(f"[RHS] std_ratio   = {std_ratio:.6f}")
                    print(f"[RHS] range_ratio = {range_ratio:.6f}")

            comp = s_chunk > s_gold.unsqueeze(1)

            # build mask for filtered entities (excluding gold)
            mask = torch.zeros_like(comp, dtype=torch.bool)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = rhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low_idx = bisect.bisect_left(filt_sorted, start)
                high_idx = bisect.bisect_left(filt_sorted, end)
                gold_val = t_cpu[i]
                for idx in range(low_idx, high_idx):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask[rows, cols] = True

            greater += (comp & (~mask)).sum(dim=1)

        rank = greater + 1
        update_stats("rhs", rank, t)

        # ======================
        # LHS: predict h
        # ======================
        anchor_emb = get_anchor_emb(t)
        conj_flag = torch.ones(B, dtype=torch.bool, device=device)

        s_gold_struct = rotate_model.score_from_head_emb(anchor_emb, r, h.unsqueeze(1), conj=conj_flag).squeeze(1)
        s_gold = s_gold_struct

        if apply_sem_lhs:
            # NOTE: sem model still uses (h,r,t) triple; here gold is same triple, only prediction side changes.
            delta_g2, lam_g2, s_gold_sem = sem_forward(h, r, t, r)
            s_gold = s_gold + s_gold_sem
        else:
            delta_g2 = lam_g2 = s_gold_sem = None

        lhs_filters_sorted = []
        for i in range(B):
            filt = to_skip["lhs"].get((t_cpu[i], r_cpu[i]), set())
            lhs_filters_sorted.append(sorted(list(filt)))

        greater = torch.zeros(B, dtype=torch.long, device=device)

        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand = all_ent_ids[start:end]

            s_chunk_struct = rotate_model.score_from_head_emb(anchor_emb, r, cand, conj=conj_flag)
            s_chunk = s_chunk_struct

            if apply_sem_lhs:
                s_chunk_sem = sem_lhs_scores(t, r, cand)
                s_chunk = s_chunk + s_chunk_sem
            else:
                s_chunk_sem = None

            if print_sem_stats and b_idx == 0 and start == 0:
                print("\n[LHS] Predict head h")
                print(f"s_gold_struct:  {_stat_line(s_gold_struct)}")
                print(f"s_chunk_struct: {_stat_line(s_chunk_struct)}")
                if apply_sem_lhs:
                    print(f"s_gold_sem:     {_stat_line(s_gold_sem)}")
                    print(f"s_chunk_sem:    {_stat_line(s_chunk_sem)}")
                    std_ratio = (s_chunk_sem.std(unbiased=False) / (s_chunk_struct.std(unbiased=False) + 1e-12)).item()
                    range_ratio = ((s_chunk_sem.max()-s_chunk_sem.min()) / ((s_chunk_struct.max()-s_chunk_struct.min()) + 1e-12)).item()
                    print(f"[LHS] std_ratio   = {std_ratio:.6f}")
                    print(f"[LHS] range_ratio = {range_ratio:.6f}")
                    print("\n==========================================================================\n")

            comp = s_chunk > s_gold.unsqueeze(1)

            mask = torch.zeros_like(comp, dtype=torch.bool)
            rows, cols = [], []
            for i in range(B):
                filt_sorted = lhs_filters_sorted[i]
                if not filt_sorted:
                    continue
                low_idx = bisect.bisect_left(filt_sorted, start)
                high_idx = bisect.bisect_left(filt_sorted, end)
                gold_val = h_cpu[i]
                for idx in range(low_idx, high_idx):
                    fid = filt_sorted[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask[rows, cols] = True

            greater += (comp & (~mask)).sum(dim=1)

        rank = greater + 1
        update_stats("lhs", rank, h)

        if verbose_every and (b_idx + 1) % verbose_every == 0:
            print(f"Evaluated {b_idx + 1} batches...")

    # --- Reporting ---
    print("\n" + "=" * 60)
    print(f"{'Metric':<10} | {'Set':<8} | {'MRR':<8} | {'H@1':<8} | {'H@10':<8} | {'Count':<8}")
    print("-" * 60)

    results = {"rhs": {}, "lhs": {}, "avg": {}}

    for side in ["rhs", "lhs"]:
        n_total = max(stats[side]["total"]["n"], 1)
        results[side]["total"] = {
            "MRR": stats[side]["total"]["mrr"] / n_total,
            "H1": stats[side]["total"]["h1"] / n_total,
            "H10": stats[side]["total"]["h10"] / n_total
        }

        for bid in [0, 1, 2]:
            n_b = max(stats[side][bid]["n"], 1)
            bname = bucket_names[bid]
            results[side][bname] = {
                "MRR": stats[side][bid]["mrr"] / n_b,
                "H1": stats[side][bid]["h1"] / n_b,
                "H10": stats[side][bid]["h10"] / n_b
            }

            row_name = f"{side.upper()} {bname}"
            print(
                f"{row_name:<10} | {bname:<8} | {results[side][bname]['MRR']:.4f}   | "
                f"{results[side][bname]['H1']:.4f}   | {results[side][bname]['H10']:.4f}   | {n_b:<8}"
            )

        print(
            f"{side.upper()} TOTAL | ALL      | {results[side]['total']['MRR']:.4f}   | "
            f"{results[side]['total']['H1']:.4f}   | {results[side]['total']['H10']:.4f}   | {n_total:<8}"
        )
        print("-" * 60)

    avg_mrr = (results["rhs"]["total"]["MRR"] + results["lhs"]["total"]["MRR"]) / 2
    avg_h1 = (results["rhs"]["total"]["H1"] + results["lhs"]["total"]["H1"]) / 2
    avg_h10 = (results["rhs"]["total"]["H10"] + results["lhs"]["total"]["H10"]) / 2

    print(f"OVERALL    | AVG      | {avg_mrr:.4f}   | {avg_h1:.4f}   | {avg_h10:.4f}   | {n_total * 2:<8}")
    print("=" * 60 + "\n")

    return results


# =========================================================
# 3. main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/fb15k_custom")

    parser.add_argument("--pretrained_rotate", type=str, required=True)
    parser.add_argument("--pretrained_refiner", type=str, default=None)
    parser.add_argument("--pretrained_semres", type=str, default=None)

    parser.add_argument("--disable_refiner", action="store_true")
    parser.add_argument("--disable_semres", action="store_true")

    parser.add_argument("--sem_rhs_only", action="store_true")
    parser.add_argument("--sem_lhs_only", action="store_true")
    parser.add_argument("--print_sem_stats", action="store_true")

    parser.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])

    parser.add_argument("--emb_dim", type=int, default=500)
    parser.add_argument("--margin", type=float, default=9.0)

    parser.add_argument("--lambda_sem", type=float, default=0.05)
    parser.add_argument("--K", type=int, default=16)

    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--sem_subchunk", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Bidirectional Evaluation on {device} (split={args.eval_split}) ===")

    processor = KGProcessor(args.data_path, max_neighbors=args.K)
    processor.load_files()

    if not args.disable_refiner:
        processor.nbr_ent = processor.nbr_ent.to(device)
        processor.nbr_rel = processor.nbr_rel.to(device)
        processor.nbr_dir = processor.nbr_dir.to(device)
        processor.nbr_mask = processor.nbr_mask.to(device)
        processor.freq = processor.freq.to(device)

    print(f"Loading RotatE: {args.pretrained_rotate}")
    rotate_model = RotatEModel(processor.num_entities, processor.num_relations, args.emb_dim, args.margin).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))

    refiner = None
    if args.pretrained_refiner and not args.disable_refiner:
        print(f"Loading StructRefiner: {args.pretrained_refiner}")
        refiner = StructRefiner(args.emb_dim, args.K).to(device)
        refiner.load_state_dict(torch.load(args.pretrained_refiner, map_location=device))

    semres_model = None
    ent_embs, rel_embs = None, None
    if args.pretrained_semres and not args.disable_semres:
        print(f"Loading SemRes: {args.pretrained_semres}")
        ent_embs, rel_embs = load_embeddings(processor, args, device)
        ckpt = torch.load(args.pretrained_semres, map_location="cpu")

        # case A: new biencoder ckpt
        if isinstance(ckpt, dict) and ckpt.get("model_type", "") == "biencoder":
            cfg = ckpt["model_args"]
            semres_model = SemanticBiEncoderScorer(
                text_dim=cfg["text_dim"],
                num_relations=cfg["num_relations"],
                proj_dim=cfg.get("proj_dim", 256),
                dropout=cfg.get("dropout", 0.1),
                text_norm=cfg.get("text_norm", True),
            ).to(device)
            semres_model.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            # case B: old semres state_dict
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            semres_model = SemanticResidualScorerV2(
                ent_embs.size(1),
                processor.num_relations,
                args.lambda_sem,
            ).to(device)
            semres_model.load_state_dict(ckpt, strict=True)

    split_triplets = processor.valid_triplets if args.eval_split == "valid" else processor.test_triplets
    to_skip = build_to_skip(processor, split=args.eval_split)

    test_loader = DataLoader(
        TrainDataset(split_triplets),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    _ = eval_chunked_bidirectional(
        processor=processor,
        rotate_model=rotate_model,
        test_loader=test_loader,
        device=device,
        to_skip=to_skip,
        eval_split=args.eval_split,
        refiner=refiner,
        semres_model=semres_model,
        ent_text_embs=ent_embs,
        rel_text_embs=rel_embs,
        chunk_size=args.chunk_size,
        sem_subchunk=args.sem_subchunk,
        disable_refiner=args.disable_refiner,
        disable_semres=args.disable_semres,
        sem_rhs_only=args.sem_rhs_only,
        sem_lhs_only=args.sem_lhs_only,
        print_sem_stats=args.print_sem_stats,
    )


if __name__ == "__main__":
    main()