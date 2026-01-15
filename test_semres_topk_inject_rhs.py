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
    q = sem_model.encode_query(h_txt, r_txt)  # [B,d]
    v = sem_model.encode_entity(ent_embs[cand_t_2d.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q, v)


@torch.no_grad()
def sem_score_rhs_biencoder_pos(sem_model, ent_embs, rel_embs, h, r, t):
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    t_txt = ent_embs[t]
    q = sem_model.encode_query(h_txt, r_txt)
    v = sem_model.encode_entity(t_txt)
    return (q * v).sum(dim=-1)  # [B]


@torch.no_grad()
def eval_rhs_topk_inject(
    processor,
    rotate_model,
    sem_model,
    ent_text_embs,
    rel_text_embs,
    loader,
    to_skip,
    split,
    device,
    topk=500,
    chunk_size=2048,
    b_scale=0.0,
    get_b_fn=None,
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

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

        # gold total
        s_gold_struct = rotate_model(h, r, t, mode="single")  # [B]
        s_gold_sem = sem_score_rhs_biencoder_pos(sem_model, ent_text_embs, rel_text_embs, h, r, t)  # [B]
        if get_b_fn is None:
            b = torch.full((B,), float(b_scale), device=device, dtype=s_gold_struct.dtype)
        else:
            b = get_b_fn(r).to(device, non_blocking=True).to(s_gold_struct.dtype)

        s_gold_total = s_gold_struct + b * s_gold_sem

        # precompute sorted filters
        rhs_filters_sorted = []
        for i in range(B):
            filt = to_skip["rhs"].get((h_cpu[i], r_cpu[i]), set())
            rhs_filters_sorted.append(sorted(list(filt)))

        # for rank counting with base struct (non-topk entities)
        greater_struct = torch.zeros(B, dtype=torch.long, device=device)

        # maintain topK by struct (excluding gold + filtered)
        top_scores = torch.full((B, topk), -1e9, device=device)
        top_ids = torch.full((B, topk), -1, device=device, dtype=torch.long)

        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_1d = all_ent_ids[start:end]  # contiguous chunk [C]

            # struct scores for this chunk: [B,C]
            s_chunk = rotate_model(h, r, cand_1d, mode="batch_neg")

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

        # correction inside topK: replace struct comparison with total comparison
        # compute sem on top_ids
        sem_topk = sem_score_rhs_biencoder(sem_model, ent_text_embs, rel_text_embs, h, r, top_ids)
        total_topk = top_scores + b.unsqueeze(1) * sem_topk

        greater_struct_topk = (top_scores > s_gold_total.unsqueeze(1)).sum(dim=1)
        greater_total_topk = (total_topk > s_gold_total.unsqueeze(1)).sum(dim=1)

        greater = greater_struct - greater_struct_topk + greater_total_topk
        rank = greater + 1  # [B]

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
    for key in [0, 1, 2, "total"]:
        n = stats[key]["n"]
        name = "TOTAL" if key == "total" else bucket_names[key]
        print(f"RHS {name:<5} | MRR={safe_div(stats[key]['mrr'], n):.4f} | "
              f"H@1={safe_div(stats[key]['h1'], n):.4f} | H@10={safe_div(stats[key]['h10'], n):.4f} | n={n}")
    print("==========================================================================\n")


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

    ap.add_argument("--calib_path", type=str, default=None)
    ap.add_argument("--calib_b_max", type=float, default=None)

    ap.add_argument("--b_scale", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    sem = load_sem_model(args.pretrained_sem, ent_embs.size(1), processor.num_relations, device)

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

    eval_rhs_topk_inject(
        processor=processor,
        rotate_model=rotate,
        sem_model=sem,
        ent_text_embs=ent_embs,
        rel_text_embs=rel_embs,
        loader=loader,
        to_skip=to_skip,
        split=args.eval_split,
        device=device,
        topk=args.topk,
        chunk_size=args.chunk_size,
        b_scale=args.b_scale,
        get_b_fn=get_b_fn,
    )


if __name__ == "__main__":
    main()
