import argparse
import json
import os
import time
import bisect
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.semantic_biencoder import SemanticBiEncoderScorer
from eval.eval_full_entity_filtered import build_to_skip, load_embeddings
from tools.run_meta import write_run_metadata


def _tensor_size_mb(t: torch.Tensor) -> float:
    return (t.numel() * t.element_size()) / (1024 * 1024)


def _model_size_mb(model: torch.nn.Module) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 * 1024)


@torch.no_grad()
def build_sem_index(sem_model, ent_text_embs, rel_text_embs, device, batch_size=1024):
    sem_model.eval()
    num_ent = ent_text_embs.size(0)
    num_rel = rel_text_embs.size(0)

    ent_index = torch.empty((num_ent, sem_model.proj_dim), dtype=torch.float32)
    for start in range(0, num_ent, batch_size):
        end = min(start + batch_size, num_ent)
        batch = ent_text_embs[start:end].to(device, non_blocking=True)
        vec = sem_model.encode_entity(batch).to("cpu")
        ent_index[start:end] = vec

    # precompute relation vectors for both directions
    rel_index_dir0 = torch.empty((num_rel, sem_model.proj_dim), dtype=torch.float32)
    rel_index_dir1 = torch.empty((num_rel, sem_model.proj_dim), dtype=torch.float32)
    dir0 = sem_model.dir_emb.weight[0].detach().to(device)
    dir1 = sem_model.dir_emb.weight[1].detach().to(device)
    for start in range(0, num_rel, batch_size):
        end = min(start + batch_size, num_rel)
        rel_batch = rel_text_embs[start:end].to(device, non_blocking=True)
        rel0 = sem_model.encode_relation(rel_batch + dir0).to("cpu")
        rel1 = sem_model.encode_relation(rel_batch + dir1).to("cpu")
        rel_index_dir0[start:end] = rel0
        rel_index_dir1[start:end] = rel1

    return ent_index, rel_index_dir0, rel_index_dir1


@torch.no_grad()
def eval_sem_only(
    processor,
    sem_model,
    ent_index,
    rel_index_dir0,
    rel_index_dir1,
    loader,
    to_skip,
    device,
    topk=200,
    chunk_size=2048,
    eval_side="rhs",
):
    ent_index = ent_index.to(device, non_blocking=True)
    rel_index_dir0 = rel_index_dir0.to(device, non_blocking=True)
    rel_index_dir1 = rel_index_dir1.to(device, non_blocking=True)

    num_ent = processor.num_entities
    all_ent_ids = torch.arange(num_ent, device=device, dtype=torch.long)

    stats = {"mrr": 0.0, "h1": 0, "h3": 0, "h10": 0, "rec": 0, "n": 0}

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h.size(0)

        if eval_side == "rhs":
            anchor_ids, rel_ids, gold_ids = h, r, t
            dir_flag = 0
            filters = to_skip["rhs"]
        else:
            anchor_ids, rel_ids, gold_ids = t, r, h
            dir_flag = 1
            filters = to_skip["lhs"]

        anchor_vec = ent_index[anchor_ids]
        rel_vec = (rel_index_dir0 if dir_flag == 0 else rel_index_dir1)[rel_ids]

        feat = torch.cat([anchor_vec, rel_vec, anchor_vec * rel_vec, anchor_vec - rel_vec], dim=-1)
        q = F.normalize(sem_model.q_fuse(feat), p=2, dim=-1)

        gold_vec = ent_index[gold_ids]
        s_gold = (q * gold_vec).sum(dim=-1)

        # filters sorted for masking
        anchor_cpu = anchor_ids.tolist()
        rel_cpu = rel_ids.tolist()
        gold_cpu = gold_ids.tolist()
        filt_sorted = []
        for i in range(B):
            key = (anchor_cpu[i], rel_cpu[i])
            filt_sorted.append(sorted(list(filters.get(key, set()))))

        top_scores = torch.full((B, topk), -1e9, device=device)
        top_ids = torch.full((B, topk), -1, device=device, dtype=torch.long)

        greater = torch.zeros(B, dtype=torch.long, device=device)
        for start in range(0, num_ent, chunk_size):
            end = min(start + chunk_size, num_ent)
            cand_ids = all_ent_ids[start:end]
            cand_vec = ent_index[start:end]  # [C, d]
            s_chunk = torch.matmul(q, cand_vec.t())  # [B, C]

            # mask filtered (excluding gold)
            rows, cols = [], []
            for i in range(B):
                lst = filt_sorted[i]
                if not lst:
                    continue
                lo = bisect.bisect_left(lst, start)
                hi = bisect.bisect_left(lst, end)
                gold_val = gold_cpu[i]
                for idx in range(lo, hi):
                    fid = lst[idx]
                    if fid == gold_val:
                        continue
                    rows.append(i)
                    cols.append(fid - start)
            if rows:
                mask = torch.zeros_like(s_chunk, dtype=torch.bool)
                mask[rows, cols] = True
                s_chunk = s_chunk.masked_fill(mask, -1e9)

            # update topK
            cand_2d = cand_ids.unsqueeze(0).expand(B, -1)
            merged_scores = torch.cat([top_scores, s_chunk], dim=1)
            merged_ids = torch.cat([top_ids, cand_2d], dim=1)
            top_scores, idx = torch.topk(merged_scores, k=topk, dim=1)
            top_ids = torch.gather(merged_ids, 1, idx)

            # count greater than gold (excluding filtered + gold)
            comp = s_chunk > s_gold.unsqueeze(1)
            greater += comp.sum(dim=1)

        # recall@K
        hit = (top_ids == gold_ids.unsqueeze(1)).any(dim=1)
        stats["rec"] += int(hit.sum().item())

        rank = greater + 1
        mrr = (1.0 / rank.float())
        h1 = (rank <= 1)
        h3 = (rank <= 3)
        h10 = (rank <= 10)

        stats["mrr"] += mrr.sum().item()
        stats["h1"] += h1.sum().item()
        stats["h3"] += h3.sum().item()
        stats["h10"] += h10.sum().item()
        stats["n"] += int(B)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_sem", type=str, required=True)
    ap.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"])
    ap.add_argument("--eval_sides", type=str, default="both", choices=["rhs", "lhs", "both"])
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("artifacts", f"eval_sem_only_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)
    write_run_metadata(args.out_dir, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Sem-only Full-Entity Evaluation on {device} (split={args.eval_split}) ===")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    ent_text_embs, rel_text_embs = load_embeddings(processor, args, device=torch.device("cpu"))

    # load biencoder
    ckpt = torch.load(args.pretrained_sem, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("model_type", "") == "biencoder":
        cfg = ckpt["model_args"]
        sem_model = SemanticBiEncoderScorer(
            text_dim=cfg["text_dim"],
            num_relations=cfg["num_relations"],
            proj_dim=cfg.get("proj_dim", 256),
            dropout=cfg.get("dropout", 0.1),
            text_norm=cfg.get("text_norm", True),
        ).to(device)
        sem_model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        raise ValueError("Expected a biencoder checkpoint with model_type='biencoder'.")

    # build index
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ent_index, rel_index_dir0, rel_index_dir1 = build_sem_index(
        sem_model,
        ent_text_embs,
        rel_text_embs,
        device=device,
        batch_size=1024,
    )
    index_time = time.time() - t0

    to_skip = build_to_skip(processor, split=args.eval_split)
    loader = DataLoader(
        TrainDataset(getattr(processor, f"{args.eval_split}_triplets")),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results = {}
    total_time = 0.0
    avg_mrr = None
    if args.eval_sides in ["rhs", "both"]:
        t1 = time.time()
        stats = eval_sem_only(
            processor,
            sem_model,
            ent_index,
            rel_index_dir0,
            rel_index_dir1,
            loader,
            to_skip,
            device,
            topk=args.topk,
            chunk_size=args.chunk_size,
            eval_side="rhs",
        )
        total_time += time.time() - t1
        results["rhs"] = stats
    if args.eval_sides in ["lhs", "both"]:
        t1 = time.time()
        stats = eval_sem_only(
            processor,
            sem_model,
            ent_index,
            rel_index_dir0,
            rel_index_dir1,
            loader,
            to_skip,
            device,
            topk=args.topk,
            chunk_size=args.chunk_size,
            eval_side="lhs",
        )
        total_time += time.time() - t1
        results["lhs"] = stats

    # summary
    def safe_div(a, b): return a / max(1, b)
    if "rhs" in results:
        s = results["rhs"]
        print(f"[RHS] MRR={safe_div(s['mrr'], s['n']):.4f} | "
              f"H@1={safe_div(s['h1'], s['n']):.4f} | "
              f"H@3={safe_div(s['h3'], s['n']):.4f} | "
              f"H@10={safe_div(s['h10'], s['n']):.4f} | "
              f"Rec@{args.topk}={safe_div(s['rec'], s['n']):.4f}")
    if "lhs" in results:
        s = results["lhs"]
        print(f"[LHS] MRR={safe_div(s['mrr'], s['n']):.4f} | "
              f"H@1={safe_div(s['h1'], s['n']):.4f} | "
              f"H@3={safe_div(s['h3'], s['n']):.4f} | "
              f"H@10={safe_div(s['h10'], s['n']):.4f} | "
              f"Rec@{args.topk}={safe_div(s['rec'], s['n']):.4f}")

    if "rhs" in results and "lhs" in results:
        avg_mrr = 0.5 * (safe_div(results["rhs"]["mrr"], results["rhs"]["n"]) +
                         safe_div(results["lhs"]["mrr"], results["lhs"]["n"]))
        print(f"[AVG] MRR={avg_mrr:.4f}")

    index_size_mb = (
        _tensor_size_mb(ent_index) +
        _tensor_size_mb(rel_index_dir0) +
        _tensor_size_mb(rel_index_dir1)
    )
    model_size_mb = _model_size_mb(sem_model)
    peak_mem = None
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    summary = {
        "metrics": results,
        "avg_mrr": avg_mrr,
        "index_build_time_sec": index_time,
        "eval_time_sec": total_time,
        "index_size_mb": index_size_mb,
        "model_size_mb": model_size_mb,
        "peak_cuda_mem_mb": peak_mem,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Metrics] saved to {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
