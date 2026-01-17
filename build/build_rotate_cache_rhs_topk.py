import argparse
import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor
from models.rotate import RotatEModel
from tools.run_meta import write_run_metadata


@torch.no_grad()
def build_true_tail_sets(processor: KGProcessor, filtered_splits: str):
    """
    filtered_splits:
      - "train"
      - "train_valid"
      - "train_valid_test"
    """
    mapping = {
        "train": [processor.train_triplets],
        "train_valid": [processor.train_triplets, processor.valid_triplets],
        "train_valid_test": [processor.train_triplets, processor.valid_triplets, processor.test_triplets],
    }
    if filtered_splits not in mapping:
        raise ValueError(f"Unknown filtered_splits={filtered_splits}")
    true_tail = {}
    for triplets in mapping[filtered_splits]:
        for h, r, t in triplets.tolist():
            h, r, t = int(h), int(r), int(t)
            true_tail.setdefault((h, r), set()).add(t)
    return true_tail


@torch.no_grad()
def topk_for_batch(rotate_model, h, r, num_entities, K, chunk_size=2048, true_tail=None, gold_t=None, device="cuda"):
    """
    Return topK negative tails for each (h,r, gold_t), filtered, excluding gold_t.
    """
    B = h.size(0)
    # maintain running topK
    top_scores = torch.full((B, K), -1e9, device=device)
    top_ids = torch.full((B, K), -1, device=device, dtype=torch.long)

    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)

    for start in range(0, num_entities, chunk_size):
        end = min(start + chunk_size, num_entities)
        cand = ent_ids[start:end].unsqueeze(0).expand(B, -1)  # [B, c]
        s = rotate_model(h, r, cand, mode="batch_neg")        # [B, c]

        if true_tail is not None and gold_t is not None:
            # mask out all true tails (except gold) and also mask out gold
            s_mask = torch.zeros_like(s, dtype=torch.bool)
            h_cpu = h.detach().cpu().tolist()
            r_cpu = r.detach().cpu().tolist()
            gold_cpu = gold_t.detach().cpu().tolist()
            for i in range(B):
                forb = true_tail.get((int(h_cpu[i]), int(r_cpu[i])), set())
                if len(forb) > 0:
                    # mark forbidden in this chunk
                    forb_in_chunk = [x for x in forb if start <= x < end]
                    if forb_in_chunk:
                        idx = torch.tensor([x - start for x in forb_in_chunk], device=device)
                        s_mask[i, idx] = True
                # ensure gold is excluded
                gt = int(gold_cpu[i])
                if start <= gt < end:
                    s_mask[i, gt - start] = True
            s = s.masked_fill(s_mask, -1e9)

        # merge with current topK
        merged_scores = torch.cat([top_scores, s], dim=1)  # [B, K+c]
        merged_ids = torch.cat([top_ids, cand], dim=1)

        new_scores, idx = torch.topk(merged_scores, k=K, dim=1)
        new_ids = torch.gather(merged_ids, 1, idx)

        top_scores, top_ids = new_scores, new_ids

    return top_ids  # [B, K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_filtered", action="store_true")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--filtered_splits", type=str, default="train_valid",choices=["train", "train_valid", "train_valid_test"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"[CacheV2] K={args.K} filtered={args.use_filtered}")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate_model = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate_model.eval()

    true_tail = build_true_tail_sets(processor, args.filtered_splits) if args.use_filtered else None

    train = processor.train_triplets.to(torch.long)
    N = train.size(0)
    neg_t = torch.empty((N, args.K), dtype=torch.int16)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    step = 0
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        batch = train[start:end].to(device)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        top_ids = topk_for_batch(
            rotate_model, h, r,
            num_entities=processor.num_entities,
            K=args.K,
            chunk_size=args.chunk_size,
            true_tail=true_tail,
            gold_t=t,
            device=device
        ).to("cpu")

        neg_t[start:end] = top_ids.to(torch.int16)
        step += 1
        if step % 50 == 0:
            print(f"[CacheV2] step={step} done={end}/{N} ({end/N*100:.1f}%)")

    torch.save({"neg_t": neg_t}, args.out_path)
    meta = {
        "K": args.K,
        "filtered": bool(args.use_filtered),
        "num_train": int(N),
        "num_entities": int(processor.num_entities),
        "note": "neg_t excludes gold tail; aligned with processor.train_triplets order"
    }
    with open(args.out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    out_dir = os.path.dirname(args.out_path) or "."
    write_run_metadata(out_dir, args)

    print(f"[CacheV2] Saved: {args.out_path}")
    print(f"[CacheV2] Meta:  {args.out_path}.json")


if __name__ == "__main__":
    main()
