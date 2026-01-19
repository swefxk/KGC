import argparse
import os
import sys
import json
import torch

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor
from models.rotate import RotatEModel
from tools.run_meta import write_run_metadata


@torch.no_grad()
def build_true_head_sets_trainvalid(processor: KGProcessor):
    """
    filtered uses all known true heads from train+valid (NO test).
    map: (t, r) -> {h}
    """
    true_head = {}
    for triplets in [processor.train_triplets, processor.valid_triplets]:
        for h, r, t in triplets.tolist():
            h, r, t = int(h), int(r), int(t)
            true_head.setdefault((t, r), set()).add(h)
    return true_head


@torch.no_grad()
def topk_head_for_batch(
    rotate_model: RotatEModel,
    h: torch.LongTensor,
    r: torch.LongTensor,
    t: torch.LongTensor,
    num_entities: int,
    K: int,
    chunk_size: int,
    true_head=None,
    device="cuda",
):
    """
    Return topK negative heads for each (h,r,t), filtered, excluding gold h.
    Output: [B,K] candidate head ids
    """
    B = h.size(0)
    top_scores = torch.full((B, K), -1e9, device=device)
    top_ids = torch.full((B, K), -1, device=device, dtype=torch.long)

    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)

    # We'll score head candidates: score_head_batch(r, t, cand_h)
    for start in range(0, num_entities, chunk_size):
        end = min(start + chunk_size, num_entities)
        cand = ent_ids[start:end].unsqueeze(0).expand(B, -1)  # [B,c]

        s = rotate_model.score_head_batch(r, t, cand)  # [B,c]

        if true_head is not None:
            # mask out all true heads for (t,r) and also mask out gold h
            s_mask = torch.zeros_like(s, dtype=torch.bool)
            t_cpu = t.detach().cpu().tolist()
            r_cpu = r.detach().cpu().tolist()
            h_cpu = h.detach().cpu().tolist()

            for i in range(B):
                forb = true_head.get((int(t_cpu[i]), int(r_cpu[i])), set())
                if forb:
                    forb_in_chunk = [x for x in forb if start <= x < end]
                    if forb_in_chunk:
                        idx = torch.tensor([x - start for x in forb_in_chunk], device=device)
                        s_mask[i, idx] = True

                gh = int(h_cpu[i])
                if start <= gh < end:
                    s_mask[i, gh - start] = True

            s = s.masked_fill(s_mask, -1e9)

        merged_scores = torch.cat([top_scores, s], dim=1)     # [B,K+c]
        merged_ids = torch.cat([top_ids, cand], dim=1)        # [B,K+c]

        new_scores, idx = torch.topk(merged_scores, k=K, dim=1)
        new_ids = torch.gather(merged_ids, 1, idx)

        top_scores, top_ids = new_scores, new_ids

    return top_ids  # [B,K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_filtered_trainvalid", action="store_true")
    ap.add_argument("--out_path", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"[LHS-CacheV2] K={args.K} filtered_trainvalid={args.use_filtered_trainvalid}")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate_model = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=1000, margin=9.0).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate_model.eval()

    true_head = build_true_head_sets_trainvalid(processor) if args.use_filtered_trainvalid else None

    train = processor.train_triplets.to(torch.long)
    N = train.size(0)
    neg_h = torch.empty((N, args.K), dtype=torch.int16)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    step = 0
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        batch = train[start:end].to(device)
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        top_ids = topk_head_for_batch(
            rotate_model=rotate_model,
            h=h, r=r, t=t,
            num_entities=processor.num_entities,
            K=args.K,
            chunk_size=args.chunk_size,
            true_head=true_head,
            device=device
        ).to("cpu")

        neg_h[start:end] = top_ids.to(torch.int16)
        step += 1
        if step % 50 == 0:
            print(f"[LHS-CacheV2] step={step} done={end}/{N} ({end/N*100:.1f}%)")

    torch.save({"neg_h": neg_h}, args.out_path)
    meta = {
        "K": args.K,
        "filtered_trainvalid": bool(args.use_filtered_trainvalid),
        "num_train": int(N),
        "num_entities": int(processor.num_entities),
        "note": "neg_h excludes gold head; aligned with processor.train_triplets order; filtered uses train+valid truths"
    }
    with open(args.out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    out_dir = os.path.dirname(args.out_path) or "."
    write_run_metadata(out_dir, args)

    print(f"[LHS-CacheV2] Saved: {args.out_path}")
    print(f"[LHS-CacheV2] Meta:  {args.out_path}.json")


if __name__ == "__main__":
    main()
