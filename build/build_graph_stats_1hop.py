import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor


def _read_split(processor: KGProcessor, name: str) -> torch.LongTensor:
    return processor._read_triplets(f"{name}.txt")


def build_topology(triplets: torch.LongTensor, num_entities: int, max_neighbors: int):
    adj = defaultdict(list)
    freq = torch.zeros(num_entities, dtype=torch.float)

    for h, r, t in triplets.tolist():
        # OUT edge: h -> t (direction=True)
        adj[h].append((t, r, True))
        # IN edge: t <- h (direction=False)
        adj[t].append((h, r, False))
        freq[h] += 1.0
        freq[t] += 1.0

    nbr_ent = torch.full((num_entities, max_neighbors), -1, dtype=torch.long)
    nbr_rel = torch.full((num_entities, max_neighbors), -1, dtype=torch.long)
    nbr_dir = torch.zeros((num_entities, max_neighbors), dtype=torch.bool)
    nbr_mask = torch.zeros((num_entities, max_neighbors), dtype=torch.bool)

    for i in range(num_entities):
        neighbors = adj.get(i, [])
        neighbors.sort(key=lambda x: (x[0], x[1]))
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]
        for k, (n_id, r_id, direction) in enumerate(neighbors):
            nbr_ent[i, k] = n_id
            nbr_rel[i, k] = r_id
            nbr_dir[i, k] = direction
            nbr_mask[i, k] = True

    deg = nbr_mask.sum(dim=1).float()
    return nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq, deg


def build_rel_hist(nbr_rel: torch.LongTensor, nbr_mask: torch.BoolTensor, num_relations: int, topm: int):
    num_entities = nbr_rel.size(0)
    rel_hist_ids = torch.full((num_entities, topm), -1, dtype=torch.long)
    rel_hist_counts = torch.zeros((num_entities, topm), dtype=torch.float)

    for i in range(num_entities):
        rels = nbr_rel[i][nbr_mask[i]]
        if rels.numel() == 0:
            continue
        counts = torch.bincount(rels, minlength=num_relations).float()
        k = min(topm, num_relations)
        top_counts, top_ids = torch.topk(counts, k=k)
        keep = top_counts > 0
        top_counts = top_counts[keep]
        top_ids = top_ids[keep]
        m = top_ids.numel()
        if m > 0:
            rel_hist_ids[i, :m] = top_ids
            rel_hist_counts[i, :m] = top_counts

    return rel_hist_ids, rel_hist_counts


def infer_dir_constants(
    nbr_ent: torch.LongTensor,
    nbr_rel: torch.LongTensor,
    nbr_dir: torch.BoolTensor,
    nbr_mask: torch.BoolTensor,
    train_triplets: torch.LongTensor,
    sample_size: int = 1000,
):
    n = min(sample_size, train_triplets.size(0))
    idx = torch.randperm(train_triplets.size(0))[:n]
    sample = train_triplets[idx].tolist()

    rhs_counts = {0: 0, 1: 0}
    lhs_counts = {0: 0, 1: 0}

    for h, r, t in sample:
        # RHS: h -> t
        mask = (nbr_ent[h] == t) & (nbr_rel[h] == r) & nbr_mask[h]
        if mask.any():
            vals = nbr_dir[h][mask].long().tolist()
            for v in vals:
                rhs_counts[int(v)] += 1

        # LHS: t -> h
        mask = (nbr_ent[t] == h) & (nbr_rel[t] == r) & nbr_mask[t]
        if mask.any():
            vals = nbr_dir[t][mask].long().tolist()
            for v in vals:
                lhs_counts[int(v)] += 1

    if rhs_counts[0] + rhs_counts[1] == 0:
        raise RuntimeError("dir calibration failed: no RHS matches found in sampled triplets.")
    if lhs_counts[0] + lhs_counts[1] == 0:
        raise RuntimeError("dir calibration failed: no LHS matches found in sampled triplets.")

    dir_rhs = 1 if rhs_counts[1] >= rhs_counts[0] else 0
    dir_lhs = 1 if lhs_counts[1] >= lhs_counts[0] else 0

    print(f"[DirCalib] RHS counts: {rhs_counts} -> dir_rhs={dir_rhs}")
    print(f"[DirCalib] LHS counts: {lhs_counts} -> dir_lhs={dir_lhs}")
    return int(dir_rhs), int(dir_lhs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--max_neighbors", type=int, default=16)
    ap.add_argument("--topm", type=int, default=32)
    ap.add_argument("--splits", type=str, default="train_valid",
                    choices=["train", "train_valid"],
                    help="Triples used for graph stats (never include test).")
    ap.add_argument("--dir_sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.out_path is None:
        args.out_path = os.path.join(args.data_path, "graph_stats_1hop.pt")

    processor = KGProcessor(args.data_path, max_neighbors=args.max_neighbors)
    processor._load_or_build_dicts()

    train_triplets = _read_split(processor, "train")
    if args.splits == "train_valid":
        valid_triplets = _read_split(processor, "valid")
        triplets = torch.cat([train_triplets, valid_triplets], dim=0)
    else:
        triplets = train_triplets

    num_entities = processor.num_entities
    num_relations = processor.num_relations

    print(f"[GraphStats] building from splits={args.splits}, max_neighbors={args.max_neighbors}, topm={args.topm}")
    nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq, deg = build_topology(
        triplets, num_entities, args.max_neighbors
    )
    rel_hist_ids, rel_hist_counts = build_rel_hist(
        nbr_rel, nbr_mask, num_relations, args.topm
    )

    dir_rhs, dir_lhs = infer_dir_constants(
        nbr_ent, nbr_rel, nbr_dir, nbr_mask, train_triplets, sample_size=args.dir_sample
    )

    obj = {
        "splits": args.splits,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "max_neighbors": args.max_neighbors,
        "topm": args.topm,
        "nbr_ent": nbr_ent,
        "nbr_rel": nbr_rel,
        "nbr_dir": nbr_dir,
        "nbr_mask": nbr_mask,
        "freq": freq,
        "deg": deg,
        "rel_hist_ids": rel_hist_ids,
        "rel_hist_counts": rel_hist_counts,
        "dir_rhs": dir_rhs,
        "dir_lhs": dir_lhs,
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(obj, args.out_path)
    print(f"[GraphStats] saved to {args.out_path}")


if __name__ == "__main__":
    main()
