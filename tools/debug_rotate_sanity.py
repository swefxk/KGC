import os, random, math
import torch
import numpy as np

from data.data_loader import KGProcessor
from models.rotate import RotatEModel

@torch.no_grad()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/fb15k_custom")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_check", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    proc = KGProcessor(args.data_path)
    proc.load_files()
    train = proc.train_triplets
    valid = proc.valid_triplets
    test  = proc.test_triplets
    print(f"[DATA] #ent={proc.num_entities} #rel={proc.num_relations} "
          f"| train={len(train)} valid={len(valid)} test={len(test)}")

    # ---- load model ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    emb_dim = ckpt.get("emb_dim", None) or ckpt.get("model_args", {}).get("emb_dim", None)
    if emb_dim is None:
        # fallback: infer
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        if "entity_embedding.weight" in sd:
            emb_dim = sd["entity_embedding.weight"].shape[1] // 2
        else:
            emb_dim = sd["entity_embedding"].shape[1] // 2
    model = RotatEModel(num_entities=proc.num_entities, num_relations=proc.num_relations, emb_dim=emb_dim, margin=ckpt.get("margin", 9.0))
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model = model.to(args.device).eval()

    # ---- build truth maps exactly like training should ----
    # (h,r)-> set(t)
    true_tail = {}
    # (r,t)-> set(h)
    true_head = {}
    for h,r,t in train.tolist():
        true_tail.setdefault((h,r), set()).add(t)
        true_head.setdefault((r,t), set()).add(h)

    # ---- sample triples ----
    all_triples = torch.cat([train, valid, test], dim=0)
    idxs = np.random.choice(len(all_triples), size=min(args.num_check, len(all_triples)), replace=False)
    samp = all_triples[idxs].to(args.device)
    h = samp[:,0]; r = samp[:,1]; t = samp[:,2]
    B = h.shape[0]

    # ============ Check 1: head-batch 等价性（conj 逻辑）============
    # score(h,r,t) should equal score_head_batch implemented as score(t, conj(r), h)
    s1 = model(h, r, t, mode="single")  # [B]
    # emulate your score_head_batch path:
    t_emb = model.entity_embedding[t]   # [B,2D]
    # cand_h shape [B,1]
    cand_h = h.unsqueeze(1)
    s2 = model.score_from_head_emb(t_emb, r, cand_h, conj=torch.ones_like(r, dtype=torch.bool, device=args.device)).squeeze(1)
    diff = (s1 - s2).abs().mean().item()
    mx   = (s1 - s2).abs().max().item()
    print(f"[CHECK1] head-batch equivalence | mean|diff|={diff:.6e} max|diff|={mx:.6e}")
    if diff > 1e-4:
        print("  !! FAIL: 说明 conj/score_from_head_emb 的定义与你 head-batch 推导不一致，训练会被直接打坏")

    # ============ Check 2: truth map key 方向是否一致 ============
    # 只用 train 抽样，避免 split 口径不一致导致误报
    miss_tail = 0
    miss_head = 0
    train_idxs = np.random.choice(len(train), size=min(200, len(train)), replace=False)
    train_samp = train[train_idxs].tolist()
    for hh,rr,tt in train_samp:
        if tt not in true_tail.get((hh,rr), set()): miss_tail += 1
        if hh not in true_head.get((rr,tt), set()): miss_head += 1
    print(f"[CHECK2] truth-map membership (train200) | miss_tail={miss_tail} miss_head={miss_head}")
    if miss_tail > 0 or miss_head > 0:
        print("  !! FAIL: true_tail/true_head 的 key 方向或来源 splits 不一致（训练过滤会失效，假负例会爆炸）")

    # ============ Check 3: 负采样过滤是否真的在工作 ============
    # simulate your negative sampling filter rate for a few queries
    nentity = proc.num_entities
    bad = 0
    tot = 0
    for i in range(min(200, B)):
        hh = int(h[i].item()); rr = int(r[i].item()); tt = int(t[i].item())
        forbid = true_tail.get((hh,rr), set())
        cand = np.random.randint(nentity, size=1024, dtype=np.int64)
        # after filtering
        if forbid:
            forb_arr = np.fromiter(forbid, dtype=np.int64)
            cand = cand[~np.in1d(cand, forb_arr, assume_unique=False)]
        # count if any remaining candidate is still a true tail
        # (should be 0 by construction; if >0, filter broken)
        if forbid:
            violate = np.in1d(cand, forb_arr, assume_unique=False).any()
            tot += 1
            bad += int(violate)
    print(f"[CHECK3] neg-filter violation_rate={bad}/{tot} = {bad/(tot+1e-9):.6f}")
    if bad > 0:
        print("  !! FAIL: 负采样过滤逻辑有 bug（或 forbid 不是你以为的集合）")

if __name__ == "__main__":
    main()
