import argparse
import os
import sys
import json
import random
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from eval.eval_full_entity_filtered import load_embeddings, build_to_skip, eval_chunked_bidirectional
from eval.eval_topk_inject import eval_rhs_topk_inject, eval_lhs_topk_inject
from tools.run_meta import write_run_metadata
from models.semantic_biencoder import SemanticBiEncoderScorer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IndexedTripletDataset(Dataset):
    def __init__(self, triplets: torch.LongTensor):
        self.triplets = triplets

    def __len__(self):
        return self.triplets.size(0)

    def __getitem__(self, idx: int):
        return self.triplets[idx], idx


def indexed_collate_fn(batch):
    triples = torch.stack([x[0] for x in batch], dim=0)
    idxs = torch.tensor([x[1] for x in batch], dtype=torch.long)
    return triples, idxs


def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved args to {path}")


@torch.no_grad()
def build_true_tails_map(triplets: torch.LongTensor):
    """
    Build mapping (h,r) -> sorted unique list of true tails (from given triplets only).
    """
    mp = defaultdict(set)
    for h, r, t in triplets.tolist():
        mp[(int(h), int(r))].add(int(t))
    out = {}
    for k, s in mp.items():
        out[k] = sorted(list(s))
    return out


def sample_hard_from_cache_cpu(cache_neg_t_cpu: torch.Tensor, row_idx_cpu: torch.Tensor, k_sample: int):
    """
    cache_neg_t_cpu: [N, Kcache] on CPU
    row_idx_cpu: [B] on CPU
    return: [B, k_sample] on CPU
    """
    B = row_idx_cpu.size(0)
    Kcache = cache_neg_t_cpu.size(1)
    if k_sample >= Kcache:
        return cache_neg_t_cpu[row_idx_cpu]
    cols = torch.randint(0, Kcache, (B, k_sample), dtype=torch.long)  # CPU
    out = cache_neg_t_cpu[row_idx_cpu.unsqueeze(1), cols]
    return out


def make_forbidden_padded(h_ids: torch.Tensor, r_ids: torch.Tensor, true_tails: dict, device, max_forb: int):
    """
    Returns:
      forb: [B, M] padded with -1
      forb_mask: [B, M] bool
    """
    h_cpu = h_ids.detach().cpu().tolist()
    r_cpu = r_ids.detach().cpu().tolist()
    lists = []
    mmax = 0
    for i in range(len(h_cpu)):
        lst = true_tails.get((int(h_cpu[i]), int(r_cpu[i])), [])
        if max_forb > 0 and len(lst) > max_forb:
            lst = lst[:max_forb]
        lists.append(lst)
        mmax = max(mmax, len(lst))
    if mmax == 0:
        return None, None
    forb = torch.full((len(h_cpu), mmax), -1, dtype=torch.long, device=device)
    for i, lst in enumerate(lists):
        if lst:
            forb[i, :len(lst)] = torch.tensor(lst, dtype=torch.long, device=device)
    forb_mask = forb.ne(-1)
    return forb, forb_mask


class PreBatchQueue:
    def __init__(self, capacity: int, dim: int, device):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.vecs = torch.zeros((self.capacity, self.dim), device=device, dtype=torch.float32)
        self.ids = torch.full((self.capacity,), -1, device=device, dtype=torch.long)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def enqueue(self, ids: torch.Tensor, vecs: torch.Tensor):
        # ids: [B], vecs: [B,d]
        B = ids.size(0)
        if B == 0:
            return
        if B >= self.capacity:
            ids = ids[-self.capacity:]
            vecs = vecs[-self.capacity:]
            B = ids.size(0)

        end = self.ptr + B
        if end <= self.capacity:
            self.ids[self.ptr:end] = ids
            self.vecs[self.ptr:end] = vecs
        else:
            first = self.capacity - self.ptr
            self.ids[self.ptr:] = ids[:first]
            self.vecs[self.ptr:] = vecs[:first]
            rest = B - first
            self.ids[:rest] = ids[first:]
            self.vecs[:rest] = vecs[first:]

        self.ptr = (self.ptr + B) % self.capacity
        self.filled = min(self.capacity, self.filled + B)

    def get(self):
        if self.filled <= 0:
            return None, None
        # return first filled portion (order not important for negatives)
        return self.ids[:self.filled], self.vecs[:self.filled]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_rotate", type=str, default=None)
    ap.add_argument("--emb_dim", type=int, default=1000, help="RotatE embedding dim")
    ap.add_argument("--train_cache", type=str, default=None)  # legacy RHS cache
    ap.add_argument("--train_cache_rhs", type=str, default=None)
    ap.add_argument("--train_cache_lhs", type=str, default=None)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--pretrained_sem", type=str, default=None)
    ap.add_argument("--neg_source", type=str, default="cache", choices=["cache", "random"],
                    help="negative source: cache (RotatE hard) or random")
    ap.add_argument("--random_lhs", action="store_true",
                    help="if set and neg_source=random, also train LHS with random negatives")

    # bi-encoder
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--text_norm", action="store_true")

    # negatives
    ap.add_argument("--hard_k", type=int, default=128)
    ap.add_argument("--use_inbatch", action="store_true")
    ap.add_argument("--queue_size", type=int, default=4096)
    ap.add_argument("--max_forb", type=int, default=64)  # cap tails per (h,r) when masking
    ap.add_argument("--disable_true_tail_mask", action="store_true")
    ap.add_argument("--loss_lhs_weight", type=float, default=1.0)

    # train
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    # eval
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--eval_chunk_size", type=int, default=2048)
    ap.add_argument("--eval_sem_rhs_only", action="store_true")  # strongly recommended for bi-encoder first stage
    ap.add_argument("--eval_mode", type=str, default="full", choices=["full", "topk"])
    ap.add_argument("--eval_metric", type=str, default="avg", choices=["rhs", "avg"])
    ap.add_argument("--eval_topk", type=int, default=200)
    ap.add_argument("--eval_b_rhs", type=float, default=1.0)
    ap.add_argument("--eval_b_lhs", type=float, default=1.0)
    ap.add_argument("--eval_struct_weight_rhs", type=float, default=0.0)
    ap.add_argument("--eval_struct_weight_lhs", type=float, default=0.0)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_run_metadata(args.save_dir, args)
    print(f"=== Train Bi-Encoder Sem (SimKGC-style) on {device} ===")
    save_args(args, args.save_dir)

    # ---- data ----
    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    # true tails for masking (train only)
    # train+valid truths for masking (avoid false negatives from valid)
    true_tails = None
    if not args.disable_true_tail_mask:
        to_skip_tv = build_to_skip(processor, split="valid")
        true_tails = {k: sorted(list(v)) for k, v in to_skip_tv["rhs"].items()}

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        IndexedTripletDataset(processor.train_triplets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
        collate_fn=indexed_collate_fn,
    )

    valid_loader = DataLoader(
        TrainDataset(processor.valid_triplets),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # ---- cache (CPU!) ----
    cache_neg_t = None
    if args.neg_source == "cache":
        cache_rhs_path = args.train_cache_rhs or args.train_cache
        if not cache_rhs_path:
            raise ValueError("Need --train_cache_rhs (or legacy --train_cache) when neg_source=cache.")
        cache = torch.load(cache_rhs_path, map_location="cpu")
        cache_neg_t = cache["neg_t"] if isinstance(cache, dict) and "neg_t" in cache else cache
        cache_neg_t = cache_neg_t.to(torch.long).contiguous()  # CPU
        if cache_neg_t.size(0) != processor.train_triplets.size(0):
            raise ValueError(f"RHS cache rows={cache_neg_t.size(0)} != num_train={processor.train_triplets.size(0)}")
        print(f"[Cache] neg_t CPU: {tuple(cache_neg_t.shape)} dtype={cache_neg_t.dtype}")
    else:
        print("[Cache] neg_source=random (no RotatE cache used)")

    cache_neg_h = None
    if args.train_cache_lhs:
        cache_h = torch.load(args.train_cache_lhs, map_location="cpu")
        cache_neg_h = cache_h["neg_h"] if isinstance(cache_h, dict) and "neg_h" in cache_h else cache_h
        cache_neg_h = cache_neg_h.to(torch.long).contiguous()
        if cache_neg_h.size(0) != processor.train_triplets.size(0):
            raise ValueError(f"LHS cache rows={cache_neg_h.size(0)} != num_train={processor.train_triplets.size(0)}")
        print(f"[Cache] neg_h CPU: {tuple(cache_neg_h.shape)} dtype={cache_neg_h.dtype}")

    # ---- RotatE (frozen, for eval only) ----
    rotate_model = None
    if args.pretrained_rotate:
        rotate_model = RotatEModel(
            processor.num_entities,
            processor.num_relations,
            emb_dim=args.emb_dim,
            margin=9.0,
        ).to(device)
        rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
        rotate_model.eval()
        for p in rotate_model.parameters():
            p.requires_grad = False

    # ---- embeddings (frozen tensors) ----
    ent_embs, rel_embs = load_embeddings(processor, args, device)
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    # ---- bi-encoder ----
    sem_model = SemanticBiEncoderScorer(
        text_dim=ent_embs.size(1),
        num_relations=processor.num_relations,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        text_norm=args.text_norm,
    ).to(device)
    if args.pretrained_sem:
        ckpt = torch.load(args.pretrained_sem, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        sem_model.load_state_dict(ckpt, strict=False)

    optimizer = optim.AdamW(sem_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ce = nn.CrossEntropyLoss()

    queue = PreBatchQueue(args.queue_size, args.proj_dim, device=device) if args.queue_size > 0 else None

    best_rhs = -1.0
    save_path = os.path.join(args.save_dir, "biencoder_best.pth")
    to_skip_valid = build_to_skip(processor, split="valid")

    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        sem_model.train()
        total_loss = 0.0
        n_steps = 0

        # running stats
        stat_p = 0.0
        stat_margin = 0.0
        stat_p_hard = 0.0
        stat_margin_hard = 0.0
        stat_p_lhs = 0.0
        stat_margin_lhs = 0.0
        stat_cnt = 0

        for triples, row_idx in train_loader:
            triples = triples.to(device, non_blocking=True)
            h = triples[:, 0]
            r = triples[:, 1]
            t = triples[:, 2]

            # sample negatives on CPU then move
            row_idx_cpu = row_idx.cpu()
            if cache_neg_t is not None:
                hard_cpu = sample_hard_from_cache_cpu(cache_neg_t, row_idx_cpu, args.hard_k)  # CPU [B,Kh]
            else:
                hard_cpu = torch.randint(
                    0, processor.num_entities, (h.size(0), args.hard_k), dtype=torch.long
                )
            hard = hard_cpu.to(device, non_blocking=True)  # GPU

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                tau = max(args.tau, 1e-6)

                h_txt = ent_embs[h]
                r_txt = rel_embs[r]
                t_txt = ent_embs[t]

                q = sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))  # [B,d]
                pos_v = sem_model.encode_entity(t_txt)            # [B,d]
                pos_logit = (q * pos_v).sum(dim=-1)               # [B]

                def mask_value_for(tensor: torch.Tensor):
                    # fp16-safe mask value: (mask / tau) ~ -50000 for near-zero softmax
                    if tensor.dtype in (torch.float16, torch.bfloat16):
                        return -5e4 * tau
                    return -1e9

                # hard logits: [B,Kh]
                B, Kh = hard.shape
                hard_v = sem_model.encode_entity(ent_embs[hard.reshape(-1)]).view(B, Kh, -1)
                hard_logits = torch.einsum("bd,bkd->bk", q, hard_v)

                # in-batch logits: [B,B] (diagonal masked)
                inbatch_logits = None
                inbatch_ids = None
                if args.use_inbatch:
                    inbatch_logits = q @ pos_v.t()  # [B,B]
                    inbatch_logits.fill_diagonal_(mask_value_for(inbatch_logits))
                    inbatch_ids = t.unsqueeze(0).expand(B, B)  # [B,B]

                # queue logits: [B,Q]
                q_ids, q_vecs = (None, None)
                queue_logits = None
                if queue is not None:
                    q_ids, q_vecs = queue.get()
                    if q_ids is not None:
                        queue_logits = q @ q_vecs.t()  # [B,Q]

                # ---- false-negative mask (train true tails) ----
                forb = forb_mask = None
                if true_tails is not None:
                    forb, forb_mask = make_forbidden_padded(h, r, true_tails, device=device, max_forb=args.max_forb)
                if forb is not None:
                    # mask hard
                    # [B,Kh,M]
                    mh = (hard.unsqueeze(-1) == forb.unsqueeze(1)) & forb_mask.unsqueeze(1)
                    mh = mh.any(dim=-1)  # [B,Kh]
                    hard_logits = hard_logits.masked_fill(mh, mask_value_for(hard_logits))

                    # mask in-batch
                    if inbatch_logits is not None:
                        mib = (inbatch_ids.unsqueeze(-1) == forb.unsqueeze(1)) & forb_mask.unsqueeze(1)
                        mib = mib.any(dim=-1)  # [B,B]
                        inbatch_mask_val = mask_value_for(inbatch_logits)
                        inbatch_logits = inbatch_logits.masked_fill(mib, inbatch_mask_val)
                        inbatch_logits.fill_diagonal_(inbatch_mask_val)

                    # mask queue
                    if queue_logits is not None and q_ids is not None:
                        # [B,Q,M]
                        mq = (q_ids.unsqueeze(0).unsqueeze(-1) == forb.unsqueeze(1)) & forb_mask.unsqueeze(1)
                        mq = mq.any(dim=-1)  # [B,Q]
                        queue_logits = queue_logits.masked_fill(mq, mask_value_for(queue_logits))

                # ---- assemble logits: [B, 1+Kh + (B if inbatch) + (Q if queue)] ----
                parts = [pos_logit.unsqueeze(1), hard_logits]
                if inbatch_logits is not None:
                    parts.append(inbatch_logits)
                if queue_logits is not None:
                    parts.append(queue_logits)

                logits = torch.cat(parts, dim=1) / tau
                labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)

                loss = ce(logits, labels)

                # ---- optional LHS (hard-only or random) ----
                if cache_neg_h is not None or (args.neg_source == "random" and args.random_lhs):
                    if cache_neg_h is not None:
                        hard_h_cpu = sample_hard_from_cache_cpu(cache_neg_h, row_idx_cpu, args.hard_k)
                    else:
                        hard_h_cpu = torch.randint(
                            0, processor.num_entities, (h.size(0), args.hard_k), dtype=torch.long
                        )
                    hard_h = hard_h_cpu.to(device, non_blocking=True)
                    q_lhs = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(h))  # [B,d]
                    pos_v_lhs = sem_model.encode_entity(h_txt)
                    pos_logit_lhs = (q_lhs * pos_v_lhs).sum(dim=-1)  # [B]
                    hard_v_lhs = sem_model.encode_entity(ent_embs[hard_h.reshape(-1)]).view(B, Kh, -1)
                    hard_logits_lhs = torch.einsum("bd,bkd->bk", q_lhs, hard_v_lhs)
                    logits_lhs = torch.cat([pos_logit_lhs.unsqueeze(1), hard_logits_lhs], dim=1) / tau
                    loss_lhs = ce(logits_lhs, labels)
                    loss = loss + args.loss_lhs_weight * loss_lhs

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(sem_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_steps += 1

            # ---- stats: p(pos > maxneg), margin ----
            with torch.no_grad():
                neg_max_hard = hard_logits.max(dim=1).values
                neg_max = neg_max_hard
                if inbatch_logits is not None:
                    neg_max = torch.maximum(neg_max, inbatch_logits.max(dim=1).values)
                if queue_logits is not None:
                    neg_max = torch.maximum(neg_max, queue_logits.max(dim=1).values)

                p = (pos_logit > neg_max).float().mean().item()
                m = (pos_logit - neg_max).mean().item()
                p_hard = (pos_logit > neg_max_hard).float().mean().item()
                m_hard = (pos_logit - neg_max_hard).mean().item()
                stat_p += p
                stat_margin += m
                stat_p_hard += p_hard
                stat_margin_hard += m_hard
                if cache_neg_h is not None or (args.neg_source == "random" and args.random_lhs):
                    neg_max_hard_lhs = hard_logits_lhs.max(dim=1).values
                    p_hard_lhs = (pos_logit_lhs > neg_max_hard_lhs).float().mean().item()
                    m_hard_lhs = (pos_logit_lhs - neg_max_hard_lhs).mean().item()
                    stat_p_lhs += p_hard_lhs
                    stat_margin_lhs += m_hard_lhs
                stat_cnt += 1

            # enqueue after update (no grad)
            if queue is not None:
                with torch.no_grad():
                    queue.enqueue(t.detach(), pos_v.detach())

        msg = (f"Epoch {epoch} | loss={total_loss/max(1,n_steps):.6f} | "
               f"p(pos>maxneg)={stat_p/max(1,stat_cnt):.4f} | margin={stat_margin/max(1,stat_cnt):.4f} | "
               f"p_hard={stat_p_hard/max(1,stat_cnt):.4f} | margin_hard={stat_margin_hard/max(1,stat_cnt):.4f}")
        if cache_neg_h is not None:
            msg += (f" | p_hard_lhs={stat_p_lhs/max(1,stat_cnt):.4f} | "
                    f"margin_hard_lhs={stat_margin_lhs/max(1,stat_cnt):.4f}")
        print(msg)

        # ---- Eval (full-entity filtered, but recommend sem_rhs_only during this phase) ----
        if args.eval_every and (epoch % args.eval_every == 0):
            if rotate_model is None:
                raise ValueError("eval_every requires --pretrained_rotate for full-entity eval.")
            sem_model.eval()
            if args.eval_mode == "full":
                print("Evaluating on VALID (full-entities filtered)...")
                metrics = eval_chunked_bidirectional(
                    processor=processor,
                    rotate_model=rotate_model,
                    test_loader=valid_loader,
                    device=device,
                    to_skip=to_skip_valid,
                    eval_split="valid",
                    refiner=None,
                    semres_model=sem_model,       # bi-encoder implements forward(delta,lam)
                    ent_text_embs=ent_embs,
                    rel_text_embs=rel_embs,
                    chunk_size=args.eval_chunk_size,
                    sem_subchunk=256,
                    disable_refiner=True,
                    disable_semres=False,
                    sem_rhs_only=args.eval_sem_rhs_only,
                    verbose_every=0,
                )
                rhs = metrics["rhs"]["total"]["MRR"]
                lhs = metrics["lhs"]["total"]["MRR"]
            else:
                print("Evaluating on VALID (topK, sem-only rerank)...")
                eval_loader = DataLoader(
                    TrainDataset(processor.valid_triplets),
                    batch_size=args.eval_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=False,
                )
                rhs_stats, _, _ = eval_rhs_topk_inject(
                    processor=processor,
                    rotate_model=rotate_model,
                    sem_model=sem_model,
                    ent_text_embs=ent_embs,
                    rel_text_embs=rel_embs,
                    refiner=None,
                    nbr_ent=None,
                    nbr_rel=None,
                    nbr_dir=None,
                    nbr_mask=None,
                    freq=None,
                    gate_model=None,
                    gate_ent_temp=1.0,
                    loader=eval_loader,
                    to_skip=to_skip_valid,
                    split="valid",
                    device=device,
                    topk=args.eval_topk,
                    chunk_size=args.eval_chunk_size,
                    b_scale=args.eval_b_rhs,
                    get_b_fn=None,
                    refiner_gamma=0.0,
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
                    refiner_diag=False,
                    struct_weight=args.eval_struct_weight_rhs,
                )
                lhs_stats, _, _ = eval_lhs_topk_inject(
                    processor=processor,
                    rotate_model=rotate_model,
                    sem_model=sem_model,
                    ent_text_embs=ent_embs,
                    rel_text_embs=rel_embs,
                    refiner=None,
                    nbr_ent=None,
                    nbr_rel=None,
                    nbr_dir=None,
                    nbr_mask=None,
                    freq=None,
                    gate_model=None,
                    gate_ent_temp=1.0,
                    loader=eval_loader,
                    to_skip=to_skip_valid,
                    split="valid",
                    device=device,
                    topk=args.eval_topk,
                    chunk_size=args.eval_chunk_size,
                    b_scale=args.eval_b_lhs,
                    get_b_fn=None,
                    refiner_gamma=0.0,
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
                    refiner_diag=False,
                    sem_union_topk=0,
                    struct_weight=args.eval_struct_weight_lhs,
                )
                rhs = rhs_stats["total"]["mrr"] / max(1, rhs_stats["total"]["n"])
                lhs = lhs_stats["total"]["mrr"] / max(1, lhs_stats["total"]["n"])

            avg = 0.5 * (rhs + lhs)
            print(f"[VALID] RHS MRR: {rhs:.4f} | LHS MRR: {lhs:.4f} | AVG: {avg:.4f}")
            metric = avg if args.eval_metric == "avg" else rhs

            if metric > best_rhs:
                best_rhs = metric
                ckpt = {
                    "model_type": "biencoder",
                    "model_args": {
                        "text_dim": int(ent_embs.size(1)),
                        "num_relations": int(processor.num_relations),
                        "proj_dim": int(args.proj_dim),
                        "dropout": float(args.dropout),
                        "text_norm": bool(args.text_norm),
                    },
                    "state_dict": sem_model.state_dict(),
                    "train_args": vars(args),
                    "best_valid_metric": float(best_rhs),
                    "best_valid_rhs": float(rhs),
                    "best_valid_lhs": float(lhs),
                    "best_valid_avg": float(avg),
                }
                torch.save(ckpt, save_path)
                print(f"Saved best to {save_path} (best_valid_metric={best_rhs:.4f})")

    print(f"Done. Best VALID RHS MRR: {best_rhs:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
