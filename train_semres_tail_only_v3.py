import argparse
import os
import sys
import json
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from models.semantic_residual import SemanticResidualScorerV2

from test_semres import eval_chunked_bidirectional, load_embeddings, build_to_skip


# -------------------------
# Reproducibility
# -------------------------
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


# -------------------------
# Train-only truth sets (optional filtered negatives)
# -------------------------
def build_true_sets_train(train_triplets: torch.LongTensor):
    true_tail = {}
    true_head = {}
    for h, r, t in train_triplets.tolist():
        h, r, t = int(h), int(r), int(t)
        true_tail.setdefault((h, r), set()).add(t)
        true_head.setdefault((t, r), set()).add(h)
    return true_head, true_tail


def _sample_one_filtered(num_entities: int, num_neg: int, target: int, forbidden_set, max_tries: int = 2000):
    need = num_neg
    negs = []

    forb_arr = None
    if forbidden_set:
        forb_arr = np.fromiter(forbidden_set, dtype=np.int64)

    oversample = 4
    tries = 0

    while need > 0 and tries < max_tries:
        tries += 1
        cand = np.random.randint(num_entities, size=need * oversample, dtype=np.int64)

        cand = cand[cand != target]
        if cand.size == 0:
            continue

        if forb_arr is not None and forb_arr.size > 0:
            mask = ~np.isin(cand, forb_arr, assume_unique=False)
            cand = cand[mask]
            if cand.size == 0:
                continue

        take = cand[:need]
        negs.append(take)
        need -= take.size

    if need > 0:
        while need > 0:
            cand = np.random.randint(num_entities, size=need * oversample, dtype=np.int64)
            cand = cand[cand != target]
            if cand.size == 0:
                continue
            take = cand[:need]
            negs.append(take)
            need -= take.size

    out = np.concatenate(negs, axis=0)[:num_neg]
    return out


def sample_filtered_negatives_batch_tail(
    h_ids: torch.LongTensor,
    r_ids: torch.LongTensor,
    t_ids: torch.LongTensor,
    num_entities: int,
    num_neg: int,
    true_tail_train: dict,
    device: torch.device
):
    h_cpu = h_ids.detach().cpu().numpy()
    r_cpu = r_ids.detach().cpu().numpy()
    t_cpu = t_ids.detach().cpu().numpy()

    neg_np = np.empty((h_cpu.shape[0], num_neg), dtype=np.int64)
    for i in range(h_cpu.shape[0]):
        key = (int(h_cpu[i]), int(r_cpu[i]))
        forbidden = true_tail_train.get(key, set())
        neg_np[i] = _sample_one_filtered(num_entities, num_neg, int(t_cpu[i]), forbidden)

    return torch.from_numpy(neg_np).to(device=device, dtype=torch.long)


def sample_uniform_excluding_target(num_entities: int, target_ids: torch.LongTensor, shape, device):
    neg = torch.randint(0, num_entities, shape, device=device, dtype=torch.long)
    if target_ids is None:
        return neg

    mask = neg.eq(target_ids.unsqueeze(1))
    while mask.any():
        neg[mask] = torch.randint(0, num_entities, (int(mask.sum().item()),), device=device, dtype=torch.long)
        mask = neg.eq(target_ids.unsqueeze(1))
    return neg


@torch.no_grad()
def mine_struct_hard_from_pool(rotate_model, h, r, cand_t, k):
    # cand_t: [B, P]
    s = rotate_model(h, r, cand_t, mode="batch_neg")          # [B, P]
    top_idx = torch.topk(s, k=k, dim=1).indices               # [B, k]
    neg = torch.gather(cand_t, 1, top_idx)                    # [B, k]
    s_neg = torch.gather(s, 1, top_idx)                       # [B, k]
    return neg, s_neg


@torch.no_grad()
def sem_scores_for_pool(semres_model, ent_embs, rel_embs, h, r, cand_t, sem_subchunk=256):
    # return s_sem = lam * delta  [B, P]
    device = h.device
    B, P = cand_t.shape
    out = torch.empty((B, P), device=device)

    for start in range(0, P, sem_subchunk):
        end = min(start + sem_subchunk, P)
        sub = cand_t[:, start:end]  # [B, p]

        h_rep = h.unsqueeze(1).expand(-1, sub.size(1)).reshape(-1)
        r_rep = r.unsqueeze(1).expand(-1, sub.size(1)).reshape(-1)
        t_flat = sub.reshape(-1)

        h_txt = ent_embs[h_rep]
        r_txt = rel_embs[r_rep]
        t_txt = ent_embs[t_flat]

        delta, lam = semres_model(h_txt, r_txt, t_txt, r_rep)
        s_sem = (lam * delta).view(B, -1)
        out[:, start:end] = s_sem

    return out


@torch.no_grad()
def mine_sem_hard_from_pool(semres_model, ent_embs, rel_embs, h, r, cand_t, k, sem_subchunk=256):
    s_sem = sem_scores_for_pool(semres_model, ent_embs, rel_embs, h, r, cand_t, sem_subchunk=sem_subchunk)  # [B,P]
    top_idx = torch.topk(s_sem, k=k, dim=1).indices
    neg = torch.gather(cand_t, 1, top_idx)
    s_neg = torch.gather(s_sem, 1, top_idx)
    return neg, s_neg


def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved args to {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/fb15k_custom")
    parser.add_argument("--pretrained_rotate", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/fair_experiment/semres_tail_only_v3")

    parser.add_argument("--emb_dim", type=int, default=500)
    parser.add_argument("--margin", type=float, default=9.0)

    # SemRes core
    parser.add_argument("--lambda_sem", type=float, default=0.01)
    parser.add_argument("--text_norm", action="store_true")
    parser.add_argument("--delta_bound", type=float, default=10.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_neg", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    # Neg sampling
    parser.add_argument("--hard_neg", action="store_true")
    parser.add_argument("--neg_pool_mult", type=float, default=16.0)
    parser.add_argument("--filtered_neg", action="store_true")

    # NEW: sem-hard negatives
    parser.add_argument("--sem_hard_neg", action="store_true", help="mix sem-hard negatives into training")
    parser.add_argument("--sem_hard_ratio", type=float, default=0.5, help="fraction of negatives from sem-hard (0~1)")
    parser.add_argument("--sem_subchunk", type=int, default=256)

    # NEW: sem-only auxiliary loss
    parser.add_argument("--sem_only_loss_weight", type=float, default=1.0)

    # NEW: margin-aware weighting (optional)
    parser.add_argument("--use_margin_weight", action="store_true")
    parser.add_argument("--margin_tau", type=float, default=1.0)
    parser.add_argument("--margin_temp", type=float, default=1.0)

    # Eval
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_chunk_size", type=int, default=2048)
    parser.add_argument("--eval_sem_rhs_only", action="store_true", help="validate in sem_rhs_only mode if supported")

    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_num_workers", type=int, default=2)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)

    save_path = os.path.join(args.save_dir, "semres_best.pth")

    print(f"=== Training SemRes V3 on {device_type} ===")
    print(f"[SemRes] lambda_base={args.lambda_sem}, text_norm={args.text_norm}, delta_bound={args.delta_bound}")
    print(f"[Neg] hard_neg={args.hard_neg}, pool_mult={args.neg_pool_mult}, filtered={args.filtered_neg}, "
          f"sem_hard={args.sem_hard_neg}, sem_ratio={args.sem_hard_ratio}")
    print(f"[Loss] sem_only_weight={args.sem_only_loss_weight}, margin_weight={args.use_margin_weight}")
    print(f"[Opt] lr={args.lr}, wd={args.weight_decay}, amp={args.amp}, grad_accum={args.grad_accum}")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    true_head_train, true_tail_train = build_true_sets_train(processor.train_triplets)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        TrainDataset(processor.train_triplets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )

    valid_loader = DataLoader(
        TrainDataset(processor.valid_triplets),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.eval_num_workers,
        pin_memory=True,
    )

    print(f"Loading frozen RotatE from {args.pretrained_rotate}")
    rotate_model = RotatEModel(
        processor.num_entities, processor.num_relations, args.emb_dim, args.margin
    ).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate_model.eval()
    for p in rotate_model.parameters():
        p.requires_grad = False

    # text embeddings
    ent_embs, rel_embs = load_embeddings(processor, args, device)
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    semres_model = SemanticResidualScorerV2(
        text_dim=ent_embs.size(1),
        num_relations=processor.num_relations,
        lambda_base=args.lambda_sem,
        text_norm=args.text_norm,
        delta_bound=args.delta_bound
    ).to(device)

    optimizer = optim.AdamW(semres_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_none = nn.CrossEntropyLoss(reduction="none")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_valid = -1.0
    to_skip_valid = build_to_skip(processor, split="valid")

    def sem_score(h_ids, r_ids, t_ids):
        h_txt = ent_embs[h_ids]
        r_txt = rel_embs[r_ids]
        t_txt = ent_embs[t_ids]
        delta, lam = semres_model(h_txt, r_txt, t_txt, r_ids)
        return lam * delta  # [B]

    def sem_score_batchneg(h_ids, r_ids, neg_t):  # neg_t: [B,K]
        B, K = neg_t.shape
        t_flat = neg_t.reshape(-1)
        h_rep = h_ids.repeat_interleave(K)
        r_rep = r_ids.repeat_interleave(K)

        h_txt = ent_embs[h_rep]
        r_txt = rel_embs[r_rep]
        t_txt = ent_embs[t_flat]

        delta, lam = semres_model(h_txt, r_txt, t_txt, r_rep)
        return (lam * delta).view(B, K)

    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        semres_model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            B = h.size(0)

            # ---- build candidate pool ----
            pool = int(args.num_neg * args.neg_pool_mult)
            if args.filtered_neg:
                cand_t = sample_filtered_negatives_batch_tail(
                    h_ids=h, r_ids=r, t_ids=t,
                    num_entities=processor.num_entities,
                    num_neg=pool,
                    true_tail_train=true_tail_train,
                    device=device
                )  # [B,P]
            else:
                cand_t = sample_uniform_excluding_target(processor.num_entities, t, (B, pool), device)

            # ---- mine negatives ----
            if args.hard_neg:
                # struct-hard part
                k_sem = int(round(args.num_neg * args.sem_hard_ratio)) if args.sem_hard_neg else 0
                k_sem = max(0, min(args.num_neg, k_sem))
                k_str = args.num_neg - k_sem

                neg_parts = []
                score_parts_str = []

                if k_str > 0:
                    neg_str, s_neg_str_part = mine_struct_hard_from_pool(rotate_model, h, r, cand_t, k=k_str)
                    neg_parts.append(neg_str)
                    score_parts_str.append(s_neg_str_part)

                if k_sem > 0:
                    # sem-hard mining uses current semres weights but no grad
                    semres_model.eval()
                    neg_sem, _ = mine_sem_hard_from_pool(
                        semres_model, ent_embs, rel_embs, h, r, cand_t, k=k_sem, sem_subchunk=args.sem_subchunk
                    )
                    semres_model.train()
                    neg_parts.append(neg_sem)

                neg_t = torch.cat(neg_parts, dim=1) if len(neg_parts) > 1 else neg_parts[0]  # [B, num_neg]

                # recompute struct scores on selected negs (cheap, consistent)
                s_neg_str = rotate_model(h, r, neg_t, mode="batch_neg")  # [B, num_neg]
            else:
                # uniform / filtered only
                if args.filtered_neg:
                    neg_t = sample_filtered_negatives_batch_tail(
                        h_ids=h, r_ids=r, t_ids=t,
                        num_entities=processor.num_entities,
                        num_neg=args.num_neg,
                        true_tail_train=true_tail_train,
                        device=device
                    )
                else:
                    neg_t = sample_uniform_excluding_target(processor.num_entities, t, (B, args.num_neg), device)

                s_neg_str = rotate_model(h, r, neg_t, mode="batch_neg")

            s_pos_str = rotate_model(h, r, t, mode="single")  # [B]

            # ---- forward with AMP ----
            with torch.cuda.amp.autocast(enabled=args.amp):
                s_pos_sem = sem_score(h, r, t)                   # [B]
                s_neg_sem = sem_score_batchneg(h, r, neg_t)      # [B, K]

                logits_total = torch.cat([(s_pos_str + s_pos_sem).unsqueeze(1), (s_neg_str + s_neg_sem)], dim=1)
                labels = torch.zeros(B, dtype=torch.long, device=device)

                loss_total_vec = ce_none(logits_total, labels)

                # sem-only aux
                logits_sem = torch.cat([s_pos_sem.unsqueeze(1), s_neg_sem], dim=1)
                loss_sem_vec = ce_none(logits_sem, labels)

                # margin-aware weight (optional)
                if args.use_margin_weight:
                    hardest_neg = s_neg_str.max(dim=1).values
                    margin_struct = s_pos_str - hardest_neg
                    w = torch.sigmoid((args.margin_tau - margin_struct) / max(args.margin_temp, 1e-6))
                    loss_vec = loss_total_vec + args.sem_only_loss_weight * loss_sem_vec
                    loss = (loss_vec * w).mean()
                else:
                    loss = loss_total_vec.mean() + args.sem_only_loss_weight * loss_sem_vec.mean()

            loss = loss / max(args.grad_accum, 1)
            scaler.scale(loss).backward()

            total_loss += float(loss.item()) * max(args.grad_accum, 1)
            n_batches += 1

            if (step % max(args.grad_accum, 1)) == 0:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(semres_model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch} | AvgLoss: {avg_loss:.6f}")

        # ---- Eval ----
        if args.eval_every and (epoch % args.eval_every == 0):
            semres_model.eval()
            print("Evaluating on VALID...")

            # signature-safe pass-through
            import inspect
            sig = inspect.signature(eval_chunked_bidirectional)

            kwargs = dict(
                processor=processor,
                rotate_model=rotate_model,
                test_loader=valid_loader,
                device=device,
                to_skip=to_skip_valid,
                eval_split="valid",
                refiner=None,
                semres_model=semres_model,
                ent_text_embs=ent_embs,
                rel_text_embs=rel_embs,
                chunk_size=args.eval_chunk_size,
                disable_refiner=True,
                disable_semres=False,
                verbose_every=0,
            )
            if "sem_subchunk" in sig.parameters:
                kwargs["sem_subchunk"] = args.sem_subchunk
            if "sem_rhs_only" in sig.parameters:
                kwargs["sem_rhs_only"] = args.eval_sem_rhs_only

            metrics = eval_chunked_bidirectional(**kwargs)

            rhs = metrics["rhs"]["total"]["MRR"]
            lhs = metrics["lhs"]["total"]["MRR"]
            avg = (rhs + lhs) / 2.0
            print(f"[VALID] RHS: {rhs:.4f} | LHS: {lhs:.4f} | AVG: {avg:.4f}")

            if avg > best_valid:
                best_valid = avg
                torch.save(semres_model.state_dict(), save_path)
                print(f"Saved best SemRes to {save_path} (Best VALID AVG={best_valid:.4f})")

    print(f"Done. Best VALID AVG MRR: {best_valid:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
