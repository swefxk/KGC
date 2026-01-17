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
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from models.semantic_residual import SemanticResidualScorerV2
from eval.eval_full_entity_filtered import eval_chunked_bidirectional, load_embeddings, build_to_skip


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
# Dataset with row index (for cache alignment)
# -------------------------
class IndexedTripletDataset(Dataset):
    """Return (triplet_tensor[3], row_idx_int). row_idx aligns with original ordering."""
    def __init__(self, triplets: torch.LongTensor):
        assert isinstance(triplets, torch.Tensor) and triplets.ndim == 2 and triplets.size(1) == 3
        self.triplets = triplets

    def __len__(self):
        return self.triplets.size(0)

    def __getitem__(self, idx: int):
        return self.triplets[idx], idx


def indexed_collate_fn(batch):
    triples = torch.stack([x[0] for x in batch], dim=0)         # [B,3]
    idxs = torch.tensor([x[1] for x in batch], dtype=torch.long) # [B]
    return triples, idxs


# -------------------------
# Utils
# -------------------------
def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved args to {path}")


def sample_uniform_excluding_target(num_entities: int, target_ids: torch.LongTensor, shape, device):
    neg = torch.randint(0, num_entities, shape, device=device, dtype=torch.long)
    if target_ids is None:
        return neg
    mask = neg.eq(target_ids.unsqueeze(1))
    while mask.any():
        neg[mask] = torch.randint(0, num_entities, (int(mask.sum().item()),), device=device, dtype=torch.long)
        mask = neg.eq(target_ids.unsqueeze(1))
    return neg


def sample_from_cache(cache_neg_t: torch.LongTensor, row_idx: torch.LongTensor, k_sample: int):
    """
    cache_neg_t: [N, Kcache]
    row_idx: [B]
    return: sampled [B, k_sample]
    """
    B = row_idx.size(0)
    Kcache = cache_neg_t.size(1)
    if k_sample >= Kcache:
        return cache_neg_t[row_idx]
    cols = torch.randint(0, Kcache, (B, k_sample), device=row_idx.device, dtype=torch.long)
    out = cache_neg_t[row_idx.unsqueeze(1), cols]
    return out


@torch.no_grad()
def rotate_score_pos(rotate_model, h, r, t):
    return rotate_model(h, r, t, mode="single")  # [B]


@torch.no_grad()
def rotate_score_batchneg(rotate_model, h, r, cand_t):
    return rotate_model(h, r, cand_t, mode="batch_neg")  # [B,K]


def sem_score_flat(semres_model, ent_embs, rel_embs, h_ids, r_ids, t_ids):
    h_txt = ent_embs[h_ids]
    r_txt = rel_embs[r_ids]
    t_txt = ent_embs[t_ids]
    delta, lam = semres_model(h_txt, r_txt, t_txt, r_ids)
    return lam * delta  # [M]


def sem_score_batchneg(semres_model, ent_embs, rel_embs, h_ids, r_ids, cand_t, sem_subchunk=256):
    device = h_ids.device
    B, K = cand_t.shape
    out = torch.empty((B, K), device=device, dtype=torch.float32)

    for start in range(0, K, sem_subchunk):
        end = min(start + sem_subchunk, K)
        sub = cand_t[:, start:end]
        k = sub.size(1)

        h_rep = h_ids.unsqueeze(1).expand(-1, k).reshape(-1)
        r_rep = r_ids.unsqueeze(1).expand(-1, k).reshape(-1)
        t_flat = sub.reshape(-1)

        s = sem_score_flat(semres_model, ent_embs, rel_embs, h_rep, r_rep, t_flat)
        out[:, start:end] = s.view(B, k)

    return out


# -------------------------
# Relation alpha: positive bounded gate
# alpha = alpha_max * sigmoid(alpha_raw)
# -------------------------
class RelationAlpha(nn.Module):
    def __init__(self, num_relations: int, alpha_init: float, alpha_max: float):
        super().__init__()
        self.alpha_raw = nn.Embedding(num_relations, 1)

        alpha_init = float(alpha_init)
        alpha_max = float(alpha_max)
        if not (0.0 < alpha_init < alpha_max):
            raise ValueError(f"Require 0 < alpha_init < alpha_max, got alpha_init={alpha_init}, alpha_max={alpha_max}")

        # sigmoid(x) = alpha_init/alpha_max  => x = log(p/(1-p))
        p = alpha_init / alpha_max
        x0 = math.log(p / (1.0 - p))
        nn.init.constant_(self.alpha_raw.weight, x0)

        self.alpha_max = alpha_max

    def forward(self, r_ids: torch.LongTensor):
        a = torch.sigmoid(self.alpha_raw(r_ids).squeeze(-1))  # [B]
        return self.alpha_max * a


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="data/fb15k_custom")
    parser.add_argument("--pretrained_rotate", type=str, required=True)
    parser.add_argument("--train_cache", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # semres
    parser.add_argument("--lambda_sem", type=float, default=0.01)
    parser.add_argument("--text_norm", action="store_true")
    parser.add_argument("--delta_bound", type=float, default=10.0)

    # candidates
    parser.add_argument("--cache_K_sample", type=int, default=256)
    parser.add_argument("--rand_neg", type=int, default=0)  # IMPORTANT: default 0 (avoid positive contamination)
    parser.add_argument("--sem_subchunk", type=int, default=256)

    # loss
    parser.add_argument("--total_loss_weight", type=float, default=0.05)
    parser.add_argument("--sem_only_loss_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # alpha gate
    parser.add_argument("--alpha_init", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--alpha_lr_mult", type=float, default=1.0)
    parser.add_argument("--alpha_l2", type=float, default=1e-4)

    # optim
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    # train
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    # eval
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_chunk_size", type=int, default=2048)
    parser.add_argument("--eval_sem_rhs_only", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    save_args(args, args.save_dir)
    save_path = os.path.join(args.save_dir, "semres_best.pth")

    print(f"=== Train SemRes Rerank V3 (FIXIDX V2) on {device_type} ===")
    print(f"[Sem] lambda_sem={args.lambda_sem} text_norm={args.text_norm} delta_bound={args.delta_bound}")
    print(f"[Cand] cache_K_sample={args.cache_K_sample} rand_neg={args.rand_neg}")
    print(f"[Loss] total_w={args.total_loss_weight} sem_w={args.sem_only_loss_weight} temp={args.temperature}")
    print(f"[Alpha] init={args.alpha_init} max={args.alpha_max} lr_mult={args.alpha_lr_mult} l2={args.alpha_l2}")
    print(f"[Opt] lr={args.lr} wd={args.weight_decay} amp={args.amp}")

    # ---- data ----
    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

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

    # ---- cache ----
    cache_neg_t = torch.load(args.train_cache, map_location="cpu")
    if isinstance(cache_neg_t, dict) and "neg_t" in cache_neg_t:
        cache_neg_t = cache_neg_t["neg_t"]
    if not isinstance(cache_neg_t, torch.Tensor):
        raise ValueError(f"train_cache must be Tensor (or dict with key 'neg_t'), got {type(cache_neg_t)}")
    cache_neg_t = cache_neg_t.to(torch.long)

    if cache_neg_t.size(0) != processor.train_triplets.size(0):
        raise ValueError(
            f"Cache mismatch: cache rows={cache_neg_t.size(0)} vs num_train={processor.train_triplets.size(0)}"
        )

    print(f"[Cache] Loaded neg_t: {tuple(cache_neg_t.shape)} dtype={cache_neg_t.dtype}")

    # ---- RotatE ----
    print(f"Loading frozen RotatE from {args.pretrained_rotate}")
    rotate_model = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate_model.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate_model.eval()
    for p in rotate_model.parameters():
        p.requires_grad = False

    # ---- embeddings ----
    ent_embs, rel_embs = load_embeddings(processor, args, device)
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    # ---- SemRes ----
    semres_model = SemanticResidualScorerV2(
        text_dim=ent_embs.size(1),
        num_relations=processor.num_relations,
        lambda_base=args.lambda_sem,
        text_norm=args.text_norm,
        delta_bound=args.delta_bound
    ).to(device)

    rel_alpha = RelationAlpha(processor.num_relations, args.alpha_init, args.alpha_max).to(device)

    optimizer = optim.AdamW(
        [
            {"params": semres_model.parameters(), "lr": args.lr},
            {"params": rel_alpha.parameters(), "lr": args.lr * args.alpha_lr_mult},
        ],
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ce = nn.CrossEntropyLoss()

    best_valid = -1.0
    to_skip_valid = build_to_skip(processor, split="valid")

    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        semres_model.train()
        rel_alpha.train()

        total_loss = 0.0
        n_batches = 0

        # stats
        alpha_abs_sum = 0.0
        alpha_max = 0.0
        sem_pos_sum = 0.0
        sem_neg_sum = 0.0
        stat_batches = 0

        for triples, row_idx in train_loader:
            triples = triples.to(device, non_blocking=True)
            row_idx = row_idx.to(device, non_blocking=True)

            h = triples[:, 0]
            r = triples[:, 1]
            t = triples[:, 2]
            B = h.size(0)

            # cand negatives: cache hard negatives (+ optional random)
            cand_cache = sample_from_cache(cache_neg_t.to(device), row_idx, args.cache_K_sample)  # [B,Kc]
            if args.rand_neg > 0:
                cand_rand = sample_uniform_excluding_target(processor.num_entities, t, (B, args.rand_neg), device)
                cand_neg = torch.cat([cand_cache, cand_rand], dim=1)
            else:
                cand_neg = cand_cache

            # struct scores (frozen)
            with torch.no_grad():
                s_pos_str = rotate_score_pos(rotate_model, h, r, t)               # [B]
                s_neg_str = rotate_score_batchneg(rotate_model, h, r, cand_neg)   # [B,K]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # raw sem scores (NO alpha)
                s_pos_sem_raw = sem_score_flat(semres_model, ent_embs, rel_embs, h, r, t)  # [B]
                s_neg_sem_raw = sem_score_batchneg(
                    semres_model, ent_embs, rel_embs, h, r, cand_neg, sem_subchunk=args.sem_subchunk
                )  # [B,K]

                alpha = rel_alpha(r)  # [B] positive bounded

                # total uses gated sem
                s_pos_sem = alpha * s_pos_sem_raw
                s_neg_sem = alpha.unsqueeze(1) * s_neg_sem_raw

                logits_total = torch.cat(
                    [(s_pos_str + s_pos_sem).unsqueeze(1), (s_neg_str + s_neg_sem)],
                    dim=1
                ) / max(args.temperature, 1e-6)

                # sem-only loss uses RAW sem (关键：保证 semres 有梯度)
                logits_sem = torch.cat(
                    [s_pos_sem_raw.unsqueeze(1), s_neg_sem_raw],
                    dim=1
                ) / max(args.temperature, 1e-6)

                labels = torch.zeros(B, device=device, dtype=torch.long)

                loss_total = ce(logits_total, labels)
                loss_sem = ce(logits_sem, labels)

                # alpha l2 on raw parameters
                alpha_l2 = (rel_alpha.alpha_raw.weight.squeeze(-1) ** 2).mean()

                loss = args.total_loss_weight * loss_total + args.sem_only_loss_weight * loss_sem + args.alpha_l2 * alpha_l2

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(semres_model.parameters()) + list(rel_alpha.parameters()),
                    args.grad_clip
                )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1

            # stats (light)
            if stat_batches < 50:
                with torch.no_grad():
                    a = alpha.detach()
                    alpha_abs_sum += float(a.abs().mean().item())
                    alpha_max = max(alpha_max, float(a.max().item()))
                    sem_pos_sum += float(s_pos_sem_raw.mean().item())
                    sem_neg_sum += float(s_neg_sem_raw.mean().item())
                    stat_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        alpha_abs_mean = alpha_abs_sum / max(1, stat_batches)
        sem_pos_mean = sem_pos_sum / max(1, stat_batches)
        sem_neg_mean = sem_neg_sum / max(1, stat_batches)

        print(f"Epoch {epoch} | AvgLoss: {avg_loss:.6f} | "
              f"alpha_abs_mean~{alpha_abs_mean:.4f} alpha_max~{alpha_max:.4f} | "
              f"sem_pos_raw_mean~{sem_pos_mean:.5f} sem_neg_raw_mean~{sem_neg_mean:.5f}")

        # ---- Eval ----
        if args.eval_every and (epoch % args.eval_every == 0):
            semres_model.eval()
            rel_alpha.eval()
            print("Evaluating on VALID (full-entities, filtered)...")

            class SemResWithAlpha(nn.Module):
                def __init__(self, sem, alpha):
                    super().__init__()
                    self.sem = sem
                    self.alpha = alpha

                def forward(self, h_txt, r_txt, t_txt, r_ids):
                    delta, lam = self.sem(h_txt, r_txt, t_txt, r_ids)
                    # eval uses lam*delta; we want alpha*(lam*delta)
                    a = self.alpha(r_ids)  # [M]
                    new_lam = a * lam
                    return delta, new_lam

            sem_eval = SemResWithAlpha(semres_model, rel_alpha).to(device)
            sem_eval.eval()

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
                semres_model=sem_eval,
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
                ckpt = {
                    "semres": semres_model.state_dict(),
                    "rel_alpha": rel_alpha.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, save_path)
                print(f"Saved best to {save_path} (Best AVG={best_valid:.4f})")

    print(f"Done. Best VALID AVG MRR: {best_valid:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    main()
