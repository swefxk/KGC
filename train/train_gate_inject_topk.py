import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from models.struct_refiner import StructRefiner
from models.semantic_biencoder import SemanticBiEncoderScorer
from models.gate_injector import ConfidenceGate, entropy_from_logits
from eval.eval_full_entity_filtered import load_embeddings
from tools.run_meta import write_run_metadata


@torch.no_grad()
def sem_score_rhs_biencoder(sem_model, ent_embs, rel_embs, h, r, cand_t_2d):
    B, K = cand_t_2d.shape
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    q = sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))  # [B,d]
    v = sem_model.encode_entity(ent_embs[cand_t_2d.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q, v)


@torch.no_grad()
def sem_score_rhs_biencoder_pos(sem_model, ent_embs, rel_embs, h, r, t):
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    t_txt = ent_embs[t]
    q = sem_model.encode_query(h_txt, r_txt, dir_ids=torch.zeros_like(h))
    v = sem_model.encode_entity(t_txt)
    return (q * v).sum(dim=-1)  # [B]


@torch.no_grad()
def sem_score_lhs_biencoder(sem_model, ent_embs, rel_embs, t, r, cand_h_2d):
    B, K = cand_h_2d.shape
    t_txt = ent_embs[t]
    r_txt = rel_embs[r]
    q = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))  # [B,d]
    v = sem_model.encode_entity(ent_embs[cand_h_2d.reshape(-1)]).view(B, K, -1)
    return torch.einsum("bd,bkd->bk", q, v)


@torch.no_grad()
def sem_score_lhs_biencoder_pos(sem_model, ent_embs, rel_embs, t, r, h):
    t_txt = ent_embs[t]
    r_txt = rel_embs[r]
    h_txt = ent_embs[h]
    q = sem_model.encode_query(t_txt, r_txt, dir_ids=torch.ones_like(t))
    v = sem_model.encode_entity(h_txt)
    return (q * v).sum(dim=-1)  # [B]


def sample_from_topk(cand_2d: torch.Tensor, k_sample: int, prefix_k: int):
    """
    Return sampled candidates AND their indices in the topK list.
    Strategy: take top prefix_k, and random sample from the rest.
    """
    B, K = cand_2d.shape
    if k_sample <= 0 or k_sample >= K:
        idx = torch.arange(K, device=cand_2d.device).unsqueeze(0).expand(B, -1)
        return cand_2d, idx

    prefix_k = max(0, min(prefix_k, k_sample, K))
    rest_k = k_sample - prefix_k

    if prefix_k > 0:
        idx_prefix = torch.arange(prefix_k, device=cand_2d.device).unsqueeze(0).expand(B, -1)
    else:
        idx_prefix = torch.empty((B, 0), dtype=torch.long, device=cand_2d.device)

    if rest_k > 0 and prefix_k < K:
        idx_rest = torch.randint(prefix_k, K, (B, rest_k), device=cand_2d.device)
        idx = torch.cat([idx_prefix, idx_rest], dim=1)
    else:
        idx = idx_prefix

    return cand_2d.gather(1, idx), idx


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if float(denom.item()) < 1e-12:
        return 0.0
    return float((x * y).mean().item() / denom.item())


def save_args(args, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "args.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[IO] Saved args to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--pretrained_refiner", type=str, default=None)
    ap.add_argument("--pretrained_sem", type=str, required=True)
    ap.add_argument("--train_cache_rhs", type=str, required=True)
    ap.add_argument("--train_cache_lhs", type=str, required=True)

    ap.add_argument("--b_rhs", type=float, default=2.0)
    ap.add_argument("--b_lhs", type=float, default=2.5)
    ap.add_argument("--sample_k", type=int, default=128)

    ap.add_argument("--gate_rel_dim", type=int, default=16)
    ap.add_argument("--gate_dir_dim", type=int, default=8)
    ap.add_argument("--gate_hidden_dim", type=int, default=64)
    ap.add_argument("--gate_g_min", type=float, default=0.0)
    ap.add_argument("--gate_g_max", type=float, default=2.0)
    ap.add_argument("--gate_init_bias", type=float, default=0.5413)
    ap.add_argument("--gate_ent_temp", type=float, default=1.0)
    ap.add_argument("--lambda_g", type=float, default=0.05)
    ap.add_argument("--lambda_mono", type=float, default=0.1)
    ap.add_argument("--lambda_risk", type=float, default=0.05)
    ap.add_argument("--risk_temp", type=float, default=1.0)
    ap.add_argument("--sample_prefix_k", type=int, default=64)
    ap.add_argument("--diag_every", type=int, default=10)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_run_metadata(args.save_dir, args)
    save_args(args, args.save_dir)

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()
    train = processor.train_triplets.to(torch.long)

    # caches
    rhs = torch.load(args.train_cache_rhs, map_location="cpu")
    lhs = torch.load(args.train_cache_lhs, map_location="cpu")
    neg_t = (rhs["neg_t"] if isinstance(rhs, dict) else rhs).to(torch.long)
    neg_h = (lhs["neg_h"] if isinstance(lhs, dict) else lhs).to(torch.long)
    if neg_t.size(0) != train.size(0) or neg_h.size(0) != train.size(0):
        raise ValueError("Cache rows do not match train size.")

    # models (frozen)
    rotate = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()
    for p in rotate.parameters():
        p.requires_grad = False

    refiner = None
    nbr_ent = nbr_rel = nbr_dir = nbr_mask = freq = None
    if args.pretrained_refiner:
        refiner = StructRefiner(
            emb_dim=500,
            K=16,
            num_relations=processor.num_relations,
        ).to(device)
        refiner.load_state_dict(torch.load(args.pretrained_refiner, map_location=device), strict=False)
        refiner.eval()
        for p in refiner.parameters():
            p.requires_grad = False
        nbr_ent = processor.nbr_ent.to(device, non_blocking=True)
        nbr_rel = processor.nbr_rel.to(device, non_blocking=True)
        nbr_dir = processor.nbr_dir.to(device, non_blocking=True)
        nbr_mask = processor.nbr_mask.to(device, non_blocking=True)
        freq = processor.freq.to(device, non_blocking=True)

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    sem = SemanticBiEncoderScorer(
        text_dim=ent_embs.size(1),
        num_relations=processor.num_relations,
        proj_dim=256,
        dropout=0.1,
        text_norm=True,
    ).to(device)
    sem_ckpt = torch.load(args.pretrained_sem, map_location="cpu")
    if isinstance(sem_ckpt, dict) and "state_dict" in sem_ckpt:
        sem_ckpt = sem_ckpt["state_dict"]
    sem.load_state_dict(sem_ckpt, strict=False)
    sem.eval()
    for p in sem.parameters():
        p.requires_grad = False

    gate = ConfidenceGate(
        num_relations=processor.num_relations,
        rel_emb_dim=args.gate_rel_dim,
        dir_emb_dim=args.gate_dir_dim,
        hidden_dim=args.gate_hidden_dim,
        g_min=args.gate_g_min,
        g_max=args.gate_g_max,
        init_bias=args.gate_init_bias,
    ).to(device)

    opt = optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    loader = DataLoader(
        torch.arange(train.size(0)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    for ep in range(1, args.epochs + 1):
        gate.train()
        loss_sum = 0.0
        g_rhs_sum = 0.0
        g_lhs_sum = 0.0
        recall_rhs = 0.0
        recall_lhs = 0.0
        corr_m12_rhs = 0.0
        corr_ent_rhs = 0.0
        corr_m12_lhs = 0.0
        corr_ent_lhs = 0.0
        g_pos_rhs = 0.0
        g_neg_rhs = 0.0
        g_pos_lhs = 0.0
        g_neg_lhs = 0.0
        n_diag = 0
        n_batches = 0

        for row_idx in loader:
            row_idx_cpu = row_idx  # CPU indices
            triples = train[row_idx_cpu].to(device)
            h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
            B = h.size(0)

            cand_t = neg_t[row_idx_cpu].to(device)  # [B,K]
            cand_h = neg_h[row_idx_cpu].to(device)  # [B,K]

            if refiner is None:
                top_scores_rhs = rotate(h, r, cand_t, mode="batch_neg")
                s_gold_struct_rhs = rotate(h, r, t, mode="single")
            else:
                anchor_emb = refiner.refine_anchor(h, rotate, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq)
                conj_flag = torch.zeros(B, dtype=torch.bool, device=device)
                top_scores_rhs = rotate.score_from_head_emb(anchor_emb, r, cand_t, conj=conj_flag)
                s_gold_struct_rhs = rotate.score_from_head_emb(anchor_emb, r, t.unsqueeze(1), conj=conj_flag).squeeze(1)

            if refiner is None:
                top_scores_lhs = rotate.score_head_batch(r, t, cand_h)
                s_gold_struct_lhs = rotate.score_head_batch(r, t, h.unsqueeze(1)).squeeze(1)
            else:
                anchor_emb = refiner.refine_anchor(t, rotate, nbr_ent, nbr_rel, nbr_dir, nbr_mask, freq)
                conj_flag = torch.ones(B, dtype=torch.bool, device=device)
                top_scores_lhs = rotate.score_from_head_emb(anchor_emb, r, cand_h, conj=conj_flag)
                s_gold_struct_lhs = rotate.score_from_head_emb(anchor_emb, r, h.unsqueeze(1), conj=conj_flag).squeeze(1)

            dir0 = torch.zeros_like(h)
            dir1 = torch.ones_like(h)
            g_rhs = gate(top_scores_rhs, r, dir0, ent_temp=args.gate_ent_temp)
            g_lhs = gate(top_scores_lhs, r, dir1, ent_temp=args.gate_ent_temp)

            # monotonic prior: confident -> smaller g
            m12_rhs = top_scores_rhs[:, 0] - top_scores_rhs[:, 1]
            m12_lhs = top_scores_lhs[:, 0] - top_scores_lhs[:, 1]
            perm = torch.randperm(B, device=device)
            mono_mask_rhs = m12_rhs > m12_rhs[perm]
            mono_mask_lhs = m12_lhs > m12_lhs[perm]
            loss_mono_rhs = torch.nn.functional.softplus(g_rhs - g_rhs[perm])[mono_mask_rhs].mean() if mono_mask_rhs.any() else 0.0
            loss_mono_lhs = torch.nn.functional.softplus(g_lhs - g_lhs[perm])[mono_mask_lhs].mean() if mono_mask_lhs.any() else 0.0
            loss_mono = loss_mono_rhs + loss_mono_lhs

            cand_t_s, idx_t = sample_from_topk(cand_t, args.sample_k, args.sample_prefix_k)
            cand_h_s, idx_h = sample_from_topk(cand_h, args.sample_k, args.sample_prefix_k)

            with torch.no_grad():
                sem_topk_rhs = sem_score_rhs_biencoder(sem, ent_embs, rel_embs, h, r, cand_t_s)
                sem_gold_rhs = sem_score_rhs_biencoder_pos(sem, ent_embs, rel_embs, h, r, t)

                sem_topk_lhs = sem_score_lhs_biencoder(sem, ent_embs, rel_embs, t, r, cand_h_s)
                sem_gold_lhs = sem_score_lhs_biencoder_pos(sem, ent_embs, rel_embs, t, r, h)

                sem_topk_rhs_full = sem_score_rhs_biencoder(sem, ent_embs, rel_embs, h, r, cand_t)
                sem_topk_lhs_full = sem_score_lhs_biencoder(sem, ent_embs, rel_embs, t, r, cand_h)

            top_scores_rhs_s = top_scores_rhs.gather(1, idx_t)
            top_scores_lhs_s = top_scores_lhs.gather(1, idx_h)

            logits_rhs = torch.cat(
                [(s_gold_struct_rhs + (args.b_rhs * g_rhs) * sem_gold_rhs).unsqueeze(1),
                 (top_scores_rhs_s + (args.b_rhs * g_rhs).unsqueeze(1) * sem_topk_rhs)],
                dim=1
            )
            logits_lhs = torch.cat(
                [(s_gold_struct_lhs + (args.b_lhs * g_lhs) * sem_gold_lhs).unsqueeze(1),
                 (top_scores_lhs_s + (args.b_lhs * g_lhs).unsqueeze(1) * sem_topk_lhs)],
                dim=1
            )

            labels = torch.zeros(B, dtype=torch.long, device=device)
            loss_rhs = ce(logits_rhs, labels)
            loss_lhs = ce(logits_lhs, labels)
            reg = args.lambda_g * 0.5 * (g_rhs.mean() + g_lhs.mean())
            # risk regularizer: penalize gate when sem disagrees with struct
            T = max(args.risk_temp, 1e-6)
            p_rhs = torch.softmax(top_scores_rhs / T, dim=1)
            q_rhs = torch.softmax(sem_topk_rhs_full / T, dim=1)
            p_lhs = torch.softmax(top_scores_lhs / T, dim=1)
            q_lhs = torch.softmax(sem_topk_lhs_full / T, dim=1)
            kl_rhs = (p_rhs * (p_rhs.clamp_min(1e-12).log() - q_rhs.clamp_min(1e-12).log())).sum(dim=1)
            kl_lhs = (p_lhs * (p_lhs.clamp_min(1e-12).log() - q_lhs.clamp_min(1e-12).log())).sum(dim=1)
            loss_risk = (g_rhs * kl_rhs).mean() + (g_lhs * kl_lhs).mean()

            loss = loss_rhs + loss_lhs + reg + args.lambda_mono * loss_mono + args.lambda_risk * loss_risk

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            g_rhs_sum += float(g_rhs.mean().item())
            g_lhs_sum += float(g_lhs.mean().item())
            kth_rhs = top_scores_rhs.min(dim=1).values
            kth_lhs = top_scores_lhs.min(dim=1).values
            recall_rhs += float((s_gold_struct_rhs >= kth_rhs).float().mean().item())
            recall_lhs += float((s_gold_struct_lhs >= kth_lhs).float().mean().item())
            n_batches += 1

            if args.diag_every > 0 and (n_batches % args.diag_every == 0):
                with torch.no_grad():
                    m12_rhs = top_scores_rhs[:, 0] - top_scores_rhs[:, 1]
                    ent_rhs = entropy_from_logits(top_scores_rhs, temp=args.gate_ent_temp)
                    m12_lhs = top_scores_lhs[:, 0] - top_scores_lhs[:, 1]
                    ent_lhs = entropy_from_logits(top_scores_lhs, temp=args.gate_ent_temp)

                    corr_m12_rhs += pearson_corr(g_rhs, m12_rhs)
                    corr_ent_rhs += pearson_corr(g_rhs, ent_rhs)
                    corr_m12_lhs += pearson_corr(g_lhs, m12_lhs)
                    corr_ent_lhs += pearson_corr(g_lhs, ent_lhs)

                    # topK rank improvement diagnostic
                    sem_topk_rhs_full = sem_score_rhs_biencoder(sem, ent_embs, rel_embs, h, r, cand_t)
                    sem_topk_lhs_full = sem_score_lhs_biencoder(sem, ent_embs, rel_embs, t, r, cand_h)
                    total_topk_rhs = top_scores_rhs + (args.b_rhs * g_rhs).unsqueeze(1) * sem_topk_rhs_full
                    total_topk_lhs = top_scores_lhs + (args.b_lhs * g_lhs).unsqueeze(1) * sem_topk_lhs_full

                    s_gold_total_rhs = s_gold_struct_rhs + (args.b_rhs * g_rhs) * sem_gold_rhs
                    s_gold_total_lhs = s_gold_struct_lhs + (args.b_lhs * g_lhs) * sem_gold_lhs

                    rank_struct_rhs = (top_scores_rhs > s_gold_struct_rhs.unsqueeze(1)).sum(dim=1) + 1
                    rank_total_rhs = (total_topk_rhs > s_gold_total_rhs.unsqueeze(1)).sum(dim=1) + 1
                    rank_struct_lhs = (top_scores_lhs > s_gold_struct_lhs.unsqueeze(1)).sum(dim=1) + 1
                    rank_total_lhs = (total_topk_lhs > s_gold_total_lhs.unsqueeze(1)).sum(dim=1) + 1

                    in_topk_rhs = s_gold_struct_rhs >= top_scores_rhs.min(dim=1).values
                    in_topk_lhs = s_gold_struct_lhs >= top_scores_lhs.min(dim=1).values

                    delta_rhs = (rank_struct_rhs - rank_total_rhs)
                    delta_lhs = (rank_struct_lhs - rank_total_lhs)

                    if in_topk_rhs.any():
                        g_pos_rhs += float(g_rhs[(in_topk_rhs & (delta_rhs > 0))].mean().item()) if (in_topk_rhs & (delta_rhs > 0)).any() else 0.0
                        g_neg_rhs += float(g_rhs[(in_topk_rhs & (delta_rhs <= 0))].mean().item()) if (in_topk_rhs & (delta_rhs <= 0)).any() else 0.0
                    if in_topk_lhs.any():
                        g_pos_lhs += float(g_lhs[(in_topk_lhs & (delta_lhs > 0))].mean().item()) if (in_topk_lhs & (delta_lhs > 0)).any() else 0.0
                        g_neg_lhs += float(g_lhs[(in_topk_lhs & (delta_lhs <= 0))].mean().item()) if (in_topk_lhs & (delta_lhs <= 0)).any() else 0.0

                    n_diag += 1

        print(f"Epoch {ep} | loss={loss_sum/max(1,n_batches):.6f} | "
              f"g_rhs_mean={g_rhs_sum/max(1,n_batches):.4f} | g_lhs_mean={g_lhs_sum/max(1,n_batches):.4f} | "
              f"recall@K_rhs={recall_rhs/max(1,n_batches):.4f} | recall@K_lhs={recall_lhs/max(1,n_batches):.4f}")
        if n_diag > 0:
            print(f"[Diag] corr(g,m12): rhs={corr_m12_rhs/max(1,n_diag):.4f} lhs={corr_m12_lhs/max(1,n_diag):.4f} | "
                  f"corr(g,entropy): rhs={corr_ent_rhs/max(1,n_diag):.4f} lhs={corr_ent_lhs/max(1,n_diag):.4f}")
            print(f"[Diag] g_mean | Δrank>0 vs ≤0: rhs={g_pos_rhs/max(1,n_diag):.4f}/{g_neg_rhs/max(1,n_diag):.4f} "
                  f"lhs={g_pos_lhs/max(1,n_diag):.4f}/{g_neg_lhs/max(1,n_diag):.4f}")

        save_path = os.path.join(args.save_dir, f"gate_ep{ep}.pth")
        torch.save({"state_dict": gate.state_dict(), "args": vars(args)}, save_path)
        print(f"Saved gate to {save_path}")


if __name__ == "__main__":
    main()
