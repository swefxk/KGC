import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import bisect
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.rotate import RotatEModel
from test_semres import load_embeddings, build_to_skip
from models.semantic_biencoder import SemanticBiEncoderScorer


@torch.no_grad()
def topk_negatives_filtered(rotate_model, h, r, t, to_skip_rhs, num_entities, K=500, chunk_size=2048):
    """
    Return topK negative tails per query under filtered setting:
      - exclude all filtered true tails (except gold)
      - exclude gold
    """
    device = h.device
    B = h.size(0)

    top_scores = torch.full((B, K), -1e9, device=device)
    top_ids = torch.full((B, K), -1, device=device, dtype=torch.long)

    ent_ids = torch.arange(num_entities, device=device, dtype=torch.long)
    h_cpu, r_cpu, t_cpu = h.tolist(), r.tolist(), t.tolist()

    # precompute sorted filters for bisect
    filt_sorted = []
    for i in range(B):
        s = to_skip_rhs.get((h_cpu[i], r_cpu[i]), set())
        filt_sorted.append(sorted(list(s)))

    for start in range(0, num_entities, chunk_size):
        end = min(start + chunk_size, num_entities)
        cand = ent_ids[start:end]  # [C]
        s = rotate_model(h, r, cand, mode="batch_neg")  # [B,C]

        # mask filtered (except gold) + mask gold
        s_mask = torch.zeros_like(s, dtype=torch.bool)
        rows, cols = [], []
        for i in range(B):
            fs = filt_sorted[i]
            if fs:
                low = bisect.bisect_left(fs, start)
                high = bisect.bisect_left(fs, end)
                gold = t_cpu[i]
                for idx in range(low, high):
                    eid = fs[idx]
                    if eid == gold:
                        continue
                    rows.append(i)
                    cols.append(eid - start)
            gold = t_cpu[i]
            if start <= gold < end:
                rows.append(i)
                cols.append(gold - start)
        if rows:
            s_mask[rows, cols] = True
            s = s.masked_fill(s_mask, -1e9)

        cand2d = cand.unsqueeze(0).expand(B, -1)
        merged_scores = torch.cat([top_scores, s], dim=1)
        merged_ids = torch.cat([top_ids, cand2d], dim=1)
        top_scores, idx = torch.topk(merged_scores, k=K, dim=1)
        top_ids = torch.gather(merged_ids, 1, idx)

    return top_ids, top_scores


@torch.no_grad()
def sem_scores_biencoder(sem_model, ent_embs, rel_embs, h, r, t, neg_t):
    """
    pos_sem: [B]
    neg_sem: [B,K]
    """
    h_txt = ent_embs[h]
    r_txt = rel_embs[r]
    t_txt = ent_embs[t]

    q = sem_model.encode_query(h_txt, r_txt)
    pos_v = sem_model.encode_entity(t_txt)
    pos = (q * pos_v).sum(dim=-1)

    B, K = neg_t.shape
    neg_v = sem_model.encode_entity(ent_embs[neg_t.reshape(-1)]).view(B, K, -1)
    neg = torch.einsum("bd,bkd->bk", q, neg_v)
    return pos, neg


class Calibrator(nn.Module):
    def __init__(self, num_relations, per_relation: bool, b_max: float):
        super().__init__()
        self.per_relation = per_relation
        self.b_max = float(b_max)
        if per_relation:
            self.w = nn.Embedding(num_relations, 1)
            nn.init.constant_(self.w.weight, -4.0)  # sigmoid ~ 0.018 -> near zero start
        else:
            self.w = nn.Parameter(torch.tensor(-4.0))

    def forward(self, r_ids):
        if self.per_relation:
            x = self.w(r_ids).squeeze(-1)
        else:
            x = self.w.expand_as(r_ids)
        b = self.b_max * torch.sigmoid(x)  # b in [0, b_max]
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/fb15k_custom")
    ap.add_argument("--pretrained_rotate", type=str, required=True)
    ap.add_argument("--pretrained_sem", type=str, required=True)

    ap.add_argument("--per_relation", action="store_true")
    ap.add_argument("--b_max", type=float, default=20.0)

    ap.add_argument("--K", type=int, default=500)
    ap.add_argument("--chunk_size", type=int, default=2048)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--l2", type=float, default=1e-4)

    ap.add_argument("--save_path", type=str, default="checkpoints/sem_biencoder_simkgc_v1/calib.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    rotate = RotatEModel(processor.num_entities, processor.num_relations, emb_dim=500, margin=9.0).to(device)
    rotate.load_state_dict(torch.load(args.pretrained_rotate, map_location=device))
    rotate.eval()
    for p in rotate.parameters():
        p.requires_grad = False

    ent_embs, rel_embs = load_embeddings(processor, args, device)
    ent_embs = ent_embs.to(device, non_blocking=True)
    rel_embs = rel_embs.to(device, non_blocking=True)

    ckpt = torch.load(args.pretrained_sem, map_location="cpu")
    if not (isinstance(ckpt, dict) and ckpt.get("model_type") == "biencoder"):
        raise ValueError("Expect biencoder checkpoint with model_type='biencoder'")
    cfg = ckpt["model_args"]
    sem = SemanticBiEncoderScorer(
        text_dim=cfg["text_dim"],
        num_relations=cfg["num_relations"],
        proj_dim=cfg.get("proj_dim", 256),
        dropout=cfg.get("dropout", 0.1),
        text_norm=cfg.get("text_norm", True),
    ).to(device)
    sem.load_state_dict(ckpt["state_dict"], strict=True)
    sem.eval()
    for p in sem.parameters():
        p.requires_grad = False

    # filtered sets for VALID
    to_skip = build_to_skip(processor, split="valid")
    to_skip_rhs = to_skip["rhs"]

    loader = DataLoader(
        TrainDataset(processor.valid_triplets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    calib = Calibrator(processor.num_relations, args.per_relation, args.b_max).to(device)
    opt = optim.AdamW(calib.parameters(), lr=args.lr, weight_decay=args.l2)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        calib.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            with torch.no_grad():
                neg_ids, _ = topk_negatives_filtered(
                    rotate, h, r, t,
                    to_skip_rhs=to_skip_rhs,
                    num_entities=processor.num_entities,
                    K=args.K,
                    chunk_size=args.chunk_size
                )
                s_pos_str = rotate(h, r, t, mode="single")
                s_neg_str = rotate(h, r, neg_ids, mode="batch_neg")  # [B,K]
                s_pos_sem, s_neg_sem = sem_scores_biencoder(sem, ent_embs, rel_embs, h, r, t, neg_ids)

            b = calib(r)  # [B], non-negative

            logits = torch.cat(
                [(s_pos_str + b * s_pos_sem).unsqueeze(1), (s_neg_str + b.unsqueeze(1) * s_neg_sem)],
                dim=1
            )
            labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)

            loss = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            n += 1

        # report mean b
        with torch.no_grad():
            if args.per_relation:
                b_mean = float((args.b_max * torch.sigmoid(calib.w.weight.squeeze(-1))).mean().item())
            else:
                b_mean = float((args.b_max * torch.sigmoid(calib.w)).item())
        print(f"Epoch {ep} | calib_loss={total/max(1,n):.6f} | mean_b~{b_mean:.4f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    obj = {
        "per_relation": bool(args.per_relation),
        "b_max": float(args.b_max),
        "state_dict": calib.state_dict(),
    }
    torch.save(obj, args.save_path)
    print(f"Saved calibrator to: {args.save_path}")


if __name__ == "__main__":
    main()
