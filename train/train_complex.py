import argparse
import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())

from data.data_loader import KGProcessor, TrainDataset
from models.struct_backbone_complex import ComplExBackbone
from eval.eval_full_entity_filtered import eval_chunked_bidirectional, build_to_skip
from tools.run_meta import write_run_metadata


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


def _count_frequency(triples: np.ndarray, start: int = 4):
    count = {}
    for h, r, t in triples:
        key_hr = (int(h), int(r))
        key_tr = (int(t), -int(r) - 1)
        count[key_hr] = count.get(key_hr, start) + 1
        count[key_tr] = count.get(key_tr, start) + 1
    return count


def _get_true_head_and_tail(triples: np.ndarray):
    true_tail = {}
    true_head = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        true_tail.setdefault((h, r), set()).add(t)
        true_head.setdefault((r, t), set()).add(h)
    return true_head, true_tail


class OfficialTrainDataset(Dataset):
    def __init__(
        self,
        triples: np.ndarray,
        nentity: int,
        negative_sample_size: int,
        mode: str,
        true_head: dict,
        true_tail: dict,
        count: dict,
    ):
        assert mode in ("head-batch", "tail-batch")
        self.triples = triples
        self.nentity = int(nentity)
        self.negative_sample_size = int(negative_sample_size)
        self.mode = mode
        self.true_head = true_head
        self.true_tail = true_tail
        self.count = count

    def __len__(self):
        return self.triples.shape[0]

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx]
        h, r, t = int(h), int(r), int(t)

        w = self.count[(h, r)] + self.count[(t, -r - 1)]
        subsampling_weight = math.sqrt(1.0 / w)

        negs = []
        need = self.negative_sample_size
        forbidden = self.true_head.get((r, t), set()) if self.mode == "head-batch" else self.true_tail.get((h, r), set())

        while need > 0:
            cand = np.random.randint(self.nentity, size=need * 2, dtype=np.int64)
            if forbidden:
                forb_arr = np.fromiter(forbidden, dtype=np.int64)
                mask = ~np.in1d(cand, forb_arr, assume_unique=False)
                cand = cand[mask]
            if cand.size == 0:
                continue
            take = cand[:need]
            negs.append(take)
            need -= take.size

        negative_sample = np.concatenate(negs, axis=0)[: self.negative_sample_size]
        positive_sample = np.array([h, r, t], dtype=np.int64)

        return (
            torch.from_numpy(positive_sample).long(),
            torch.from_numpy(negative_sample).long(),
            torch.tensor([subsampling_weight], dtype=torch.float),
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        pos = torch.stack([d[0] for d in data], dim=0)
        neg = torch.stack([d[1] for d in data], dim=0)
        w = torch.cat([d[2] for d in data], dim=0)
        mode = data[0][3]
        return pos, neg, w, mode


def _one_shot_iterator(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


class BidirectionalOneShotIterator:
    def __init__(self, dataloader_head: DataLoader, dataloader_tail: DataLoader, disable_head_batch: bool = False, only_head_batch: bool = False):
        self.iterator_head = _one_shot_iterator(dataloader_head)
        self.iterator_tail = _one_shot_iterator(dataloader_tail)
        self.step = 0
        self.disable_head_batch = disable_head_batch
        self.only_head_batch = only_head_batch

    def __next__(self):
        self.step += 1
        if self.only_head_batch:
            return next(self.iterator_head)
        if self.disable_head_batch:
            return next(self.iterator_tail)
        return next(self.iterator_head) if (self.step % 2 == 0) else next(self.iterator_tail)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== ComplEx Stage3 (official-ish) on {device} ===")
    print(f"[NegSampling] adversarial={args.negative_adversarial_sampling}, T={args.adversarial_temperature}")
    print(f"[Weighting] uni_weight={args.uni_weight}")
    print(f"[Reg] L3={args.regularization}")
    print(f"[HeadBatch] disabled={args.disable_head_batch}, only_head={args.only_head_batch}")

    processor = KGProcessor(args.data_path, max_neighbors=16)
    processor.load_files()

    train_np = processor.train_triplets.cpu().numpy()
    true_head, true_tail = _get_true_head_and_tail(train_np)
    count = _count_frequency(train_np, start=4)

    ds_head = OfficialTrainDataset(
        triples=train_np,
        nentity=processor.num_entities,
        negative_sample_size=args.num_neg,
        mode="head-batch",
        true_head=true_head,
        true_tail=true_tail,
        count=count,
    )
    ds_tail = OfficialTrainDataset(
        triples=train_np,
        nentity=processor.num_entities,
        negative_sample_size=args.num_neg,
        mode="tail-batch",
        true_head=true_head,
        true_tail=true_tail,
        count=count,
    )

    g_head = torch.Generator()
    g_head.manual_seed(args.seed)
    g_tail = torch.Generator()
    g_tail.manual_seed(args.seed + 1)

    dl_head = DataLoader(
        ds_head,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_head,
        collate_fn=OfficialTrainDataset.collate_fn,
        drop_last=True,
    )
    dl_tail = DataLoader(
        ds_tail,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_tail,
        collate_fn=OfficialTrainDataset.collate_fn,
        drop_last=True,
    )

    train_it = BidirectionalOneShotIterator(
        dl_head,
        dl_tail,
        disable_head_batch=args.disable_head_batch,
        only_head_batch=args.only_head_batch,
    )
    steps_per_epoch = max(1, len(train_np) // args.batch_size)

    model = ComplExBackbone(processor.num_entities, processor.num_relations, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eval_loader = DataLoader(
        TrainDataset(processor.valid_triplets if args.eval_split == "valid" else processor.test_triplets),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    to_skip = build_to_skip(processor, split=args.eval_split)

    best_metric = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    write_run_metadata(args.save_dir, args)
    best_path = os.path.join(args.save_dir, "best_model.pth")

    def score_head_batch(r, t, cand_h):
        t_emb = model.entity_embedding[t]
        return model.score_from_head_emb(t_emb, r, cand_h, conj=torch.ones_like(r, dtype=torch.bool, device=device))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        mode_counter = {"head-batch": 0, "tail-batch": 0}
        global_step = (epoch - 1) * steps_per_epoch

        for _ in range(steps_per_epoch):
            global_step += 1
            pos, neg, subsampling_weight, mode = next(train_it)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            subsampling_weight = subsampling_weight.to(device, non_blocking=True)

            mode_counter[mode] += 1
            h = pos[:, 0]
            r = pos[:, 1]
            t = pos[:, 2]

            if mode == "tail-batch":
                pos_score = model(h, r, t, mode="single")
                neg_score = model(h, r, neg, mode="batch_neg")
            else:
                pos_score = score_head_batch(r, t, h.unsqueeze(1)).squeeze(1)
                neg_score = score_head_batch(r, t, neg)

            pos_log = F.logsigmoid(pos_score)
            if args.negative_adversarial_sampling:
                w = F.softmax(neg_score * args.adversarial_temperature, dim=1).detach()
                neg_log = (w * F.logsigmoid(-neg_score)).sum(dim=1)
            else:
                neg_log = F.logsigmoid(-neg_score).mean(dim=1)

            loss_vec = -(pos_log + neg_log)
            if args.uni_weight:
                loss = loss_vec.mean()
            else:
                loss = (subsampling_weight * loss_vec).sum() / subsampling_weight.sum()

            if args.regularization and args.regularization > 0:
                reg = (model.entity_embedding.norm(p=3) ** 3) + (model.relation_embedding.norm(p=3) ** 3)
                loss = loss + args.regularization * reg

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(steps_per_epoch, 1)
        head_ratio = mode_counter["head-batch"] / max(steps_per_epoch, 1) * 100
        tail_ratio = mode_counter["tail-batch"] / max(steps_per_epoch, 1) * 100
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.6f} | "
              f"head-batch: {mode_counter['head-batch']} ({head_ratio:.1f}%) | "
              f"tail-batch: {mode_counter['tail-batch']} ({tail_ratio:.1f}%)")

        if args.eval_every and (epoch % args.eval_every == 0):
            print(f"Evaluating on {args.eval_split} (filtered, bidirectional)...")
            metrics = eval_chunked_bidirectional(
                processor=processor,
                rotate_model=model,
                test_loader=eval_loader,
                device=device,
                to_skip=to_skip,
                eval_split=args.eval_split,
                refiner=None,
                semres_model=None,
                ent_text_embs=None,
                rel_text_embs=None,
                chunk_size=args.eval_chunk_size,
                sem_subchunk=256,
                disable_refiner=True,
                disable_semres=True,
                sem_rhs_only=False,
                sem_lhs_only=False,
                refiner_topk_only=False,
                refiner_topk=args.eval_topk,
                print_sem_stats=False,
                refiner_diag=False,
                recall_k=args.eval_topk,
                rel_bucket_map=None,
                score_eps=args.score_eps,
            )
            mrr = 0.5 * (metrics["rhs"]["total"]["MRR"] + metrics["lhs"]["total"]["MRR"])
            if mrr > best_metric:
                best_metric = mrr
                torch.save(model.state_dict(), best_path)
                print(f"[Save] new best MRR={mrr:.4f} -> {best_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/fb15k_custom")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--emb_dim", type=int, default=500)

    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_neg", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_split", type=str, default="valid", choices=["valid", "test"])
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--eval_chunk_size", type=int, default=2048)
    p.add_argument("--eval_topk", type=int, default=200)
    p.add_argument("--score_eps", type=float, default=0.0)

    p.add_argument("--negative_adversarial_sampling", action="store_true")
    p.add_argument("--adversarial_temperature", type=float, default=1.0)
    p.add_argument("--uni_weight", action="store_true")
    p.add_argument("--regularization", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--disable_head_batch", action="store_true")
    p.add_argument("--only_head_batch", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
