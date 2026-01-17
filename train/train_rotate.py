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
from models.rotate import RotatEModel

# 复用你已有的严格 filtered bidirectional eval
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
    """
    关键：DataLoader 多进程时，numpy/random 必须按 worker 独立设种子。
    否则你所谓“负采样”高度重复，训练直接废一半。
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _count_frequency(triples: np.ndarray, start: int = 4):
    """
    官方常用：对 (h,r) 与 (t, -r-1) 计数，用于 subsampling_weight
    """
    count = {}
    for h, r, t in triples:
        key_hr = (int(h), int(r))
        key_tr = (int(t), -int(r) - 1)
        count[key_hr] = count.get(key_hr, start) + 1
        count[key_tr] = count.get(key_tr, start) + 1
    return count


def _get_true_head_and_tail(triples: np.ndarray):
    """
    true_tail[(h,r)] = set(t)
    true_head[(r,t)] = set(h)
    """
    true_tail = {}
    true_head = {}
    for h, r, t in triples:
        h, r, t = int(h), int(r), int(t)
        true_tail.setdefault((h, r), set()).add(t)
        true_head.setdefault((r, t), set()).add(h)
    return true_head, true_tail


class OfficialTrainDataset(Dataset):
    """
    输出格式对齐常见官方实现：
    positive_sample, negative_sample, subsampling_weight, mode
    mode in {"head-batch", "tail-batch"}
    """
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

        # subsampling weight
        w = self.count[(h, r)] + self.count[(t, -r - 1)]
        subsampling_weight = math.sqrt(1.0 / w)

        # 过滤假负例：排除 train 真值集合
        negs = []
        need = self.negative_sample_size

        if self.mode == "head-batch":
            forbidden = self.true_head.get((r, t), set())
        else:
            forbidden = self.true_tail.get((h, r), set())

        # 主循环：够用就行，别写“聪明但脆弱”的向量化
        while need > 0:
            cand = np.random.randint(self.nentity, size=need * 2, dtype=np.int64)

            if forbidden:
                # forbidden 通常不大，np.in1d 足够
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
            torch.from_numpy(positive_sample).long(),              # [3]
            torch.from_numpy(negative_sample).long(),              # [Neg]
            torch.tensor([subsampling_weight], dtype=torch.float), # [1]
            self.mode,
        )

    @staticmethod
    def collate_fn(data):
        pos = torch.stack([d[0] for d in data], dim=0)   # [B,3]
        neg = torch.stack([d[1] for d in data], dim=0)   # [B,Neg]
        w = torch.cat([d[2] for d in data], dim=0)       # [B]
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

        # 交替 head/tail
        if self.step % 2 == 0:
            return next(self.iterator_head)
        return next(self.iterator_tail)


@torch.no_grad()
def sanity_check_scores(model: RotatEModel, triples: torch.Tensor, device: torch.device, tol: float = 1e-4):
    """
    自检：训练用 score 与 eval/eval_full_entity_filtered.py 用 score_from_head_emb 数学一致。
    不一致就别训了，指标没有意义。
    """
    model.eval()
    triples = triples.to(device)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]

    s_single = model(h, r, t, mode="single")

    h_emb = model.entity_embedding[h]
    s_tail = model.score_from_head_emb(
        h_emb, r, t.unsqueeze(1),
        conj=torch.zeros_like(r, dtype=torch.bool, device=device)
    ).squeeze(1)

    t_emb = model.entity_embedding[t]
    s_head = model.score_from_head_emb(
        t_emb, r, h.unsqueeze(1),
        conj=torch.ones_like(r, dtype=torch.bool, device=device)
    ).squeeze(1)

    e1 = (s_single - s_tail).abs().max().item()
    e2 = (s_single - s_head).abs().max().item()

    print(f"[SanityCheck] max|single - tail_equiv| = {e1:.6g}")
    print(f"[SanityCheck] max|single - head_equiv| = {e2:.6g}")

    if e1 > tol or e2 > tol:
        raise RuntimeError(
            "评分函数不一致：你训练用的 score 和 eval/eval_full_entity_filtered.py 用的 score_from_head_emb 不一致。\n"
            "先修 RotatEModel：保证 forward(single) 与 score_from_head_emb 完全一致。"
        )


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== RotatE Stage3 (official-ish) on {device} ===")
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

    # 独立的generator，避免head/tail shuffle耦合
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
        collate_fn=OfficialTrainDataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g_head,
        persistent_workers=(args.num_workers > 0),
    )
    dl_tail = DataLoader(
        ds_tail,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=OfficialTrainDataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=g_tail,
        persistent_workers=(args.num_workers > 0),
    )

    train_it = BidirectionalOneShotIterator(dl_head, dl_tail, 
                                             disable_head_batch=args.disable_head_batch,
                                             only_head_batch=args.only_head_batch)

    model = RotatEModel(processor.num_entities, processor.num_relations, args.emb_dim, args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 一致性自检：不一致直接死
    with torch.no_grad():
        sample_idx = torch.randint(0, processor.train_triplets.size(0), (32,))
        sanity_triples = processor.train_triplets[sample_idx]
        sanity_check_scores(model, sanity_triples, device, tol=1e-4)

    # eval loader（复用你项目里的 triplet dataset）
    valid_triplets = processor.valid_triplets if args.eval_split == "valid" else processor.test_triplets
    g_eval = torch.Generator()
    g_eval.manual_seed(args.seed + 2)
    eval_loader = DataLoader(
        TrainDataset(valid_triplets),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g_eval,
        persistent_workers=(args.num_workers > 0),
    )
    to_skip = build_to_skip(processor, split=args.eval_split)

    best_avg = -1.0
    os.makedirs(args.save_dir, exist_ok=True)
    write_run_metadata(args.save_dir, args)
    best_path = os.path.join(args.save_dir, "best_model.pth")

    # 根据训练模式自动设置 steps_per_epoch
    if args.only_head_batch:
        # 只跑head-batch（调试用）
        steps_per_epoch = len(dl_head)
    elif args.disable_head_batch:
        # tail-only: 只用 tail-batch 数据
        steps_per_epoch = len(dl_tail)
    else:
        # 双向训练: head/tail 交替，需要 *2 保证每个epoch用完所有数据
        steps_per_epoch = max(len(dl_head), len(dl_tail)) * 2
    
    # 允许手动覆盖
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    
    print(f"[Training] steps_per_epoch={steps_per_epoch}, dl_head={len(dl_head)}, dl_tail={len(dl_tail)}")

    def score_head_batch(r: torch.Tensor, t: torch.Tensor, cand_h: torch.Tensor):
        # score(h', r, t) 等价：score(t, conj(r), h')
        t_emb = model.entity_embedding[t]  # [B,2D]
        return model.score_from_head_emb(
            t_emb, r, cand_h, conj=torch.ones_like(r, dtype=torch.bool, device=device)
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # 诊断：统计head-batch和tail-batch的步数
        mode_counter = {"head-batch": 0, "tail-batch": 0}
        global_step = (epoch - 1) * steps_per_epoch

        for step_in_epoch in range(steps_per_epoch):
            global_step += 1
            pos, neg, subsampling_weight, mode = next(train_it)
            pos = pos.to(device, non_blocking=True)                      # [B,3]
            neg = neg.to(device, non_blocking=True)                      # [B,Neg]
            subsampling_weight = subsampling_weight.to(device, non_blocking=True)  # [B]
            
            # 统计mode
            mode_counter[mode] += 1

            h = pos[:, 0]
            r = pos[:, 1]
            t = pos[:, 2]

            if mode == "tail-batch":
                pos_score = model(h, r, t, mode="single")            # [B]
                neg_score = model(h, r, neg, mode="batch_neg")       # [B,Neg]
            else:
                pos_score = score_head_batch(r, t, h.unsqueeze(1)).squeeze(1)  # [B]
                neg_score = score_head_batch(r, t, neg)                         # [B,Neg]
            
            # 诊断：每100步打印score分布
            if global_step % 100 == 0:
                with torch.no_grad():
                    print(f"[{mode}] step={global_step} "
                          f"pos: {pos_score.mean():.3f}±{pos_score.std():.3f} "
                          f"neg: {neg_score.mean():.3f}±{neg_score.std():.3f}")

            pos_log = F.logsigmoid(pos_score)  # [B]

            if args.negative_adversarial_sampling:
                w = F.softmax(neg_score * args.adversarial_temperature, dim=1).detach()
                neg_log = (w * F.logsigmoid(-neg_score)).sum(dim=1)
                
                # 诊断：打印adversarial权重信息
                if global_step % 100 == 0:
                    with torch.no_grad():
                        w_max = w.max(dim=1)[0].mean()
                        w_entropy = -(w * (w + 1e-10).log()).sum(dim=1).mean()
                        print(f"  w_max={w_max:.4f} entropy={w_entropy:.3f}")
            else:
                neg_log = F.logsigmoid(-neg_score).mean(dim=1)

            loss_vec = -(pos_log + neg_log)  # [B]

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
            print(f"Evaluating on {args.eval_split} (filtered, bidirectional) using eval/eval_full_entity_filtered.py evaluator...")
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
                disable_refiner=True,
                disable_semres=True,
                verbose_every=50,
            )
            rhs = metrics["rhs"]["total"]["MRR"]
            lhs = metrics["lhs"]["total"]["MRR"]
            avg = (rhs + lhs) / 2.0
            print(f"[{args.eval_split.upper()}] RHS: {rhs:.4f} | LHS: {lhs:.4f} | AVG: {avg:.4f}")

            if avg > best_avg:
                best_avg = avg
                torch.save(model.state_dict(), best_path)
                print(f"New Best Saved: {best_path} (AVG MRR={best_avg:.4f})")

    print(f"Done. Best {args.eval_split.upper()} AVG MRR: {best_avg:.4f}")
    print(f"Best checkpoint: {best_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="data/fb15k_custom")
    p.add_argument("--save_dir", type=str, required=True)

    p.add_argument("--emb_dim", type=int, default=500)
    p.add_argument("--margin", type=float, default=9.0)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_neg", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)

    # official knobs
    p.add_argument("--negative_adversarial_sampling", action="store_true")
    p.add_argument("--adversarial_temperature", type=float, default=1.0)
    p.add_argument("--uni_weight", action="store_true")
    p.add_argument("--regularization", type=float, default=0.0)

    # head-batch
    p.add_argument("--disable_head_batch", action="store_true")
    p.add_argument("--only_head_batch", action="store_true", help="Only train head-batch (for debugging)")

    # eval
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_split", type=str, default="valid", choices=["valid", "test"])
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--eval_chunk_size", type=int, default=2048)

    # infra
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="<=0 means auto=len(dl_tail)")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
