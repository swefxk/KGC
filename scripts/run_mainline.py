#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.getcwd())

from tools.run_meta import write_run_metadata


def run(cmd, env=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="fb15k_custom")
    ap.add_argument("--run_root", type=str, default="runs")
    ap.add_argument("--epochs_sem", type=int, default=10)
    ap.add_argument("--epochs_refiner", type=int, default=5)
    ap.add_argument("--epochs_gate", type=int, default=5)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--b_rhs", type=float, default=2.0)
    ap.add_argument("--b_lhs", type=float, default=2.5)
    ap.add_argument("--gamma_rhs", type=float, default=2.0)
    ap.add_argument("--gamma_lhs", type=float, default=0.0)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.run_root, ts)
    os.makedirs(run_dir, exist_ok=True)
    write_run_metadata(run_dir, args)

    data_path = os.path.join("data", args.dataset)
    ckpt_root = os.path.join("checkpoints", f"mainline_{args.dataset}")
    rotate_dir = os.path.join(ckpt_root, "rotate")
    sem_dir = os.path.join(ckpt_root, "sem_biencoder")
    gate_dir = os.path.join(ckpt_root, "gate")
    refiner_dir = os.path.join(ckpt_root, "refiner")
    os.makedirs(rotate_dir, exist_ok=True)
    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(gate_dir, exist_ok=True)
    os.makedirs(refiner_dir, exist_ok=True)

    cache_dir = os.path.join(data_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    rhs_cache = os.path.join(cache_dir, "train_rhs_top500_neg_filtered_trainvalid.pt")
    lhs_cache = os.path.join(cache_dir, "train_lhs_top500_neg_filtered_trainvalid.pt")

    rotate_ckpt = os.path.join(rotate_dir, "best_model.pth")
    sem_ckpt = os.path.join(sem_dir, "biencoder_best.pth")
    gate_ckpt = os.path.join(gate_dir, "gate_best.pth")
    refiner_ckpt = os.path.join(refiner_dir, "refiner_best.pth")

    run([sys.executable, "train/train_rotate.py",
         "--data_path", data_path,
         "--save_dir", rotate_dir])

    run([sys.executable, "build/build_rotate_cache_rhs_topk.py",
         "--data_path", data_path,
         "--pretrained_rotate", rotate_ckpt,
         "--K", "500",
         "--use_filtered",
         "--filtered_splits", "train_valid",
         "--out_path", rhs_cache])

    run([sys.executable, "build/build_rotate_cache_lhs_topk.py",
         "--data_path", data_path,
         "--pretrained_rotate", rotate_ckpt,
         "--K", "500",
         "--use_filtered_trainvalid",
         "--out_path", lhs_cache])

    run([sys.executable, "train/train_sem_biencoder.py",
         "--data_path", data_path,
         "--pretrained_rotate", rotate_ckpt,
         "--train_cache_rhs", rhs_cache,
         "--train_cache_lhs", lhs_cache,
         "--save_dir", sem_dir,
         "--epochs", str(args.epochs_sem)])

    run([sys.executable, "train/train_struct_refiner.py",
         "--data_path", data_path,
         "--pretrained_rotate", rotate_ckpt,
         "--train_cache_rhs", rhs_cache,
         "--train_cache_lhs", lhs_cache,
         "--save_dir", refiner_dir,
         "--epochs", str(args.epochs_refiner)])

    run([sys.executable, "train/train_gate_inject_topk.py",
         "--data_path", data_path,
         "--pretrained_rotate", rotate_ckpt,
         "--pretrained_sem", sem_ckpt,
         "--train_cache_rhs", rhs_cache,
         "--train_cache_lhs", lhs_cache,
         "--b_rhs", str(args.b_rhs),
         "--b_lhs", str(args.b_lhs),
         "--save_dir", gate_dir,
         "--epochs", str(args.epochs_gate)])

    run([sys.executable, "eval/eval_topk_inject.py",
         "--data_path", data_path,
         "--eval_split", "test",
         "--eval_sides", "both",
         "--pretrained_rotate", rotate_ckpt,
         "--pretrained_sem", sem_ckpt,
         "--pretrained_refiner", refiner_ckpt,
         "--pretrained_gate", gate_ckpt,
         "--topk", str(args.topk),
         "--b_rhs", str(args.b_rhs),
         "--b_lhs", str(args.b_lhs),
         "--refiner_gamma_rhs", str(args.gamma_rhs),
         "--refiner_gamma_lhs", str(args.gamma_lhs),
         "--bootstrap_samples", "2000",
         "--out_dir", os.path.join(run_dir, "eval")])

    print(f"[DONE] run_dir={run_dir}")


if __name__ == "__main__":
    main()
