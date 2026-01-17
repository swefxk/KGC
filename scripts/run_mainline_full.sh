#!/usr/bin/env bash
set -euo pipefail

# =========================
# Mainline Repro Script
# =========================
# Usage:
#   bash scripts/run_mainline_full.sh fb15k_custom
#
# Run from repo root. This script does NOT download datasets.

DATASET_NAME="${1:-fb15k_custom}"
DATA_PATH="data/${DATASET_NAME}"

# ---- checkpoint dirs ----
CKPT_ROOT="checkpoints/mainline_${DATASET_NAME}"
ROTATE_DIR="${CKPT_ROOT}/rotate"
SEM_DIR="${CKPT_ROOT}/sem_biencoder"
GATE_DIR="${CKPT_ROOT}/gate"
REFINER_DIR="${CKPT_ROOT}/refiner"

mkdir -p "${ROTATE_DIR}" "${SEM_DIR}" "${GATE_DIR}" "${REFINER_DIR}"

# ---- caches ----
CACHE_DIR="${DATA_PATH}/cache"
RHS_CACHE_TV="${CACHE_DIR}/train_rhs_top500_neg_filtered_trainvalid.pt"
LHS_CACHE_TV="${CACHE_DIR}/train_lhs_top500_neg_filtered_trainvalid.pt"

# ---- model paths ----
ROTATE_CKPT="${ROTATE_DIR}/best_model.pth"
SEM_CKPT="${SEM_DIR}/biencoder_best.pth"
GATE_CKPT="${GATE_DIR}/gate_best.pth"
REFINER_CKPT="${REFINER_DIR}/refiner_best.pth"

# ---- final eval hyperparams ----
TOPK=200
B_RHS=2.0
B_LHS=2.5
GAMMA_RHS=2.0
GAMMA_LHS=0.0

echo "[1/6] Train RotatE"
python train/train_rotate.py \
  --data_path "${DATA_PATH}" \
  --save_dir "${ROTATE_DIR}"

echo "[2/6] Build TopK hard-negative caches (train+valid filtered)"
python build/build_rotate_cache_rhs_topk.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --K 500 \
  --use_filtered \
  --filtered_splits train_valid \
  --out_path "${RHS_CACHE_TV}"

python build/build_rotate_cache_lhs_topk.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --K 500 \
  --use_filtered_trainvalid \
  --out_path "${LHS_CACHE_TV}"

echo "[3/6] Train Semantic Bi-Encoder (SimKGC-style, bidirectional)"
python train/train_sem_biencoder.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --train_cache_rhs "${RHS_CACHE_TV}" \
  --train_cache_lhs "${LHS_CACHE_TV}" \
  --save_dir "${SEM_DIR}" \
  --epochs 10

echo "[4/6] Train Struct Refiner"
python train/train_struct_refiner.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --train_cache_rhs "${RHS_CACHE_TV}" \
  --train_cache_lhs "${LHS_CACHE_TV}" \
  --save_dir "${REFINER_DIR}" \
  --epochs 5

echo "[5/6] Train Gate (topK-only, aligned with eval)"
python train/train_gate_inject_topk.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --pretrained_sem "${SEM_CKPT}" \
  --train_cache_rhs "${RHS_CACHE_TV}" \
  --train_cache_lhs "${LHS_CACHE_TV}" \
  --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --save_dir "${GATE_DIR}" \
  --epochs 5

echo "[6/6] Final Evaluation (TopK-only injection + Gate + Delta)"
python eval/eval_topk_inject.py \
  --data_path "${DATA_PATH}" \
  --eval_split test \
  --eval_sides both \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --pretrained_sem "${SEM_CKPT}" \
  --pretrained_refiner "${REFINER_CKPT}" \
  --pretrained_gate "${GATE_CKPT}" \
  --topk "${TOPK}" \
  --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --bootstrap_samples 2000

echo "Done. Artifacts in: ${CKPT_ROOT}"
