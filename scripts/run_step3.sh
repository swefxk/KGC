#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Continue from Step 3 (after caches built)
# Usage:
#   bash scripts/run_step3.sh fb15k_custom
# =========================================

DATASET_NAME="${1:-fb15k_custom}"
DATA_PATH="data/${DATASET_NAME}"
CKPT_ROOT="checkpoints/mainline_${DATASET_NAME}"

ROTATE_CKPT="${CKPT_ROOT}/rotate/best_model.pth"
SEM_DIR="${CKPT_ROOT}/sem_biencoder"
REFINER_DIR="${CKPT_ROOT}/refiner"
GATE_DIR="${CKPT_ROOT}/gate"

RHS_CACHE="${DATA_PATH}/cache/train_rhs_top500_neg_filtered_trainvalid.pt"
LHS_CACHE="${DATA_PATH}/cache/train_lhs_top500_neg_filtered_trainvalid.pt"

TOPK=200
B_RHS=2.0
B_LHS=2.5
GAMMA_RHS=2.0
GAMMA_LHS=0.0

mkdir -p "${SEM_DIR}" "${REFINER_DIR}" "${GATE_DIR}"

echo "[3/6] Train Semantic Bi-Encoder (SimKGC-style, bidirectional)"
/home/dell/anaconda3/envs/Metric-Distill/bin/python train/train_sem_biencoder.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --emb_dim 1000 \
  --train_cache_rhs "${RHS_CACHE}" \
  --train_cache_lhs "${LHS_CACHE}" \
  --save_dir "${SEM_DIR}" \
  --eval_metric avg \
  --epochs 10

echo "[4/6] Train Struct Refiner"
/home/dell/anaconda3/envs/Metric-Distill/bin/python train/train_struct_refiner.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --emb_dim 1000 \
  --train_cache_rhs "${RHS_CACHE}" \
  --train_cache_lhs "${LHS_CACHE}" \
  --save_dir "${REFINER_DIR}" \
  --epochs 5

echo "[5/6] Train Gate (topK-only, aligned with eval)"
/home/dell/anaconda3/envs/Metric-Distill/bin/python train/train_gate_inject_topk.py \
  --data_path "${DATA_PATH}" \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --pretrained_sem "${SEM_DIR}/biencoder_best.pth" \
  --emb_dim 1000 \
  --train_cache_rhs "${RHS_CACHE}" \
  --train_cache_lhs "${LHS_CACHE}" \
  --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --save_dir "${GATE_DIR}" \
  --epochs 5

echo "[6/6] Final Evaluation (TopK-only injection + Gate + Delta)"
/home/dell/anaconda3/envs/Metric-Distill/bin/python eval/eval_topk_inject.py \
  --data_path "${DATA_PATH}" \
  --eval_split test \
  --eval_sides both \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --pretrained_sem "${SEM_DIR}/biencoder_best.pth" \
  --pretrained_refiner "${REFINER_DIR}/refiner_best.pth" \
  --pretrained_gate "${GATE_DIR}/gate_best.pth" \
  --topk "${TOPK}" \
  --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --bootstrap_samples 2000

echo "Done."
