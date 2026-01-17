#!/usr/bin/env bash
set -euo pipefail

# Safety mainline: identical to run_mainline_full.sh, but enables entropy gating for Delta at eval.
# Usage:
#   bash scripts/run_mainline_safety.sh fb15k_custom

DATASET_NAME="${1:-fb15k_custom}"
DATA_PATH="data/${DATASET_NAME}"

CKPT_ROOT="checkpoints/mainline_${DATASET_NAME}"
ROTATE_DIR="${CKPT_ROOT}/rotate"
SEM_DIR="${CKPT_ROOT}/sem_biencoder"
GATE_DIR="${CKPT_ROOT}/gate"
REFINER_DIR="${CKPT_ROOT}/refiner"

ROTATE_CKPT="${ROTATE_DIR}/best_model.pth"
SEM_CKPT="${SEM_DIR}/biencoder_best.pth"
GATE_CKPT="${GATE_DIR}/gate_best.pth"
REFINER_CKPT="${REFINER_DIR}/refiner_best.pth"

# Your current best defaults (edit if needed)
TOPK=200
B_RHS=2.0
B_LHS=2.5
REFINER_GAMMA_RHS=2.0
REFINER_GAMMA_LHS=0.0

# Safety gating
DELTA_GATE_ENT_Q=0.6

echo "[SAFETY] Final Evaluation with entropy gating (q=${DELTA_GATE_ENT_Q})"

python eval/eval_topk_inject.py \
  --data_path "${DATA_PATH}" \
  --eval_split test \
  --eval_sides both \
  --pretrained_rotate "${ROTATE_CKPT}" \
  --pretrained_sem "${SEM_CKPT}" \
  --pretrained_gate "${GATE_CKPT}" \
  --pretrained_refiner "${REFINER_CKPT}" \
  --topk "${TOPK}" \
  --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --refiner_gamma_rhs "${REFINER_GAMMA_RHS}" --refiner_gamma_lhs "${REFINER_GAMMA_LHS}" \
  --delta_gate_ent_q "${DELTA_GATE_ENT_Q}" \
  --bootstrap_samples 2000
