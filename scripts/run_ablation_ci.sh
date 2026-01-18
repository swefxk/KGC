#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_ablation_ci.sh fb15k_custom_r2_20260117_seed42

RUN="${1:-fb15k_custom_r2_20260117_seed42}"
ROOT="artifacts/${RUN}"
DATA="data/fb15k_custom"

TOPK=200
B_RHS=2.0
B_LHS=2.5
GAMMA_RHS=2.0
GAMMA_LHS=0.0
BOOT_SAMPLES=2000
BOOT_SEED=42
BOOT_CI=0.95

ROTATE="${ROOT}/checkpoints/rotate/best_model.pth"
SEM="${ROOT}/checkpoints/sem_biencoder/biencoder_best.pth"
GATE="${ROOT}/checkpoints/gate/gate_best.pth"
REFINER="${ROOT}/checkpoints/refiner/refiner_best.pth"

mkdir -p "${ROOT}/eval"

echo "[Ablation] R0 full-entity RotatE"
python eval/eval_full_entity_filtered.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --recall_k "${TOPK}" \
  --eval_split test --eval_sides both \
  --out_dir "${ROOT}/eval/R0_rotate_full"

echo "[Ablation] R0-topK (topK-only, b=0, gamma=0)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --topk "${TOPK}" --b_rhs 0.0 --b_lhs 0.0 \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/R0_rotate_topk"

echo "[Ablation] Sem-only (no gate, no delta)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/Sem_only"

echo "[Ablation] C: Sem+Gate (no delta)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --pretrained_gate "${GATE}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/C_sem_gate" \
  --save_ranks_path "${ROOT}/eval/C_sem_gate/ranks.pt"

echo "[Ablation] D: Sem+Gate+Delta"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --pretrained_gate "${GATE}" \
  --pretrained_refiner "${REFINER}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --out_dir "${ROOT}/eval/D_sem_gate_delta" \
  --save_ranks_path "${ROOT}/eval/D_sem_gate_delta/ranks.pt"

echo "[Ablation] Δ-only (b=0, no gate)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_refiner "${REFINER}" \
  --topk "${TOPK}" --b_rhs 0.0 --b_lhs 0.0 \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --out_dir "${ROOT}/eval/Delta_only"

echo "[Ablation] Safety: entropy gating (q=0.6)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --pretrained_gate "${GATE}" \
  --pretrained_refiner "${REFINER}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --delta_gate_ent_q 0.6 \
  --out_dir "${ROOT}/eval/Safety_entropy_q06"

echo "[Ablation] Paired bootstrap D−C (ΔMRR)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --pretrained_gate "${GATE}" \
  --pretrained_refiner "${REFINER}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs "${GAMMA_RHS}" --refiner_gamma_lhs "${GAMMA_LHS}" \
  --out_dir "${ROOT}/eval/BOOT_D_minus_C" \
  --bootstrap_samples "${BOOT_SAMPLES}" --bootstrap_seed "${BOOT_SEED}" --bootstrap_ci "${BOOT_CI}" \
  --paired_bootstrap \
  --paired_baseline_ranks "${ROOT}/eval/C_sem_gate/ranks.pt" \
  | tee "${ROOT}/eval/BOOT_D_minus_C/paired_bootstrap.txt"

echo "[Ablation] Done: ${ROOT}/eval"
