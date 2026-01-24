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

CKPT_ROOT="${ROOT}/checkpoints"
if [ ! -f "${CKPT_ROOT}/rotate/best_model.pth" ]; then
  echo "[Error] Missing checkpoints under ${CKPT_ROOT}. Run mainline first."
  exit 1
fi

ROTATE="${CKPT_ROOT}/rotate/best_model.pth"
SEM="${CKPT_ROOT}/sem_biencoder/biencoder_best.pth"
SEM_INDEP="${CKPT_ROOT}/sem_biencoder_indep/biencoder_best.pth"
GATE="${CKPT_ROOT}/gate/gate_best.pth"
REFINER="${CKPT_ROOT}/refiner/refiner_best.pth"

mkdir -p "${ROOT}/eval"

echo "[Ablation] R0 full-entity RotatE"
python eval/eval_full_entity_filtered.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --recall_k "${TOPK}" \
  --eval_split test \
  --out_dir "${ROOT}/eval/R0_rotate_full" \
  --save_ranks_path "${ROOT}/eval/R0_rotate_full/ranks.pt"

echo "[Ablation] R0-topK (topK-only, b=0, gamma=0)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --strict_r0 \
  --topk "${TOPK}" --b_rhs 0.0 --b_lhs 0.0 \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/R0_rotate_topk" \
  --save_ranks_path "${ROOT}/eval/R0_rotate_topk/ranks.pt"

echo "[Check] R0 ranks consistency (full vs topK strict)"
python - <<PY
import sys
import torch

full = torch.load("${ROOT}/eval/R0_rotate_full/ranks.pt", map_location="cpu")
topk = torch.load("${ROOT}/eval/R0_rotate_topk/ranks.pt", map_location="cpu")

for key in ("rhs_ranks", "lhs_ranks"):
    diff = (full[key] != topk[key]).sum().item()
    if diff:
        print(f"[Error] R0 mismatch in {key}: {diff} diffs")
        sys.exit(1)
print("[OK] R0 full-entity and topK strict ranks match")
PY

echo "[Ablation] Sem (TopK rerank; candidates from RotatE, struct_weight=0)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --struct_weight_rhs 0.0 --struct_weight_lhs 0.0 \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides both \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/Sem_rerank_topk"

if [ -f "${SEM_INDEP}" ]; then
  echo "[Ablation] Sem-only Full-Entity (independent biencoder)"
  python eval/eval_sem_biencoder_full.py \
    --data_path "${DATA}" \
    --pretrained_sem "${SEM_INDEP}" \
    --topk "${TOPK}" \
    --eval_split test --eval_sides both \
    --out_dir "${ROOT}/eval/Sem_only_full"
else
  echo "[Ablation] Sem-only Full-Entity skipped (missing ${SEM_INDEP})"
fi

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

echo "[Ablation] Hybrid LHS (RotatE union Sem, unionK=400)"
python eval/eval_topk_inject.py \
  --data_path "${DATA}" \
  --pretrained_rotate "${ROTATE}" \
  --pretrained_sem "${SEM}" \
  --topk "${TOPK}" --b_rhs "${B_RHS}" --b_lhs "${B_LHS}" \
  --eval_split test --eval_sides lhs \
  --lhs_union_sem_topk "${TOPK}" \
  --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
  --out_dir "${ROOT}/eval/Hybrid_union_lhs"

echo "[Ablation] Safety: entropy gating q=0.6"
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

echo "[Ablation] Paired bootstrap D-C (Delta MRR)"
mkdir -p "${ROOT}/eval/BOOT_D_minus_C"
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
