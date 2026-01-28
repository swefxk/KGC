#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/check_eval_equivalence.sh <run_id> [dataset_name] [topk] [struct_type] [emb_dim]
# Example:
#   bash scripts/check_eval_equivalence.sh fb15k_custom_main_seed42 fb15k_custom 200 rotate 1000

RUN_ID="${1:?Usage: bash scripts/check_eval_equivalence.sh <run_id> [dataset_name] [topk] [struct_type] [emb_dim]}"
DATASET="${2:-fb15k_custom}"
TOPK="${3:-200}"
STRUCT_TYPE="${4:-rotate}"

ROOT="artifacts/${RUN_ID}"
DATA="data/${DATASET}"
if [ "${STRUCT_TYPE}" = "complex" ]; then
  STRUCT_CKPT="${ROOT}/checkpoints/complex/best_model.pth"
  EMB_DIM="${5:-500}"
else
  STRUCT_TYPE="rotate"
  STRUCT_CKPT="${ROOT}/checkpoints/rotate/best_model.pth"
  EMB_DIM="${5:-1000}"
fi

if [ ! -f "${STRUCT_CKPT}" ]; then
  echo "[Error] Missing struct ckpt: ${STRUCT_CKPT}"
  exit 1
fi

BATCH=8
SCORE_EPS=0

run_mode() {
  local name="$1"
  shift
  local extra_flags=("$@")

  local out_full="${ROOT}/eval/equiv_${name}_full"
  local out_topk="${ROOT}/eval/equiv_${name}_topk"

  echo "[Run] mode=${name} (full-entity)"
  python eval/eval_full_entity_filtered.py \
    --data_path "${DATA}" \
    --struct_type "${STRUCT_TYPE}" \
    --pretrained_struct "${STRUCT_CKPT}" \
    --emb_dim "${EMB_DIM}" \
    --recall_k "${TOPK}" \
    --eval_split test \
    --out_dir "${out_full}" \
    --save_ranks_path "${out_full}/ranks.pt" \
    --test_batch_size "${BATCH}" \
    --score_eps "${SCORE_EPS}" \
    "${extra_flags[@]}"

  echo "[Run] mode=${name} (topK strict)"
  python eval/eval_topk_inject.py \
    --data_path "${DATA}" \
    --struct_type "${STRUCT_TYPE}" \
    --pretrained_struct "${STRUCT_CKPT}" \
    --emb_dim "${EMB_DIM}" \
    --strict_r0 \
    --topk "${TOPK}" --b_rhs 0.0 --b_lhs 0.0 \
    --eval_split test --eval_sides both \
    --refiner_gamma_rhs 0.0 --refiner_gamma_lhs 0.0 \
    --out_dir "${out_topk}" \
    --save_ranks_path "${out_topk}/ranks.pt" \
    --batch_size "${BATCH}" \
    --score_eps "${SCORE_EPS}" \
    "${extra_flags[@]}"

  echo "[Check] mode=${name} ranks + AVG MRR"
  python - <<PY
import json
import sys
import torch

full = torch.load("${out_full}/ranks.pt", map_location="cpu")
topk = torch.load("${out_topk}/ranks.pt", map_location="cpu")
for key in ("rhs_ranks", "lhs_ranks"):
    diff = (full[key] != topk[key]).sum().item()
    print(f"{key} diff {diff}")
    if diff:
        sys.exit(1)

with open("${out_full}/metrics.json", "r", encoding="utf-8") as f:
    m_full = json.load(f)
with open("${out_topk}/metrics.json", "r", encoding="utf-8") as f:
    m_topk = json.load(f)
avg_full = m_full["avg"]["MRR"]
avg_topk = m_topk["avg"]["MRR"]
print(f"avg_full {avg_full} avg_topk {avg_topk}")
if abs(avg_full - avg_topk) > 1e-12:
    sys.exit(1)
print("[OK] equivalence PASS0 strict passed")
PY
}

# Default mode
run_mode "default"

# Repro mode
run_mode "repro" --deterministic --disable_tf32 --matmul_precision highest --seed 42

echo "[PASS] default + repro modes are equivalent"
