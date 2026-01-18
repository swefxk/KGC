#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Full pipeline reproduction + ablation CI
# Usage:
#   bash scripts/run_full_and_ablation.sh fb15k_custom
# =========================================

DATASET_NAME="${1:-fb15k_custom}"

echo "[A] Full pipeline reproduction (mainline)"
bash scripts/run_mainline_full.sh "${DATASET_NAME}"

echo "[B] Ablation + paired bootstrap CI"
bash scripts/run_ablation_ci.sh "${DATASET_NAME}"

echo "Done."
