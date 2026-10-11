#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GPU_IDS=${GPU_IDS:-0,1}
METHODS=${METHODS:-opd,prr}
SIZES=${SIZES:-0.6b,1.7b}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
IFS=, read -ra SIZE_LIST <<< "$SIZES"
IFS=, read -ra METHOD_LIST <<< "$METHODS"
for size in "${SIZE_LIST[@]}"; do
  for method in "${METHOD_LIST[@]}"; do
    bash "$REPO_ROOT/scripts/train.sh" --size "$size" --method "$method" \
      --gpus "$GPU_IDS" --output "$REPO_ROOT/outputs/$RUN_NAME/${size}_$method" "$@"
  done
done
