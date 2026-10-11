#!/usr/bin/env bash
set -euo pipefail
GPU_IDS=${GPU_IDS:-0,1}
exec docker exec -u "$(id -u):$(id -g)" -w /workspace/PRR_OPD \
  -e CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CONTAINER:-prr-opd-public}" \
  bash /workspace/PRR_OPD/scripts/train.sh --gpus "$GPU_IDS" "$@"
