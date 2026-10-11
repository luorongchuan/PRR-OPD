#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PRR_TRAIN_PYTHON:-python}" -B "$REPO_ROOT/scripts/train.py" "$@"
