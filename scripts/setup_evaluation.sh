#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
EVAL_ENV=${EVAL_ENV:-$REPO_ROOT/runtime/eval}
EVAL_BASE_PYTHON=${EVAL_BASE_PYTHON:-python3.12}
"$EVAL_BASE_PYTHON" -m venv "$EVAL_ENV"
"$EVAL_ENV/bin/python" -m pip install -r "$REPO_ROOT/requirements-eval.txt"
"$EVAL_ENV/bin/python" -B "$REPO_ROOT/scripts/check_environment.py" --mode evaluation
printf 'Set PRR_EVAL_PYTHON=%s\n' "$EVAL_ENV/bin/python"
