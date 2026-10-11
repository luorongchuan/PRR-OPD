#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BIGCODE_BASE_PYTHON=${BIGCODE_BASE_PYTHON:-python3.10}
ENV_DIR="$REPO_ROOT/evaluation/runtime/bigcode_py310"
"$BIGCODE_BASE_PYTHON" -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/evaluation/repairs/scoring_v4/requirements-bigcode-py310.txt"
"$ENV_DIR/bin/python" -B "$REPO_ROOT/evaluation/setup_bigcode_resources.py"
"$ENV_DIR/bin/python" -B "$REPO_ROOT/scripts/download_scoring_resources.py"
"$ENV_DIR/bin/python" -B -c 'import numpy,pandas,tensorflow; print("BIGCODE_IMPORTS_OK",numpy.__version__,pandas.__version__,tensorflow.__version__)'
printf 'Installation complete. The scorer still requires all 148 canonical controls to pass.\n'
