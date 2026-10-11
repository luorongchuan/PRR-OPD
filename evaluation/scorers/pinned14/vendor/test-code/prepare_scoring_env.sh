#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$(command -v python)}
# Dependencies are installed explicitly into the evaluation environment at setup.
PYTHONPATH="$HERE/vendor/livecodebench${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -B -c 'import numpy,pyarrow,sympy; from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness; print("LOCAL_CODE_DEPENDENCIES_OK")'
