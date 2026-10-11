#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PRR_EVAL_PYTHON:-python}" -B "$HERE/run.py" "$@"
