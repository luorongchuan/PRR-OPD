#!/usr/bin/env bash
set -euo pipefail
exec docker exec -it -u "$(id -u):$(id -g)" -w /workspace/PRR_OPD \
  "${CONTAINER:-prr-opd-public}" bash
