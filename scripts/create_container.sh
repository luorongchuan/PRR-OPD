#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-songmzhang/kdflow:sgl0.5.12_torch2.11_cu129}
CONTAINER=${CONTAINER:-prr-opd-public}
mounts=(-v "$REPO_ROOT:/workspace/PRR_OPD")
if [[ -n ${HOST_MODEL_DIR:-} ]]; then
  HOST_MODEL_DIR=$(cd -- "$HOST_MODEL_DIR" && pwd)
  mounts+=(-v "$HOST_MODEL_DIR:/workspace/PRR_OPD/models:ro")
fi
if [[ -n ${HOST_DATA_DIR:-} ]]; then
  HOST_DATA_DIR=$(cd -- "$HOST_DATA_DIR" && pwd)
  mounts+=(-v "$HOST_DATA_DIR:/workspace/PRR_OPD/data:ro")
fi
docker run -d --name "$CONTAINER" --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PRR_HOST_UID="$(id -u)" -e PRR_HOST_GID="$(id -g)" \
  "${mounts[@]}" -w /workspace/PRR_OPD --entrypoint /bin/bash "$IMAGE" \
  -lc 'getent group "$PRR_HOST_GID" >/dev/null || groupadd -g "$PRR_HOST_GID" prruser
       getent passwd "$PRR_HOST_UID" >/dev/null || useradd -m -u "$PRR_HOST_UID" -g "$PRR_HOST_GID" -s /bin/bash prruser
       exec sleep infinity'
