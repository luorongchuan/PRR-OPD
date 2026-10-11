#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:?MODE PROBLEMS PREDICTIONS CACHE OUTPUT PYTHON}
PROBLEMS=$(realpath "${2:?}")
PREDICTIONS=$(realpath "${3:?}")
CACHE=$(realpath -m "${4:?}")
OUTPUT=$(realpath -m "${5:?}")
PYTHON=$(realpath "${6:?}")
VENDOR=$(realpath "$HERE/../bigcodebench")
mkdir -p "$CACHE" "$OUTPUT"
if [[ ${PRR_BIGCODE_BACKEND:-host} == docker ]]; then
  exec "$PYTHON" -B "$HERE/docker_sandbox.py" "$MODE" "$PROBLEMS" "$PREDICTIONS" "$CACHE" "$OUTPUT" "$PYTHON"
fi
if [[ ${PRR_BIGCODE_SANDBOX_INNER:-0} != 1 ]]; then
  exec unshare --user --map-root-user --net --mount --pid --fork --mount-proc \
    env PRR_BIGCODE_SANDBOX_INNER=1 HOST_HERE="$HERE" HOST_PYTHON="$PYTHON" \
    HOST_PROBLEMS="$PROBLEMS" HOST_PREDICTIONS="$PREDICTIONS" HOST_VENDOR="$VENDOR" \
    HOST_CACHE="$CACHE" HOST_OUTPUT="$OUTPUT" \
    bash "$0" "$MODE" "$PROBLEMS" "$PREDICTIONS" "$CACHE" "$OUTPUT" "$PYTHON"
fi
mount --make-rprivate /
JAIL=$(mktemp -d)
trap 'umount -R "$JAIL" 2>/dev/null || true; rmdir "$JAIL" 2>/dev/null || true' EXIT
mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL"
mkdir -p "$JAIL"/{usr,bin,lib,lib64,etc,proc,tmp,dev,work,vendor,controls,data,input,out}
for path in usr bin lib lib64 etc; do
  mount --rbind "/$path" "$JAIL/$path"
  mount -o remount,ro,bind "$JAIL/$path"
done
PY_ENV=$(dirname "$(dirname "$HOST_PYTHON")")
mkdir -p "$JAIL$PY_ENV"
mount --rbind "$PY_ENV" "$JAIL$PY_ENV"
mount -o remount,ro,bind "$JAIL$PY_ENV"
if [[ -n ${PRR_BIGCODE_EXTRA_SITE:-} ]]; then
  mkdir -p "$JAIL/extra"
  mount --bind "$PRR_BIGCODE_EXTRA_SITE" "$JAIL/extra"
  mount -o remount,ro,bind "$JAIL/extra"
fi
mount --bind "$HOST_HERE" "$JAIL/work"
mount -o remount,ro,bind "$JAIL/work"
mkdir -p "$JAIL/vendor/bigcodebench"
mount --bind "$HOST_VENDOR" "$JAIL/vendor/bigcodebench"
mount -o remount,ro,bind "$JAIL/vendor/bigcodebench"
touch "$JAIL/data/problems.parquet" "$JAIL/input/predictions.jsonl"
mount --bind "$HOST_PROBLEMS" "$JAIL/data/problems.parquet"
mount -o remount,ro,bind "$JAIL/data/problems.parquet"
mount --bind "$HOST_PREDICTIONS" "$JAIL/input/predictions.jsonl"
mount -o remount,ro,bind "$JAIL/input/predictions.jsonl"
mount --bind "$HOST_CACHE" "$JAIL/controls"
if [[ "$MODE" == score ]]; then
  mount -o remount,ro,bind "$JAIL/controls"
fi
mount --bind "$HOST_OUTPUT" "$JAIL/out"
mount -t proc proc "$JAIL/proc"
mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL/tmp"
mount -t tmpfs -o size=1G,nosuid,noexec tmpfs "$JAIL/dev"
for node in null zero random urandom; do
  touch "$JAIL/dev/$node"
  chmod 666 "$JAIL/dev/$node"
done
mkdir -p "$JAIL/dev/shm"
mount -t tmpfs -o size=2G,nosuid,nodev,noexec tmpfs "$JAIL/dev/shm"
export HOME=/tmp/home TMPDIR=/tmp XDG_CONFIG_HOME=/tmp/config
export XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=/vendor/bigcodebench:/work PYTHONDONTWRITEBYTECODE=1
if [[ -n ${PRR_BIGCODE_EXTRA_SITE:-} ]]; then
  export PYTHONPATH="/extra:$PYTHONPATH"
  export NLTK_DATA=/extra/nltk_data
fi
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset BIGCODEBENCH_TIMEOUT_PER_TASK
mkdir -p "$JAIL/tmp/home" "$JAIL/tmp/config" "$JAIL/tmp/cache" "$JAIL/tmp/matplotlib"
exec chroot "$JAIL" "$HOST_PYTHON" -B /work/score_local.py \
  --mode "$MODE" --problems /data/problems.parquet --predictions /input/predictions.jsonl \
  --cache /controls --output /out --vendor /vendor/bigcodebench
