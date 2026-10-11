#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-$(command -v python)}
DATA_ROOT=${DATA_ROOT:-${PRR_DATA_ROOT:?Set PRR_DATA_ROOT to the evaluation data directory}}
RESULT_ROOT=${1:?usage: sandbox_score.sh RESULT_ROOT [WORKERS]}
WORKERS=${2:-8}

if [[ ${PRR_CODE_SANDBOX_INNER:-0} != 1 ]]; then
  RESULT_ROOT=$(realpath "$RESULT_ROOT")
  exec unshare --user --map-root-user --net --mount --pid --fork --mount-proc \
    env PRR_CODE_SANDBOX_INNER=1 HOST_HERE="$HERE" HOST_PYTHON="$PYTHON" \
    HOST_DATA_ROOT="$DATA_ROOT" HOST_RESULT_ROOT="$RESULT_ROOT" WORKERS="$WORKERS" \
    PRR_CODE_SANDBOX_SMOKE="${PRR_CODE_SANDBOX_SMOKE:-0}" \
    bash "$0" "$RESULT_ROOT" "$WORKERS"
fi

mount --make-rprivate /
JAIL=$(mktemp -d)
cleanup() {
  set +e
  umount -R "$JAIL" 2>/dev/null
  rmdir "$JAIL" 2>/dev/null
}
trap cleanup EXIT

mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL"
mkdir -p "$JAIL"/{usr,bin,lib,lib64,etc,proc,tmp,dev,work,datasets,out}

for path in usr bin lib lib64 etc; do
  mount --rbind "/$path" "$JAIL/$path"
  mount -o remount,ro,bind "$JAIL/$path"
done

PY_ENV=$(dirname "$(dirname "$HOST_PYTHON")")
mkdir -p "$JAIL$PY_ENV"
mount --rbind "$PY_ENV" "$JAIL$PY_ENV"
mount -o remount,ro,bind "$JAIL$PY_ENV"

mount --bind "$HOST_HERE" "$JAIL/work"
mount -o remount,ro,bind "$JAIL/work"
mount --bind "$HOST_DATA_ROOT" "$JAIL/datasets"
mount -o remount,ro,bind "$JAIL/datasets"
mount --bind "$HOST_RESULT_ROOT" "$JAIL/out"

mount -t proc proc "$JAIL/proc"
mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL/tmp"
mount -t tmpfs -o size=1G,nosuid,noexec tmpfs "$JAIL/dev"
for node in null zero random urandom; do
  touch "$JAIL/dev/$node"
  mount --bind "/dev/$node" "$JAIL/dev/$node"
done
mkdir -p "$JAIL/dev/shm"
mount -t tmpfs -o size=2G,nosuid,nodev,noexec tmpfs "$JAIL/dev/shm"

export HOME=/tmp/home
export TMPDIR=/tmp
export XDG_CACHE_HOME=/tmp/cache
export MPLCONFIGDIR=/tmp/matplotlib
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH=/work
export PRR_CODE_DATA_ROOT=/datasets
mkdir -p "$JAIL/tmp/home" "$JAIL/tmp/cache" "$JAIL/tmp/matplotlib"

if [[ ${PRR_CODE_SANDBOX_SMOKE:-0} == basic ]]; then
  exec chroot "$JAIL" "$HOST_PYTHON" /work/sandbox_smoke.py
fi
if [[ ${PRR_CODE_SANDBOX_SMOKE:-0} == judges ]]; then
  exec chroot "$JAIL" "$HOST_PYTHON" /work/judge_smoke.py
fi

exec chroot "$JAIL" "$HOST_PYTHON" /work/score.py \
  --result-root /out --data-root /datasets --workers "$WORKERS"
