#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVALUATOR=$(realpath "$HERE/../..")
KIND=${1:?KIND MODE DATASET PREDICTIONS DATA OUTPUT CACHE PYTHON}
MODE=${2:?}
DATASET=${3:?}
PREDICTIONS=$(realpath "${4:?}")
DATA=$(realpath "${5:?}")
OUTPUT=$(realpath -m "${6:?}")
CACHE=$(realpath -m "${7:?}")
PYTHON=$(realpath -s "${8:?}")
shift 8
mkdir -p "$OUTPUT" "$CACHE"
if [[ ${PRR_V4_SANDBOX_INNER:-0} != 1 ]]; then
  exec unshare --user --map-root-user --net --mount --pid --fork --mount-proc \
    env PRR_V4_SANDBOX_INNER=1 HOST_REPAIR="$HERE" HOST_ARCHIVE="$EVALUATOR/scorers" \
      HOST_PYTHON="$PYTHON" HOST_PREDICTIONS="$PREDICTIONS" HOST_DATA="$DATA" \
      HOST_OUTPUT="$OUTPUT" HOST_CACHE="$CACHE" HOST_NLTK_DATA="$EVALUATOR/runtime/bigcode_packages/nltk_data" \
      HOST_DEVICES=/dev \
      bash "$0" "$KIND" "$MODE" "$DATASET" "$PREDICTIONS" "$DATA" "$OUTPUT" "$CACHE" "$PYTHON" "$@"
fi
mount --make-rprivate /
JAIL=$(mktemp -d)
trap 'umount -R "$JAIL" 2>/dev/null || true; rmdir "$JAIL" 2>/dev/null || true' EXIT
mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL"
mkdir -p "$JAIL"/{usr,bin,lib,lib64,etc,proc,tmp,dev,archive,repair,input,data,out,cache}
for path in usr bin lib lib64 etc; do
  mount --rbind "/$path" "$JAIL/$path"
  mount -o remount,ro,bind "$JAIL/$path"
done
PY_ENV=$(dirname "$(dirname "$HOST_PYTHON")")
mkdir -p "$JAIL$PY_ENV"
mount --rbind "$PY_ENV" "$JAIL$PY_ENV"
mount -o remount,ro,bind "$JAIL$PY_ENV"
for entry in "archive:$HOST_ARCHIVE" "repair:$HOST_REPAIR"; do
  target=${entry%%:*}; source=${entry#*:}
  mount --bind "$source" "$JAIL/$target"
  mount -o remount,ro,bind "$JAIL/$target"
done
touch "$JAIL/input/predictions.jsonl" "$JAIL/data/tasks"
mount --bind "$HOST_PREDICTIONS" "$JAIL/input/predictions.jsonl"
mount -o remount,ro,bind "$JAIL/input/predictions.jsonl"
mount --bind "$HOST_DATA" "$JAIL/data/tasks"
mount -o remount,ro,bind "$JAIL/data/tasks"
mount --bind "$HOST_OUTPUT" "$JAIL/out"
mount --bind "$HOST_CACHE" "$JAIL/cache"
if [[ "$MODE" == score ]]; then
  mount -o remount,ro,bind "$JAIL/cache"
fi
mount -t proc proc "$JAIL/proc"
mount -t tmpfs -o size=8G,nosuid,nodev tmpfs "$JAIL/tmp"
mount -t tmpfs -o size=1G,nosuid,noexec tmpfs "$JAIL/dev"
for node in null zero random urandom; do
  touch "$JAIL/dev/$node"
  if [[ ! -c "$HOST_DEVICES/$node" ]]; then
    printf 'Missing private character devices; run repairs/scoring_v4/setup_devices.sh\n' >&2
    exit 1
  fi
  mount --bind "$HOST_DEVICES/$node" "$JAIL/dev/$node"
  test -c "$JAIL/dev/$node"
done
mkdir -p "$JAIL/dev/shm"
mount -t tmpfs -o size=2G,nosuid,nodev,noexec tmpfs "$JAIL/dev/shm"
# Socket-based official tests need loopback, but this namespace has no external interface.
ip link set lo up
export HOME=/tmp/home TMPDIR=/tmp XDG_CONFIG_HOME=/tmp/config
export XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=/repair:/archive/pinned14/vendor/test-code/official_validation:/archive/pinned14/vendor/test-code/vendor/evalplus:/archive/bigcodebench
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PRR_BIGCODE_SANDBOX_INNER=1
if [[ "$KIND" == bigcode && -d "$HOST_NLTK_DATA" ]]; then
  mkdir -p "$JAIL/nltk_data"
  mount --bind "$HOST_NLTK_DATA" "$JAIL/nltk_data"
  mount -o remount,ro,bind "$JAIL/nltk_data"
  export NLTK_DATA=/nltk_data
fi
mkdir -p "$JAIL/tmp/home" "$JAIL/tmp/config" "$JAIL/tmp/cache" "$JAIL/tmp/matplotlib"
chroot "$JAIL" "$HOST_PYTHON" -B -c \
  'import os,stat; assert all(stat.S_ISCHR(os.stat("/dev/"+n).st_mode) for n in ("null","zero","random","urandom")); assert len(open("/dev/urandom","rb").read(16))==16; assert open("/dev/null","rb").read()==b""; open("/dev/null","wb").write(b"device-check"); print("SANDBOX_DEVICES_OK",flush=True)'
if [[ "$KIND" == evalplus ]]; then
  CONTROL_ARGS=()
  if [[ "$MODE" == controls ]]; then
    CONTROL_ARGS=(--canonical-controls)
  fi
  exec chroot "$JAIL" "$HOST_PYTHON" -B /repair/score_evalplus.py \
    --dataset "$DATASET" --predictions /input/predictions.jsonl --predictions-origin "$HOST_PREDICTIONS" \
    --official-data /data/tasks --official-data-origin "$HOST_DATA" --output /out --cache /cache --workers 1 "${CONTROL_ARGS[@]}"
elif [[ "$KIND" == bigcode ]]; then
  exec chroot "$JAIL" "$HOST_PYTHON" -B /repair/score_local.py \
    --mode "$MODE" --problems /data/tasks --predictions /input/predictions.jsonl \
    --cache /cache --output /out --vendor /archive/bigcodebench "$@"
else
  exit 2
fi
