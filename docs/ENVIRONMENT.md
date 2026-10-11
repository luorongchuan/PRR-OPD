# Environments and Containers

Training and evaluation use separate environments. The training framework uses
SGLang for online responses and teacher targets. Evaluation uses vLLM and local
CPU scorers. Installing evaluation's Torch/vLLM dependencies into the training
image changes its tested package combination.

## Training: Recommended Container

```bash
bash scripts/create_container.sh
bash scripts/enter_container.sh
python -m pip install -e /workspace/PRR_OPD/training --no-deps
python -B scripts/check_environment.py --mode training
```

Default image: `songmzhang/kdflow:sgl0.5.12_torch2.11_cu129`.
The completed 1.7B main-method argument receipt records:

| Package | Recorded version |
| --- | --- |
| Torch | 2.11.0+cu129 |
| Transformers | 5.6.2 |
| Ray | 2.55.1 |
| SGLang | 0.5.12.post1 |

These versions are also stored in `configs/environment_training.json`.
Image tags can change; record the resolved image ID/digest when reproducing:

```bash
docker image inspect songmzhang/kdflow:sgl0.5.12_torch2.11_cu129 \
  --format '{{.Id}} {{json .RepoDigests}}'
```

The container mounts the checkout at `/workspace/PRR_OPD`, uses the host UID/GID
for commands, and stays alive after an interactive shell exits. It exposes all
host GPUs; each training command selects its two assigned GPU indices.
GPU visibility is not a scheduler reservation. Use the GPU pair allocated to you.

The experimental setup uses two NVIDIA GPUs with approximately 80 GiB memory
each. Actual host RAM requirements depend on CPU offloading, Ray and colocated
jobs. Confirm available memory before a full run. The bundled public launchers
do not start jobs to reserve idle GPUs.

## Training: Native Alternative

Create a dedicated Python environment with CUDA-compatible versions of the
recorded packages. Install the framework dependencies and FlashAttention after
Torch. For example, in an already CUDA-compatible environment:

```bash
python -m pip install -r training/requirements.txt -c constraints-training.txt
python -m pip install flash_attn==2.8.3 --no-build-isolation
python -m pip install -e training --no-deps
python -B scripts/check_environment.py --mode training
```

`training/requirements.txt` is the upstream broad dependency list; the constraints
pin the four core versions observed in the completed training receipt.
This is not a complete transitive lockfile or a guarantee that every host driver
can use that CUDA build. The original image is the preferred starting point.
Training entry points record the actual versions in each output directory.

## Evaluation Environment

The experiments' host evaluation environment uses Python 3.12, Torch 2.8.0,
vLLM 0.11.0 and Transformers 4.56.1. Relevant versions are pinned in
`requirements-eval.txt`.

```bash
bash scripts/setup_evaluation.sh
export PRR_EVAL_PYTHON="$(pwd)/runtime/eval/bin/python"
export PRR_DATA_ROOT="$(pwd)/data/evaluation"
"$PRR_EVAL_PYTHON" -B scripts/check_environment.py \
  --mode evaluation --check-sandbox
```

An existing compatible environment can be used by setting `PRR_EVAL_PYTHON`
to its absolute Python path. The setup scripts install only into newly specified
private virtual environments. BigCodeBench additionally uses Python 3.10:

```bash
bash scripts/setup_bigcodebench.sh
```

Local code execution requires Linux user/mount/PID/network namespaces,
`unshare`, `mount`, `chroot`, and `ip` (util-linux and iproute2). Run scoring on
a host where unprivileged user namespaces are enabled. A Docker shell may have
different namespace restrictions from the host. Do not disable isolation to
work around missing namespace support. BigCodeBench uses real `/dev` character
devices bound read-only into the private filesystem, with isolated loopback.

## Keep Jobs Running After Disconnecting

For host-triggered container training:

```bash
tmux new -s prr-training
GPU_IDS=0,1 bash scripts/train_in_container.sh --method prr --size 1.7b \
  --output outputs/1.7b_prr
# Detach with Ctrl-b, then d; reattach with: tmux attach -t prr-training
```

The evaluation runner's `--launch` option detaches the queue and records its
PID and log path. A local laptop disconnect does not stop detached jobs on a
running server; a foreground process tied to a disconnected terminal can stop.

## Common Failures

| Symptom | Check |
| --- | --- |
| GPU already occupied | Release or select your allocated GPU; the launcher never kills tasks |
| Ray reports low host memory | Check machine-wide RAM and the active scorers before resuming |
| Frozen training SHA256 differs | Use the exact prompt snapshot and ordering; do not suppress the check |
| FlashAttention import fails | Check Torch/CUDA/FlashAttention ABI compatibility |
| Code sandbox cannot `unshare` | Enable user namespaces on the execution host |
| BigCodeBench canonical controls fail | Fix private dependencies/resources; a failed environment is not model accuracy 0 |
| Resume rejected | Use the same output folder, complete rank states, recipe, and data |
