# PRR-OPD

Research code for **progress-retention reweighting in on-policy distillation**.
This release contains the current PRR-OPD method, the training framework,
size-matched baseline launchers, and the local mathematical and code evaluation
pipeline. The supported students are **Qwen3-0.6B** and **Qwen3-1.7B**, with
**Qwen3-8B** as the frozen teacher.

The current method uses **clipped-gradient (SGD) calibration candidates**.
The actual student optimizer is **AdamW**. The older AdamW-candidate variant is
available explicitly as an ablation; it is not the default method.

No model weights, training or evaluation datasets, generated responses,
checkpoints, local environments, experiment logs, or GPU reservation processes
are distributed in this repository.

## Contents

| Directory | Purpose |
| --- | --- |
| `training/kdflow/` | Training framework used by the completed SGD-feedback runs |
| `configs/training/` | Recorded recipes for the two student sizes |
| `baselines/` | IW-OPD implementation and cached-teacher-response generation |
| `evaluation/` | Persistent GPU generation with independent guarded CPU scoring |
| `scripts/` | Training, container, environment, data preparation, and evaluation entry points |
| `tests/` | CPU numerical checks and portable launcher checks |
| `docs/` | Method, data, environment, evaluation, and reproduction details |

Start with [the method](docs/METHOD.md), [environment setup](docs/ENVIRONMENT.md),
[data preparation](docs/DATA.md), and [evaluation](docs/EVALUATION.md).
[Reproduction notes](docs/REPRODUCIBILITY.md) distinguish the experimental source
snapshot from the new public launchers.

## Quick Start

Clone the repository with the local checkout name used in the examples:

```bash
git clone https://github.com/luorongchuan/PRR-OPD.git PRR_OPD
cd PRR_OPD
```

Run the commands from the root of your local checkout. Every launcher resolves
its repository directory and passes **absolute paths** to training. Configuration
paths use `${REPO_ROOT}`, expanded by the launcher; no machine-specific username
or home directory is embedded in the code. Inside the provided container, the
absolute repository root is `/workspace/PRR_OPD`.

### 1. Create the Training Container

Training uses KDFlow, Ray, FSDP2, and SGLang. The recommended environment is the
same image tag used in the experiments:

```bash
bash scripts/create_container.sh
bash scripts/enter_container.sh
```

The default container name is `prr-opd-public`; it can be overridden using
`CONTAINER`. Existing model/data folders can be mounted read-only during creation:

```bash
HOST_MODEL_DIR=/absolute/path/to/models \
HOST_DATA_DIR=/absolute/path/to/data \
bash scripts/create_container.sh
```

Then, inside the container:

```bash
python -m pip install -e /workspace/PRR_OPD/training --no-deps
python -B scripts/check_environment.py --mode training
```

See [environment setup](docs/ENVIRONMENT.md) for observed package versions,
Docker details, a native environment alternative, and background execution.

### 2. Prepare External Inputs

Download the official Qwen3 models separately. The default directory layout is:

```text
models/
  Qwen3-0.6B/
  Qwen3-1.7B/
  Qwen3-8B/
data/
  train_frozen_17629_seed42.parquet
  evaluation/                 # downloaded benchmark inputs
```

Students are the official **Qwen3-0.6B and Qwen3-1.7B** checkpoints, not a
separately named `Base` model variant. Models require their tokenizer files,
`config.json`, and all Safetensors shards. Custom locations are accepted through
`--student`, `--teacher`, and `--data`.

The frozen training input has 17,629 cleaned DAPO-Math prompts. Its SHA256 is:

```text
2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76
```

This file is **not included**. [Data preparation](docs/DATA.md) documents the
cleaning procedure, original input fingerprints, frozen row order, benchmark
versions, schema, and which external inputs must be obtained separately.

### 3. Train PRR-OPD or OPD

Inside the training container or a compatible native environment:

```bash
# PRR-OPD, Qwen3-0.6B.
bash scripts/train.sh --method prr --size 0.6b --gpus 0,1 \
  --output outputs/0.6b_prr

# Uniform OPD, Qwen3-1.7B.
bash scripts/train.sh --method opd --size 1.7b --gpus 0,1 \
  --output outputs/1.7b_opd
```

The equivalent host-side Docker entry is:

```bash
GPU_IDS=0,1 bash scripts/train_in_container.sh \
  --method prr --size 1.7b --output outputs/1.7b_prr
```

Train both methods and both sizes sequentially on the same GPU pair:

```bash
GPU_IDS=0,1 bash scripts/train_main_table.sh
```

Inspect exact commands and resolved absolute paths without loading a model:

```bash
bash scripts/train.sh --method prr --size 1.7b --gpus 0,1 --dry-run
```

The launchers require two available GPUs and never terminate existing jobs.
No training is started automatically by cloning, installing, or testing the code.

## Training Recipe

| Setting | Qwen3-0.6B | Qwen3-1.7B |
| --- | ---: | ---: |
| Student GPUs | 2 | 2 |
| Global batch size | 32 | 32 |
| Per-GPU microbatch size | 8 | 4 |
| Gradient accumulation | 2 | 4 |
| Training prompts | 17,629 | 17,629 |
| Epochs / optimizer updates | 1 / 551 | 1 / 551 |
| Prompt token limit | 1,024 | 1,024 |
| Generated response token limit | 4,096 | 4,096 |
| Total training context limit | 5,120 | 5,120 |
| Learning rate / minimum LR | 3e-6 / 3e-6 | 3e-6 / 3e-6 |
| AdamW betas | (0.9, 0.95) | (0.9, 0.95) |
| Weight decay | 0 | 0 |
| Global gradient clipping | 1 | 1 |
| Precision | BF16 | BF16 |
| Seed | 42 | 42 |
| Rollout temperature / top-p | 1 / 1 | 1 / 1 |
| Responses per training prompt | 1 | 1 |
| PRR gamma / calibration interval | 2 / 10 | 2 / 10 |
| PRR candidate / assignment | SGD / aligned | SGD / aligned |
| Periodic save / retained checkpoints | every 10 updates / 2 | every 10 updates / 2 |

The student receives full-vocabulary reverse-KL targets from a frozen teacher.
Prompt shuffling and dropping the last batch are disabled. The final 29 real
examples are preserved through zero-loss padding, giving 551 optimizer updates.
Teacher tensor parallelism is 2; student training uses FSDP2 on the two GPUs.
Teacher and rollout engines are colocated with the student and use the
framework's memory-saving sleep/wake schedule.

PRR divides each response into four ordered token regions. Calibration uses
one uniform gradient and four regional gradients on the same rollout. It assigns
larger weights to regions whose progress is less retained by the uniform
candidate, projects the weights to `[0.5, 2]` with sum 4, and holds them until the
next calibration. See [METHOD.md](docs/METHOD.md) for the exact definitions.

## Baselines

OPD uses the same response-normalized reverse-KL objective with all region
weights fixed at 1. Other baseline entries share the size-matched training recipe:

```bash
bash scripts/train.sh --method iw_opd --size 0.6b --gpus 0,1
bash scripts/train.sh --method entropy_opd --size 1.7b --gpus 0,1
bash scripts/train.sh --method sft --size 1.7b --gpus 0,1 \
  --data data/teacher_sft.parquet
```

IW-OPD is the experiment's **full-vocabulary reverse-KL adaptation**, with
`IW_GAMMA=0.5`, no mean-one normalization, and no regional calibration.
Entropy-Aware OPD uses the existing teacher-entropy threshold of 0.8 nats and
soft-term coefficient 1. SFT uses fixed teacher responses and cross-entropy;
its step count depends on the supplied target data. These objectives differ by
design. [Baseline notes](docs/BASELINES.md) describe target generation and scope.

## Checkpoints, Logs, and Resume

For a full 551-update distillation run:

```text
outputs/RUN/
  launch.json
  arguments.json
  train.log
  prr_diagnostics.jsonl
  step_metrics.jsonl
  latest
  epoch_1_end_global_step_551/
    model/
    training_state/rank_0.pt
    training_state/rank_1.pt
  train_complete.json
```

Resume using a complete saved checkpoint in the same output folder:

```bash
bash scripts/train.sh --method prr --size 1.7b --gpus 0,1 \
  --output outputs/1.7b_prr \
  --resume outputs/1.7b_prr/epoch_1_global_step_100
```

The launcher checks that data and training settings match before resuming.
For detached execution, use `tmux` or `nohup` as described in the environment
guide. Output folders are excluded by `.gitignore`.

## Evaluate the Main Table

Evaluation uses a **separate vLLM environment**. Do not replace the training
container's SGLang/Torch packages with the evaluation requirements.

```bash
bash scripts/setup_evaluation.sh
export PRR_EVAL_PYTHON="$(pwd)/runtime/eval/bin/python"
export PRR_DATA_ROOT="$(pwd)/data/evaluation"
```

With benchmarks and local scoring dependencies prepared:

```bash
bash scripts/eval_main_table.sh --gpu 0 --launch \
  --session outputs/evaluation/1.7b_pair \
  --model opd=outputs/1.7b_opd/epoch_1_end_global_step_551/model \
  --model prr=outputs/1.7b_prr/epoch_1_end_global_step_551/model
```

This covers the 11 main-table tasks: MATH-500, AMC23, AIME24, HMMT25, Minerva,
Olympiad-Bench, GSM8K, HumanEvalPlus, MBPPPlus, LiveCodeBench, and
BigCodeBench-Hard. The general runner additionally supports AIME25 and both
CRUXEval tasks. Use `--datasets AMC23,MBPPPlus` for selected tasks, or `all` for
14 tasks. Repeated `--model` options evaluate several models serially.

Each question receives 16 sampled responses with temperature 0.6, top-p 0.95,
top-k 20, seed 42, thinking disabled, and a 2,048-token response limit. The metric
is sampled Pass@1: the mean per-question correct-response fraction.

GPU generation immediately continues with the next dataset/model while CPU
scoring runs independently. Results update in `results.csv` and `RESULTS.md`.
Multiple queues require distinct GPU indices and session directories. Scoring
uses shared locks and bounded memory to prevent the historical uncontrolled
LiveCodeBench memory growth. BigCodeBench scores locally and uploads nothing;
all 148 canonical controls must pass before a model score is accepted.

See [evaluation details](docs/EVALUATION.md) for environment setup, scoring
rules, CPU resource limits, reference checks, and incomplete-result handling.

## Optional Component and Parameter Checks

The maintained launcher exposes the relevant changes without distributing the
machine-specific experiment queues:

```bash
# Older optimizer-aware calibration; actual optimizer is still AdamW.
bash scripts/train.sh --method prr --size 0.6b --candidate adamw --gpus 0,1

# Shuffle the current clipped-gradient feedback weights across regions.
bash scripts/train.sh --method prr --size 0.6b --shuffle-weights --gpus 0,1

# Feedback sensitivity at 1.7B.
bash scripts/train.sh --method prr --size 1.7b --gamma 1 --interval 10 --gpus 0,1
```

Motivation replay, Pass@k expansion jobs, plotting scripts, historical reruns,
multi-seed queues, and GPU handoff watchers are omitted from this main-experiment
release. The main training and sampled Pass@1 evaluation do not depend on them.

## Validation and Attribution

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests scripts/test_clean.py
# In the evaluation environment:
python -B evaluation/test_pipeline.py
python -B evaluation/test_scorers.py
```

Numerical tests exercise clipped-gradient retention, read-only calibration,
projection constraints, fallback behavior, and response normalization. Pipeline
tests exercise overlapping generation/scoring and resource guards using tiny
synthetic jobs. These checks do not establish benchmark reproduction or GPU
throughput in a fresh environment.

Training is derived from KDFlow; code scoring includes pinned EvalPlus,
LiveCodeBench, and BigCodeBench sources. See [third-party notices](THIRD_PARTY.md)
and [source provenance](docs/REPRODUCIBILITY.md). [Preparation checks](docs/VALIDATION.md)
record the CPU-only verification and remaining full-run prerequisites.
Upstream MIT and Apache-2.0
license files are retained. Datasets and Qwen3 model weights retain their own terms.
