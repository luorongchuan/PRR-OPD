# Baselines and Teacher Targets

All supported baseline students are Qwen3-0.6B and Qwen3-1.7B. Distillation
launchers reuse the recorded student-size recipe, frozen prompt input, teacher,
batch sizes, optimizer, token limits, and training duration. Only the objective
or calibration setting indicated below changes.

## Uniform OPD

```bash
bash scripts/train.sh --method opd --size 0.6b --gpus 0,1
```

The response-normalized full-vocabulary reverse KL uses unit token-region
weights. It skips the PRR calibration passes. Training is still AdamW.

## IW-OPD

```bash
bash scripts/train.sh --method iw_opd --size 1.7b --gpus 0,1
```

The experiment's adaptation is in `baselines/iw_opd_full_kl.py`. It uses
full-vocabulary reverse KL weighted by importance derived from the teacher's
and rollout student's sampled-token log probabilities. The launcher supplies
`IW_GAMMA=0.5`, `IW_NORMALIZATION=none`, and `IW_EPS=1e-8`.
The objective consumes aligned teacher/student response tokens and preserves
the underlying response normalization. There is no regional PRR calibration.

This is explicitly a **full-KL adaptation**, not a claim of byte-identical
reproduction of every upstream IW-OPD setting. Its startup diagnostics and
launch metadata should accompany reported results.

## Entropy-Aware OPD

```bash
bash scripts/train.sh --method entropy_opd --size 0.6b --gpus 0,1
```

This uses the entropy-aware branch of the recorded `PRRV1` implementation,
with teacher-entropy threshold 0.8 nats and soft-term coefficient 1. It does not
compute PRR region-retention weights. The implementation remains in the common
framework so batching, response masks, and optimizer behavior are shared.
Keep its precise objective/settings in the report instead of treating any
entropy-based distillation method as interchangeable.

## Fixed-Response SFT

SFT consumes a fixed Qwen3-8B answer per training prompt and minimizes
cross-entropy. It does not call the teacher during student training. This is a
different training mechanism from online reverse-KL distillation.

First prepare the exact frozen prompts. In the training environment, start a
teacher server on an available GPU pair, from the checkout root:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path "$(pwd)/models/Qwen3-8B" --tp-size 2 \
  --host 127.0.0.1 --port 31005
```

After the server is ready, in another terminal in the same environment:

```bash
mkdir -p "$(pwd)/data/sft"
python -B baselines/generate_teacher_responses.py \
  --input "$(pwd)/data/train_frozen_17629_seed42.parquet" \
  --output "$(pwd)/data/sft/teacher_sft.parquet" \
  --jsonl "$(pwd)/data/sft/teacher_responses.jsonl" \
  --metadata "$(pwd)/data/sft/teacher_metadata.json" \
  --tokenizer "$(pwd)/models/Qwen3-8B" \
  --server-url http://127.0.0.1:31005 \
  --batch-size 32 --max-prompt-tokens 1024 --max-response-tokens 4096 \
  --temperature 1 --top-p 1 --seed 42
```

The client applies the Qwen3 non-thinking chat template, verifies the frozen
input SHA256, saves responses incrementally, and records the generation
metadata. The recorded helper retains truncated outputs with a clipping flag;
it does not silently filter for answer correctness.

Stop the teacher server after target generation, then train with:

```bash
bash scripts/train.sh --method sft --size 0.6b --gpus 0,1 \
  --data data/sft/teacher_sft.parquet --output outputs/0.6b_sft
bash scripts/train.sh --method sft --size 1.7b --gpus 0,1 \
  --data data/sft/teacher_sft.parquet --output outputs/1.7b_sft
```

Global/per-GPU batch sizes, AdamW, LR, precision, seed and context limits are
aligned with the corresponding distillation recipe. SFT update counts depend
on retained rows and the SFT loader's length checks; they are not necessarily
551. Distinguish raw teacher-response SFT from completion-cleaned or
correctness-filtered SFT variants. The private exploration queues for those
variants are not part of this main-method release.

## Current Method Versus Historical Variants

`prr` means the current SGD-candidate PRR-OPD. `--candidate adamw` names the
older optimizer-aware variant; `--shuffle-weights` shuffles the chosen variant's
newly computed region weights. Specify both flags when reproducing an
AdamW-candidate shuffle rather than the current method's shuffle.
These flags change calibration only; all variants still train with AdamW.
