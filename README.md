# PRR-OPD

Code for progress-retention reweighting in on-policy distillation, using
Qwen3-0.6B or Qwen3-1.7B students and a frozen Qwen3-8B teacher.
PRR calibration uses clipped-gradient candidates; actual updates use AdamW.
Model weights, datasets, generated responses, and checkpoints are not included.

## Setup

Run from the checkout root. Create the training container on the host:

```bash
bash scripts/create_container.sh
bash scripts/enter_container.sh
```

Inside the container:

```bash
python -m pip install -e training --no-deps
python -B scripts/check_environment.py --mode training
```

Place the official Qwen3 checkpoints in `models/Qwen3-0.6B`,
`models/Qwen3-1.7B`, and `models/Qwen3-8B`, and the frozen 17,629-prompt
file at `data/train_frozen_17629_seed42.parquet`.
See [data preparation](docs/DATA.md) for required inputs and checksums.
Training and evaluation use separate environments; recorded training versions
are in `configs/environment_training.json`.

## Training

Use two allocated GPUs:

```bash
bash scripts/train.sh --method prr --size 0.6b --gpus 0,1 \
  --output outputs/0.6b_prr
bash scripts/train.sh --method opd --size 0.6b --gpus 0,1 \
  --output outputs/0.6b_opd
```

Use `--size 1.7b` for the larger student. To train both methods at both sizes
sequentially, run `GPU_IDS=0,1 bash scripts/train_main_table.sh`.
For host-side container launch, use `scripts/train_in_container.sh`.

The size-specific recipes in `configs/training/` use a fixed prompt order,
one epoch (551 updates), batch size 32, a 4,096-token response budget, BF16,
and full-vocabulary reverse KL. Add `--dry-run` to inspect a command,
or `--resume CHECKPOINT` to restore a complete checkpoint in the same output
directory. Final models are saved under
`epoch_1_end_global_step_551/model/`.

Optional training switches are `--candidate adamw`, `--shuffle-weights`,
`--gamma`, and `--interval`. Other methods are `iw_opd`, `entropy_opd`,
and `sft --data PATH`. IW-OPD is a full-KL adaptation with importance exponent
0.5. The entropy control uses gated teacher-student probability overlap
(threshold 0.8 nats, coefficient 1), not gated forward KL.
SFT requires fixed teacher responses and does not use an online teacher.

## Evaluation

Prepare benchmarks using [data preparation](docs/DATA.md), then set up a
separate evaluation environment:

```bash
bash scripts/setup_evaluation.sh
export PRR_EVAL_PYTHON="$(pwd)/runtime/eval/bin/python"
export PRR_DATA_ROOT="$(pwd)/data/evaluation"
```

Evaluate the final checkpoints:

```bash
bash scripts/eval_main_table.sh --gpu 0 --launch \
  --session outputs/evaluation/0.6b_pair \
  --model opd=outputs/0.6b_opd/epoch_1_end_global_step_551/model \
  --model prr=outputs/0.6b_prr/epoch_1_end_global_step_551/model
```

This wrapper selects the repository's eleven-task default, including GSM8K.
For a different subset, use `evaluation/run_eval.sh --datasets LIST` with
the same model, GPU, and session options. Evaluation uses 16 responses per
question, seed 42, temperature 0.6, top-p 0.95, top-k 20, BF16, and a 2,048-token
response budget, with thinking disabled. Results are saved in the session's
`results.csv` and `RESULTS.md`.
[Evaluation notes](docs/EVALUATION.md) cover local scoring, BigCodeBench setup,
and resuming interrupted sessions.

## Tests and Licenses

```bash
python -m pytest -q tests scripts/test_clean.py
```

Training code is in `training/kdflow/`, launchers in `scripts/`, and local
generation/scoring in `evaluation/`. See [third-party notices](THIRD_PARTY.md)
for KDFlow and the vendored scorer licenses and versions.
