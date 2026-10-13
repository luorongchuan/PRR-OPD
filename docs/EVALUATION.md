# Local Evaluation

Use `evaluation/run_eval.sh` for selected tasks, or
`scripts/eval_main_table.sh` for the existing eleven-task suite.
Setup and final-checkpoint launch commands are in the [main README](../README.md).
Use a separate vLLM environment and prepare [benchmark inputs](DATA.md) first.

## Setup and Launch

BigCodeBench-Hard additionally needs its isolated Python 3.10 environment and
hash-pinned offline resources:

```bash
bash scripts/setup_bigcodebench.sh
"$PRR_EVAL_PYTHON" -B scripts/check_environment.py \
  --mode evaluation --check-sandbox
```

Selected-task example:

```bash
bash evaluation/run_eval.sh --gpu 0 --launch \
  --session outputs/evaluation/selected \
  --datasets AMC23,HumanEvalPlus,MBPPPlus \
  --model prr=outputs/0.6b_prr/epoch_1_end_global_step_551/model
```

Each question receives 16 responses with a 2,048-token budget, BF16,
temperature 0.6, top-p 0.95, top-k 20, seed 42, and thinking disabled.
Sampled Pass@1 is the mean per-question correct-response fraction, not Pass@16.
Use the same generation profile for comparisons; `--profile fast` can alter
the sequence of seeded samples.

## Scoring and Safety

Mathematics uses the recorded final-answer extraction and Math-Verify.
Code tasks use the pinned local benchmark checkers. HumanEvalPlus retains all
164 tasks, including its recorded canonical negative control.
BigCodeBench requires all 148 canonical controls to pass before accepting
model scores. Failed prerequisites remain blocked, not model accuracy zero.

Generation and bounded CPU scoring run independently. Use different GPUs and
session directories for concurrent queues, while retaining the shared scoring
locks. Defaults allow one heavy scoring job, two ordinary checker workers,
and one BigCodeBench worker. Do not bypass these limits.
Generated code executes locally in isolated Linux namespaces. The host needs
`unshare`, `mount`, `chroot`, `ip`, and enabled user namespaces; do not disable
the sandbox to work around environment failures. Scoring uploads nothing.

## Outputs and Resume

Each session contains `results.csv`, `RESULTS.md`, `status.json`, and
per-model protocols, generated batches, and per-dataset score summaries.
Pending or blocked scores are missing, not zero. GPU generation completion
does not imply scoring completion.

Repeat the same command and session to resume validated outputs. Changed
models, data, code, or decoding settings require a new session. Add
`--resume-evaluations` only after resolving a recorded resource hold.
