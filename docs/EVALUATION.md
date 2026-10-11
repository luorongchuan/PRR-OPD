# Local Evaluation

Use `evaluation/run_eval.sh` as the unified entry. It serializes generation
on one GPU, keeps one vLLM engine loaded per model, and starts independent
CPU scoring as soon as a dataset's generation is complete. It supports one
dataset, several datasets, and several models without changing judgment rules.
No model code or responses are uploaded for scoring.

## Setup

Create a separate evaluation environment; do not overwrite the training image:

```bash
bash scripts/setup_evaluation.sh
export PRR_EVAL_PYTHON="$(pwd)/runtime/eval/bin/python"
export PRR_DATA_ROOT="$(pwd)/data/evaluation"
"$PRR_EVAL_PYTHON" -B scripts/check_environment.py \
  --mode evaluation --check-sandbox
```

An existing compatible environment can be used instead. The recorded relevant
versions are Python 3.12, Torch 2.8.0, vLLM 0.11.0, Transformers 4.56.1,
Math-Verify 0.9.0, and SymPy 1.14.0. Prepare external benchmark inputs as
described in [DATA.md](DATA.md).

For BigCodeBench-Hard, install the private Python 3.10 dependencies and resources:

```bash
bash scripts/setup_bigcodebench.sh
```

This creates `evaluation/runtime/bigcode_py310`, not a replacement for the
generation or training environment. It downloads NLTK and hash-pinned task
resources. If a resource's bytes differ, setup fails explicitly; obtain the
recorded resource before scoring. Completing setup does not itself certify
the 148 canonical controls: that check occurs before prediction scoring.

## Generation and Metrics

| Setting | Value |
| --- | --- |
| Responses per problem | 16 |
| Response limit | 2,048 tokens |
| Context limit | 8,192 tokens |
| Temperature | 0.6 |
| Top-p / decoding top-k | 0.95 / 20 |
| Seed | 42 |
| Thinking | Disabled |
| Precision | BF16 |
| Comparable request batch / max sequences | 4 / 64 |
| Default GPU memory utilization | 0.875 |

On an 80-GiB GPU the engine's memory budget is approximately 70 GiB, not a
promise of constant allocation/utilization. `--profile fast` changes request
batch size/max sequences to 16/256; it can change the sequence of seeded
samples and is recorded in the protocol. Use the same profile for comparisons.

The main-table metric is sampled Pass@1:

```text
100 * mean_over_questions(number_correct / 16)
```

It is not greedy accuracy or Pass@16. A partial sample round must not be
reported as the final model score. The full dataset and all 16 responses are
required; changes to the model, source code, decoder or data need a new session.

## Commands

Main-table evaluation on one allocated GPU:

```bash
bash scripts/eval_main_table.sh --gpu 0 --launch \
  --session outputs/evaluation/0.6b_pair \
  --model opd=outputs/0.6b_opd/epoch_1_end_global_step_551/model \
  --model prr=outputs/0.6b_prr/epoch_1_end_global_step_551/model
```

Selected tasks:

```bash
bash evaluation/run_eval.sh --gpu 1 --launch \
  --session outputs/evaluation/1.7b_selected \
  --datasets AMC23,HumanEvalPlus,MBPPPlus \
  --model prr=outputs/1.7b_prr/epoch_1_end_global_step_551/model
```

The first command covers the eleven main-table tasks. `--datasets all` includes
AIME25 and CRUXEval-I/O as well. Model paths can be original student/teacher
folders or complete exported Hugging Face checkpoints. Use unique model labels
and session folders. The runner refuses an occupied GPU and never kills jobs.

For a CPU-only input/protocol check without loading model weights or a vLLM
engine:

```bash
"$PRR_EVAL_PYTHON" -B evaluation/generate.py \
  --datasets AMC23,MBPPPlus \
  --model-path "$(pwd)/models/Qwen3-0.6B" \
  --data-root "$PRR_DATA_ROOT" \
  --output "$(pwd)/outputs/evaluation/input_check" --stage prepare
```

Use a fresh output directory if code/data/model settings change.

## Judgment Rules

| Tasks | Judgment |
| --- | --- |
| Mathematics | Last complete boxed/fboxed final answer, with the recorded extraction fallback, then Math-Verify/symbolic equivalence against the reference |
| HumanEvalPlus, MBPPPlus | Official pinned expanded Plus test sets; official checker in an isolated local sandbox |
| LiveCodeBench | Pinned v6-delta task IDs, public/private cases, code extraction, execution limits and checker |
| CRUXEval-I/O | Pinned input/output parsing and execution judgment |
| BigCodeBench-Hard | Official local Hard/Instruct test harness, calibrated=false, fixed dependency/resource environment |

The historical HumanEval+ release contains one canonical negative control,
HumanEval/32. The checker expects that exact failure and successful canonical
execution for the other 163 tasks. All **164** tasks remain in model evaluation;
none is dropped and the denominator is not changed. Expectations are pinned by
dataset and task hashes in `evaluation/repairs/scoring_v4/reference_expectations.json`.

BigCodeBench requires **148/148** canonical controls before model responses can
be judged. Missing dependencies, namespace failures, changed test resources,
and control failures block a score instead of becoming model accuracy zero.
The published scoring entry has no automatic remote fallback or upload.

## Concurrent Queues and CPU Safety

Run independent generation queues on different GPU indices with distinct
session folders. The default shared lock root is
`runtime/evaluation_locks`. Processes use file locks, not a scheduler-wide GPU
reservation. GPU allocation still requires coordination with other users.

Heavy CPU scoring is bounded and uses a shared lane across these queues.
The safety default is deliberate: a scoring backlog does not block generation,
but unrestricted parallel LiveCodeBench processes can exhaust host RAM.
Do not give different queues different lock roots merely to bypass the limit.

| Guard | Default |
| --- | --- |
| Simultaneous heavy scoring jobs | 1 |
| Ordinary checker workers | 2 |
| Ordinary per-process address-space limit | 4 GiB |
| BigCodeBench workers / address-space limit | 1 / 30 GiB |
| Process-tree RSS stop threshold | 24 GiB |
| Minimum available host memory | 128 GiB |
| Sample draws | Serial |

RSS supervision is polled, not an atomic cgroup cap. Shared-server jobs still
require monitoring. Evaluation subprocesses set thread limits, hide GPUs from
CPU workers, and record `resource_guard.json`. A low-memory incident can set an
evaluation hold; resume explicitly after verifying the host is safe.

Generated code runs in Linux user/mount/PID/network namespaces with isolated
filesystems and offline task resources. User namespaces and the utilities
listed in [ENVIRONMENT.md](ENVIRONMENT.md) are required. Do not disable the
sandbox as a workaround, and do not run arbitrary generated code on a host
with valuable credentials. These guards do not constitute a formal security
guarantee against all hostile code.

## Outputs and Resume

```text
outputs/evaluation/SESSION/
  plan.json, launcher.json, queue.log, status.json
  results.csv, RESULTS.md
  models/LABEL/
    protocol.json, generation.log, generation_complete.json
    generation/DATASET/batch_000000.json
    scores/DATASET/summary.json
    scores/DATASET/resource_guard.json
    scores/DATASET/scoring.log
```

The plan and protocol fingerprint code, model metadata, benchmark inputs, and
decoder settings. Validated batches and completed scores are reused on resume.
Malformed or failed summaries are quarantined. Pending/blocked scores remain
missing, not zero. Inspect `status.json` and per-dataset scoring logs rather
than assuming a GPU generation completion marker means scoring is finished.

Repeat the same launch command/session after an ordinary interruption. If a
memory hold is active, add `--resume-evaluations` only after resolving it.
Do not reuse cached results across changed protocols.
