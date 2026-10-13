# Models and Data

External model weights, training prompts, teacher responses, and benchmark
inputs are not included. Use the official `Qwen/Qwen3-0.6B`,
`Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-8B` checkpoints, with their original
tokenizers and all weight shards, in the corresponding `models/` folders.
Custom paths are accepted by `scripts/train.sh --student --teacher --data`.

## Training Inputs

| Snapshot | Rows | SHA256 |
| --- | ---: | --- |
| Processed source | 17,917 | `500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39` |
| Cleaned source | 17,902 | `45011c3651109088a1132fc41d7c5a91e41b9c2fdcaa106dc68ef9b54b074234` |
| Frozen training input | 17,629 | `2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76` |

With the exact processed training source and MATH-500/AMC23 test inputs present:

```bash
python -B scripts/clean_training_data.py \
  --train data/dapo-math-17k-processed.parquet \
  --math500 data/evaluation/MATH-500/test.parquet \
  --amc23 data/evaluation/AMC23/test.parquet \
  --output-dir data/dapo_math_clean
python -B scripts/freeze_training_data.py \
  --input data/dapo_math_clean/train.parquet
```

Cleaning removes 15 exact question overlaps after NFKC/whitespace normalization
and removal of the recorded trailing reasoning instructions.
`configs/training_row_order.json` selects the fixed 17,629-row training stream.
Use PyArrow 24.0.0 to reconstruct the byte-identical frozen Parquet.
The launcher verifies its hash. A generic DAPO download is not a substitute for
the exact historical processed source, which must be obtained separately.

Training data contains `prompt` messages, `reward_model.ground_truth`, and
`extra_info.index`. Distillation consumes prompts only. SFT additionally
requires `teacher_response`; use `baselines/generate_teacher_responses.py --help`
for the fixed-response generation entry.

## Evaluation Inputs

Set `PRR_DATA_ROOT` to `data/evaluation`. Mathematical benchmarks use
`DATASET/test.parquet` with the same prompt/reference/index schema.
Their recorded fingerprints are in
`evaluation/scorers/pinned14/vendor/other-test/datasets_manifest.json`.

```bash
python -B scripts/prepare_code_data.py
python -B scripts/download_evalplus.py
python -B scripts/prepare_math_data.py --download-missing
```

The math helper downloads HMMT25 and GSM8K; other processed math inputs must
already be present. Code preparation pins revisions and checksums.
HumanEvalPlus v0.1.10 and MBPPPlus v0.2.0 require the official expanded Plus
tests, not only the processed task folders. LiveCodeBench uses the v6 delta
(175 questions), not the cumulative release. BigCodeBench-Hard has 148 tasks
and needs the additional [scoring setup](EVALUATION.md).

Record model revisions, data hashes, and the resolved container digest in new
runs. The public launchers and CPU checks do not guarantee exact reproduction
of historical scores.
