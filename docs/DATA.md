# Models and Data

This repository distributes code and metadata, not model weights, examples,
test cases, teacher responses, or checkpoints. Obtain external inputs under
their respective licenses. Default paths are inside the checkout. All launch
paths are resolved to absolute paths at runtime.

## Models

Use the official `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, and `Qwen/Qwen3-8B`
repositories. For example, with the Hugging Face CLI available:

```bash
hf download Qwen/Qwen3-0.6B --local-dir "$(pwd)/models/Qwen3-0.6B"
hf download Qwen/Qwen3-1.7B --local-dir "$(pwd)/models/Qwen3-1.7B"
hf download Qwen/Qwen3-8B --local-dir "$(pwd)/models/Qwen3-8B"
```

Record the downloaded revisions. The experimental receipts preserve local
model paths and configuration, not a recoverable immutable Hub revision;
this release does not invent one. Keep the original tokenizer, chat template,
config, and every shard named in `model.safetensors.index.json`.
Thinking is disabled for training and evaluation. The supported student models
are not checkpoints with a separate `Base` suffix.

## Training Prompt Snapshot

The experiment used processed DAPO-Math prompts in Parquet format:

| Stage | Rows | SHA256 |
| --- | ---: | --- |
| Processed source | 17,917 | `500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39` |
| Question-overlap-cleaned source | 17,902 | `45011c3651109088a1132fc41d7c5a91e41b9c2fdcaa106dc68ef9b54b074234` |
| Fixed training subset and order | 17,629 | `2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76` |

The cleaning helper removes training questions that exactly match MATH-500 or
AMC23 after Unicode NFKC normalization, whitespace normalization, and removal
of one of two literal trailing reasoning instructions. Matching is case
sensitive; answers and model predictions are not used. It removes 15 source
rows: 4 matching MATH-500 and 11 matching AMC23. It preserves the retained
rows' contents, schema, metadata, and order. It does not remove every internal
duplicate or claim decontamination against every possible benchmark.

Place the exact processed source at `data/dapo-math-17k-processed.parquet`
and the two processed evaluation inputs at their paths below, then run:

```bash
python -B scripts/clean_training_data.py \
  --train "$(pwd)/data/dapo-math-17k-processed.parquet" \
  --math500 "$(pwd)/data/evaluation/MATH-500/test.parquet" \
  --amc23 "$(pwd)/data/evaluation/AMC23/test.parquet" \
  --output-dir "$(pwd)/data/dapo_math_clean"

python -B scripts/freeze_training_data.py \
  --input "$(pwd)/data/dapo_math_clean/train.parquet"
```

Use PyArrow 24.0.0 for the byte-identical frozen snapshot. The cleaned source
includes rows outside the training prompt-length budget. The recorded
`configs/training_row_order.json` selects and orders 17,629 distinct rows from
that 17,902-row source. It is **not** a permutation of 17,629 source rows.
The framework also checks the 1,024-token prompt limit when loading data.

The reconstruction script requires the exact cleaned input hash and verifies
the resulting frozen hash. Different serialization or an alternative processed
DAPO download is not silently accepted as the paper's exact input. No download
URL for the historical processed training file was preserved in the receipts;
obtaining that exact input remains an external prerequisite. A generic public
DAPO download cannot be claimed to reproduce the frozen bytes.

The training schema includes:

```text
prompt: list of {role, content}, one user message per example
reward_model: {ground_truth, style}
extra_info: {index}
data_source, ability: strings
```

The distillation objective consumes prompts, not the ground-truth answer.
SFT instead requires a separately generated `teacher_response` column; see
[baseline instructions](BASELINES.md).

## Evaluation Inputs

Set `PRR_DATA_ROOT` to an absolute data directory, normally:

```bash
export PRR_DATA_ROOT="$(pwd)/data/evaluation"
```

| Task | Questions | Input relative to `PRR_DATA_ROOT` |
| --- | ---: | --- |
| MATH-500 | 500 | `MATH-500/test.parquet` |
| AMC23 | 83 | `AMC23/test.parquet` |
| AIME24 | 30 | `AIME24/test.parquet` |
| HMMT25 | 30 | `HMMT25/test.parquet` |
| Minerva | 272 | `Minerva/test.parquet` |
| Olympiad-Bench | 675 | `Olympiad-Bench/test.parquet` |
| GSM8K | 1,319 | `GSM8K/test.parquet` |
| HumanEvalPlus | 164 | `EvalPlusOfficial/HumanEvalPlus-v0.1.10.jsonl.gz` plus the prepared task folder |
| MBPPPlus | 378 | `EvalPlusOfficial/MbppPlus-v0.2.0.jsonl.gz` plus the prepared task folder |
| LiveCodeBench | 175 | `LiveCodeBench/raw/test6.jsonl` |
| BigCodeBench-Hard | 148 | `BigCodeBench-Hard/test.parquet` |
| AIME25 (additional) | 30 | `AIME25/test.parquet` |
| CRUXEval-I/O (additional) | 800 each | `CRUXEval/test.parquet` |

The main table uses the first eleven tasks. `all` evaluates all fourteen.
LiveCodeBench is the **v6 delta**, not the cumulative v6 release.

### Code Benchmarks

```bash
python -B scripts/prepare_code_data.py
python -B scripts/download_evalplus.py
# Verify already prepared files without downloading:
python -B scripts/prepare_code_data.py --verify-only
```

`prepare_code_data.py` pins the actual dataset revisions, creates local task
folders, and records checksums. It also prepares CRUXEval for the optional tasks.
The EvalPlus helper downloads the official HumanEval+ v0.1.10 and MBPP+ v0.2.0
releases, checking `configs/evalplus_releases.json`. The processed Hugging Face
task folders alone are not substitutes for the official expanded Plus tests.

### Mathematics Benchmarks

The generator expects `prompt`, `reward_model.ground_truth`, and
`extra_info.index` in each processed mathematical test file. Preserve the
reasoning instruction in each prompt and the complete set of question IDs.
MATH-500 and AMC23 fingerprints used during cleaning are:

```text
MATH-500: 2e4c43a3d52175a773816f99f327d3ed5e3d15af6f479d87196a702cdb4f5411
AMC23:    0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c
```

Obtain the processed AIME24, AIME25, Minerva and Olympiad-Bench snapshots
separately. Their historical fingerprints and schema metadata are in
`evaluation/scorers/pinned14/vendor/other-test/datasets_manifest.json`.
After these files are present:

```bash
python -B scripts/prepare_math_data.py --download-missing
```

This helper downloads hash-pinned HMMT25 and GSM8K raw data, converts those two
datasets, and validates the six additional mathematical test inputs. It is not
a complete downloader for all mathematics benchmarks. Regenerated Parquet
serialization can have a different byte hash; retain the raw revision and
verify logical examples, IDs, answers, and prompt templates before comparing
with historical results.

## BigCodeBench Resources

BigCodeBench requires an isolated Python 3.10 environment and offline task
resources in addition to the benchmark itself. Follow
[evaluation setup](EVALUATION.md). Resource downloads are hash checked; a
changed web page must not be substituted for the recorded resource. The scorer
refuses to publish accuracy unless every canonical control passes.
