# Release Preparation Checks

Validation performed during preparation, without launching GPU training or
benchmark generation and without modifying the experimental environments.

| Check | Result |
| --- | --- |
| PRR numerical, launcher, path and cleaning regression suite | 42 passed |
| Generation/scoring pipeline synthetic suite | 9 passed |
| Local scorer synthetic suite, including official BigCodeBench toy checker | 9 passed |
| Main-table evaluation input/protocol preflight | All 11 datasets loaded |
| Frozen prompt reconstruction | Byte-identical SHA256 verified |
| Python source syntax | 235 files parsed |
| Shell source syntax | 14 scripts checked individually |
| Original training framework preservation | 80 Python files byte-identical to the experimental snapshot |
| Published scorer source manifest | 134 source files fingerprinted |
| Private home paths and obsolete release-name scan | No matches in published source/configuration/documentation |

The actual KDFlow argument parser accepted both sizes for PRR, OPD, IW-OPD,
Entropy-Aware OPD and SFT. Tests also verify that relative training arguments
are resolved against the checkout root, even when launched from another
working directory. The default student configurations are only 0.6B and 1.7B.

CPU checks used a compatible existing evaluation interpreter with GPUs hidden.
The official toy checker needs local multiprocessing IPC; it passed when run
with that capability. It initially failed in a restricted tooling sandbox
that forbids local sockets, rather than because of a scoring-rule failure.

The input preflight reads external datasets and tokenizer files but does not
load model weights or initialize a vLLM engine. Its temporary files and the
reconstructed training Parquet are outside this public repository.

These checks do **not** certify a fresh full GPU experiment, an exact match to
historical paper scores, a fresh dependency installation, or all 148 real
BigCodeBench canonical controls in a new environment. Those require the
external data, model revisions, resources and suitable GPU/runtime setup.

## Recheck

```bash
python -m pytest -q tests scripts/test_clean.py
python -B evaluation/test_pipeline.py
python -B evaluation/test_scorers.py
```

After intentional scorer edits, review the changed judgment behavior and run
`python -B scripts/refresh_source_manifest.py` before starting a new evaluation
session. Do not refresh fingerprints simply to accept unexplained changes or
reuse results from an incompatible protocol.

## Publishing

The repository has a `.gitignore` for external inputs, model/checkpoint files,
generated outputs, virtual environments and runtime locks. Before uploading,
inspect the staged file list. Do not force-add ignored experiment artifacts.

```bash
git init
git add .
git status --short
git diff --cached --stat
```

Create the remote repository and commit/push under the intended paper-release
account after review. This release preparation does not set a remote, commit
on your behalf, or upload any files.
