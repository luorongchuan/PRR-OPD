# Reproduction Scope and Provenance

## What This Release Preserves

`training/kdflow` is the complete Python framework snapshot used by the
completed current SGD-feedback experiments, including FSDP2, teacher/rollout
actors, exact loss reduction, PRR calibration, checkpoint state, and diagnostics.
Its source is derived from KDFlow base commit
`3cfe26cdf284b4e117ee0479dc974235aec49b5a` and the local experimental modifications.
The individual original file hashes are in `source_snapshot.json`.

`configs/training/0.6b.json` and `1.7b.json` are exported from the actual verified
training argument receipts, not reconstructed from an old generic example
launcher. Private machine paths are replaced by `${REPO_ROOT}` entries.
Published launchers expand these to the current checkout's absolute paths.
The current default is clipped-gradient calibration and aligned region weights.

The evaluation source is copied from the unified local experimental evaluator,
including its final EvalPlus/BigCodeBench reference repairs. See
`evaluation/scorers/SOURCE_MANIFEST.json` for the distributed scorer fingerprints
and `configs/third_party_commits.json` for upstream versions.

## New Public Integration

The public `scripts/train.py`, wrapper scripts, container helpers, data
reconstruction entry, environment checks, tests, and documentation are new
release integration code. They replace machine-specific GPU queues and mounts.
They were checked without launching new full GPU experiments.

Changes around evaluation remove private paths, external environments, and
remote-upload dependencies. Selected-task loading now reads only requested
benchmarks; the judgment rules remain in the vendored scorers. Sandbox device
mounts use actual character devices rather than regular-file substitutes.
Source fingerprints are refreshed for the exported code; old cached sessions
are not automatically accepted as matching this release.

## Exact Versus Functional Reproduction

Exact reproduction requires the original frozen training Parquet, initial
weights/tokenizers, decoder settings, benchmark splits, prompts, and package
combination. The training launcher enforces the frozen data hash. The row-index
metadata can reconstruct it from the exact cleaned source without distributing
examples. See [DATA.md](DATA.md) for the historical input availability limits.

Model Hub revisions and an immutable container digest were not present in the
selected receipts. The image tag and observed package versions are recorded;
an undocumented revision/digest is not invented. Archive the resolved values
in each new run. Stochastic kernels, concurrency, hardware and sampling can
affect results even when seeds agree.

Functional reproduction using newly processed data or different initial
revisions should be reported as such, not claimed to reproduce the exact
historical checkpoint. The current release is not a promise that a fresh GPU
run will exactly match a previously selected evaluation result.

## Reports to Retain

Keep each new run's `launch.json`, `arguments.json`, `train.log`, step metrics,
PRR diagnostics, final checkpoint identity, and evaluation plan/protocol and
per-response outcomes. The training launcher records input and framework hashes,
checks two-GPU occupancy, and verifies final model/training-state files.

For the main-table macro averages, average the seven mathematical benchmark
scores for ID, the four code benchmark scores for OOD, and all eleven for All.
These are benchmark macro averages, not pooled per-question accuracy. Fix the
reported evaluation run policy before comparisons; do not select better seeds
for PRR and worse seeds for baselines. A best-of-runs table must identify that
policy and apply the same evaluation budget to every method.

## Included and Omitted Experiments

The release retains the two-size main training recipes, OPD, full-KL IW-OPD,
Entropy-Aware OPD, teacher-response SFT, local sampled Pass@1 scoring, and flags
for candidate/shuffle/feedback-parameter checks.

It excludes private GPU handoff queues, watchers, old multi-seed reruns,
motivation replay, model interpolation, Pass@k expansion queues, plotting
artifacts, all dataset/model copies, and checkpoint archives. Those extras are
not dependencies of the main method. Only 0.6B and 1.7B students are configured.

## CPU Validation

```bash
python -m pytest -q tests scripts/test_clean.py
python -B evaluation/test_pipeline.py
python -B evaluation/test_scorers.py
```

Run scorer tests in the evaluation environment, on a Linux host that permits
local multiprocessing IPC. They exercise small synthetic code, not full
benchmark accuracy. Full BigCodeBench canonical certification and GPU
throughput still require the external inputs and suitable runtime environment.

See [VALIDATION.md](VALIDATION.md) for the release preparation checks. Nothing
in preparation installs into the existing experimental environments or changes
the original running jobs.
