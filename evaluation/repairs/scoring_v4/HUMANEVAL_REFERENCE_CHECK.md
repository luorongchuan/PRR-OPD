# HumanEvalPlus Reference-Control Repair

2026-10-09. Official data, test inputs, tolerances, and the pinned EvalPlus
prediction checker are unchanged. No task is dropped and no predicted answer
receives credit because of a reference failure.

## Reproduction

The v0.1.10 data SHA256 is
`272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101`.
Its official Newton reference for `HumanEval/32` passes all 100 base inputs but
fails seven of the 788 expanded inputs: indices 119, 177, 185, 399, 616, 650,
and 694. The first failure caused the previous fast self-check to report
163/164 and abort before prediction scoring.

For expanded input 119, the coefficients are
`[9450000,9,-7,3,2,6,1,-4,-10,6,17,-1]`. The official reference returns
`17.3124550475086`; the official polynomial residual is
`-0.0023150274059275944`, outside `atol=0.0001`. Other failing cases also include
nonconvergence of the fixed 1000-iteration Newton reference. This is a failure
of the reference program against the frozen tests, not evidence that the
prediction checker crashed.

Raw diagnostic evidence is stored under
`checks/humaneval_controls_20261009_full32/controls/canonical_failure.json`.

## Self-Check Contract

The self-check evaluates all 164 reference programs. For HumanEval/32 it
collects the full 788 expanded-test outcomes rather than stopping at the first
failure. The expected negative-control outcome is allowed only when the data
hash, reference-code hash, expanded-input hash, tolerance, task ID, exact seven
failed input indices, and complete checked-input counts all match the pinned
fixture in `reference_expectations.json`.

Any other failure, timeout, incomplete check, or changed release still aborts.
The summary reports **163 passed reference programs and 164 validated reference
outcomes**, with the known failure retained explicitly. It never reports that
all 164 official reference programs passed.

Model predictions still run through the unchanged official oracle. In
particular, a model failing HumanEval/32 receives a failure as before. The
denominator remains `164 * 16`. Only the reference self-check's expected-outcome
contract changes; official grading is not made more permissive.

## Provenance And Resources

New scores include the control-policy revision/hash and repaired source hashes.
Previously failed logs are preserved and retries use separate scoring output
directories. The CPU guard still limits each process to 4 GiB address space,
the process tree to 24 GiB RSS, and requires 128 GiB host memory available.
Scoring is CPU-only and uses the global single-heavy-job lock.
