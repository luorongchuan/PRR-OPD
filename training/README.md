# Training Framework

This is the KDFlow source snapshot used by the completed SGD-feedback
PRR-OPD runs. See the [repository README](../README.md) for supported Qwen3
students, data requirements, and launch commands. Install into the training
image with `python -m pip install -e training --no-deps`.

The public entry is `../scripts/train.sh`. It explicitly supplies the paper's
clipped-gradient candidate, response budget, and size-specific microbatch.
Calling the upstream CLI directly without these options may select different
defaults, including the historical AdamW candidate.

Important modules:

- `algorithms/prr_core.py`: response regions, clipping, calibration candidates,
  retention ratios, projection, and persistent regional weight state.
- `algorithms/prr_v1.py`: full-vocabulary reverse-KL and the existing
  Entropy-Aware OPD objective.
- `ray/train/student_actor.py`: gradient accumulation, calibration, and AdamW
  updates on the actual fixed rollout batch.
- `trainer/on_policy_kd_trainer.py`: rollout/teacher/student orchestration,
  real-example normalization, final partial-batch handling, and diagnostics.
- `backend/fsdp/`: FSDP training, optimizer, and checkpoint state.
- `ray/` and `backend/sglang/`: actor placement, teacher inference, rollout
  inference, and memory-saving engine operation.
- `datasets/`: prompt and cached-teacher-response data preprocessing.

The numerical framework files are retained byte-for-byte from the experiment
snapshot. Public portability changes are in launchers and evaluation adapters.
The accompanying MIT license is retained from KDFlow.
