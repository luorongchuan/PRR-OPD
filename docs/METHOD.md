# Method and Implementation

PRR-OPD generates one response per prompt from the current student and obtains
full-vocabulary teacher distributions on those same student-generated prefixes.
The teacher is frozen. The distillation loss is reverse KL, `KL(student || teacher)`.

For response `i`, let `T_i` be its number of valid response tokens. The four
regions use boundaries `floor(k * T_i / 4)`, for `k = 0,...,4`, among valid
response-token positions. Prompt and padding tokens are excluded. Regional
loss `L_k` keeps each token's coefficient `1 / T_i`, followed by averaging over
real examples. It does not renormalize a region to unit mass and does not
multiply its gradient by four. With uniform weights, the four regional losses
sum to the response-normalized OPD objective.

## Calibration

The joint gradient is `g = sum_k a_k`, where `a_k = grad L_k`. On one fixed
rollout batch, the current method constructs:

```text
C(z)       = z * min(1, 1 / (||z||_2 + 1e-6))
d_uniform  = -learning_rate * C(g)
d_solo[k]  = -learning_rate * C(a_k)
P[k]       = -dot(a_k, d_uniform)
S[k]       = -dot(a_k, d_solo[k])
r[k]       = P[k] / S[k]
```

Clipping uses the global norm across trainable parameter shards. Calibration
performs one uniform backward pass and four region-only backward passes with
the same tokens, masks, and teacher targets. It reads the optimizer learning
rate but uses no optimizer moments in the SGD candidates.

`r[k]` is a signed local first-order progress ratio, not a probability or an
accuracy score. Positive `S[k]` is required. Undefined/nonfinite ratios leave
the stored weights unchanged. Calibration does not apply any candidate to the
model and verifies that trainable parameters and optimizer state were unchanged.

## Feedback and Actual Update

Let `mean_r` be the mean retention across the four regions. Feedback scores are
`u[k] = exp(gamma * (mean_r - r[k]))`. The code projects `u` onto:

```text
0.5 <= w[k] <= 2, for k = 1,...,4
sum_k w[k] = 4
```

This is a Euclidean projection onto a box-constrained sum hyperplane, implemented
with a scalar shift found by bisection. It is not simple divide-by-sum
normalization. The default is `gamma=2`, with calibration before the first update
and every 10 completed updates thereafter (one-based training updates 1, 11,
21, ...). Initial weights are all 1. Weights are held between calibrations.

A fresh weighted backward pass computes the actual training gradient. The
student then performs the usual globally clipped **AdamW** update and scheduler
step. Diagnostics and PRR state are checkpointed with training state. Changing
calibration candidates to SGD does not replace the actual optimizer with SGD.

## Code Map

| Concept | Implementation |
| --- | --- |
| Four response regions | `training/kdflow/algorithms/prr_core.py:response_token_weights` |
| Global clipped gradients | `training/kdflow/algorithms/prr_core.py:clipped_gradients` |
| Joint / solo progress | `training/kdflow/algorithms/prr_core.py:calibrate` |
| Feedback projection | `training/kdflow/algorithms/prr_core.py:feedback_weights` |
| Periodic stored weights | `training/kdflow/algorithms/prr_core.py:PRRState` |
| Full-vocabulary loss | `training/kdflow/algorithms/prr_v1.py:PRRV1` |
| Actual weighted AdamW update | `training/kdflow/ray/train/student_actor.py:_fit_prr` |

## Variants

Uniform OPD skips calibration and fixes all weights at 1. `--candidate adamw`
uses the old read-only optimizer-aware candidate for comparison.
`--shuffle-weights` deterministically permutes the four newly projected weights
at each valid calibration, preserving their values and sum. Its seed depends
on the training seed and completed update count. Gamma and interval can be
changed with `--gamma` and `--interval`; these changes are recorded in run metadata.

The first-order diagnostic does not guarantee a loss decrease under the later
AdamW update, reasoning correctness, or improvements on every benchmark.
