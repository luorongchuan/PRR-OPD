"""Independent numerical checks for clipped-gradient progress retention."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('public_prr_core', ROOT / 'training/kdflow/algorithms/prr_core.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def test_gradient_candidates_match_formula_and_do_not_mutate_adamw():
    theta = torch.nn.Parameter(torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64))
    optimizer = torch.optim.AdamW([theta], lr=0.001, foreach=False, fused=False, weight_decay=0)
    theta.grad = torch.tensor([0.3, 0.1, -0.6], dtype=torch.float64)
    optimizer.step()
    gradients = torch.tensor([[0.1, 0.2, 0.3], [-0.3, 0.4, 0.2],
                              [0.1, 0.6, -0.2], [0.2, -0.1, 0.4]], dtype=torch.float64)

    def backward(region):
        optimizer.zero_grad(set_to_none=True)
        (theta * (gradients.sum(0) if region is None else gradients[region])).sum().backward()

    before, state = theta.detach().clone(), core._state_digest(optimizer)
    diagnostic = core.calibrate(optimizer, backward, candidate_mode='sgd')
    joint = gradients.sum(0)
    displacement = -0.001 * joint * min(1, 1 / (joint.norm().item() + 1e-6))
    expected = []
    for gradient in gradients:
        solo = -0.001 * gradient * min(1, 1 / (gradient.norm().item() + 1e-6))
        expected.append((gradient.dot(displacement) / gradient.dot(solo)).item())
    assert diagnostic['retentions'] == pytest.approx(expected)
    assert core._state_digest(optimizer) == state
    torch.testing.assert_close(theta, before, rtol=0, atol=0)


@pytest.mark.parametrize('gamma', [0, 1, 2, 4])
def test_projected_weights(gamma):
    weights = core.feedback_weights([0.1, 0.4, 0.8, 1.3], gamma=gamma)
    assert core.valid_weights(weights)
    if gamma == 0:
        assert weights == pytest.approx([1, 1, 1, 1])


def test_undefined_retention_keeps_previous_weights():
    state = core.PRRState(interval=10)
    state.refresh([0.1, 0.4, 0.8, 1.3])
    previous = state.weights[:]
    state.completed_steps = 10
    state.refresh([None, 1, 1, 1])
    assert state.weights == previous


def test_uniform_response_normalization():
    mask = torch.tensor([[False, True, True, True, True, False], [True, True, False, False, False, False]])
    assert core.response_token_weights(mask).sum().item() == pytest.approx(2)
    total = sum(core.response_token_weights(mask, region=k) for k in range(4))
    torch.testing.assert_close(total, core.response_token_weights(mask))
