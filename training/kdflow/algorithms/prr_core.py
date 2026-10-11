"""Numerical core of the clean-v1 four-region PRR-V1 update.

The caller owns fixed rollout tokens, full-vocabulary reverse-KL forwards and
microbatch accumulation. Candidate AdamW displacements read, but never change,
the current optimizer state. Distributed sharded candidates are not supported.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


UNIFORM = (1.0, 1.0, 1.0, 1.0)


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _include_local_shard(tensor):
    if isinstance(tensor, DTensor):
        if any(not isinstance(placement, Replicate) for placement in tensor.placements):
            return True
        return tensor.device_mesh.get_local_rank() == 0
    if _distributed():
        raise RuntimeError("Multi-rank PRR requires DTensor gradients")
    return True


def _global_sum(value):
    if _distributed():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def response_token_weights(mask, weights=UNIFORM, region=None):
    """Return flattened token coefficients; each response has equal total mass."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("Expected a [batch, tokens] boolean response mask")
    if region is not None and region not in range(4):
        raise ValueError("Invalid PRR region")
    if not valid_weights(weights):
        raise ValueError("Invalid PRR weights")
    coefficients = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    for row in range(mask.shape[0]):
        positions = mask[row].nonzero(as_tuple=True)[0]
        length = positions.numel()
        if not length:
            raise ValueError("Every response must contain at least one valid token")
        bounds = [k * length // 4 for k in range(5)]
        for k in range(4):
            if region is None or region == k:
                coefficients[row, positions[bounds[k]:bounds[k + 1]]] = (1.0 if region is not None else weights[k]) / length
    return coefficients[mask]


def valid_weights(weights):
    return (len(weights) == 4 and all(math.isfinite(w) and 0.5 - 1e-10 <= w <= 2.0 + 1e-10 for w in weights)
            and abs(sum(weights) - 4.0) <= 1e-10)


def feedback_weights(retentions, gamma=2.0):
    if len(retentions) != 4 or not all(math.isfinite(x) for x in retentions):
        raise ValueError("Invalid retention rates")
    mean = sum(retentions) / 4
    values = [math.exp(gamma * (mean - x)) for x in retentions]
    left = min(value - 2.0 for value in values)
    right = max(value - 0.5 for value in values)
    for _ in range(100):
        middle = (left + right) / 2
        if sum(min(2.0, max(0.5, value - middle)) for value in values) > 4.0:
            left = middle
        else:
            right = middle
    result = [min(2.0, max(0.5, value - (left + right) / 2)) for value in values]
    if not valid_weights(result):
        raise RuntimeError("PRR projection did not converge")
    return result


@dataclass
class PRRState:
    weights: list = field(default_factory=lambda: list(UNIFORM))
    last_valid_retention: list = field(default_factory=list)
    last_calibration_step: int = -1
    fallback_reason: str = ""
    completed_steps: int = 0
    gamma: float = 2.0
    interval: int = 10
    feedback_mode: str = "aligned"
    feedback_seed: int = 42

    def due(self, interval=None):
        interval = self.interval if interval is None else interval
        return self.completed_steps % interval == 0 and self.completed_steps != self.last_calibration_step

    def refresh(self, retentions, interval=None):
        if not self.due(interval):
            raise ValueError("Calibration is not due")
        self.fallback_reason = ""
        if len(retentions) != 4 or any(x is None or not math.isfinite(x) for x in retentions):
            self.fallback_reason = "undefined_retention_keep_previous_or_initial_uniform"
        else:
            try:
                candidate = feedback_weights(retentions, gamma=self.gamma)
                if self.feedback_mode == "shuffled":
                    random.Random(self.feedback_seed + self.completed_steps).shuffle(candidate)
                elif self.feedback_mode != "aligned":
                    raise RuntimeError("Unknown feedback assignment")
                if not valid_weights(candidate):
                    raise ValueError("Invalid projected weights")
                self.weights = candidate
                self.last_valid_retention = list(retentions)
            except (OverflowError, ValueError, FloatingPointError):
                self.fallback_reason = "invalid_weight_formula_keep_previous_or_initial_uniform"
        if not valid_weights(self.weights):
            raise RuntimeError("Corrupt PRR state")
        self.last_calibration_step = self.completed_steps


def _parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _gradients(optimizer):
    gradients = [parameter.grad for parameter in _parameters(optimizer)]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("PRR requires a gradient for every trainable parameter")
    return gradients


@torch.no_grad()
def clipped_gradients(gradients, max_norm=1.0):
    if not gradients:
        raise ValueError("Cannot clip an empty gradient list")
    copies = [_local(gradient).detach().clone() for gradient in gradients]
    norm_sq = torch.zeros((), dtype=torch.float64, device=copies[0].device)
    for original, gradient in zip(gradients, copies, strict=True):
        if _include_local_shard(original):
            norm_sq.add_(gradient.detach().double().square().sum())
    norm = _global_sum(norm_sq).sqrt()
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite global gradient norm")
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for gradient in copies:
        gradient.mul_(scale.to(gradient.device))
    return copies, float(norm), float(scale)


@torch.no_grad()
def adamw_candidate(optimizer, gradients):
    """Return local theta_after - theta_before from one read-only AdamW step."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("PRR-V1 requires torch.optim.AdamW")
    parameters = _parameters(optimizer)
    if len(gradients) != len(parameters):
        raise ValueError("Gradient list does not match optimizer")
    output = []
    index = 0
    for group in optimizer.param_groups:
        if group.get("foreach") is not False or group.get("fused") is not False:
            raise ValueError("PRR-V1 requires AdamW(foreach=False, fused=False)")
        if group.get("capturable", False) or group.get("differentiable", False):
            raise ValueError("Unsupported AdamW execution mode")
        beta1, beta2 = group["betas"]
        lr, eps, decay = float(group["lr"]), float(group["eps"]), float(group["weight_decay"])
        for parameter in group["params"]:
            value = _local(parameter).detach()
            gradient = gradients[index]
            index += 1
            if gradient is None or gradient.shape != value.shape or not torch.isfinite(gradient).all():
                raise ValueError("Candidate requires finite explicit gradients")
            state = optimizer.state.get(parameter, {})
            step = float(_local(state.get("step", 0))) + 1
            first = state.get("exp_avg")
            second = state.get("exp_avg_sq")
            first = torch.zeros_like(value) if first is None else _local(first)
            second = torch.zeros_like(value) if second is None else _local(second)
            grad = gradient.to(value.dtype)
            if group.get("maximize", False):
                grad = -grad
            moment = first.to(value.dtype).mul(beta1).add(grad, alpha=1 - beta1)
            variance = second.to(value.dtype).mul(beta2).addcmul(grad, grad, value=1 - beta2)
            if group.get("amsgrad", False):
                previous = state.get("max_exp_avg_sq")
                variance = variance if previous is None else torch.maximum(_local(previous).to(value.dtype), variance)
            denominator = variance.sqrt().div(math.sqrt(1 - beta2**step)).add(eps)
            displacement = moment.div(denominator).mul(-lr / (1 - beta1**step))
            if decay:
                displacement = displacement.add(value, alpha=-lr * decay)
            output.append(displacement)
    return output


def _dot(left, right):
    if not left:
        raise ValueError("Cannot dot empty tensor lists")
    total = torch.zeros((), dtype=torch.float64, device=_local(left[0]).device)
    for original, other in zip(left, right, strict=True):
        local = _local(original)
        if local.shape != other.shape:
            raise ValueError("PRR dot-product shard shapes do not match")
        if _include_local_shard(original):
            total.add_((local.detach().double() * other.detach().double()).sum())
    return float(_global_sum(total))


def _state_digest(optimizer):
    state = optimizer.state_dict()
    digest = hashlib.sha256(json.dumps(state["param_groups"], sort_keys=True, default=str).encode())
    for key, values in sorted(state["state"].items()):
        digest.update(str(key).encode())
        for name, value in sorted(values.items()):
            digest.update(name.encode())
            if torch.is_tensor(value):
                local = _local(value).detach().cpu().contiguous()
                digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def calibrate(optimizer, backward_pass, max_norm=1.0, candidate_mode="adamw"):
    """Run five backward passes; backward_pass(region) must use fixed tokens."""
    parameters = _parameters(optimizer)
    versions = tuple(parameter._version for parameter in parameters)
    before = _state_digest(optimizer)
    def candidate(gradients):
        if candidate_mode == "adamw":
            return adamw_candidate(optimizer, gradients)
        if candidate_mode != "sgd":
            raise ValueError("Unknown candidate mode")
        # Only the diagnostic changes; the actual update still uses AdamW.
        lrs = [float(group["lr"]) for group in optimizer.param_groups for _ in group["params"]]
        return [gradient.mul(-lr) for gradient, lr in zip(gradients, lrs, strict=True)]
    try:
        backward_pass(None)
        full = _gradients(optimizer)
        clipped, full_norm, scale = clipped_gradients(full, max_norm)
        displacement = candidate(clipped)
        full_progress = -_dot(full, displacement)
        del full, clipped
        progress, solo, norms = [], [], []
        for region in range(4):
            backward_pass(region)
            gradients = _gradients(optimizer)
            clipped, norm, _ = clipped_gradients(gradients, max_norm)
            progress.append(-_dot(gradients, displacement))
            solo.append(-_dot(gradients, candidate(clipped)))
            norms.append(norm)
            del gradients, clipped
    finally:
        optimizer.zero_grad(set_to_none=True)
    if tuple(parameter._version for parameter in parameters) != versions or _state_digest(optimizer) != before:
        raise RuntimeError("PRR calibration mutated parameters or optimizer state")
    retentions = [p / s if s > 0 else None for p, s in zip(progress, solo, strict=True)]
    return {"retentions": retentions, "progress": progress, "solo_progress": solo,
            "region_grad_norms": norms, "uniform_progress": full_progress,
            "uniform_preclip_norm": full_norm, "uniform_clip_scale": scale}
