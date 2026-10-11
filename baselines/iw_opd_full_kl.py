"""IW-OPD token weighting adapted to KDFlow's full-vocabulary reverse KL."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.prr_core import UNIFORM, response_token_weights
from kdflow.loss.chunked_loss import chunked_loss


EPS = 1e-8


def reverse_kl(student_logits, teacher_logits, reduction="none"):
    """The unchanged full-vocabulary reverse KL used by the OPD baseline."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("IW-OPD requires identical student/teacher vocabularies")
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    loss = (student.exp() * (student - teacher)).sum(dim=-1)
    if reduction != "none":
        raise ValueError("IW-OPD requires per-token reverse KL")
    return loss


def iw_token_weights(
    sampled_discrepancy: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 0.5,
    normalization: str = "none",
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached full-shape IW weights and per-response fallback flags."""
    if sampled_discrepancy.shape != response_mask.shape:
        raise ValueError("sampled_discrepancy and response_mask must have equal shapes")
    if response_mask.ndim != 2 or response_mask.dtype != torch.bool:
        raise ValueError("response_mask must be a [batch, sequence] boolean tensor")
    if not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if normalization not in ("none", "mean1"):
        raise ValueError("normalization must be 'none' or 'mean1'")
    if eps <= 0:
        raise ValueError("eps must be positive")

    discrepancy = sampled_discrepancy.detach().float().abs()
    weights = torch.ones_like(discrepancy)
    zero_denom = torch.zeros(
        response_mask.shape[0], dtype=torch.bool, device=response_mask.device
    )
    with torch.no_grad():
        for row in range(response_mask.shape[0]):
            positions = response_mask[row].nonzero(as_tuple=True)[0]
            length = positions.numel()
            if length <= 1:
                zero_denom[row] = True
                continue
            values = discrepancy[row, positions]
            denom = values[:-1].sum()
            if not torch.isfinite(denom) or denom <= eps:
                zero_denom[row] = True
                continue
            prefix = torch.zeros_like(values)
            prefix[1:] = torch.cumsum(values[:-1], dim=0)
            current = 1.0 + float(gamma) * (1.0 - prefix / denom)
            if normalization == "mean1":
                current = current / current.mean()
            weights[row, positions] = current
    return weights.detach(), zero_denom.detach()


def _quartile_means(weights, response_mask, active_samples):
    buckets = [[] for _ in range(4)]
    for row in range(response_mask.shape[0]):
        if not bool(active_samples[row]):
            continue
        positions = response_mask[row].nonzero(as_tuple=True)[0]
        length = positions.numel()
        bounds = [k * length // 4 for k in range(5)]
        for region in range(4):
            selected = positions[bounds[region] : bounds[region + 1]]
            if selected.numel():
                buckets[region].append(weights[row, selected])
    return [
        float(torch.cat(values).mean()) if values else float("nan")
        for values in buckets
    ]


def iw_statistics(
    weights,
    sampled_discrepancy,
    response_mask,
    zero_denom,
    sample_weights,
    weighted_rkl,
    unweighted_rkl,
    gamma,
):
    """Compute detached diagnostics over real, non-padding responses."""
    active_samples = sample_weights.detach().to(response_mask.device) > 0
    active_mask = response_mask & active_samples[:, None]
    valid_weights = weights[active_mask]
    valid_discrepancy = sampled_discrepancy.detach().abs()[active_mask]
    if valid_weights.numel() == 0:
        raise ValueError("IW diagnostics require at least one real response token")

    first, last, discrepancy_sums = [], [], []
    for row in range(response_mask.shape[0]):
        if not bool(active_samples[row]):
            continue
        positions = response_mask[row].nonzero(as_tuple=True)[0]
        first.append(weights[row, positions[0]])
        last.append(weights[row, positions[-1]])
        discrepancy_sums.append(
            sampled_discrepancy[row, positions].detach().abs().sum()
        )
    quartiles = _quartile_means(weights, response_mask, active_samples)
    real_count = max(int(active_samples.sum()), 1)
    metrics = {
        "iw/gamma": float(gamma),
        "iw/weight_mean": float(valid_weights.mean()),
        "iw/weight_min": float(valid_weights.min()),
        "iw/weight_max": float(valid_weights.max()),
        "iw/first_token_weight_mean": float(torch.stack(first).mean()),
        "iw/last_token_weight_mean": float(torch.stack(last).mean()),
        "iw/discrepancy_abs_mean": float(valid_discrepancy.mean()),
        "iw/discrepancy_abs_sum_mean": float(torch.stack(discrepancy_sums).mean()),
        "iw/zero_denom_count": float((zero_denom & active_samples).sum()),
        "iw/valid_token_count": float(active_mask.sum()),
        "iw/weighted_rkl": float(weighted_rkl.detach()) / real_count,
        "iw/unweighted_rkl": float(unweighted_rkl.detach()) / real_count,
    }
    metrics.update(
        {f"iw/weight_r{k + 1}_mean": value for k, value in enumerate(quartiles)}
    )
    return metrics


@register_algorithm("prr_v1")
class IWOPDFullKL:
    """Drop-in PRR-V1 registration used only when IW_OPD_ENABLE=1."""

    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        del kwargs
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.gamma = float(os.environ.get("IW_GAMMA", "0.5"))
        self.normalization = os.environ.get("IW_NORMALIZATION", "none")
        self.eps = float(os.environ.get("IW_EPS", str(EPS)))
        self.metric_call = 0

        if isinstance(teacher_lm_head, dict):
            raise ValueError("IW-OPD supports exactly one teacher")
        if self.args.kd.prr_method != "opd":
            raise ValueError("IW-OPD bridge requires --prr_method opd")
        if self.args.kd.kd_temperature != 1.0 or self.args.kd.kd_ratio != 1.0:
            raise ValueError("IW-OPD requires temperature=1 and kd_ratio=1")
        if self.args.model.ring_attn_size != 1:
            raise ValueError("IW-OPD does not support ring attention")
        if self.student.model.config.vocab_size != self.teacher_lm_head.out_features:
            raise ValueError("IW-OPD requires the same full vocabulary")
        if self.normalization not in ("none", "mean1"):
            raise ValueError("IW_NORMALIZATION must be none or mean1")

    @torch.no_grad()
    def _teacher_sampled_log_probs(self, teacher_hiddens, labels):
        values = []
        chunk_size = self.args.train.chunked_loss_size or teacher_hiddens.shape[0]
        for start in range(0, teacher_hiddens.shape[0], chunk_size):
            end = min(start + chunk_size, teacher_hiddens.shape[0])
            logits = self.teacher_lm_head(teacher_hiddens[start:end])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            values.append(
                log_probs.gather(-1, labels[start:end, None]).squeeze(-1)
            )
        return torch.cat(values).detach()

    def _record_metrics(self, metrics):
        self.metric_call += 1
        rank = int(os.environ.get("RANK", "0"))
        record = {"microbatch_call": self.metric_call, "rank": rank, **metrics}
        root = Path(self.args.ckpt.save_path) / "iw_metrics"
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"rank_{rank}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.metric_call == 1:
            summary = " ".join(
                f"{key}={value:.6g}" for key, value in metrics.items()
            )
            self.strategy.log(f"IW_SANITY {summary}")

    def region_loss(self, micro_batch, region=None, weights=UNIFORM):
        if region is not None or tuple(weights) != UNIFORM:
            raise ValueError("IW-OPD uses token weights, not PRR region weights")
        student_mask = micro_batch["stu_loss_mask"].bool()
        teacher_mask = micro_batch["tea_loss_mask"].bool()
        student_labels = micro_batch["stu_input_ids"].roll(
            shifts=-1, dims=1
        )[student_mask]
        teacher_labels = micro_batch["tea_input_ids"].roll(
            shifts=-1, dims=1
        )[teacher_mask]
        if not torch.equal(student_labels, teacher_labels):
            raise ValueError(
                "IW-OPD requires aligned student and teacher response tokens"
            )

        sample_weights = torch.as_tensor(
            micro_batch.get(
                "_sample_weights", torch.ones(student_mask.shape[0])
            ),
            dtype=torch.float32,
            device=student_mask.device,
        )
        if (
            sample_weights.ndim != 1
            or sample_weights.numel() != student_mask.shape[0]
        ):
            raise ValueError("IW sample weights must match the microbatch size")

        teacher_hiddens = micro_batch.get("teacher_hiddens")
        if (
            teacher_hiddens is None
            or teacher_hiddens.shape[0] != student_labels.numel()
        ):
            raise ValueError("Teacher hidden states do not match response tokens")
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)

        rollout_log_probs = micro_batch.get("rollout_log_probs")
        if rollout_log_probs is None:
            raise ValueError(
                "IW-OPD requires rollout-time student log probabilities"
            )
        rollout_sampled = rollout_log_probs[student_mask].detach().float()
        teacher_sampled = self._teacher_sampled_log_probs(
            teacher_hiddens, student_labels
        )
        if rollout_sampled.shape != teacher_sampled.shape:
            raise ValueError(
                "Rollout and teacher sampled log probabilities are misaligned"
            )

        discrepancy_full = torch.zeros_like(
            rollout_log_probs, dtype=torch.float32
        )
        discrepancy_full[student_mask] = teacher_sampled - rollout_sampled
        iw_full, zero_denom = iw_token_weights(
            discrepancy_full,
            student_mask,
            gamma=self.gamma,
            normalization=self.normalization,
            eps=self.eps,
        )

        base_coefficients = response_token_weights(student_mask)
        token_sample_weights = torch.repeat_interleave(
            sample_weights, student_mask.sum(dim=1)
        )
        base_coefficients = base_coefficients * token_sample_weights
        iw_flat = iw_full[student_mask]

        student_output = self.student(
            micro_batch["stu_input_ids"],
            attention_mask=micro_batch["stu_attn_mask"],
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **(micro_batch.get("stu_multi_modal_inputs") or {}),
        )
        student_hiddens = student_output["hidden_states"][-1][student_mask]
        del student_output
        token_losses = chunked_loss(
            student_hiddens,
            self.student.model.lm_head,
            reverse_kl,
            teacher_hidden=teacher_hiddens,
            teacher_head=self.teacher_lm_head,
            chunk_size=(
                self.args.train.chunked_loss_size
                or student_hiddens.shape[0]
            ),
            reduction="none",
        )
        coefficients = base_coefficients.to(token_losses.device)
        unweighted = (token_losses * coefficients).sum()
        weighted = (
            token_losses
            * coefficients
            * iw_flat.to(token_losses.device)
        ).sum()

        metrics = iw_statistics(
            iw_full,
            discrepancy_full,
            student_mask,
            zero_denom,
            sample_weights,
            weighted,
            unweighted,
            self.gamma,
        )
        self._record_metrics(metrics)
        return weighted

    def training_step(self, micro_batch):
        loss = self.region_loss(micro_batch)
        loss = loss / micro_batch["stu_loss_mask"].shape[0]
        return {"train/loss": loss, "train/kd_loss": loss}
