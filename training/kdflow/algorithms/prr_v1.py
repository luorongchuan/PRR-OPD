"""Exact clean-v1 OPD/PRR-V1 loss on KDFlow's fixed rollout microbatches."""

import torch
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.prr_core import UNIFORM, response_token_weights
from kdflow.loss.chunked_loss import chunked_loss


def reverse_kl(student_logits, teacher_logits, reduction="none"):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Exact PRR requires identical student/teacher vocabulary shapes")
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    loss = (student.exp() * (student - teacher)).sum(dim=-1)
    if reduction != "none":
        raise ValueError("PRR requires per-token KL")
    return loss


def entropy_aware_reverse_kl(
    student_logits,
    teacher_logits,
    reduction="none",
    entropy_threshold=0.8,
    soft_kd_coef=1.0,
    force_all=False,
):
    """Full-vocabulary counterpart of MCD's Entropy-Aware OPD reward."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Exact Entropy-Aware OPD requires identical vocabulary shapes")
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_prob = student.exp()
    teacher_prob = teacher.exp()

    rkl = (student_prob * (student - teacher)).sum(dim=-1)
    teacher_entropy = -(teacher_prob * teacher).sum(dim=-1)
    if force_all:
        high_entropy = torch.ones_like(teacher_entropy)
    else:
        high_entropy = (teacher_entropy >= float(entropy_threshold)).to(rkl.dtype)

    # MCD adds p_teacher(v) to the sampled-token reward. Its exact
    # expectation under p_student gives this full-vocabulary overlap term.
    teacher_overlap = (student_prob * teacher_prob).sum(dim=-1)
    loss = rkl - float(soft_kd_coef) * high_entropy * teacher_overlap
    if reduction != "none":
        raise ValueError("Entropy-Aware OPD requires per-token losses")
    return loss


@register_algorithm("prr_v1")
class PRRV1:
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        if isinstance(teacher_lm_head, dict):
            raise ValueError("PRR-V1 currently supports exactly one teacher")
        if self.args.kd.kd_temperature != 1.0 or self.args.kd.kd_ratio != 1.0:
            raise ValueError("PRR-V1 requires temperature=1 and kd_ratio=1")
        if self.args.model.ring_attn_size != 1:
            raise ValueError("PRR-V1 does not support ring attention")
        if self.student.model.config.vocab_size != self.teacher_lm_head.out_features:
            raise ValueError("Exact PRR requires the same full vocabulary for student and teacher")

    def region_loss(self, micro_batch, region=None, weights=UNIFORM):
        student_mask = micro_batch["stu_loss_mask"].bool()
        teacher_mask = micro_batch["tea_loss_mask"].bool()
        student_labels = micro_batch["stu_input_ids"].roll(shifts=-1, dims=1)[student_mask]
        teacher_labels = micro_batch["tea_input_ids"].roll(shifts=-1, dims=1)[teacher_mask]
        if not torch.equal(student_labels, teacher_labels):
            raise ValueError("PRR requires exactly aligned student and teacher response tokens")
        coefficients = response_token_weights(student_mask, weights, region)
        sample_weights = micro_batch.get("_sample_weights")
        if sample_weights is not None:
            sample_weights = torch.as_tensor(
                sample_weights, dtype=coefficients.dtype, device=coefficients.device
            )
            if sample_weights.ndim != 1 or sample_weights.numel() != student_mask.shape[0]:
                raise ValueError("PRR sample weights must match the microbatch size")
            if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
                raise ValueError("PRR sample weights must be finite and non-negative")
            token_sample_weights = torch.repeat_interleave(sample_weights, student_mask.sum(dim=1))
            coefficients = coefficients * token_sample_weights
        teacher_hiddens = micro_batch.get("teacher_hiddens")
        if teacher_hiddens is None or teacher_hiddens.shape[0] != coefficients.numel():
            raise ValueError("Teacher hidden states do not match response tokens")
        student_output = self.student(
            micro_batch["stu_input_ids"],
            attention_mask=micro_batch["stu_attn_mask"],
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **(micro_batch.get("stu_multi_modal_inputs") or {}),
        )
        student_hiddens = student_output["hidden_states"][-1][student_mask]
        del student_output
        if student_hiddens.shape[0] != coefficients.numel():
            raise ValueError("Student hidden states do not match response tokens")
        loss_fn = reverse_kl
        loss_kwargs = {}
        kd_args = getattr(self.args, "kd", None)
        if getattr(kd_args, "prr_method", "opd") == "entropy_opd":
            loss_fn = entropy_aware_reverse_kl
            loss_kwargs = {
                "entropy_threshold": self.args.kd.eopd_entropy_threshold,
                "soft_kd_coef": self.args.kd.eopd_soft_kd_coef,
                "force_all": self.args.kd.eopd_force_all,
            }
        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        token_losses = chunked_loss(
            student_hiddens,
            self.student.model.lm_head,
            loss_fn,
            teacher_hidden=teacher_hiddens,
            teacher_head=self.teacher_lm_head,
            chunk_size=self.args.train.chunked_loss_size or student_hiddens.shape[0],
            reduction="none",
            **loss_kwargs,
        )
        return (token_losses * coefficients.to(token_losses.device)).sum()

    def training_step(self, micro_batch):
        loss = self.region_loss(micro_batch)
        loss = loss / micro_batch["stu_loss_mask"].shape[0]
        return {"train/loss": loss, "train/kd_loss": loss}
