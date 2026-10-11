"""Multi-modal field helpers (verl-inspired)."""
from typing import Iterable, Optional

import torch


def extract_multi_modal_inputs(
    multi_modal_inputs_list: Optional[Iterable[Optional[dict]]],
) -> dict:
    """Concat per-sample mm dicts into a batched dict for HF VLM forward.

    Args:
        multi_modal_inputs_list: an iterable of per-sample multi_modal_inputs
            dicts (each dict maps field name -> tensor with leading dim
            corresponding to that sample's patches/tokens). ``None`` entries
            (e.g. text-only samples) are skipped.

    Returns:
        A dict mapping field name -> tensor concatenated along ``dim=0``
        across all non-empty samples. Empty input returns ``{}``.
    """
    if not multi_modal_inputs_list:
        return {}
    collected: dict = {}
    for d in multi_modal_inputs_list:
        if not d:
            continue
        for k, v in d.items():
            if v is not None:
                collected.setdefault(k, []).append(v)
    return {k: torch.cat(vs, dim=0) for k, vs in collected.items()}
