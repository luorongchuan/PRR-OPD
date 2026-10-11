"""Preserve Qwen3 default RoPE when reading Transformers 5 exports with 4.x."""
import json
import math
from pathlib import Path


def qwen3_hf_overrides(model_path):
    raw = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return qwen3_rope_overrides(raw)


def qwen3_rope_overrides(raw):
    if raw.get("model_type") != "qwen3" or "rope_parameters" not in raw:
        return {}
    params = raw["rope_parameters"]
    if not isinstance(params, dict) or params.get("rope_type") != "default":
        raise ValueError("Unsupported Qwen3 rope_parameters; refusing silent legacy RoPE fallback")
    if set(params) - {"rope_type", "rope_theta"}:
        raise ValueError("Unrecognized Qwen3 RoPE fields; explicit compatibility handling required")
    theta = params.get("rope_theta")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not math.isfinite(theta) or theta <= 0:
        raise ValueError("Qwen3 rope_parameters.rope_theta must be a finite positive number")
    if "rope_theta" in raw and raw["rope_theta"] != theta:
        raise ValueError("Conflicting Qwen3 top-level and nested rope_theta")
    if raw.get("rope_scaling") is not None:
        raise ValueError("Conflicting Qwen3 rope_scaling and default rope_parameters")
    return {"rope_theta": theta, "rope_scaling": None}


def verify_loaded_rope(config, overrides):
    for name, expected in overrides.items():
        if not hasattr(config, name) or getattr(config, name) != expected:
            raise RuntimeError(f"Loaded model config does not honor {name}={expected!r}")
