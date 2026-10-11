"""Conditional registration bridge for the isolated IW-OPD experiment."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path


if os.environ.get("IW_OPD_ENABLE") == "1":
    # Stock OPD registers first. The override exists only in this opt-in process.
    importlib.import_module("kdflow.algorithms.prr_v1")
    source = os.environ.get("IW_OPD_IMPL")
    if not source:
        raise RuntimeError("IW_OPD_IMPL must point to iw_opd_full_kl.py")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "_isolated_iw_opd_full_kl"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load IW-OPD implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
