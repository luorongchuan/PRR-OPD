"""Shared helpers for the corrected, official EvalPlus validation."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
TEST_CODE = HERE.parent
if str(TEST_CODE) not in sys.path:
    sys.path.insert(0, str(TEST_CODE))

import common as legacy  # noqa: E402


DATA_ROOT = Path(os.environ.get("PRR_CODE_DATA_ROOT", Path(os.environ.get("PRR_DATA_ROOT", "data/evaluation")))) / "EvalPlusOfficial"
OFFICIAL_DATA = {
    "HumanEvalPlus": DATA_ROOT / "HumanEvalPlus-v0.1.10.jsonl.gz",
    "MBPPPlus": DATA_ROOT / "MbppPlus-v0.2.0.jsonl.gz",
}
EXPECTED_ROWS = {"HumanEvalPlus": 164, "MBPPPlus": 378}
LEGACY_RESULTS = {
    "opd": TEST_CODE / "results" / "opd_step551" / "353ea3318bfb30ce",
    "prr": TEST_CODE / "results" / "prr_step551" / "b29c98cdef0e0a58",
}
RESULTS_ROOT = HERE / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    legacy.atomic_jsonl(path, rows)


def official_rows(dataset: str) -> list[dict[str, Any]]:
    path = OFFICIAL_DATA[dataset]
    if not path.is_file():
        raise FileNotFoundError(f"Missing official EvalPlus release: {path}")
    rows = load_jsonl_gz(path)
    if len(rows) != EXPECTED_ROWS[dataset]:
        raise RuntimeError(f"Unexpected {dataset} row count: {len(rows)}")
    ids = [str(row["task_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate task ids in {dataset}")
    required = {
        "task_id", "prompt", "entry_point", "canonical_solution",
        "base_input", "plus_input", "atol",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"{dataset} {row.get('task_id')} is missing {sorted(missing)}")
    return rows


def build_evalplus_records(tokenizer, dataset: str) -> list[dict[str, Any]]:
    records = []
    for row in official_rows(dataset):
        prompt = row["prompt"].strip()
        entry_point = row["entry_point"]
        if entry_point not in prompt:
            raise RuntimeError(f"Official prompt does not expose entry point: {row['task_id']}")
        if dataset == "MBPPPlus" and "assert" not in prompt:
            raise RuntimeError(f"Official MBPP+ prompt has no signature-bearing assertion: {row['task_id']}")
        user = f"{legacy.INSTRUCTION_PREFIX}\n```\n{prompt}\n```"
        records.append(
            {
                "dataset": dataset,
                "id": str(row["task_id"]),
                "prompt": legacy.chat_prompt(tokenizer, user, prefill=legacy.RESPONSE_PREFIX),
                "stop": "code_fence",
                "entry_point": entry_point,
            }
        )
    return records
