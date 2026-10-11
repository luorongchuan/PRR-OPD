"""Prepare and verify the six additional evaluation datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PRR_DATA_ROOT", HERE.parent / "data/evaluation"))
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

DOWNLOADS = {
    "HMMT25": {
        "revision": "6fdc4277120810ff75aa22d2d5489b91f7a262a1",
        "url": "https://huggingface.co/datasets/MathArena/hmmt_feb_2025/resolve/6fdc4277120810ff75aa22d2d5489b91f7a262a1/data/train-00000-of-00001.parquet",
        "raw": DATA_ROOT / "HMMT25/raw/train-00000-of-00001.parquet",
        "sha256": "ccca64d248b89dc0765e40b893fe4ebeab242226a4336027d95d256886a3d5f2",
        "count": 30,
    },
    "GSM8K": {
        "revision": "e53f048856ff4f594e959d75785d2c2d37b678ee",
        "url": "https://huggingface.co/datasets/openai/gsm8k/resolve/e53f048856ff4f594e959d75785d2c2d37b678ee/main/test-00000-of-00001.parquet",
        "raw": DATA_ROOT / "GSM8K/raw/test-00000-of-00001.parquet",
        "sha256": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
        "count": 1319,
    },
}

DATASETS = [
    ("AIME24", 30),
    ("AIME25", 30),
    ("HMMT25", 30),
    ("Minerva", 272),
    ("Olympiad-Bench", 675),
    ("GSM8K", 1319),
]


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"Refusing to overwrite a different artifact: {path}")
        return
    pending = path.with_name(path.name + f".pending-{os.getpid()}")
    pending.write_text(encoded)
    os.replace(pending, path)


def ensure_download(name: str, allow_download: bool) -> Path:
    item = DOWNLOADS[name]
    path = item["raw"]
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"Missing {path}; rerun with --download-missing")
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + f".pending-{os.getpid()}")
        urllib.request.urlretrieve(item["url"], pending)
        os.replace(pending, path)
    actual = file_sha(path)
    if actual != item["sha256"]:
        raise RuntimeError(f"Raw {name} SHA256 mismatch: {actual}")
    frame = pd.read_parquet(path)
    if len(frame) != item["count"]:
        raise RuntimeError(f"Raw {name} count mismatch: {len(frame)}")
    return path


def as_record(question: str, answer: str, dataset: str, sample_id: str) -> dict:
    question = str(question).strip()
    answer = str(answer).strip()
    if not question or not answer:
        raise ValueError(f"Empty question/answer in {dataset}:{sample_id}")
    return {
        "prompt": [{"content": f"{question} {INSTRUCTION}", "role": "user"}],
        "data_source": dataset,
        "ability": "math",
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {"index": sample_id, "split": "test"},
    }


def converted_frame(name: str, raw: Path) -> pd.DataFrame:
    source = pd.read_parquet(raw)
    rows = []
    if name == "HMMT25":
        required = {"problem_idx", "problem", "answer"}
        if not required.issubset(source.columns):
            raise RuntimeError(f"Unexpected HMMT25 columns: {list(source.columns)}")
        for row in source.itertuples(index=False):
            rows.append(as_record(row.problem, row.answer, name, f"HMMT25-{int(row.problem_idx)}"))
    elif name == "GSM8K":
        if not {"question", "answer"}.issubset(source.columns):
            raise RuntimeError(f"Unexpected GSM8K columns: {list(source.columns)}")
        for index, row in source.iterrows():
            if "####" not in row["answer"]:
                raise RuntimeError(f"GSM8K row {index} has no final-answer marker")
            gold = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
            rows.append(as_record(row["question"], gold, name, f"GSM8K-{index}"))
    else:  # pragma: no cover
        raise ValueError(name)
    return pd.DataFrame(rows, columns=["prompt", "data_source", "ability", "reward_model", "extra_info"])


def canonical_records(frame: pd.DataFrame) -> list[dict]:
    def canonical(value):
        if hasattr(value, "tolist"):
            return canonical(value.tolist())
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(item) for item in value]
        return value
    return [canonical(row) for row in frame.to_dict(orient="records")]


def write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = pd.read_parquet(path)
        left = canonical_records(current)
        right = canonical_records(frame)
        if left != right:
            raise RuntimeError(f"Refusing to overwrite a different dataset: {path}")
        return
    pending = path.with_name(path.name + f".pending-{os.getpid()}")
    frame.to_parquet(pending, index=False)
    os.replace(pending, path)


def validate_dataset(name: str, expected: int) -> dict:
    path = DATA_ROOT / name / "test.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"prompt", "reward_model", "extra_info"}
    if len(frame) != expected or not required.issubset(frame.columns):
        raise RuntimeError(f"Invalid {name}: rows={len(frame)} columns={list(frame.columns)}")
    ids, answers = [], []
    for index, row in frame.iterrows():
        extra = row["extra_info"]
        reward = row["reward_model"]
        ids.append(str(extra.get("index", index)))
        answers.append(str(reward.get("ground_truth", "")))
    if len(set(ids)) != expected or any(not value for value in answers):
        raise RuntimeError(f"Invalid IDs or references in {name}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha(path),
        "rows": len(frame),
        "unique_ids": len(set(ids)),
        "schema": str(pq.read_schema(path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    raw_manifest = {}
    for name in ("HMMT25", "GSM8K"):
        raw = ensure_download(name, args.download_missing)
        target = DATA_ROOT / name / "test.parquet"
        write_parquet_once(target, converted_frame(name, raw))
        raw_manifest[name] = {
            "url": DOWNLOADS[name]["url"],
            "revision": DOWNLOADS[name]["revision"],
            "path": str(raw.resolve()),
            "sha256": file_sha(raw),
        }
    datasets = {name: validate_dataset(name, count) for name, count in DATASETS}
    manifest = {
        "schema_version": 1,
        "instruction": INSTRUCTION,
        "dataset_order": [name for name, _ in DATASETS],
        "raw_sources": raw_manifest,
        "datasets": datasets,
        "total_rows_per_model": sum(row["rows"] for row in datasets.values()),
    }
    write_json_once(DATA_ROOT / "prepared_math_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
