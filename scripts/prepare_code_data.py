#!/usr/bin/env python3
"""Download and verify fixed code-benchmark snapshots for local evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.json as pajson
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


ROOT = Path(os.environ.get("PRR_DATA_ROOT", Path(__file__).resolve().parents[1] / "data/evaluation"))

SOURCES: dict[str, dict[str, Any]] = {
    "HumanEvalPlus": {
        "repo": "evalplus/humanevalplus",
        "revision": "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
        "files": [
            "README.md",
            "data/test-00000-of-00001-5973903632b82d40.parquet",
            "test.jsonl",
        ],
        "parquet": "data/test-00000-of-00001-5973903632b82d40.parquet",
        "expected_rows": 164,
        "release": "EvalPlus HumanEval+ fixed test split",
    },
    "MBPPPlus": {
        "repo": "evalplus/mbppplus",
        "revision": "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
        "files": [
            "README.md",
            "data/test-00000-of-00001-d5781c9c51e02795.parquet",
        ],
        "parquet": "data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "expected_rows": 378,
        "release": "EvalPlus MBPP+ v0.2.0 test split",
    },
    "LiveCodeBench": {
        "repo": "livecodebench/code_generation_lite",
        "revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        "files": [
            "README.md",
            "code_generation_lite.py",
            "test6.jsonl",
        ],
        "release_files": ["test6.jsonl"],
        "expected_rows": 175,
        "release": "code_generation_lite v6 delta (2025-01-04 to 2025-04-06)",
    },
    "CRUXEval": {
        "repo": "cruxeval-org/cruxeval",
        "revision": "b96af0450242eb4da433032b90998f25588a5d0f",
        "files": ["README.md", "test.jsonl"],
        "jsonl": "test.jsonl",
        "expected_rows": 800,
        "release": "official test split",
    },
    "BigCodeBench-Hard": {
        "repo": "bigcode/bigcodebench-hard",
        "revision": "298d2cc7b96612e15e47313c3603ee124cee0c1f",
        "files": ["README.md", "data/v0.1.4-00000-of-00001.parquet"],
        "parquet": "data/v0.1.4-00000-of-00001.parquet",
        "expected_rows": 148,
        "release": "v0.1.4",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite a different file: {path}")
    path.write_text(text, encoding="utf-8")


def copy_exact(source: Path, target: Path) -> None:
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Existing output differs: {target}")
        return
    shutil.copy2(source, target)


def table_summary(path: Path) -> tuple[int, list[dict[str, str]]]:
    table = pq.read_table(path)
    schema = [{"name": field.name, "type": str(field.type)} for field in table.schema]
    return table.num_rows, schema


def iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def scalar_or_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_lcb_index(raw: Path, target: Path, names: list[str]) -> tuple[int, int]:
    paths = [raw / name for name in names]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    omitted = {
        "public_test_cases",
        "private_test_cases",
        "generated_test_cases",
        "test_cases",
    }
    for source, line_number, record in iter_jsonl(paths):
        question_id = str(record.get("question_id", ""))
        if not question_id:
            raise RuntimeError(f"Missing question_id in {source}:{line_number}")
        if question_id in seen:
            raise RuntimeError(f"Duplicate LiveCodeBench question_id: {question_id}")
        seen.add(question_id)
        compact = {
            key: scalar_or_json(value)
            for key, value in record.items()
            if key not in omitted
        }
        compact["source_file"] = source.name
        compact["source_line"] = line_number
        rows.append(compact)

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, target, compression="zstd")
    return len(rows), len(seen)


def prepare_dataset(name: str, spec: dict[str, Any], verify_only: bool) -> dict[str, Any]:
    target = ROOT / name
    raw = target / "raw"
    target.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    if not verify_only:
        def download(filename: str) -> None:
            if (raw / filename).is_file() and (raw / filename).stat().st_size > 0:
                return
            hf_hub_download(
                repo_id=spec["repo"],
                repo_type="dataset",
                revision=spec["revision"],
                filename=filename,
                local_dir=raw,
            )

        workers = min(6 if name == "LiveCodeBench" else 2, len(spec["files"]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(download, spec["files"]))

    missing = [filename for filename in spec["files"] if not (raw / filename).is_file()]
    if missing:
        raise RuntimeError(f"{name} is missing files: {missing}")

    rows: int
    unique_tasks: int
    schema: list[dict[str, str]]
    outputs: list[Path]

    if name == "LiveCodeBench":
        index_path = target / "index.parquet"
        files_path = target / "v6_delta_files.json"
        if not verify_only and not index_path.exists():
            rows, unique_tasks = build_lcb_index(raw, index_path, spec["release_files"])
            write_json(
                files_path,
                {
                    "config": "v6_delta",
                    "ordered_files": [str(raw / item) for item in spec["release_files"]],
                    "note": "This is the 175-problem v6 delta, not the 1,055-problem cumulative release_v6 loader config.",
                },
            )
        if not index_path.is_file() or not files_path.is_file():
            raise RuntimeError("LiveCodeBench derived index/manifest is missing")
        rows, schema = table_summary(index_path)
        unique_tasks = len(set(pq.read_table(index_path, columns=["question_id"])["question_id"].to_pylist()))
        outputs = [index_path, files_path]
    elif "parquet" in spec:
        output = target / "test.parquet"
        if not verify_only:
            copy_exact(raw / spec["parquet"], output)
        if not output.is_file():
            raise RuntimeError(f"Missing processed output: {output}")
        rows, schema = table_summary(output)
        unique_tasks = rows
        outputs = [output]
    else:
        output = target / "test.parquet"
        if not verify_only and not output.exists():
            table = pajson.read_json(raw / spec["jsonl"])
            pq.write_table(table, output, compression="zstd")
        if not output.is_file():
            raise RuntimeError(f"Missing processed output: {output}")
        rows, schema = table_summary(output)
        unique_tasks = rows
        outputs = [output]

    expected = spec.get("expected_rows")
    if expected is not None and rows != expected:
        raise RuntimeError(f"{name}: expected {expected} rows, found {rows}")

    raw_files = []
    for filename in spec["files"]:
        path = raw / filename
        raw_files.append(
            {
                "relative_path": str(path.relative_to(target)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output_files = [
        {
            "relative_path": str(path.relative_to(target)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in outputs
    ]
    manifest = {
        "dataset": name,
        "source_repo": spec["repo"],
        "source_revision": spec["revision"],
        "release": spec["release"],
        "rows": rows,
        "unique_tasks": unique_tasks,
        "schema": schema,
        "raw_files": raw_files,
        "processed_files": output_files,
        "evaluation_warning": (
            "Generated code must be executed only with the official harness in an isolated, "
            "network-disabled sandbox with strict resource limits."
        ),
    }
    write_json(target / "manifest.json", manifest)
    return manifest


def write_readme(manifests: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {item['dataset']} | {item['release']} | {item['rows']} | `{ROOT / item['dataset']}` |"
        for item in manifests
    )
    text = f"""# Code benchmark snapshots

本目录保存五套代码能力评测数据的固定官方快照。下载和处理不会执行任何候选代码。

| 数据集 | 固定版本 | 任务数 | 本地目录 |
|---|---|---:|---|
{rows}

## 文件约定

- `raw/`：固定提交下的官方原始文件，不作内容修改。
- `test.parquet`：HumanEval+、MBPP+、CRUXEval、BigCodeBench-Hard 的本地统一读取入口。
- `LiveCodeBench/v6_delta_files.json`：2025 年 v6 新增 175 题的固定加载清单。
- `LiveCodeBench/index.parquet`：不含大型隐藏测试正文的轻量任务索引；正式评分必须读取 `raw/test6.jsonl`。
- `manifest.json`：来源仓库、提交 SHA、任务数、schema、文件大小和 SHA256。

## 复现与验收

```bash
python \\
  {ROOT / 'code_benchmark_tools' / 'prepare.py'}

python \\
  {ROOT / 'code_benchmark_tools' / 'prepare.py'} --verify-only
```

后续测评须使用官方 harness。HumanEval+/MBPP+ 使用 EvalPlus；LiveCodeBench 使用固定的
`code_generation_lite v6 delta`（175题，不是累计1055题）；CRUXEval 分别报告 input prediction 和 output prediction；
BigCodeBench-Hard 使用 v0.1.4。不得把数学答案提取器用于代码评测。

候选模型生成的代码是不可信输入。执行时必须使用隔离容器，关闭网络，并限制 CPU、内存、
进程数和单题超时。准备数据本身不代表执行环境已经配置完成。
"""
    path = ROOT / "CODE_BENCHMARKS_README.md"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite a different file: {path}")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifests = [
        prepare_dataset(name, spec, args.verify_only) for name, spec in SOURCES.items()
    ]
    write_readme(manifests)
    print(json.dumps({item["dataset"]: item["rows"] for item in manifests}, indent=2))


if __name__ == "__main__":
    main()
