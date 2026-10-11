#!/usr/bin/env python3
"""Score one completed local code dataset inside the existing sandbox."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from common import EXPECTED_ROWS, atomic_json, atomic_jsonl, digest, load_jsonl, read_json
from score import parallel_score, score_crux, score_lcb, summary


SUPPORTED = ("LiveCodeBench", "CRUXEval-I", "CRUXEval-O")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/datasets"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = args.result_root.resolve()
    protocol = read_json(root / "protocol.json")
    protocol_hash = protocol.get("protocol_sha256") or digest(protocol)
    generated = read_json(root / "generation_complete.json")
    if generated.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Generation/protocol mismatch")

    predictions = load_jsonl(root / "predictions.jsonl")
    if len(predictions) != EXPECTED_ROWS[args.dataset]:
        raise RuntimeError(f"Wrong prediction count for {args.dataset}")
    if any(row.get("dataset") != args.dataset for row in predictions):
        raise RuntimeError(f"Mixed datasets in {root / 'predictions.jsonl'}")

    data_root = args.data_root
    if args.dataset == "LiveCodeBench":
        source = load_jsonl(data_root / "LiveCodeBench/raw/test6.jsonl")
        expected_ids = [str(row["question_id"]) for row in source]
        tasks = list(zip(source, predictions))
        scorer = score_lcb
    else:
        source = pq.read_table(data_root / "CRUXEval/test.parquet").to_pylist()
        expected_ids = [str(row["id"]) for row in source]
        mode = "input" if args.dataset == "CRUXEval-I" else "output"
        tasks = [(mode, row, prediction) for row, prediction in zip(source, predictions)]
        scorer = score_crux

    if [str(row["id"]) for row in predictions] != expected_ids:
        raise RuntimeError(f"Prediction/source order mismatch for {args.dataset}")

    detail_path = root / "scores" / args.dataset / "per_task.jsonl"
    summary_path = root / "scores" / args.dataset / "summary.json"
    if detail_path.exists() and summary_path.exists():
        results = load_jsonl(detail_path)
        value = read_json(summary_path)
    else:
        print(
            f"SCORE_START {args.dataset} tasks={len(tasks)} workers={args.workers}",
            flush=True,
        )
        results = parallel_score(scorer, tasks, args.workers)
        if len(results) != EXPECTED_ROWS[args.dataset]:
            raise RuntimeError(f"Incomplete scores for {args.dataset}")
        if [str(row["id"]) for row in results] != expected_ids:
            raise RuntimeError(f"Reordered scores for {args.dataset}")
        value = summary(args.dataset, results)
        atomic_jsonl(detail_path, results)
        atomic_json(summary_path, value)

    print("SCORE_RESULT", json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
