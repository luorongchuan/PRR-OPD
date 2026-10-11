#!/usr/bin/env python3
"""Incrementally score one completed dataset from the 14-task protocol."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics

from eval_14_pass1 import CODE, DATA_ROOT_DEFAULT, MATH, OPD_ROOT_DEFAULT, read, write_once
from score_14_pass1 import batches, jsonl, load_refs, remote_bigcode, run


ALL_DATASETS = MATH + CODE
LOCAL_CODE = ("LiveCodeBench", "CRUXEval-I", "CRUXEval-O")


def math_score(root: Path, protocol: dict, dataset: str, correct, robust_answer) -> dict:
    n_samples = protocol["generation"]["n"]
    identity = protocol["protocol_sha256"]
    rows = []
    for prediction in batches(root, protocol, dataset):
        gold = prediction["reference_answer"]
        samples = []
        for item in prediction["samples"]:
            answer = robust_answer(item["text"])
            samples.append({
                "sample_index": item["sample_index"],
                "correct": bool(correct(answer, gold)),
                "extracted_answer": answer,
                "parse_success": answer is not None,
                "finish_reason": item["finish_reason"],
                "completion_tokens": item["completion_tokens"],
            })
        correct_samples = sum(item["correct"] for item in samples)
        rows.append({
            "id": prediction["id"],
            "reference_answer": gold,
            "correct_samples": correct_samples,
            "sampled_pass_at_1": correct_samples / n_samples,
            "samples": samples,
        })

    expected = protocol["datasets"][dataset]["rows"]
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise RuntimeError(f"Wrong math score count: {dataset}")
    all_samples = [sample for row in rows for sample in row["samples"]]
    value = {
        "dataset": dataset,
        "protocol_sha256": identity,
        "total_questions": len(rows),
        "samples_per_question": n_samples,
        "correct_samples": sum(row["correct_samples"] for row in rows),
        "sampled_pass_at_1": sum(row["sampled_pass_at_1"] for row in rows) / len(rows),
        "parse_success_rate": statistics.mean(item["parse_success"] for item in all_samples),
        "truncation_rate": statistics.mean(item["finish_reason"] == "length" for item in all_samples),
        "mean_completion_tokens": statistics.mean(item["completion_tokens"] for item in all_samples),
        "metric": "sampled Pass@1 = mean_i(c_i/16)",
    }
    jsonl(root / "scores" / dataset / "per_question.jsonl", rows)
    write_once(root / "scores" / dataset / "summary.json", value)
    return value


def prepare_code_samples(root: Path, protocol: dict, dataset: str, code_common) -> list[Path]:
    n_samples = protocol["generation"]["n"]
    sample_roots = [
        root / "incremental_code_scoring" / dataset / f"sample_{index:02d}"
        for index in range(n_samples)
    ]
    rows = [[] for _ in range(n_samples)]
    for prediction in batches(root, protocol, dataset):
        for index, item in enumerate(prediction["samples"]):
            rows[index].append({
                "dataset": dataset,
                "id": prediction["id"],
                "raw_output": item["text"],
                "solution": code_common.extract_solution(dataset, item["text"]),
                "completion_tokens": item["completion_tokens"],
                "finish_reason": item["finish_reason"],
                "stop_reason": item["stop_reason"],
            })

    expected = protocol["datasets"][dataset]["rows"]
    for index, sample_root in enumerate(sample_roots):
        if len(rows[index]) != expected:
            raise RuntimeError(f"Incomplete code sample {index}: {dataset}")
        write_once(sample_root / "protocol.json", protocol)
        jsonl(sample_root / "predictions.jsonl", rows[index])
        write_once(sample_root / "generation_complete.json", {
            "protocol_sha256": protocol["protocol_sha256"],
            "status": "GENERATION_COMPLETE",
            "dataset": dataset,
            "sample_index": index,
        })
    return sample_roots


def evalplus_score(
    root: Path,
    protocol: dict,
    dataset: str,
    sample_roots: list[Path],
    official: Path,
    data_root: Path,
    workers: int,
) -> dict:
    filename = "HumanEvalPlus-v0.1.10.jsonl.gz" if dataset == "HumanEvalPlus" else "MbppPlus-v0.2.0.jsonl.gz"
    results = []
    for sample_root in sample_roots:
        score_root = sample_root / "scores" / dataset / "official"
        if not (score_root / "summary.json").exists():
            run([
                "bash", str(official / "sandbox_score_evalplus.sh"), dataset,
                str(sample_root / "predictions.jsonl"),
                str(data_root / "EvalPlusOfficial" / filename),
                str(score_root), str(workers),
            ])
        results.append(read(score_root / "summary.json"))

    total = protocol["datasets"][dataset]["rows"]
    plus_counts = [item["plus_correct"] for item in results]
    base_counts = [item["base_correct"] for item in results]
    return {
        "dataset": dataset,
        "protocol_sha256": protocol["protocol_sha256"],
        "total_questions": total,
        "samples_per_question": protocol["generation"]["n"],
        "correct_samples": sum(plus_counts),
        "sampled_pass_at_1": sum(plus_counts) / (total * len(results)),
        "scorer": "official EvalPlus plus",
        "metric": "sampled Pass@1 = mean_i(c_i/16)",
        "base_sampled_pass_at_1": sum(base_counts) / (total * len(results)),
    }


def local_code_score(
    protocol: dict,
    dataset: str,
    sample_roots: list[Path],
    legacy: Path,
    workers: int,
) -> dict:
    run(["bash", str(legacy / "prepare_scoring_env.sh")])
    results = []
    for sample_root in sample_roots:
        summary_path = sample_root / "scores" / dataset / "summary.json"
        if not summary_path.exists():
            run([
                "bash", str(legacy / "sandbox_score_one.sh"), dataset,
                str(sample_root), str(workers),
            ])
        results.append(read(summary_path))

    total = protocol["datasets"][dataset]["rows"]
    counts = [item["correct"] for item in results]
    return {
        "dataset": dataset,
        "protocol_sha256": protocol["protocol_sha256"],
        "total_questions": total,
        "samples_per_question": protocol["generation"]["n"],
        "correct_samples": sum(counts),
        "sampled_pass_at_1": sum(counts) / (total * len(results)),
        "scorer": "pinned local sandbox",
        "metric": "sampled Pass@1 = mean_i(c_i/16)",
    }


def bigcode_score(
    protocol: dict,
    dataset: str,
    sample_roots: list[Path],
    legacy: Path,
) -> dict:
    if os.environ.get("ALLOW_BIGCODE_REMOTE_UPLOAD") != "1":
        raise RuntimeError("Set ALLOW_BIGCODE_REMOTE_UPLOAD=1 to run official remote scoring")
    total = protocol["datasets"][dataset]["rows"]
    results = [remote_bigcode(sample_root, legacy, total) for sample_root in sample_roots]
    counts = [item["correct"] for item in results]
    return {
        "dataset": dataset,
        "protocol_sha256": protocol["protocol_sha256"],
        "total_questions": total,
        "samples_per_question": protocol["generation"]["n"],
        "correct_samples": sum(counts),
        "sampled_pass_at_1": sum(counts) / (total * len(results)),
        "scorer": "official BigCodeBench remote, instruct/hard, calibrated=false",
        "metric": "sampled Pass@1 = mean_i(c_i/16)",
    }


def finalize(root: Path, protocol: dict) -> dict:
    generation_marker = read(root / "generation_complete.json")
    if generation_marker.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("Generation/protocol mismatch")
    datasets = {}
    for dataset in ALL_DATASETS:
        path = root / "scores" / dataset / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"Missing dataset score: {dataset}")
        value = read(path)
        if value.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError(f"Wrong cached score provenance: {dataset}")
        datasets[dataset] = value
    value = {
        "status": "COMPLETE",
        "protocol_sha256": protocol["protocol_sha256"],
        "model_path": protocol["model_path"],
        "checkpoint": protocol["checkpoint"],
        "metric": "sampled Pass@1 (n=16)",
        "datasets": datasets,
        "macro_avg_sampled_pass_at_1": statistics.mean(
            item["sampled_pass_at_1"] for item in datasets.values()
        ),
    }
    write_once(root / "summary.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--opd-root", type=Path, default=OPD_ROOT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if bool(args.dataset) == bool(args.finalize):
        parser.error("choose exactly one of --dataset or --finalize")

    root = args.output.expanduser().resolve()
    protocol = read(root / "protocol.json")
    if args.finalize:
        value = finalize(root, protocol)
        print("RESULT", json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
        return

    dataset = args.dataset
    summary_path = root / "scores" / dataset / "summary.json"
    with (root / f".incremental_score_{dataset}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if summary_path.exists():
            value = read(summary_path)
            if value.get("protocol_sha256") != protocol["protocol_sha256"]:
                raise RuntimeError(f"Wrong cached score provenance: {dataset}")
            print("DATASET_SCORE", json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
            return

        legacy, official, _, code_common, correct, robust_answer = load_refs(
            protocol,
            args.opd_root.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
        )
        if dataset in MATH:
            value = math_score(root, protocol, dataset, correct, robust_answer)
        else:
            sample_roots = prepare_code_samples(root, protocol, dataset, code_common)
            if dataset in ("HumanEvalPlus", "MBPPPlus"):
                value = evalplus_score(
                    root, protocol, dataset, sample_roots, official,
                    args.data_root.expanduser().resolve(), args.workers,
                )
            elif dataset in LOCAL_CODE:
                value = local_code_score(protocol, dataset, sample_roots, legacy, args.workers)
            else:
                value = bigcode_score(protocol, dataset, sample_roots, legacy)
            write_once(summary_path, value)

        print("DATASET_SCORE", json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
