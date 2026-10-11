#!/usr/bin/env python3
"""Score the frozen 14-dataset PRR/OPD validation generations with sampled Pass@1.

Adapted from OPD-Math-COLING/experiments/prr_clean_v1/sampling_pass_at_1/score.py.
The metric is exactly sampled Pass@1: for each problem c_i/16, then average.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from eval_14_pass1 import CODE, DATA_ROOT_DEFAULT, MATH, OPD_ROOT_DEFAULT, digest, read, setup_reference_paths, write_once  # noqa: E402


def jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(handle.name)
    if path.exists():
        if path.read_bytes() != temp.read_bytes():
            temp.unlink()
            raise RuntimeError(f"Refusing to replace different predictions: {path}")
        temp.unlink()
    else:
        os.replace(temp, path)


def load_refs(protocol: dict, opd_root: Path, data_root: Path):
    legacy, official, other, code_common, official_data, build_evalplus_records = setup_reference_paths(opd_root, data_root)
    math_dir = opd_root / "math_scoring"
    if not (math_dir / "scoring.py").is_file():
        raise FileNotFoundError(f"Missing vendored math scorer: {math_dir / 'scoring.py'}")
    sys.path.insert(0, str(opd_root))
    from math_scoring.scoring import correct, robust_answer  # noqa: PLC0415
    return legacy, official, other, code_common, correct, robust_answer


def batches(root: Path, protocol: dict, dataset: str):
    count = protocol["datasets"][dataset]["rows"]
    size = protocol["generation"]["request_batch_size"]
    n_samples = protocol["generation"]["n"]
    identity = protocol["protocol_sha256"]
    seen = 0
    for start in range(0, count, size):
        path = root / "generation" / dataset / f"batch_{start:06d}.json"
        batch = read(path)
        if batch.get("protocol_sha256") != identity or batch.get("start") != start:
            raise RuntimeError(f"Generation provenance mismatch: {path}")
        predictions = batch["predictions"]
        if len(predictions) != min(size, count - start):
            raise RuntimeError(f"Incomplete generation batch: {path}")
        for prediction in predictions:
            samples = prediction["samples"]
            if len(samples) != n_samples or [item["sample_index"] for item in samples] != list(range(n_samples)):
                raise RuntimeError(f"Incomplete samples: {path}")
        seen += len(predictions)
        yield from predictions
    if seen != count:
        raise RuntimeError(f"Wrong number of predictions: {dataset}")


def math_scores(root: Path, protocol: dict, correct, robust_answer) -> dict:
    summaries = {}
    n_samples = protocol["generation"]["n"]
    identity = protocol["protocol_sha256"]
    for dataset in MATH:
        detail = root / "scores" / dataset / "per_question.jsonl"
        summary_path = root / "scores" / dataset / "summary.json"
        if detail.exists() and summary_path.exists():
            value = read(summary_path)
            if value.get("protocol_sha256") != identity:
                raise RuntimeError(f"Wrong cached math score: {dataset}")
            summaries[dataset] = value
            continue
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
            rows.append({
                "id": prediction["id"],
                "reference_answer": gold,
                "correct_samples": sum(item["correct"] for item in samples),
                "sampled_pass_at_1": sum(item["correct"] for item in samples) / n_samples,
                "samples": samples,
            })
        expected = protocol["datasets"][dataset]["rows"]
        if len(rows) != expected or len({row["id"] for row in rows}) != len(rows):
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
        jsonl(detail, rows)
        write_once(summary_path, value)
        summaries[dataset] = value
        print("MATH_SCORE", json.dumps(value, ensure_ascii=False), flush=True)
    return summaries


def prepare_code_samples(root: Path, protocol: dict, code_common) -> list[Path]:
    n_samples = protocol["generation"]["n"]
    roots = [root / "code_scoring" / f"sample_{index:02d}" for index in range(n_samples)]
    rows = [[] for _ in range(n_samples)]
    for dataset in CODE:
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
    total = sum(protocol["datasets"][name]["rows"] for name in CODE)
    for index, sample_root in enumerate(roots):
        if len(rows[index]) != total:
            raise RuntimeError(f"Incomplete code sample {index}")
        write_once(sample_root / "protocol.json", protocol)
        jsonl(sample_root / "predictions.jsonl", rows[index])
        write_once(sample_root / "generation_complete.json", {
            "protocol_sha256": protocol["protocol_sha256"],
            "status": "GENERATION_COMPLETE",
            "sample_index": index,
        })
    return roots


def run(command: list[str], env=None):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, stdin=subprocess.DEVNULL, env=env)


def local_and_evalplus(sample_root: Path, workers: int, legacy: Path, official: Path, data_root: Path) -> dict:
    local = sample_root / "local_summary.json"
    if not local.exists():
        run(["bash", str(legacy / "sandbox_score.sh"), str(sample_root), str(workers)])
    for dataset in ("HumanEvalPlus", "MBPPPlus"):
        score_root = sample_root / "scores" / dataset / "official"
        if not (score_root / "summary.json").exists():
            run([
                "bash", str(official / "sandbox_score_evalplus.sh"), dataset,
                str(sample_root / "predictions.jsonl"),
                str(data_root / "EvalPlusOfficial" / ("HumanEvalPlus-v0.1.10.jsonl.gz" if dataset == "HumanEvalPlus" else "MbppPlus-v0.2.0.jsonl.gz")),
                str(score_root), str(workers),
            ])
    return read(local)


def remote_bigcode(sample_root: Path, legacy: Path, total: int) -> dict:
    raise RuntimeError("Remote scoring is disabled in the public release; use evaluation/run_eval.sh for local scoring")


def code_scores(root: Path, protocol: dict, workers: int, legacy: Path, official: Path, data_root: Path, code_common) -> dict:
    sample_roots = prepare_code_samples(root, protocol, code_common)
    run(["bash", str(legacy / "prepare_scoring_env.sh")])
    aggregate = {name: [] for name in CODE}
    for index, sample_root in enumerate(sample_roots):
        local = local_and_evalplus(sample_root, workers, legacy, official, data_root)
        for dataset in ("HumanEvalPlus", "MBPPPlus"):
            aggregate[dataset].append(read(sample_root / "scores" / dataset / "official" / "summary.json"))
        for dataset in ("LiveCodeBench", "CRUXEval-I", "CRUXEval-O"):
            aggregate[dataset].append(local["datasets"][dataset])
        print(f"CODE_LOCAL_SCORE sample={index + 1}/{len(sample_roots)}", flush=True)
    summaries = {}
    identity = protocol["protocol_sha256"]
    n_samples = protocol["generation"]["n"]
    for dataset in CODE[:-1]:
        total = protocol["datasets"][dataset]["rows"]
        if dataset in ("HumanEvalPlus", "MBPPPlus"):
            counts = [score["plus_correct"] for score in aggregate[dataset]]
            base_counts = [score["base_correct"] for score in aggregate[dataset]]
            scorer = "official EvalPlus plus"
        else:
            counts = [score["correct"] for score in aggregate[dataset]]
            base_counts = None
            scorer = "pinned local sandbox"
        summary = {
            "dataset": dataset,
            "protocol_sha256": identity,
            "total_questions": total,
            "samples_per_question": n_samples,
            "correct_samples": sum(counts),
            "sampled_pass_at_1": sum(counts) / (total * n_samples),
            "scorer": scorer,
            "metric": "sampled Pass@1 = mean_i(c_i/16)",
        }
        if base_counts is not None:
            summary["base_sampled_pass_at_1"] = sum(base_counts) / (total * n_samples)
        write_once(root / "scores" / dataset / "summary.json", summary)
        summaries[dataset] = summary
        print("CODE_SCORE", json.dumps(summary, ensure_ascii=False), flush=True)

    remote = []
    bcb_total = protocol["datasets"]["BigCodeBench-Hard"]["rows"]
    for index, sample_root in enumerate(sample_roots):
        remote.append(remote_bigcode(sample_root, legacy, bcb_total))
        print(f"CODE_REMOTE_SCORE sample={index + 1}/{len(sample_roots)}", flush=True)
    summary = {
        "dataset": "BigCodeBench-Hard",
        "protocol_sha256": identity,
        "total_questions": bcb_total,
        "samples_per_question": n_samples,
        "correct_samples": sum(row["correct"] for row in remote),
        "sampled_pass_at_1": sum(row["correct"] for row in remote) / (bcb_total * n_samples),
        "scorer": "official BigCodeBench remote, instruct/hard, calibrated=false",
        "metric": "sampled Pass@1 = mean_i(c_i/16)",
    }
    write_once(root / "scores" / "BigCodeBench-Hard" / "summary.json", summary)
    summaries["BigCodeBench-Hard"] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opd-root", type=Path, default=OPD_ROOT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.output.expanduser().resolve()
    with (root / ".scoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(root / "protocol.json")
        identity = protocol.get("protocol_sha256") or digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
        protocol["protocol_sha256"] = identity
        marker_path = root / "generation_complete.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Generation is not complete yet: missing {marker_path}")
        marker = read(marker_path)
        if marker.get("protocol_sha256") != identity or set(protocol["dataset_order"]) != set(MATH + CODE):
            raise RuntimeError("Generation or benchmark suite incomplete")
        if (root / "summary.json").exists():
            print(json.dumps(read(root / "summary.json"), ensure_ascii=False, indent=2), flush=True)
            return
        legacy, official, other, code_common, correct, robust_answer = load_refs(protocol, args.opd_root.expanduser().resolve(), args.data_root.expanduser().resolve())
        math = math_scores(root, protocol, correct, robust_answer)
        code = code_scores(root, protocol, args.workers, legacy, official, args.data_root.expanduser().resolve(), code_common)
        datasets = {**math, **code}
        final = {
            "status": "COMPLETE",
            "protocol_sha256": identity,
            "model_path": protocol["model_path"],
            "checkpoint": protocol["checkpoint"],
            "metric": "sampled Pass@1 (n=16)",
            "datasets": datasets,
            "macro_avg_sampled_pass_at_1": statistics.mean(item["sampled_pass_at_1"] for item in datasets.values()),
        }
        write_once(root / "summary.json", final)
        print("RESULT", json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
