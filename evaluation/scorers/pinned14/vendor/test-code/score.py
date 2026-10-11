"""Score generated samples with pinned benchmark logic inside the sandbox."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import io
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq

from common import EXPECTED_ROWS, atomic_json, atomic_jsonl, digest, load_jsonl, read_json


HERE = Path(__file__).resolve().parent
LCB_VENDOR = HERE / "vendor" / "livecodebench"


def _exec_program(program: str, queue) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                namespace: dict[str, Any] = {"__name__": "__main__"}
                exec(compile(program, "<candidate>", "exec"), namespace, namespace)
        queue.put({"passed": True, "status": "pass"})
    except BaseException as exc:
        queue.put({"passed": False, "status": type(exc).__name__, "detail": str(exc)[:1000]})


def execute(program: str, timeout: int = 25) -> dict[str, Any]:
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue(1)
    process = ctx.Process(target=_exec_program, args=(program, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(2)
        return {"passed": False, "status": "timeout"}
    if queue.empty():
        return {"passed": False, "status": f"worker_exit_{process.exitcode}"}
    return queue.get()


def score_evalplus(task: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    dataset, row, pred = task
    solution = pred["solution"]
    if dataset == "HumanEvalPlus":
        program = solution + "\n\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
    else:
        program = solution + "\n\n" + row["test"]
    result = execute(program)
    return {"id": pred["id"], **result}


def score_lcb(task: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, pred = task
    if str(LCB_VENDOR) not in sys.path:
        sys.path.insert(0, str(LCB_VENDOR))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    problem = CodeGenerationProblem(**{k: row[k] for k in (
        "question_title", "question_content", "platform", "question_id", "contest_id",
        "contest_date", "starter_code", "difficulty", "public_test_cases",
        "private_test_cases", "metadata",
    )})
    try:
        results, metadata = check_correctness(
            problem.get_evaluation_sample(), pred["solution"], timeout=6, debug=False
        )
        passed = bool(results) and all(value is True for value in results)
        status = "pass" if passed else "fail"
        detail = metadata
    except BaseException as exc:
        passed, status, detail = False, type(exc).__name__, str(exc)[:1000]
    return {"id": pred["id"], "passed": passed, "status": status, "detail": detail}


def score_crux(task: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    mode, row, pred = task
    generation = pred["solution"]
    if mode == "input":
        valid = "f(" in generation
        program = f"{row['code']}\nassert {row['output']} == {generation}" if valid else ""
    else:
        valid = bool(generation) and f"f({row['input']})" not in generation
        program = f"{row['code']}\nassert {row['output']} == {generation}" if valid else ""
    result = execute(program, timeout=5) if valid else {"passed": False, "status": "invalid_format"}
    return {"id": pred["id"], **result}


def parallel_score(function, tasks, workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, task): index for index, task in enumerate(tasks)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                results[index] = future.result()
            except BaseException as exc:
                results[index] = {"id": tasks[index][-1]["id"], "passed": False, "status": type(exc).__name__, "detail": str(exc)[:1000]}
            if done % 25 == 0 or done == len(tasks):
                print(f"SCORE_PROGRESS {done}/{len(tasks)}", flush=True)
    return [item for item in results if item is not None]


def summary(dataset: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["passed"]) for row in results)
    return {
        "dataset": dataset,
        "correct": correct,
        "total": len(results),
        "pass_at_1": correct / len(results),
        "pass_at_1_percent": 100.0 * correct / len(results),
        "status_counts": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    by_dataset = {name: [] for name in EXPECTED_ROWS}
    for pred in predictions:
        by_dataset[pred["dataset"]].append(pred)
    if {name: len(rows) for name, rows in by_dataset.items()} != EXPECTED_ROWS:
        raise RuntimeError("Prediction counts are incomplete")

    data = args.data_root
    tables = {
        "HumanEvalPlus": pq.read_table(data / "HumanEvalPlus/test.parquet").to_pylist(),
        "MBPPPlus": pq.read_table(data / "MBPPPlus/test.parquet").to_pylist(),
        "CRUXEval": pq.read_table(data / "CRUXEval/test.parquet").to_pylist(),
    }
    lcb = load_jsonl(data / "LiveCodeBench/raw/test6.jsonl")

    expected_ids = {
        "HumanEvalPlus": [str(row["task_id"]) for row in tables["HumanEvalPlus"]],
        "MBPPPlus": [f"Mbpp/{row['task_id']}" for row in tables["MBPPPlus"]],
        "LiveCodeBench": [str(row["question_id"]) for row in lcb],
        "CRUXEval-I": [str(row["id"]) for row in tables["CRUXEval"]],
        "CRUXEval-O": [str(row["id"]) for row in tables["CRUXEval"]],
    }
    for dataset, identifiers in expected_ids.items():
        predicted_ids = [str(row["id"]) for row in by_dataset[dataset]]
        if predicted_ids != identifiers:
            raise RuntimeError(f"Prediction/source order mismatch for {dataset}")

    all_summaries = {}
    specs = [
        ("HumanEvalPlus", score_evalplus, [("HumanEvalPlus", row, pred) for row, pred in zip(tables["HumanEvalPlus"], by_dataset["HumanEvalPlus"])]),
        ("MBPPPlus", score_evalplus, [("MBPPPlus", row, pred) for row, pred in zip(tables["MBPPPlus"], by_dataset["MBPPPlus"])]),
        ("LiveCodeBench", score_lcb, list(zip(lcb, by_dataset["LiveCodeBench"]))),
        ("CRUXEval-I", score_crux, [("input", row, pred) for row, pred in zip(tables["CRUXEval"], by_dataset["CRUXEval-I"])]),
        ("CRUXEval-O", score_crux, [("output", row, pred) for row, pred in zip(tables["CRUXEval"], by_dataset["CRUXEval-O"])]),
    ]
    for dataset, function, tasks in specs:
        detail_path = root / "scores" / dataset / "per_task.jsonl"
        summary_path = root / "scores" / dataset / "summary.json"
        if detail_path.exists() and summary_path.exists():
            results = load_jsonl(detail_path)
            current = read_json(summary_path)
        else:
            print(f"SCORE_START {dataset} tasks={len(tasks)} workers={args.workers}", flush=True)
            results = parallel_score(function, tasks, args.workers)
            if len(results) != EXPECTED_ROWS[dataset] or [x["id"] for x in results] != [x["id"] for x in by_dataset[dataset]]:
                raise RuntimeError(f"Incomplete or reordered scores for {dataset}")
            current = summary(dataset, results)
            atomic_jsonl(detail_path, results)
            atomic_json(summary_path, current)
        all_summaries[dataset] = current
        print("SCORE_RESULT", json.dumps(current, ensure_ascii=False, sort_keys=True), flush=True)

    final = {
        "status": "LOCAL_SCORING_COMPLETE",
        "method": protocol.get("method", protocol.get("model_path", "unknown")),
        "checkpoint": protocol["checkpoint"],
        "protocol_sha256": protocol_hash,
        "metric": "greedy pass@1",
        "datasets": all_summaries,
        "pending": ["BigCodeBench-Hard official remote evaluator"],
    }
    atomic_json(root / "local_summary.json", final)
    print("LOCAL_RESULT", json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
