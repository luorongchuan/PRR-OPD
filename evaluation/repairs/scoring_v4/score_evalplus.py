"""Score saved generations with the pinned official EvalPlus differential harness."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Any

from official_utils import EXPECTED_ROWS, load_jsonl, load_jsonl_gz, sha256


HERE = Path(__file__).resolve().parent
REFERENCE_POLICY = HERE / "reference_expectations.json"
ARCHIVE = Path("/archive") if Path("/archive").exists() else HERE.parents[1] / "scorers"
EVALPLUS_VENDOR = ARCHIVE / "pinned14/vendor/test-code/vendor/evalplus"
if str(EVALPLUS_VENDOR) not in sys.path:
    sys.path.insert(0, str(EVALPLUS_VENDOR))

from evalplus.eval import PASS, untrusted_check  # noqa: E402
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS  # noqa: E402
from evalplus.gen.util import trusted_exec  # noqa: E402


# Frozen from evalplus/data/mbpp.py at commit 26d6d00. JSON cannot preserve
# tuples/sets, so the official harness restores those task-specific inputs.
def deserialize_mbpp_inputs(task_id: str, inputs: list) -> list:
    number = int(task_id.split("/")[-1])
    if number in [2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429, 470, 560, 579, 596, 616, 630, 726, 740, 744, 809]:
        return [[tuple(item) for item in value] for value in inputs]
    if number in [63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780]:
        return [[[tuple(item) for item in group] for group in value] for value in inputs]
    if number in [75, 413, 444, 753]:
        return [[[tuple(item) for item in value[0]]] + [value[1]] for value in inputs]
    if number in [106, 750]:
        return [[value[0], tuple(value[1])] for value in inputs]
    if number == 115:
        return [[[{*item} if isinstance(item, list) and item else {} for item in value[0]]] for value in inputs]
    if number == 124:
        return [(float(value[0]), complex(value[1])) for value in inputs]
    if number in [250, 405, 446, 617, 720, 763, 808]:
        return [[tuple(value[0]), value[1]] for value in inputs]
    if number in [259, 401, 445]:
        restored = [[[tuple(item) for item in group] for group in value] for value in inputs]
        return [[tuple(group) for group in value] for value in restored]
    if number == 278:
        restored = [[[tuple(item) if isinstance(item, list) else item for item in value[0]]] for value in inputs]
        return [[tuple(group) for group in value] for value in restored]
    if number == 307:
        return [[tuple(value[0]), value[1], value[2]] for value in inputs]
    if number == 722:
        return [[{key: tuple(item) for key, item in value[0].items()}] + value[1:] for value in inputs]
    if number == 252:
        return [[complex(value[0])] for value in inputs]
    if number in [580, 615, 791]:
        def all_tuples(value):
            return tuple(all_tuples(item) for item in value) if isinstance(value, list) else value
        return [all_tuples(value) for value in inputs]
    return inputs


def prepare_problems(dataset: str, official_data: Path) -> dict[str, dict[str, Any]]:
    problems = {str(row["task_id"]): row for row in load_jsonl_gz(official_data)}
    if dataset == "MBPPPlus":
        for task_id, problem in problems.items():
            problem["base_input"] = deserialize_mbpp_inputs(task_id, problem["base_input"])
            problem["plus_input"] = deserialize_mbpp_inputs(task_id, problem["plus_input"])
    return problems


def groundtruth(problems: dict[str, dict[str, Any]], dataset: str, cache: Path) -> dict[str, Any]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)
    expected = {}
    for index, (task_id, problem) in enumerate(problems.items(), 1):
        code = problem["prompt"] + problem["canonical_solution"]
        output_not_none = dataset == "MBPPPlus" and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
        base, base_time = trusted_exec(code, deepcopy(problem["base_input"]), problem["entry_point"], record_time=True, output_not_none=output_not_none)
        plus, plus_time = trusted_exec(code, deepcopy(problem["plus_input"]), problem["entry_point"], record_time=True, output_not_none=output_not_none)
        expected[task_id] = {"base": base, "base_time": base_time, "plus": plus, "plus_time": plus_time}
        if index % 25 == 0 or index == len(problems):
            print(f"GROUNDTRUTH {dataset} {index}/{len(problems)}", flush=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(expected, handle)
    temporary.replace(cache)
    return expected


def check_one(task: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]],
              *, canonical_controls: bool = False) -> dict[str, Any]:
    dataset, problem, prediction, expected = task
    official_name = "humaneval" if dataset == "HumanEvalPlus" else "mbpp"
    solution = prediction["solution"]
    base_status, base_details = untrusted_check(
        official_name, solution, problem["base_input"], problem["entry_point"],
        expected["base"], problem["atol"], expected["base_time"], fast_check=True,
    )
    plus_status = "not_run"
    plus_details = []
    if base_status == PASS:
        plus_status, plus_details = untrusted_check(
            official_name, solution, problem["plus_input"], problem["entry_point"],
            expected["plus"], problem["atol"], expected["plus_time"],
            fast_check=not (canonical_controls and prediction["id"] == "HumanEval/32"),
        )
    return {
        "id": prediction["id"],
        "base_status": base_status,
        "plus_status": plus_status,
        "base_passed": base_status == PASS,
        "plus_passed": base_status == PASS and plus_status == PASS,
        "base_failed_input_indices": [i for i, passed in enumerate(base_details) if not passed][:32],
        "plus_failed_input_indices": [i for i, passed in enumerate(plus_details) if not passed][:32],
        "base_checked_input_count": len(base_details),
        "plus_checked_input_count": len(plus_details),
    }


def reference_policy():
    return json.loads(REFERENCE_POLICY.read_text()), sha256(REFERENCE_POLICY)


def known_reference_failure(row, problem, dataset, data_hash):
    policy, _ = reference_policy()
    encoded_inputs = json.dumps(problem["plus_input"], ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode()
    code = problem["prompt"] + problem["canonical_solution"]
    return (
        dataset == policy["dataset"] and data_hash == policy["official_data_sha256"]
        and row["id"] == policy["task_id"] and problem["atol"] == policy["atol"]
        and hashlib.sha256(code.encode()).hexdigest() == policy["canonical_code_sha256"]
        and hashlib.sha256(encoded_inputs).hexdigest() == policy["plus_inputs_sha256"]
        and row["base_status"] == policy["base_status"] and row["base_passed"]
        and row["plus_status"] == policy["plus_status"] and not row["plus_passed"]
        and row["base_failed_input_indices"] == []
        and row["plus_failed_input_indices"] == policy["plus_failed_input_indices"]
        and row["base_checked_input_count"] == len(problem["base_input"]) == policy["base_input_count"]
        and row["plus_checked_input_count"] == len(problem["plus_input"]) == policy["plus_input_count"]
    )


def require_canonical_controls(results, dataset, output, data_hash, problems=None):
    failed = [row for row in results if not row["plus_passed"]]
    known = []
    unexpected = []
    for row in failed:
        if (problems is not None and row["id"] == "HumanEval/32"
                and known_reference_failure(row, problems[row["id"]], dataset, data_hash)):
            known.append(row)
        else:
            unexpected.append(row)
    if not unexpected:
        policy, policy_hash = reference_policy()
        return {"canonical_controls_validated": len(results),
                "canonical_controls_passed": len(results) - len(failed),
                "known_canonical_failures": known,
                "canonical_control_policy_revision": policy["revision"],
                "canonical_control_policy_sha256": policy_hash}
    diagnostic = {
        "status": "FAILED", "dataset": dataset, "canonical_controls": True,
        "official_data_sha256": data_hash, "total": len(results),
        "passed": len(results) - len(failed), "failed_tasks": failed,
        "unexpected_failures": unexpected,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "canonical_failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    print("CANONICAL_FAILURE", json.dumps(diagnostic, sort_keys=True), flush=True)
    raise RuntimeError(f"Canonical controls failed: {diagnostic['passed']}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(EXPECTED_ROWS), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--official-data", type=Path, required=True)
    parser.add_argument("--official-data-origin", type=str)
    parser.add_argument("--predictions-origin", type=str)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--canonical-controls", action="store_true")
    args = parser.parse_args()

    all_predictions = load_jsonl(args.predictions)
    predictions = [row for row in all_predictions if row["dataset"] == args.dataset]
    problems = prepare_problems(args.dataset, args.official_data)
    if args.canonical_controls:
        predictions = [{"dataset": args.dataset, "id": task_id,
                        "solution": problem["prompt"] + problem["canonical_solution"]}
                       for task_id, problem in problems.items()]
    if len(predictions) != EXPECTED_ROWS[args.dataset] or len(problems) != EXPECTED_ROWS[args.dataset]:
        raise RuntimeError("Prediction or official task count mismatch")
    if [row["id"] for row in predictions] != list(problems):
        raise RuntimeError("Prediction order/ids do not match the official release")

    data_hash = sha256(args.official_data)
    predictions_hash = ("official-canonical-solutions" if args.canonical_controls else sha256(args.predictions))
    summary_path = args.output / "summary.json"
    detail_path = args.output / "per_task.jsonl"
    if summary_path.exists() or detail_path.exists():
        if not (summary_path.exists() and detail_path.exists()):
            raise RuntimeError(f"Incomplete existing score output: {args.output}")
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_identity = (args.dataset, data_hash, predictions_hash)
        existing_identity = (
            existing.get("dataset"), existing.get("official_data_sha256"),
            existing.get("predictions_sha256"),
        )
        if (existing_identity != expected_identity or existing.get("status") != "COMPLETE"
                or existing.get("scoring_revision") != "scoring_v4_serial"
                or existing.get("infrastructure_errors") != 0
                or existing.get("canonical_controls") != args.canonical_controls
                or existing.get("total") != EXPECTED_ROWS[args.dataset]):
            raise RuntimeError(f"Existing score output has different provenance: {args.output}")
        if args.canonical_controls and args.dataset == "HumanEvalPlus":
            _, policy_hash = reference_policy()
            if (existing.get("canonical_control_policy_sha256") != policy_hash
                    or existing.get("canonical_controls_validated") != EXPECTED_ROWS[args.dataset]):
                raise RuntimeError("Existing reference controls need a versioned recheck")
        print("OFFICIAL_RESULT", json.dumps(existing, ensure_ascii=False, sort_keys=True), flush=True)
        return
    expected = groundtruth(problems, args.dataset, args.cache / f"{args.dataset}_{data_hash[:16]}.pkl")
    tasks = [(args.dataset, problems[row["id"]], row, expected[row["id"]]) for row in predictions]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    began = time.perf_counter()
    # Avoid pickling hundreds of large test-input/oracle objects into a process pool.
    # The official checker still isolates each submitted solution in a subprocess.
    for index, task in enumerate(tasks):
        results[index] = check_one(task, canonical_controls=args.canonical_controls)
        if results[index]["base_status"] not in ("pass", "fail", "timeout"):
            raise RuntimeError(f"Infrastructure failure, not a wrong answer: {results[index]}")
        if results[index]["plus_status"] not in ("pass", "fail", "timeout", "not_run"):
            raise RuntimeError(f"Infrastructure failure, not a wrong answer: {results[index]}")
        done = index + 1
        if done % 25 == 0 or done == len(tasks):
            print(f"SCORE {args.dataset} {done}/{len(tasks)}", flush=True)
    final_results = [row for row in results if row is not None]
    base_correct = sum(row["base_passed"] for row in final_results)
    plus_correct = sum(row["plus_passed"] for row in final_results)
    controls = {}
    if args.canonical_controls:
        controls = require_canonical_controls(final_results, args.dataset, args.output, data_hash, problems)
    summary = {
        "status": "COMPLETE",
        "dataset": args.dataset,
        "evaluator": "official EvalPlus differential harness",
        "evalplus_vendor_commit": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
        "scoring_revision": "scoring_v4_serial",
        "infrastructure_errors": 0,
        "canonical_controls": args.canonical_controls,
        "official_data_path": args.official_data_origin or str(args.official_data),
        "official_data_sha256": data_hash,
        "predictions_path": args.predictions_origin or str(args.predictions),
        "predictions_sha256": predictions_hash,
        "total": len(final_results),
        "base_correct": base_correct,
        "base_pass_at_1": base_correct / len(final_results),
        "base_pass_at_1_percent": 100.0 * base_correct / len(final_results),
        "plus_correct": plus_correct,
        "plus_pass_at_1": plus_correct / len(final_results),
        "plus_pass_at_1_percent": 100.0 * plus_correct / len(final_results),
        "base_status_counts": dict(Counter(row["base_status"] for row in final_results)),
        "plus_status_counts": dict(Counter(row["plus_status"] for row in final_results)),
        "seconds": time.perf_counter() - began,
        **controls,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as handle:
        for row in final_results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("OFFICIAL_RESULT", json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
