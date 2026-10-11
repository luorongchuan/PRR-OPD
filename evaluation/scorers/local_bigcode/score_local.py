"""Use vendored official BigCodeBench checks and local canonical/test records."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys

import pyarrow.parquet as pq


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problems(path, expected=148):
    rows = pq.read_table(path).to_pylist()
    ids = {row["task_id"] for row in rows}
    if len(rows) != expected or len(ids) != expected:
        raise ValueError("Wrong or duplicate BigCodeBench tasks")
    for row in rows:
        for key in ("complete_prompt", "canonical_solution", "test", "entry_point"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"Missing official field: {row['task_id']}/{key}")
    return {row["task_id"]: row for row in rows}


def environment(rows, data, vendor):
    required = set()
    for row in rows.values():
        tree = ast.parse(row["complete_prompt"] + "\n" + row["canonical_solution"] + "\n" + row["test"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                required.update(name.name.split(".")[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                required.add(node.module.split(".")[0])
    missing = sorted(name for name in required if importlib.util.find_spec(name) is None)
    versions = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name"))
    assets = Path(os.environ.get("NLTK_DATA", "/nonexistent-nltk-data"))
    value = {"python": platform.python_version(), "packages": versions, "missing_modules": missing,
             "nltk_resources": {str(p.relative_to(assets)): sha(p) for p in sorted(assets.rglob("*")) if p.is_file()},
             "dataset_sha256": sha(data), "official_source": {str(p.relative_to(vendor)): sha(p)
                for p in sorted((vendor / "bigcodebench").rglob("*.py"))},
             "settings": {"max_as_mib": 30720, "max_data_mib": 30720, "max_stack_mib": 10,
                          "min_time_limit": 1, "calibrated": False, "workers": 1, "network": "disabled"}}
    value["fingerprint"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def validate_controls(rows, env, cached=None, checker=None):
    if env["missing_modules"]:
        raise RuntimeError("Local BigCodeBench environment lacks modules: " + ",".join(env["missing_modules"]))
    if (cached and cached.get("status") == "COMPLETE" and cached.get("environment_fingerprint") == env["fingerprint"]
            and cached.get("passed") == cached.get("total") == len(rows)
            and set(cached.get("times", {})) == set(rows)
            and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in cached["times"].values())):
        return cached
    if checker is None:
        from bigcodebench.gen.util import trusted_check
        checker = trusted_check
    times = {}
    for identity, row in rows.items():
        result = checker(row["complete_prompt"] + "\n" + row["canonical_solution"], row["test"], identity,
                         30720, 30720, 10, 1)
        times[identity] = result["time"]
        print(f"CANONICAL {len(times)}/{len(rows)} {identity} time={result['time']}", flush=True)
    failed = [identity for identity, value in times.items()
              if value is None or not math.isfinite(value) or value < 0]
    return {"status": "FAILED" if failed else "COMPLETE", "environment_fingerprint": env["fingerprint"],
            "times": times, "failed_tasks": failed, "passed": len(rows) - len(failed), "total": len(rows)}


def score(rows, predictions, controls, checker=None):
    if controls.get("status") != "COMPLETE" or controls.get("passed") != len(rows):
        raise RuntimeError("Canonical controls did not all pass; refusing to score predictions")
    if (len(predictions) != len(rows) or {r["id"] for r in predictions} != set(rows)
            or len({r["id"] for r in predictions}) != len(predictions)):
        raise ValueError("Wrong/missing/duplicate prediction task IDs")
    if checker is None:
        from bigcodebench.eval import untrusted_check
        checker = untrusted_check
    details = []
    for prediction in predictions:
        row = rows[prediction["id"]]
        status, failures = checker(prediction["solution"], row["test"], row["entry_point"],
                                   30720, 30720, 10, 1, controls["times"][row["task_id"]] or 20)
        details.append({"id": row["task_id"], "status": status, "passed": status == "pass", "details": failures})
        print(f"SCORED {len(details)}/{len(rows)} {row['task_id']} {status}", flush=True)
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "controls", "score"), required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    if args.mode != "preflight" and not (os.environ.get("PRR_BIGCODE_SANDBOX_INNER") == "1"
                                          or os.environ.get("PRR_BIGCODE_DOCKER_SANDBOX") == "1"):
        raise RuntimeError("BigCodeBench code execution is allowed only inside the local sandbox")
    sys.path.insert(0, str(args.vendor))
    rows = problems(args.problems)
    env = environment(rows, args.problems, args.vendor)
    write(args.output / "environment.json", env)
    if args.mode == "preflight":
        print(json.dumps({"missing_modules": env["missing_modules"], "fingerprint": env["fingerprint"]}))
        raise SystemExit(1 if env["missing_modules"] else 0)
    cache_file = args.cache / "canonical_controls.json"
    cached = read(cache_file) if cache_file.exists() else None
    if args.mode == "controls":
        result = validate_controls(rows, env, cached)
        write(args.output / "canonical_controls.json", result)
        if result["status"] != "COMPLETE":
            raise RuntimeError(f"Canonical solutions failed on {result['failed_tasks']}; environment score is invalid")
        write(cache_file, result)
        return
    if not cached or cached["environment_fingerprint"] != env["fingerprint"]:
        raise RuntimeError("Missing/stale canonical-control validation")
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    details = score(rows, predictions, cached)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "per_question.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    write(args.output / "summary.json", {"status": "COMPLETE", "total": len(details),
          "correct": sum(row["passed"] for row in details), "canonical_controls_passed": cached["passed"],
          "environment_fingerprint": env["fingerprint"], "uploaded": False})


if __name__ == "__main__":
    main()
