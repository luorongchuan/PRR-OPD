"""Rescore frozen generations with bounded memory and explicit repair provenance."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE.parents[1]
REPO = EVALUATOR.parent
sys.path[:0] = [str(EVALUATOR), str(EVALUATOR / "scorers/pinned14")]
from eval_common import DATA_ROOT, POLICY, PYTHON, SAFETY, digest, read, sha, write
from run import environment, spawn
from score_14_pass1 import batches, load_refs
from score_dataset_pass1 import math_score, prepare_code_samples


def input_identity(root, dataset):
    value = read(root / "protocol.json")
    if digest({k: v for k, v in value.items() if k != "protocol_sha256"}) != value["protocol_sha256"]:
        raise ValueError("Input generation protocol hash mismatch")
    rows = list(batches(root, value, dataset))
    if len({r["id"] for r in rows}) != value["datasets"][dataset]["rows"]:
        raise ValueError("Duplicate/missing generation IDs")
    return value


def shell(kind, mode, dataset, predictions, data, output, cache, *extra):
    python = (Path(os.environ.get("PRR_V4_BIGCODE_PYTHON", str(EVALUATOR / "runtime/bigcode_py310/bin/python")))
              if kind == "bigcode" else PYTHON)
    return ["bash", str(HERE / "sandbox.sh"), kind, mode, dataset, str(predictions), str(data),
            str(output), str(cache), str(python), *extra]


def plus_summary(output, identity, dataset, roots):
    cache = EVALUATOR / "scorers/pinned14/vendor/test-code/official_validation/cache"
    filename = "MbppPlus-v0.2.0.jsonl.gz" if dataset == "MBPPPlus" else "HumanEvalPlus-v0.1.10.jsonl.gz"
    data = DATA_ROOT / "EvalPlusOfficial" / filename
    controls = output / "scores" / dataset / "controls"
    subprocess.run(shell("evalplus", "controls", dataset, roots[0] / "predictions.jsonl",
                         data, controls, cache), check=True)
    control = read(controls / "summary.json")
    total = identity["datasets"][dataset]["rows"]
    validated = control.get("canonical_controls_validated", control.get("plus_correct"))
    known = control.get("known_canonical_failures", [])
    if (not control.get("canonical_controls") or control.get("status") != "COMPLETE"
            or control.get("total") != total or validated != total
            or control.get("plus_correct") + len(known) != total
            or control.get("infrastructure_errors") != 0):
        raise RuntimeError("EvalPlus reference controls have unexpected outcomes")
    if len(roots) != identity["generation"]["n"] or len(roots) != 16:
        raise ValueError("Expected all sixteen sample draws")
    totals, base_totals = [], []
    for index, sample in enumerate(roots):
        destination = output / "scores" / dataset / f"sample_{index:02d}"
        subprocess.run(shell("evalplus", "score", dataset, sample / "predictions.jsonl", data,
                             destination, cache), check=True)
        score = read(destination / "summary.json")
        if (score["status"] != "COMPLETE" or score["infrastructure_errors"] != 0
                or score["total"] != identity["datasets"][dataset]["rows"]
                or set(score["base_status_counts"]) - {"pass", "fail", "timeout"}):
            raise RuntimeError("Invalid EvalPlus score; infrastructure errors are not wrong answers")
        totals.append(score["plus_correct"])
        base_totals.append(score["base_correct"])
    total = identity["datasets"][dataset]["rows"]
    return {"dataset": dataset, "protocol_sha256": identity["protocol_sha256"],
        "total_questions": total, "samples_per_question": 16, "correct_samples": sum(totals),
        "sampled_pass_at_1": sum(totals) / (total * 16),
        "base_sampled_pass_at_1": sum(base_totals) / (total * 16),
        "scorer": "official EvalPlus differential harness, serial isolated checks",
        "infrastructure_errors": 0, "canonical_controls_passed": control["plus_correct"],
        "canonical_controls_validated": validated, "known_canonical_failures": known,
        "canonical_control_policy_revision": control.get("canonical_control_policy_revision"),
        "canonical_control_policy_sha256": control.get("canonical_control_policy_sha256"),
        "repair_sources": {p.name: sha(p) for p in (HERE / "score_evalplus.py", HERE / "rescore.py", HERE / "reference_expectations.json")},
        "scoring_revision": "scoring_v4", "metric": "sampled Pass@1 = mean_i(c_i/16)"}


def bigcode_summary(output, identity, roots):
    if len(roots) != identity["generation"]["n"] or len(roots) != 16:
        raise ValueError("Expected all sixteen BigCodeBench sample draws")
    data = DATA_ROOT / "BigCodeBench-Hard/test.parquet"
    cache = EVALUATOR / "runtime/bigcodebench_v4"
    destination = output / "scores/BigCodeBench-Hard"
    subprocess.run(shell("bigcode", "controls", "BigCodeBench-Hard", roots[0] / "predictions.jsonl",
                         data, destination / "controls", cache), check=True)
    totals, fingerprints = [], set()
    for index, sample in enumerate(roots):
        draw = destination / f"sample_{index:02d}"
        subprocess.run(shell("bigcode", "score", "BigCodeBench-Hard", sample / "predictions.jsonl",
                             data, draw, cache), check=True)
        row = read(draw / "summary.json")
        if row["status"] != "COMPLETE" or row["total"] != 148 or row["canonical_controls_passed"] != 148:
            raise RuntimeError("BigCodeBench controls/prediction count are invalid")
        totals.append(row["correct"])
        fingerprints.add(row["environment_fingerprint"])
    if len(fingerprints) != 1:
        raise RuntimeError("BigCodeBench scoring environment changed between samples")
    return {"dataset": "BigCodeBench-Hard", "protocol_sha256": identity["protocol_sha256"],
        "total_questions": 148, "samples_per_question": 16, "correct_samples": sum(totals),
        "sampled_pass_at_1": sum(totals) / (148 * 16), "canonical_controls_passed": 148,
        "uploaded": False, "environment_fingerprint": next(iter(fingerprints)),
        "scoring_revision": "scoring_v4",
        "scorer": "pinned official BigCodeBench local harness, corrected sandbox",
        "metric": "sampled Pass@1 = mean_i(c_i/16)"}


def work(args):
    identity = input_identity(args.root, args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "protocol.json", identity)
    link = args.output / "generation"
    if not link.exists():
        link.symlink_to(args.root / "generation", target_is_directory=True)
    if link.resolve() != (args.root / "generation").resolve():
        raise RuntimeError("Unexpected generation source")
    refs = load_refs(identity, EVALUATOR / "scorers/pinned14/vendor", DATA_ROOT)
    if args.dataset == "AMC23":
        summary = math_score(args.output, identity, args.dataset, refs[4], refs[5])
    else:
        roots = prepare_code_samples(args.output, identity, args.dataset, refs[3])
        summary = (bigcode_summary(args.output, identity, roots) if args.dataset == "BigCodeBench-Hard"
                   else plus_summary(args.output, identity, args.dataset, roots))
    summary.update({"scoring_revision": "scoring_v4", "generation_source": str(args.root),
        "repair_sources": {p.name: sha(p) for p in HERE.iterdir() if p.suffix in (".py", ".sh")},
        "generation_reused": True, "generation_parameters_changed": False})
    write(args.output / "scores" / args.dataset / "summary.json", summary)
    print("REPAIRED_SCORE", args.dataset, summary["sampled_pass_at_1"] * 100, flush=True)


def refresh_single_session(root, dataset, destination):
    session = root.parents[1]
    if root.parent.name != "models" or not (session / "plan.json").exists():
        return
    with (session / ".queue.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = read(session / "status.json")
        if state.get("stage") not in ("failed", "complete"):
            return
        from eval_common import ready, score_value
        from run import export
        plan = read(session / "plan.json")
        if score_value(root, dataset) is None:
            raise RuntimeError("Published repair did not pass guarded score validation")
        failures = dict(state.get("failures", {}))
        failures.pop(f"{root.name}/{dataset}", None)
        jobs = [{"key": f"{model}/{name}", "model": model, "dataset": name,
                 "output": str(session / "models" / model),
                 "ready": ready(session / "models" / model, name)}
                for model in plan["models"] for name in plan["datasets"]]
        count = export(session, jobs, failures)
        complete = count == len(jobs) and not failures and not state.get("generation_error")
        write(session / "status.json", {**state, "stage": "complete" if complete else "failed",
            "completed": count, "total": len(jobs), "failures": failures,
            "last_scoring_repair": {"dataset": dataset, "source": str(destination)},
            "updated_at": time.strftime("%F %T %Z")})


def queue(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for dataset in args.datasets.split(","):
            destination = args.output / dataset
            destination.mkdir(parents=True, exist_ok=True)
            report = destination / "scores" / dataset / "resource_guard.json"
            limit = 30 if dataset == "BigCodeBench-Hard" else 4
            argv = [str(PYTHON), "-B", str(EVALUATOR / "memory_guard.py"), "--wait-lock",
                "--report", str(report), "--per-process-gib", str(limit), "--tree-gib", "24",
                "--min-free-gib", "128", "--", str(PYTHON), "-B", str(Path(__file__).resolve()),
                "--root", str(args.root), "--output", str(destination), "--dataset", dataset, "--worker"]
            write(args.output / "status.json", {"stage": "running", "dataset": dataset, "results": results})
            with (destination / "scoring.log").open("a") as log:
                code = subprocess.call(argv, env=environment(bigcode_extra_site=EVALUATOR / "runtime/bigcode_packages"),
                                       stdout=log, stderr=subprocess.STDOUT)
            if code or not report.exists() or not read(report).get("completed"):
                results[dataset] = {"status": "failed", "returncode": code, "log": str(destination / "scoring.log")}
                write(args.output / "status.json", {"stage": "running", "results": results})
                continue
            summary = read(destination / "scores" / dataset / "summary.json")
            original = args.root / "scores" / dataset
            if args.publish:
                original.mkdir(parents=True, exist_ok=True)
                for name in ("summary.json", "resource_guard.json", "per_question.jsonl"):
                    source = destination / "scores" / dataset / name
                    if not source.exists():
                        continue
                    target = original / name
                    if target.exists():
                        target.rename(target.with_name(f"{name}.before-v4-{time.time_ns()}"))
                    temporary = target.with_name(f"{name}.tmp-{os.getpid()}")
                    shutil.copy2(source, temporary)
                    temporary.replace(target)
                write(original / "rescoring_v4_receipt.json", {"generation_source": str(args.root),
                    "score_source": str(destination), "resource_guard": read(report),
                    "summary_sha256": sha(original / "summary.json"), "scoring_revision": "scoring_v4"})
                split = args.root.parents[2]
                if (split / "split_plan.json").exists():
                    from split_eval import collect_once
                    collect_once(split, read(split / "split_plan.json"))
                else:
                    refresh_single_session(args.root, dataset, destination)
            results[dataset] = {"status": "complete", "pass1_percent": summary["sampled_pass_at_1"] * 100,
                                "summary": str(destination / "scores" / dataset / "summary.json")}
        write(args.output / "status.json", {"stage": "complete" if all(r["status"] == "complete" for r in results.values())
              else "incomplete", "results": results, "updated_at": time.strftime("%F %T %Z")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", default="AMC23,MBPPPlus,BigCodeBench-Hard")
    parser.add_argument("--dataset")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if args.worker:
        if os.environ.get("PRR_SCORING_GUARD") != "1":
            raise RuntimeError("A bounded-memory scoring supervisor is required")
        work(args)
    elif args.launch:
        args.output.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-B", str(Path(__file__).resolve()),
                   *[v for v in sys.argv[1:] if v != "--launch"]]
        process = spawn(command, args.output / "queue.log", environment())
        write(args.output / "launcher.json", {"pid": process.pid, "command": command})
        print(f"RESCORING_LAUNCHED pid={process.pid} output={args.output}", flush=True)
    else:
        queue(args)


if __name__ == "__main__":
    main()
