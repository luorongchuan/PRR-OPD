"""Stream GPU generation and guarded CPU scoring independently for 1-14 tasks."""
import argparse
import csv
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from eval_common import COUNTS, DATA_ROOT, EVAL, HERE, POLICY, PYTHON, SAFETY
from eval_common import archived_sources, digest, protocol, read, ready, score_value, selected, sha, write
from memory_guard import stop
import eval_scoring as scoring


@contextmanager
def gpu_slot(gpu, folder=SAFETY):
    with (folder / f".generation_gpu{gpu}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU {gpu} is reserved by another evaluation queue") from exc
        yield


def lift_hold(session):
    with (SAFETY / ".hold_control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hold = SAFETY / "EVALUATION_HOLD.json"
        if hold.exists():
            write(session / "evaluation_resume_receipt.json", {"previous_hold": read(hold), "pid": os.getpid(),
                  "time": time.strftime("%F %T %Z")})
            hold.unlink()


def environment(gpu="", python=PYTHON, bigcode_python=PYTHON, bigcode_backend="host", bigcode_image="",
                bigcode_extra_site=None):
    value = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
             "PYTHON": str(python), "DATA_ROOT": str(DATA_ROOT),
             "PRR_BIGCODE_PYTHON": str(bigcode_python),
             "PRR_BIGCODE_BACKEND": bigcode_backend, "PRR_BIGCODE_IMAGE": bigcode_image,
             "PRR_CODE_DATA_ROOT": str(DATA_ROOT), "PRR_DATA_ROOT": str(DATA_ROOT)}
    value.pop("ALLOW_BIGCODE_REMOTE_UPLOAD", None)
    value.pop("PRR_CODE_SANDBOX_INNER", None)
    value.pop("PRR_EVALPLUS_SANDBOX_INNER", None)
    value.pop("PRR_BIGCODE_SANDBOX_INNER", None)
    value.pop("PRR_BIGCODE_EXTRA_SITE", None)
    if bigcode_extra_site:
        value["PRR_BIGCODE_EXTRA_SITE"] = str(bigcode_extra_site)
    return value


def spawn(argv, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        return subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)


def generation_command(plan, model, output):
    batch, seqs = (16, 256) if plan["profile"] == "fast" else (4, 64)
    return [plan["python"], "-B", str(HERE / "generate.py"), "--datasets", ",".join(plan["datasets"]),
            "--model-path", model["path"], "--output", str(output), "--opd-root", str(EVAL / "vendor"),
            "--data-root", str(DATA_ROOT), "--n-samples", "16",
            "--max-tokens", str(plan["max_tokens"]), "--max-model-len", "8192",
            "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20",
            "--seed", str(plan["seed"]), "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(plan["gpu_memory_utilization"]),
            "--request-batch-size", str(batch), "--max-num-seqs", str(seqs)]


def export(session, jobs, failures):
    rows = []
    for job in jobs:
        if job["key"] in failures:
            continue
        value = score_value(Path(job["output"]), job["dataset"]) if job["ready"] else None
        if value is not None and job["key"] not in failures:
            rows.append({"model": job["model"], "dataset": job["dataset"], "pass1_percent": value,
                         "summary": str(Path(job["output"]) / "scores" / job["dataset"] / "summary.json")})
    with (session / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "dataset", "pass1_percent", "summary"))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Evaluation Results", "", "Sampled Pass@1 (%), n=16. All requested datasets are retained.",
             "CPU memory limits and generation batching are recorded in plan.json and protocol.json.",
             "Failed/blocked jobs are pending, not zero-valued scores.", "",
             "| Model | Dataset | Pass@1 (%) |", "| --- | --- | ---: |"]
    values = {(r["model"], r["dataset"]): r["pass1_percent"] for r in rows}
    for job in jobs:
        value = values.get((job["model"], job["dataset"]))
        result = f"{value:.2f}" if value is not None else ("failed" if job["key"] in failures else "pending")
        lines.append(f"| {job['model']} | {job['dataset']} | {result} |")
    (session / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return len(rows)


def quarantine(root, dataset):
    summary = Path(root) / "scores" / dataset / "summary.json"
    if summary.exists():
        summary.rename(summary.with_name(f"summary.failed-{time.time_ns()}.json"))


def run_pipeline(session, plan, generator=generation_command, scorer=scoring.command, held=None):
    held = held or (lambda: (SAFETY / "EVALUATION_HOLD.json").exists())
    jobs = [{"model": name, "dataset": dataset, "output": str(session / "models" / name),
             "key": f"{name}/{dataset}", "ready": False, "submitted": False}
            for name in plan["models"] for dataset in plan["datasets"]]
    pending, failures = [], {}
    models = iter(plan["models"].items())
    gen, score, active_job = None, None, None
    gen_known, score_known = {}, {}
    generation_failed = None
    generation_done = False
    stage = "running"
    try:
        while True:
            if held():
                raise InterruptedError("Evaluation paused by user")
            if gen is not None:
                try:
                    root = psutil.Process(gen.pid)
                    gen_known.update({p.pid: p for p in root.children(recursive=True)})
                except psutil.NoSuchProcess:
                    pass
            if score is not None:
                try:
                    root = psutil.Process(score.pid)
                    score_known.update({p.pid: p for p in root.children(recursive=True)})
                except psutil.NoSuchProcess:
                    pass
            if gen is not None and gen.poll() is not None:
                if gen.returncode:
                    generation_failed = f"{active_model}: generation exited {gen.returncode}; inspect generation.log"
                    generation_done = True
                else:
                    output = session / "models" / active_model
                    marker = read(output / "generation_complete.json")
                    if (marker.get("status") != "COMPLETE" or marker["protocol_sha256"] != protocol(output)["protocol_sha256"]
                            or marker["datasets"] != {d: COUNTS[d] for d in plan["datasets"]}):
                        raise ValueError("Invalid generation completion marker")
                stop(gen, gen_known)
                gen, gen_known = None, {}
            # Start the next engine before considering any scoring backlog.
            if gen is None and not generation_done:
                try:
                    active_model, model = next(models)
                except StopIteration:
                    generation_done = True
                else:
                    output = session / "models" / active_model
                    gen = spawn(generator(plan, model, output), output / "generation.log", environment(plan["gpu"], python=plan["python"],
                                bigcode_python=plan.get("bigcode_python", plan["python"]),
                                bigcode_backend=plan.get("bigcode_backend", "host"), bigcode_image=plan.get("bigcode_image", ""),
                                bigcode_extra_site=plan.get("bigcode_extra_site")))
            for job in jobs:
                if not job["ready"] and ready(Path(job["output"]), job["dataset"]):
                    job["ready"] = True
                    try:
                        value = score_value(Path(job["output"]), job["dataset"])
                    except Exception as exc:
                        failures[job["key"]] = f"Invalid cached score: {exc!r}"
                        quarantine(job["output"], job["dataset"])
                        job["submitted"] = True
                        continue
                    if value is not None:
                        job["submitted"] = True
                    else:
                        quarantine(job["output"], job["dataset"])
                        pending.append(job)
            if score is not None and score.poll() is not None:
                error = None
                try:
                    if score.returncode or score_value(Path(active_job["output"]), active_job["dataset"]) is None:
                        error = f"score exited {score.returncode}; inspect scoring.log/resource_guard.json"
                except Exception as exc:
                    error = f"Score validation failed: {exc!r}"
                if error:
                    failures[active_job["key"]] = error
                    quarantine(active_job["output"], active_job["dataset"])
                stop(score, score_known)
                score, score_known = None, {}
                export(session, jobs, failures)
            if score is None and pending:
                active_job = pending.pop(0)
                active_job["submitted"] = True
                root = Path(active_job["output"])
                score = spawn(scorer(plan["python"], root, active_job["dataset"]),
                              root / "scores" / active_job["dataset"] / "scoring.log",
                              environment(python=plan["python"], bigcode_python=plan.get("bigcode_python", plan["python"]),
                                          bigcode_backend=plan.get("bigcode_backend", "host"), bigcode_image=plan.get("bigcode_image", ""),
                                          bigcode_extra_site=plan.get("bigcode_extra_site")))
            write(session / "status.json", {"stage": stage, "updated_at": time.strftime("%F %T %Z"),
                  "generation": active_model if gen else ("failed" if generation_failed else "complete"),
                  "generation_pid": gen.pid if gen else None, "scoring_pid": score.pid if score else None,
                  "scoring": active_job["key"] if score else None, "pending_scoring": len(pending),
                  "failures": failures,
                  "generation_error": generation_failed})
            if generation_done and gen is None and score is None and not pending:
                count = export(session, jobs, failures)
                if count + len(failures) != len(jobs) and not generation_failed:
                    generation_failed = "Requested datasets are incomplete; no full result declared"
                stage = "failed" if failures or generation_failed else "complete"
                write(session / "status.json", {"stage": stage, "completed": count, "total": len(jobs),
                      "failures": failures, "generation_error": generation_failed,
                      "updated_at": time.strftime("%F %T %Z")})
                return 1 if stage == "failed" else 0
            time.sleep(0.2)
    except BaseException as exc:
        write(session / "status.json", {"stage": "paused" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
              "error": repr(exc), "failures": failures, "updated_at": time.strftime("%F %T %Z")})
        raise
    finally:
        for process, known in ((score, score_known), (gen, gen_known)):
            if process is not None:
                stop(process, known)


def make_plan(args):
    models = {}
    for entry in args.model:
        name, separator, raw = entry.partition("=")
        if (not separator or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
                or name in models):
            raise ValueError("Use unique NAME=/absolute/model/path entries")
        path = Path(raw).expanduser().resolve()
        if not (path / "config.json").exists() or not list(path.glob("*.safetensors")):
            raise ValueError(f"Not a complete Hugging Face checkpoint: {path}")
        models[name] = {"path": str(path), "files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for p in sorted(path.iterdir()) if p.is_file()}, "config_sha256": sha(path / "config.json")}
    sources = [p for p in HERE.glob("*.py") if not p.name.startswith("test_")]
    archived = archived_sources()
    image = args.bigcode_image
    if args.bigcode_backend == "docker" and "BigCodeBench-Hard" in selected(args.datasets):
        image = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]["Id"]
    extra = args.bigcode_extra_site.resolve() if args.bigcode_extra_site.exists() else None
    dependencies = {str(path): sha(path) for path in sorted(extra.glob("*.dist-info/METADATA"))} if extra else {}
    for python in (args.python, args.bigcode_python):
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError(f"Missing executable Python interpreter: {python}")
    return {"models": models, "datasets": selected(args.datasets), "gpu": args.gpu,
            "python": str(args.python.expanduser().absolute()), "profile": args.profile, "seed": args.seed,
            "max_tokens": args.max_tokens, "gpu_memory_utilization": args.gpu_memory_utilization,
            "bigcode_python": str(args.bigcode_python.expanduser().absolute()), "bigcode_backend": args.bigcode_backend,
            "bigcode_image": image, "bigcode_extra_site": str(extra) if extra else None,
            "bigcode_dependency_metadata": dependencies,
            "resource_policy": POLICY, "sources": {**archived, **{str(p): sha(p) for p in sorted(set(sources))}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=/absolute/checkpoint/path; repeat for multiple models")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=PYTHON)
    private = HERE / "runtime/bigcode_py310/bin/python"
    parser.add_argument("--bigcode-python", type=Path, default=private if private.exists() else PYTHON,
                        help="Isolated local BigCodeBench environment interpreter")
    parser.add_argument("--bigcode-backend", choices=("host", "docker"), default="host")
    parser.add_argument("--bigcode-image", default="bigcodebench/bigcodebench-evaluate:latest")
    parser.add_argument("--bigcode-extra-site", type=Path, default=HERE / "runtime/bigcode_packages",
                        help="Private scoring-only packages; not added to generation or training PYTHONPATH")
    parser.add_argument("--profile", choices=("comparable", "fast"), default="comparable")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.875)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-evaluations", action="store_true", help="Explicitly lift the memory-incident evaluation hold")
    parser.add_argument("--launch", action="store_true", help="Detach this queue and write queue.log/launcher.json")
    args = parser.parse_args()
    SAFETY.mkdir(parents=True, exist_ok=True)
    if args.gpu < 0 or not 0 < args.gpu_memory_utilization <= 0.90 or args.max_tokens <= 0:
        parser.error("Invalid GPU/memory/token budget; utilization must be at most 0.90")
    plan = make_plan(args)
    session = args.session.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=True)
    with (session / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = session / "plan.json"
        if previous.exists() and read(previous) != plan:
            raise RuntimeError("Plan/code/model differs; select a new session directory")
        write(previous, plan)
        hold = SAFETY / "EVALUATION_HOLD.json"
        if hold.exists() and not args.resume_evaluations:
            parser.exit(75, "Evaluation paused; use --resume-evaluations only when ready to resume.\n")
        used, total = map(int, subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
            "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip().split(","))
        if used > 500:
            raise RuntimeError(f"GPU {args.gpu} occupied ({used} MiB); no task was killed or evaluation hold lifted")
        if psutil.virtual_memory().available < POLICY["min_host_available_gib"] * 1024 ** 3:
            raise RuntimeError("Host available memory below safety floor; no evaluation launched")
        if hold.exists():
            lift_hold(session)
        if args.launch:
            argv = [str(args.python), "-B", str(Path(__file__).resolve()),
                    *[v for v in sys.argv[1:] if v not in ("--launch", "--resume-evaluations")]]
            process = spawn(argv, session / "queue.log", environment(args.gpu))
            write(session / "launcher.json", {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time(), "command": argv})
            fcntl.flock(lock, fcntl.LOCK_UN)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                marker = session / "worker_ready.json"
                if marker.exists() and read(marker).get("pid") == process.pid:
                    print(f"LAUNCHED pid={process.pid} GPU={args.gpu} session={session}", flush=True)
                    return
                if process.poll() is not None:
                    raise RuntimeError(f"Queue exited before reserving GPU; inspect {session / 'queue.log'}")
                time.sleep(0.1)
            print(f"STARTING pid={process.pid}; GPU readiness not yet confirmed; inspect {session / 'queue.log'}", flush=True)
            return
        with gpu_slot(args.gpu):
            used, total = map(int, subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip().split(","))
            if used > 500:
                raise RuntimeError(f"GPU {args.gpu} occupied ({used} MiB); no task was killed")
            print(f"GPU budget ~{total * args.gpu_memory_utilization / 1024:.1f} GiB; CPU scoring is independent", flush=True)
            write(session / "worker_ready.json", {"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                  "gpu": args.gpu, "plan_sha256": digest(plan)})

            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"signal {signum}")

            signal.signal(signal.SIGTERM, interrupted)
            raise SystemExit(run_pipeline(session, plan))


if __name__ == "__main__":
    main()
