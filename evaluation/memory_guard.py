"""CPU-only scoring supervisor; isolated from frozen benchmark sources."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import psutil

from eval_common import SAFETY

HERE = SAFETY
GIB = 1024 ** 3


def stop(process, known):
    # Snapshot descendants before killing the coordinator that owns them.
    try:
        known.update({p.pid: p for p in psutil.Process(process.pid).children(recursive=True)})
    except psutil.NoSuchProcess:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for child in known.values():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(known.values()), timeout=2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=5)


def supervise(command, report, process_limit=4 * GIB, tree_limit=24 * GIB,
              min_available=128 * GIB, interval=0.1, hold_file=None):
    if psutil.virtual_memory().available < min_available:
        raise RuntimeError("Not enough free host memory to start CPU scoring")

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = process_limit if hard == resource.RLIM_INFINITY else min(hard, process_limit)

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "EVALPLUS_MAX_MEMORY_BYTES": str(cap), "PRR_SCORING_GUARD": "1"}
    process = subprocess.Popen(command, env=env, start_new_session=True, preexec_fn=limits)
    known = {}
    peak = 0
    reason = "command_exit"
    try:
        while process.poll() is None:
            if hold_file is not None and hold_file.exists():
                reason = "evaluation_paused"
                break
            rss = 0
            try:
                root = psutil.Process(process.pid)
                known.update({p.pid: p for p in [root, *root.children(recursive=True)]})
            except psutil.NoSuchProcess:
                pass
            for child in known.values():
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, rss)
            if rss > tree_limit:
                reason = "process_tree_memory_limit"
                break
            if psutil.virtual_memory().available < min_available:
                reason = "host_low_available_memory"
                break
            time.sleep(interval)
    finally:
        stop(process, known)
    code = process.returncode if reason == "command_exit" else 137
    value = {"command": command, "returncode": code, "reason": reason,
             "resource_policy_version": "cpu-memory-guard-v2",
             "peak_tree_rss_bytes": peak, "per_process_address_space_limit_bytes": cap,
             "tree_rss_stop_threshold_bytes": tree_limit, "min_host_available_bytes": min_available,
             "tree_limit_kind": "polled conservative RSS; not an atomic cgroup limit",
             "cuda_visible_devices": "", "completed": code == 0}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(value, indent=2) + "\n")
    return code


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-process-gib", type=float, default=4)
    parser.add_argument("--tree-gib", type=float, default=24)
    parser.add_argument("--min-free-gib", type=float, default=128)
    parser.add_argument("--wait-lock", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if (HERE / "EVALUATION_HOLD.json").exists():
        parser.exit(75, "Evaluation is paused; explicit user approval is required to resume.\n")
    if not all(math.isfinite(x) and x > 0 for x in
               (args.per_process_gib, args.tree_gib, args.min_free_gib)):
        parser.error("Memory budgets must be positive and finite")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A CPU scoring command is required after --")
    with (HERE / ".cpu_scoring.lock").open("a") as lock:
        while True:
            if (HERE / "EVALUATION_HOLD.json").exists():
                parser.exit(75, "Evaluation paused while waiting for the scoring lock.\n")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not args.wait_lock:
                    parser.exit(75, "Another guarded CPU scorer is running; parallel scoring is refused.\n")
                time.sleep(0.2)
        raise SystemExit(supervise(command, args.report,
            int(args.per_process_gib * GIB), int(args.tree_gib * GIB), int(args.min_free_gib * GIB),
            hold_file=HERE / "EVALUATION_HOLD.json"))


if __name__ == "__main__":
    main()
