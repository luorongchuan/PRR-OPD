"""Build a CPU-only scoring command with global serialization and memory limits."""
from eval_common import DATA_ROOT, EVAL, HERE, POLICY


def command(python, root, dataset):
    report = root / "scores" / dataset / "resource_guard.json"
    limit = POLICY["bigcodebench"]["per_process_as_gib"] if dataset == "BigCodeBench-Hard" else POLICY["per_process_address_space_gib"]
    return [str(python), "-B", str(HERE / "memory_guard.py"), "--wait-lock",
            "--report", str(report), "--per-process-gib", str(limit),
            "--tree-gib", str(POLICY["tree_rss_stop_gib"]),
            "--min-free-gib", str(POLICY["min_host_available_gib"]), "--",
            str(python), "-B", str(HERE / "score_dataset.py"),
            "--output", str(root), "--dataset", dataset, "--opd-root", str(EVAL / "vendor"),
            "--data-root", str(DATA_ROOT), "--workers", "2"]
