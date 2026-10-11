"""Shared paths, atomic metadata and strict evaluation artifact validation."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WORKSPACE = REPO.parent
EVAL = HERE / "scorers/pinned14"
PYTHON = Path(os.environ.get("PRR_EVAL_PYTHON", sys.executable))
DATA_ROOT = Path(os.environ.get("PRR_DATA_ROOT", REPO / "data/evaluation")).expanduser().resolve()
SAFETY = Path(os.environ.get("PRR_EVAL_LOCK_ROOT", REPO / "runtime/evaluation_locks"))
COUNTS = {"MATH-500": 500, "AMC23": 83, "AIME24": 30, "AIME25": 30,
          "HMMT25": 30, "Minerva": 272, "Olympiad-Bench": 675, "GSM8K": 1319,
          "HumanEvalPlus": 164, "MBPPPlus": 378, "LiveCodeBench": 175,
          "CRUXEval-I": 800, "CRUXEval-O": 800, "BigCodeBench-Hard": 148}
POLICY = {"version": "local-scorers-v3", "workers": 2, "parallel_jobs": 1,
          "sample_draws": "serial", "per_process_address_space_gib": 4,
          "tree_rss_stop_gib": 24, "min_host_available_gib": 128,
          "tree_limit_kind": "polled RSS, not an atomic cgroup limit",
          "bigcodebench": {"backend": "local", "workers": 1, "per_process_as_gib": 30,
                           "tree_rss_stop_gib": 24, "canonical_controls": "all 148 must pass",
                           "calibrated": False, "min_time_limit": 1, "stack_limit_mib": 10}}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archived_sources():
    manifest = HERE / "scorers/SOURCE_MANIFEST.json"
    sources = {str(manifest): sha(manifest)}
    for row in read(manifest)["files"]:
        path = HERE / row["target"]
        if sha(path) != row["sha256"]:
            raise RuntimeError(f"Archived evaluator was changed: {path}")
        sources[str(path)] = row["sha256"]
    for path in (HERE / "scorers/local_bigcode").glob("*"):
        if path.suffix in (".py", ".sh"):
            sources[str(path)] = sha(path)
    sources[str(HERE / "score_dataset.py")] = sha(HERE / "score_dataset.py")
    for path in (HERE / "repairs/scoring_v4").rglob("*"):
        if path.is_file() and (path.suffix in (".py", ".sh", ".json", ".txt")
                               or path.parent.name == "assets"):
            sources[str(path)] = sha(path)
    return sources


def selected(value):
    names = list(COUNTS) if value == "all" else value.split(",")
    if not names or len(set(names)) != len(names) or not set(names) <= set(COUNTS):
        raise ValueError("Datasets must be 'all' or unique comma-separated names from the 14 tasks")
    return names


def protocol(root):
    value = read(Path(root) / "protocol.json")
    if digest({k: v for k, v in value.items() if k != "protocol_sha256"}) != value["protocol_sha256"]:
        raise ValueError("Protocol hash mismatch")
    if value.get("resource_policy") != POLICY:
        raise ValueError("Resource policy differs; historical scores need a separate session")
    return value


def ready(root, dataset):
    root = Path(root)
    if not (root / "protocol.json").exists():
        return False
    value = protocol(root)
    count = value["datasets"][dataset]["rows"]
    if count != COUNTS[dataset]:
        raise ValueError(f"Dataset count mismatch: {dataset}")
    size = value["generation"]["request_batch_size"]
    if not all((root / "generation" / dataset / f"batch_{start:06d}.json").exists()
               for start in range(0, count, size)):
        return False
    seen = set()
    for start in range(0, count, size):
        batch = read(root / "generation" / dataset / f"batch_{start:06d}.json")
        predictions = batch["predictions"]
        if (batch["protocol_sha256"] != value["protocol_sha256"] or batch["start"] != start
                or len(predictions) != min(size, count - start)):
            raise ValueError(f"Batch provenance/count mismatch: {dataset}")
        for row in predictions:
            if row["id"] in seen or len(row["samples"]) != 16:
                raise ValueError(f"Duplicate ID or missing samples: {dataset}")
            if [s["sample_index"] for s in row["samples"]] != list(range(16)):
                raise ValueError(f"Invalid sample indices: {dataset}")
            seen.add(row["id"])
    return True


def score_value(root, dataset):
    path = Path(root) / "scores" / dataset / "summary.json"
    if not path.exists():
        return None
    value, reference = read(path), protocol(root)
    total = COUNTS[dataset] * 16
    correct, score = value["correct_samples"], value["sampled_pass_at_1"]
    if (value["protocol_sha256"] != reference["protocol_sha256"]
            or value["total_questions"] != COUNTS[dataset] or value["samples_per_question"] != 16
            or not isinstance(correct, int) or not 0 <= correct <= total
            or not math.isfinite(score) or abs(score - correct / total) > 1e-10):
        raise ValueError(f"Invalid score summary: {path}")
    receipt = Path(root) / "scores" / dataset / "resource_guard.json"
    if not receipt.exists() or not read(receipt).get("completed"):
        return None
    return score * 100
