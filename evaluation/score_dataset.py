"""Use historical extraction and guarded, validated local code execution."""
from pathlib import Path
import os
import subprocess
import resource
import sys

from eval_common import DATA_ROOT, EVAL, HERE, POLICY, SAFETY, read


def repaired_evalplus(root, protocol, dataset, sample_roots, official, data_root, workers):
    from repairs.scoring_v4.rescore import plus_summary
    return plus_summary(root, protocol, dataset, sample_roots)


def repaired_bigcode(protocol, dataset, sample_roots, legacy):
    if os.environ.get("PRR_BIGCODE_BACKEND", "host") == "docker":
        return bigcode_local(protocol, dataset, sample_roots, legacy)
    from repairs.scoring_v4.rescore import bigcode_summary
    os.environ["PRR_V4_BIGCODE_PYTHON"] = os.environ["PRR_BIGCODE_PYTHON"]
    return bigcode_summary(sample_roots[0].parents[2], protocol, sample_roots)


def bigcode_local(protocol, dataset, sample_roots, legacy):
    dataset_path = DATA_ROOT / "BigCodeBench-Hard/test.parquet"
    python = os.environ["PRR_BIGCODE_PYTHON"]
    folder = HERE / "scorers/local_bigcode"
    cache = HERE / "runtime/bigcodebench"
    cache.mkdir(parents=True, exist_ok=True)
    output = sample_roots[0].parents[2] / "scores/BigCodeBench-Hard"
    output.mkdir(parents=True, exist_ok=True)
    # Both canonical controls and predictions run in the isolated local sandbox.
    subprocess.run(["bash", str(folder / "sandbox_bigcode.sh"), "controls", str(dataset_path),
                    str(sample_roots[0] / "predictions.jsonl"), str(cache), str(output / "controls"), python], check=True)
    totals, details, fingerprints = [], [], set()
    for index, sample_root in enumerate(sample_roots):
        draw = output / f"sample_{index:02d}"
        subprocess.run(["bash", str(folder / "sandbox_bigcode.sh"), "score", str(dataset_path),
                        str(sample_root / "predictions.jsonl"), str(cache), str(draw), python], check=True)
        row = read(draw / "summary.json")
        if row["total"] != 148 or row["status"] != "COMPLETE" or row["canonical_controls_passed"] != 148:
            raise RuntimeError("Incomplete or unvalidated BigCodeBench local score")
        totals.append(row["correct"])
        fingerprints.add(row["environment_fingerprint"])
        details.append(str(draw / "per_question.jsonl"))
    if len(fingerprints) != 1 or len(totals) != protocol["generation"]["n"] or protocol["datasets"][dataset]["rows"] != 148:
        raise RuntimeError("Wrong BigCodeBench sample/task count")
    return {"dataset": dataset, "protocol_sha256": protocol["protocol_sha256"],
            "total_questions": 148, "samples_per_question": len(totals), "correct_samples": sum(totals),
            "sampled_pass_at_1": sum(totals) / (148 * len(totals)),
            "scorer": "pinned official BigCodeBench local harness, instruct/hard, calibrated=false",
            "canonical_controls_passed": 148, "uploaded": False,
            "environment_fingerprint": next(iter(fingerprints)),
            "execution_policy": POLICY["bigcodebench"], "draw_details": details,
            "metric": "sampled Pass@1 = mean_i(c_i/16)"}


def main():
    if not any(flag in sys.argv for flag in ("--help", "-h")):
        if (SAFETY / "EVALUATION_HOLD.json").exists():
            raise SystemExit("Evaluation is paused; use the unified guarded runner when ready")
        if os.environ.get("PRR_SCORING_GUARD") != "1" or resource.getrlimit(resource.RLIMIT_AS)[1] == resource.RLIM_INFINITY:
            raise RuntimeError("Direct unguarded scoring is refused; use run_eval.sh")
    os.environ.pop("ALLOW_BIGCODE_REMOTE_UPLOAD", None)
    os.environ.setdefault("PRR_BIGCODE_PYTHON", sys.executable)
    sys.path.insert(0, str(EVAL))
    import score_dataset_pass1 as original
    original.evalplus_score = repaired_evalplus
    original.bigcode_score = repaired_bigcode
    original.main()


if __name__ == "__main__":
    main()
