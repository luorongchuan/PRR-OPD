"""Use the pinned generator with one persistent vLLM engine per model."""
import argparse
import os
from pathlib import Path
import sys

from eval_common import EVAL, HERE, POLICY, SAFETY, archived_sources, selected, sha


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--datasets", required=True)
    args, remaining = parser.parse_known_args()
    if (SAFETY / "EVALUATION_HOLD.json").exists():
        parser.exit(75, "Evaluation paused; no GPU generation started.\n")
    names = selected(args.datasets)
    os.environ["PRR_EVAL_DATASETS"] = ",".join(names)
    sys.path.insert(0, str(EVAL))
    import eval_14_pass1 as base
    original_protocol, original_generate = base.build_protocol, base.generate

    def build_protocol(args, model, records, eos, paths, refs):
        records = {name: records[name] for name in names}
        value = original_protocol(args, model, records, eos, paths, refs)
        value["resource_policy"] = POLICY
        value["bigcode_execution"] = {key: os.environ.get(key) for key in
            ("PRR_BIGCODE_BACKEND", "PRR_BIGCODE_IMAGE", "PRR_BIGCODE_PYTHON", "PRR_BIGCODE_EXTRA_SITE")}
        value["scoring"]["code_other"] = "pinned local sandbox; BigCodeBench official local harness with all canonical controls"
        value["source_files"].update(archived_sources())
        for path in HERE.glob("*.py"):
            if not path.name.startswith("test_"):
                value["source_files"][str(path)] = sha(path)
        value["source_files"][str(EVAL / "score_dataset_pass1.py")] = sha(EVAL / "score_dataset_pass1.py")
        return value

    def generate(args, output, model, tokenizer, records, eos, identity):
        return original_generate(args, output, model, tokenizer,
                                 {name: records[name] for name in names}, eos, identity)

    base.build_protocol, base.generate = build_protocol, generate
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
