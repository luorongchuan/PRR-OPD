#!/usr/bin/env python3
"""Generate n=16 samples for the frozen 14-dataset PRR/OPD validation protocol.

This is adapted from OPD-Math-COLING/experiments/prr_clean_v1/sampling_pass_at_1/eval.py,
but accepts an arbitrary Hugging Face model/checkpoint path instead of the old hard-coded
step-551 checkpoints. Scoring is done by score_14_pass1.py.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from model_compat import qwen3_hf_overrides, verify_loaded_rope

HERE = Path(__file__).resolve().parent
VENDOR_ROOT = HERE / "vendor"
OPD_ROOT_DEFAULT = VENDOR_ROOT
DATA_ROOT_DEFAULT = Path(os.environ.get("PRR_DATA_ROOT", HERE.parents[2] / "data/evaluation"))


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Refusing to replace different artifact: {path}")
        return
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
)


def prepare_tokenizer_path(model_path: Path, output: Path) -> Path:
    """Create a tokenizer-only compatibility copy for Transformers 4.56."""
    config_path = model_path / "tokenizer_config.json"
    if not config_path.is_file():
        return model_path

    config = read(config_path)
    if not isinstance(config.get("extra_special_tokens"), list):
        return model_path

    compat_path = output / ".tokenizer_compat"
    compat_path.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        source = model_path / name
        if not source.is_file():
            continue
        target = compat_path / name
        if name == "tokenizer_config.json":
            sanitized = dict(config)
            sanitized.pop("extra_special_tokens", None)
            write_once(target, sanitized)
            continue
        if target.exists():
            if sha256(target) != sha256(source):
                raise RuntimeError(f"Tokenizer compatibility file changed: {target}")
            continue
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)

    print(
        f"TOKENIZER_COMPAT source={model_path} tokenizer={compat_path} "
        "removed=list-valued extra_special_tokens",
        flush=True,
    )
    return compat_path


def resolve_model_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path, path / "model"]
    latest = path / "latest"
    if latest.is_file():
        latest_name = latest.read_text(encoding="utf-8").strip()
        candidates.append(path / latest_name / "model")
    for candidate in candidates:
        if (candidate / "config.json").is_file() and list(candidate.glob("*.safetensors")):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find HF model under {path}; expected MODEL_PATH, MODEL_PATH/model, "
        "or MODEL_PATH/$(cat latest)/model with config.json and safetensors."
    )


def model_fingerprint(path: Path) -> dict[str, Any]:
    names = {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "vocab.json",
        "merges.txt", "model.safetensors.index.json", "checkpoint_manifest.json",
    }
    files = {}
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        if item.name in names:
            files[item.name] = {"sha256": sha256(item), "bytes": item.stat().st_size}
        elif item.name == "model.safetensors" or item.name.startswith("model-") and item.suffix == ".safetensors":
            # Large files: store size/mtime to avoid hashing many GB every launch.
            stat = item.stat()
            files[item.name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"No model weights selected for fingerprint: {path}")
    return files


def setup_reference_paths(opd_root: Path, data_root: Path):
    os.environ.setdefault("PRR_CODE_DATA_ROOT", str(data_root))
    # In this KDFlow port, opd_root points to prr-opd-validation/vendor by default.
    legacy = opd_root / "test-code"
    official = legacy / "official_validation"
    other = opd_root / "other-test"
    for required in (legacy / "common.py", official / "official_utils.py", other / "evaluate.py"):
        if not required.is_file():
            raise FileNotFoundError(f"Missing vendored validation dependency: {required}")
    for directory in (opd_root, legacy, official):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import common as code_common  # noqa: PLC0415
    from official_utils import OFFICIAL_DATA, build_evalplus_records  # noqa: PLC0415
    return legacy, official, other, code_common, OFFICIAL_DATA, build_evalplus_records


MATH = ("MATH-500", "AMC23", "AIME24", "AIME25", "HMMT25", "Minerva", "Olympiad-Bench", "GSM8K")
CODE = ("HumanEvalPlus", "MBPPPlus", "LiveCodeBench", "CRUXEval-I", "CRUXEval-O", "BigCodeBench-Hard")


def load_math(other: Path, data_root: Path, tokenizer, max_tokens: int, max_model_len: int) -> dict[str, list[dict]]:
    spec = importlib.util.spec_from_file_location("prr_14_math_loader", other / "evaluate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load math evaluator: {other / 'evaluate.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = read(other / "datasets_manifest.json")
    rows_expected = {"MATH-500": 500, "AMC23": 83}
    rows_expected.update({name: manifest["datasets"][name]["rows"] for name in MATH[2:]})
    result = {}
    for name in MATH:
        if name not in os.environ.get("PRR_EVAL_DATASETS", ",".join(MATH + CODE)).split(","):
            continue
        rows = module.load_records(data_root / name / "test.parquet", tokenizer, max_model_len, max_tokens)
        if len(rows) != rows_expected[name] or len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError(f"Unexpected math dataset rows/ids: {name}")
        result[name] = rows
    return result


def load_code(code_common, build_evalplus_records, tokenizer, max_tokens: int, max_model_len: int) -> dict[str, list[dict]]:
    selected = set(os.environ.get("PRR_EVAL_DATASETS", ",".join(MATH + CODE)).split(","))
    records = {}
    for name in ("HumanEvalPlus", "MBPPPlus"):
        if name in selected:
            records[name] = build_evalplus_records(tokenizer, name)
    if "LiveCodeBench" in selected:
        records["LiveCodeBench"] = [{"dataset": "LiveCodeBench", "id": row["question_id"],
            "prompt": code_common.chat_prompt(tokenizer, code_common.lcb_user_prompt(row), system=code_common.SYSTEM_CODE),
            "stop": "eos"} for row in code_common.load_jsonl(code_common.DATASETS["LiveCodeBench"])]
    import pyarrow.parquet as pq
    if selected & {"CRUXEval-I", "CRUXEval-O"}:
        crux_rows = pq.read_table(code_common.DATASETS["CRUXEval"]).to_pylist()
        for name in ("CRUXEval-I", "CRUXEval-O"):
            if name in selected:
                records[name] = [{"dataset": name, "id": row["id"],
                    "prompt": code_common.chat_prompt(tokenizer,
                        code_common.crux_input_prompt(row["code"], row["output"]) if name == "CRUXEval-I"
                        else code_common.crux_output_prompt(row["code"], row["input"])),
                    "stop": "answer_tag"} for row in crux_rows]
    if "BigCodeBench-Hard" in selected:
        records["BigCodeBench-Hard"] = [{"dataset": "BigCodeBench-Hard", "id": row["task_id"],
            "prompt": code_common.chat_prompt(tokenizer,
                f"{code_common.INSTRUCTION_PREFIX}\n{row['instruct_prompt'].strip()}", prefill=code_common.RESPONSE_PREFIX),
            "stop": "code_fence", "entry_point": row["entry_point"]}
            for row in pq.read_table(code_common.DATASETS["BigCodeBench-Hard"]).to_pylist()]
    result = {}
    for name in CODE:
        if name not in selected:
            continue
        rows = records[name]
        expected = code_common.EXPECTED_ROWS[name]
        if len(rows) != expected or len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError(f"Unexpected code dataset rows/ids: {name}")
        for row in rows:
            tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
            if len(tokens) + max_tokens > max_model_len:
                raise RuntimeError(f"Code prompt exceeds context: {name}/{row['id']}")
        result[name] = rows
    return result


def build_protocol(args, model_path: Path, records: dict, eos: list[int], paths: dict[str, Path], refs: dict[str, Any]) -> dict:
    legacy = refs["legacy"]
    official = refs["official"]
    other = refs["other"]
    code_common = refs["code_common"]
    sources = [
        Path(__file__), Path(__file__).with_name("score_14_pass1.py"),
        Path(__file__).with_name("model_compat.py"),
        other / "evaluate.py", other / "datasets_manifest.json",
        refs["opd_root"] / "math_scoring/scoring.py",
        refs["opd_root"] / "math_scoring/legacy_scoring.py",
        legacy / "common.py", legacy / "score.py",
        official / "official_utils.py", legacy / "sandbox_score.sh",
    ]
    return {
        "schema_version": 2,
        "protocol_name": "prr_clean_v1_sampling_pass_at_1_14datasets_adapted_for_kdflow",
        "model_path": str(model_path),
        "checkpoint": str(model_path),
        "model_fingerprint": model_fingerprint(model_path),
        "model_loading": {"hf_overrides": qwen3_hf_overrides(model_path)},
        "dataset_order": list(records),
        "datasets": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256(paths[name]) if paths[name].is_file() else None,
                "rows": len(rows),
                "rendered_prompts_sha256": digest([(row["id"], row["prompt"]) for row in rows]),
            }
            for name, rows in records.items()
        },
        "generation": {
            "backend": "vllm",
            "backend_version": importlib.metadata.version("vllm"),
            "dtype": args.dtype,
            "thinking": False,
            "n": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "request_batch_size": args.request_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "eos_token_ids": eos,
        },
        "scoring": {
            "math": "V3 robust, last complete boxed/fbox plus legacy fallback",
            "code_plus": "official EvalPlus differential harness",
            "code_other": "pinned local benchmark harness; BigCodeBench official local checks",
            "metric": "sampled Pass@1 = mean_i(c_i/16); no majority vote",
        },
        "source_files": {str(path): code_common.sha256(path) if path.is_file() else None for path in sources},
    }


def validate_batch(path: Path, expected: list[dict], protocol_hash: str, start: int, n_samples: int):
    batch = read(path)
    if batch.get("protocol_sha256") != protocol_hash or batch.get("start") != start:
        raise RuntimeError(f"Cached batch provenance mismatch: {path}")
    if [row["id"] for row in batch.get("predictions", [])] != [row["id"] for row in expected]:
        raise RuntimeError(f"Cached batch ID mismatch: {path}")
    for item, record in zip(batch["predictions"], expected):
        if item["prompt_sha256"] != hashlib.sha256(record["prompt"].encode()).hexdigest():
            raise RuntimeError(f"Cached batch prompt mismatch: {path}")
        if len(item["samples"]) != n_samples:
            raise RuntimeError(f"Cached batch sample count mismatch: {path}")
    return batch


def generate(
    args, output: Path, model_path: Path, tokenizer_path: Path, records: dict, eos: list[int], protocol_hash: str
):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one A100 GPU for evaluation")
    from vllm import LLM, SamplingParams

    print(f"LOAD model={model_path} tokenizer={tokenizer_path} gpu={visible} output={output}", flush=True)
    overrides = qwen3_hf_overrides(model_path)
    if overrides:
        print("MODEL_CONFIG_COMPAT", json.dumps(overrides, sort_keys=True), flush=True)
    engine = LLM(
        model=str(model_path), tokenizer=str(tokenizer_path), tokenizer_mode="auto",
        trust_remote_code=True, dtype=args.dtype, tensor_parallel_size=1, seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False,
        hf_overrides=overrides,
    )
    verify_loaded_rope(engine.llm_engine.model_config.hf_config, overrides)
    for name, rows in records.items():
        for start in range(0, len(rows), args.request_batch_size):
            group = rows[start:start + args.request_batch_size]
            path = output / "generation" / name / f"batch_{start:06d}.json"
            if path.exists():
                validate_batch(path, group, protocol_hash, start, args.n_samples)
                print(f"CACHED {name} {min(start + len(group), len(rows))}/{len(rows)}", flush=True)
                continue
            stop = (["\n```\n"] if group[0].get("stop") == "code_fence" else
                    ["[/ANSWER]"] if group[0].get("stop") == "answer_tag" else None)
            sampling = SamplingParams(
                n=args.n_samples, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                max_tokens=args.max_tokens, stop=stop, stop_token_ids=eos, seed=args.seed,
            )
            began = time.perf_counter()
            outputs = engine.generate([row["prompt"] for row in group], sampling, use_tqdm=True)
            if len(outputs) != len(group):
                raise RuntimeError(f"vLLM returned wrong number of prompts for {name}")
            predictions = []
            for record, result in zip(group, outputs):
                if len(result.outputs) != args.n_samples:
                    raise RuntimeError(f"vLLM returned fewer than {args.n_samples} samples: {name}/{record['id']}")
                predictions.append({
                    "id": record["id"],
                    "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                    **({"reference_answer": record["reference_answer"]} if name in MATH else {}),
                    "samples": [
                        {
                            "sample_index": index,
                            "text": sample.text,
                            "completion_tokens": len(sample.token_ids),
                            "finish_reason": sample.finish_reason,
                            "stop_reason": sample.stop_reason,
                        }
                        for index, sample in enumerate(result.outputs)
                    ],
                })
            write_once(path, {
                "protocol_sha256": protocol_hash,
                "start": start,
                "seconds": time.perf_counter() - began,
                "predictions": predictions,
            })
            print(f"GENERATED {name} {min(start + len(group), len(rows))}/{len(rows)}", flush=True)
    write_once(output / "generation_complete.json", {
        "status": "COMPLETE",
        "protocol_sha256": protocol_hash,
        "datasets": {name: len(rows) for name, rows in records.items()},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "generate"), default="generate")
    parser.add_argument("--opd-root", type=Path, default=OPD_ROOT_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    if args.n_samples != 16:
        parser.error("This protocol is pinned to n=16 samples")
    if args.max_num_seqs < args.n_samples or args.request_batch_size < 1:
        parser.error("Invalid batching: max_num_seqs must be >= n_samples and request_batch_size >= 1")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model_path)
    tokenizer_path = prepare_tokenizer_path(model_path, output)
    refs_tuple = setup_reference_paths(args.opd_root.expanduser().resolve(), args.data_root.expanduser().resolve())
    legacy, official, other, code_common, official_data, build_evalplus_records = refs_tuple

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True, padding_side="left"
    )
    records = {
        **load_math(other, args.data_root.expanduser().resolve(), tokenizer, args.max_tokens, args.max_model_len),
        **load_code(code_common, build_evalplus_records, tokenizer, args.max_tokens, args.max_model_len),
    }
    eos = sorted({
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    } - {None})

    paths = {name: args.data_root.expanduser().resolve() / name / "test.parquet" for name in MATH}
    paths.update({
        "HumanEvalPlus": official_data["HumanEvalPlus"],
        "MBPPPlus": official_data["MBPPPlus"],
        "LiveCodeBench": code_common.DATASETS["LiveCodeBench"],
        "CRUXEval-I": code_common.DATASETS["CRUXEval"],
        "CRUXEval-O": code_common.DATASETS["CRUXEval"],
        "BigCodeBench-Hard": code_common.DATASETS["BigCodeBench-Hard"],
    })
    refs = {"legacy": legacy, "official": official, "other": other, "code_common": code_common, "opd_root": args.opd_root.expanduser().resolve()}
    protocol = build_protocol(args, model_path, records, eos, paths, refs)
    protocol_hash = digest(protocol)

    with (output / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(output / "protocol.json", {**protocol, "protocol_sha256": protocol_hash})
        print(f"PREPARED protocol={protocol_hash} datasets={len(records)} output={output}", flush=True)
        if args.stage == "prepare":
            return
        if (output / "generation_complete.json").exists():
            marker = read(output / "generation_complete.json")
            if marker.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("Existing generation uses a different protocol")
            print("GENERATION_ALREADY_COMPLETE", flush=True)
            return
        generate(args, output, model_path, tokenizer_path, records, eos, protocol_hash)


if __name__ == "__main__":
    main()
