"""Resumable single-model evaluation on six additional math benchmarks."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(value) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"Refusing to overwrite a different artifact: {path}")
        return
    pending = path.with_name(path.name + f".pending-{os.getpid()}")
    pending.write_text(encoded)
    os.replace(pending, path)


def resolve_checkpoint(item: dict) -> Path:
    path = Path(item["checkpoint"])
    if not path.is_absolute():
        path = ROOT / path
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is incomplete: {path}")
    if not list(path.glob("model*.safetensors")):
        raise FileNotFoundError(f"Checkpoint has no safetensors: {path}")
    return path.resolve()


def model_fingerprint(path: Path) -> dict:
    names = {
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "vocab.json",
        "merges.txt", "model.safetensors.index.json", "checkpoint_manifest.json",
    }
    files = [p for p in path.iterdir() if p.is_file() and (p.name in names or p.name.startswith("model-") and p.suffix == ".safetensors" or p.name == "model.safetensors")]
    if not any(p.suffix == ".safetensors" for p in files):
        raise RuntimeError(f"No model weights selected for fingerprint: {path}")
    return {p.name: file_sha(p) for p in sorted(files)}


def normalize_messages(value) -> list[dict]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, (list, tuple)):
        return [{"role": row.get("role", "user"), "content": row.get("content", "")} for row in value]
    raise TypeError(f"Unsupported prompt type: {type(value)!r}")


def load_records(path: Path, tokenizer, max_model_len: int, max_tokens: int) -> list[dict]:
    frame = pd.read_parquet(path)
    records = []
    for row_index, row in frame.iterrows():
        messages = normalize_messages(row["prompt"] if "prompt" in row else row.get("question", ""))
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) + max_tokens > max_model_len:
            raise RuntimeError(f"Prompt {row_index} requires {len(prompt_ids) + max_tokens} tokens, limit={max_model_len}")
        reward = row.get("reward_model", {}) or {}
        extra = row.get("extra_info", {}) or {}
        if hasattr(reward, "tolist"):
            reward = reward.tolist()
        if hasattr(extra, "tolist"):
            extra = extra.tolist()
        sample_id = str(extra.get("index", row_index)) if isinstance(extra, dict) else str(row_index)
        answer = reward.get("ground_truth") if isinstance(reward, dict) else None
        if answer is None:
            raise RuntimeError(f"Missing reference answer at {path}:{row_index}")
        records.append({
            "id": sample_id,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "reference_answer": str(answer),
        })
    return records


def complete_prediction(request_output, record: dict) -> dict:
    from experiments.prr_v3_calib.scoring import complete_boxes, correct, robust_answer

    if len(request_output.outputs) != 1:
        raise RuntimeError("Expected exactly one completion")
    output = request_output.outputs[0]
    text = output.text
    tokens = list(output.token_ids)
    answer = robust_answer(text)
    boxes = complete_boxes(text)
    grams = [tuple(tokens[index:index + 4]) for index in range(max(0, len(tokens) - 3))]
    finish_reason = str(output.finish_reason)
    return {
        "problem_id": record["id"],
        "prompt": record["prompt"],
        "prompt_ids": record["prompt_ids"],
        "reference_answer": record["reference_answer"],
        "generated_text": text,
        "token_ids": tokens,
        "robust_extracted_answer": answer,
        "robust_parse_success": answer is not None,
        "robust_correct": correct(answer, record["reference_answer"]),
        "response_length": len(tokens),
        "eos": finish_reason == "stop",
        "truncated": finish_reason == "length",
        "finish_reason": finish_reason,
        "stop_reason": output.stop_reason,
        "repeat_4gram_rate": 1 - len(set(grams)) / len(grams) if grams else 0.0,
        "first_complete_boxed_position": boxes[0]["start"] if boxes else None,
        "last_complete_boxed_position": boxes[-1]["start"] if boxes else None,
        "boxed_position_unit": "zero-based character offset in decoded text",
        "generation_backend": "vllm",
    }


def summarize(predictions: list[dict]) -> dict:
    total = len(predictions)
    if not total or len({row["problem_id"] for row in predictions}) != total:
        raise RuntimeError("Empty or duplicate predictions")
    correct_count = sum(row["robust_correct"] for row in predictions)
    return {
        "total": total,
        "robust_correct": correct_count,
        "robust_accuracy": correct_count / total,
        "robust_parse_success_rate": sum(row["robust_parse_success"] for row in predictions) / total,
        "eos_rate": sum(row["eos"] for row in predictions) / total,
        "truncation_rate": sum(row["truncated"] for row in predictions) / total,
        "response_length_mean": sum(row["response_length"] for row in predictions) / total,
        "repeat_4gram_rate": sum(row["repeat_4gram_rate"] for row in predictions) / total,
    }


def validate_batch(batch: dict, expected: list[dict], protocol_hash: str, start: int) -> None:
    predictions = batch["predictions"]
    if batch["protocol_sha256"] != protocol_hash or batch["start"] != start:
        raise RuntimeError("Cached batch protocol or offset mismatch")
    if [row["problem_id"] for row in predictions] != [row["id"] for row in expected]:
        raise RuntimeError("Cached batch IDs mismatch")
    if [row["reference_answer"] for row in predictions] != [row["reference_answer"] for row in expected]:
        raise RuntimeError("Cached batch references mismatch")
    if [row["prompt_ids"] for row in predictions] != [row["prompt_ids"] for row in expected]:
        raise RuntimeError("Cached batch prompts mismatch")


def build_protocol(args, model: dict, checkpoint: Path, tokenizer, records: dict) -> dict:
    manifest = read_json(HERE / "datasets_manifest.json")
    source_files = [Path(__file__), HERE / "models.json", HERE / "datasets_manifest.json", HERE / "prepare_datasets.py"]
    source_files.extend(ROOT / "experiments/prr_v3_calib" / name for name in ("scoring.py", "legacy_scoring.py"))
    eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")} - {None})
    return {
        "schema_version": 1,
        "model": model,
        "checkpoint": str(checkpoint),
        "checkpoint_files": model_fingerprint(checkpoint),
        "datasets": {
            name: {
                **manifest["datasets"][name],
                "rendered_records_sha256": digest(rows),
            }
            for name, rows in records.items()
        },
        "prompt": "Qwen chat template; enable_thinking=False; add_generation_prompt=True; dataset instruction unchanged",
        "generation": {
            "backend": "vllm",
            "backend_version": importlib.metadata.version("vllm"),
            "dtype": "bfloat16",
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "request_batch_size": args.request_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "eos_token_ids": eos_ids,
        },
        "scoring": {
            "name": "V3 robust",
            "extraction": "last complete boxed/fbox with nested braces; legacy non-boxed fallback",
            "normalization_and_verifier": "unchanged parsed_answer + math_verify",
        },
        "source_files": {str(path.resolve()): file_sha(path) for path in source_files},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.76)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be in [0.1, 0.95]")
    models = {row["key"]: row for row in read_json(HERE / "models.json")["models"]}
    if args.model_key not in models:
        parser.error(f"Unknown model key: {args.model_key}")
    model = models[args.model_key]
    checkpoint = resolve_checkpoint(model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True, padding_side="left")
    manifest = read_json(HERE / "datasets_manifest.json")
    records = {
        name: load_records(Path(manifest["datasets"][name]["path"]), tokenizer, args.max_model_len, args.max_tokens)
        for name in manifest["dataset_order"]
    }
    for name, rows in records.items():
        expected = manifest["datasets"][name]["rows"]
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise RuntimeError(f"Incomplete rendered dataset: {name}")
    protocol = build_protocol(args, model, checkpoint, tokenizer, records)
    protocol_hash = digest(protocol)
    output = HERE / "results" / args.model_key / protocol_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(output / "protocol.json", protocol)
        if args.prepare_only:
            print(f"PREPARED {args.model_key} {output}", flush=True)
            return
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdecimal() or str(int(visible)) != visible:
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to one explicit physical GPU index")
        if (output / "summary.json").exists():
            summary = read_json(output / "summary.json")
            if summary.get("status") != "COMPLETE" or summary.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("Invalid completed summary")
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return

        from vllm import LLM, SamplingParams

        print(f"LOAD model={args.model_key} checkpoint={checkpoint} physical_gpu={visible}", flush=True)
        engine = LLM(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False,
        )
        sampling = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            stop_token_ids=protocol["generation"]["eos_token_ids"],
            seed=args.seed,
        )
        dataset_summaries = {}
        for dataset, rows in records.items():
            predictions = []
            for start in range(0, len(rows), args.request_batch_size):
                expected = rows[start:start + args.request_batch_size]
                batch_path = output / dataset / f"batch_{start:06d}.json"
                if batch_path.exists():
                    batch = read_json(batch_path)
                    validate_batch(batch, expected, protocol_hash, start)
                else:
                    began = time.perf_counter()
                    print(f"GENERATE model={args.model_key} dataset={dataset} start={start} count={len(expected)}", flush=True)
                    generated = engine.generate([row["prompt"] for row in expected], sampling, use_tqdm=True)
                    if len(generated) != len(expected):
                        raise RuntimeError("vLLM result count mismatch")
                    batch = {
                        "protocol_sha256": protocol_hash,
                        "start": start,
                        "seconds": time.perf_counter() - began,
                        "physical_gpu": visible,
                        "predictions": [complete_prediction(output_row, record) for output_row, record in zip(generated, expected)],
                    }
                    validate_batch(batch, expected, protocol_hash, start)
                    write_once(batch_path, batch)
                predictions.extend(batch["predictions"])
                print(f"PROGRESS model={args.model_key} dataset={dataset} done={len(predictions)}/{len(rows)}", flush=True)
            dataset_summaries[dataset] = summarize(predictions)
            write_once(output / dataset / "summary.json", dataset_summaries[dataset])
        summary = {
            "status": "COMPLETE",
            "protocol_sha256": protocol_hash,
            "model_key": args.model_key,
            "label": model["label"],
            "checkpoint": str(checkpoint),
            "datasets": dataset_summaries,
        }
        write_once(output / "summary.json", summary)
        print("RESULT", json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
