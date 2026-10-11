#!/usr/bin/env python3
"""Generate one fixed Qwen3-8B response for every frozen DAPO prompt."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from transformers import AutoTokenizer


EXPECTED_ROWS = 17629
EXPECTED_SOURCE_SHA256 = "2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:31005")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_existing(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            record = json.loads(line)
            if record["source_index"] != expected_index:
                raise ValueError(
                    f"Resume JSONL is not sequential at line {expected_index + 1}"
                )
            records.append(record)
    return records


def response_texts(payload, expected):
    if isinstance(payload, list):
        texts = [item["text"] for item in payload]
    elif isinstance(payload, dict) and isinstance(payload.get("text"), list):
        texts = payload["text"]
    elif isinstance(payload, dict) and isinstance(payload.get("text"), str):
        texts = [payload["text"]]
    else:
        raise ValueError(f"Unexpected SGLang response shape: {type(payload)}")
    if len(texts) != expected:
        raise ValueError(f"Expected {expected} responses, received {len(texts)}")
    return texts


def main():
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)
    jsonl = Path(args.jsonl)
    metadata_path = Path(args.metadata)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(source)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Frozen source SHA256 mismatch: {source_sha}")

    source_table = pq.read_table(source)
    if source_table.num_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} source rows, found {source_table.num_rows}"
        )
    source_rows = source_table.to_pylist()
    records = read_existing(jsonl)
    if len(records) > len(source_rows):
        raise ValueError("Resume JSONL contains more rows than the source parquet")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    clipped_responses = sum(bool(record["response_clipped"]) for record in records)

    with jsonl.open("a", encoding="utf-8") as handle:
        for start in range(len(records), len(source_rows), args.batch_size):
            batch_rows = source_rows[start : start + args.batch_size]
            prompts = []
            for offset, row in enumerate(batch_rows):
                prompt = tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_tokens = len(
                    tokenizer.encode(prompt, add_special_tokens=False)
                )
                if prompt_tokens > args.max_prompt_tokens:
                    raise ValueError(
                        f"Prompt {start + offset} has {prompt_tokens} tokens, "
                        f"above the {args.max_prompt_tokens}-token protocol limit"
                    )
                prompts.append(prompt)

            request = {
                "text": prompts,
                "sampling_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_response_tokens,
                    "sampling_seed": args.seed,
                },
            }
            result = requests.post(
                f"{args.server_url.rstrip('/')}/generate",
                json=request,
                timeout=7200,
            )
            result.raise_for_status()
            texts = response_texts(result.json(), len(batch_rows))

            for offset, (row, text) in enumerate(zip(batch_rows, texts)):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                clipped = len(token_ids) >= args.max_response_tokens
                if clipped:
                    # Reserve one position for the assistant EOS inserted by SFTDataset.
                    token_ids = token_ids[: args.max_response_tokens - 1]
                    text = tokenizer.decode(token_ids, skip_special_tokens=False)
                    clipped_responses += 1
                record = {
                    "source_index": start + offset,
                    "source_id": row["extra_info"]["index"],
                    "prompt": row["prompt"],
                    "teacher_response": text,
                    "response_tokens": len(token_ids),
                    "response_clipped": clipped,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            handle.flush()
            print(f"generated {len(records)}/{len(source_rows)}", flush=True)

    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"Generated {len(records)} rows, expected {EXPECTED_ROWS}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary, compression="zstd")
    temporary.replace(output)

    response_lengths = [record["response_tokens"] for record in records]
    metadata = {
        "dataset": "Teacher-SFT on frozen cleaned DAPO-Math prompts",
        "source_path": str(source),
        "source_sha256": source_sha,
        "rows": len(records),
        "teacher_model": str(args.tokenizer),
        "enable_thinking": False,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sampling_seed": args.seed,
        "clipped_responses": clipped_responses,
        "mean_response_tokens": statistics.fmean(response_lengths),
        "max_observed_response_tokens": max(response_lengths),
        "output_sha256": sha256(output),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
