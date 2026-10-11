"""Shared, frozen protocol helpers for the step-551 code evaluation."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4] if len(HERE.parents) > 4 else Path("/")
DATA_ROOT = Path(
    os.environ.get("PRR_CODE_DATA_ROOT", ROOT.parent / "datasets" / "test_data")
)
RESULTS_ROOT = HERE / "results"

METHODS = {
    "opd": HERE.parent / "opd_20260920_191737_930141160_gpu6",
    "prr": HERE.parent / "prr_20260920_191411_124857465_gpu4",
}
CHECKPOINTS = {name: run / "checkpoints" / "step_551" for name, run in METHODS.items()}

DATASETS = {
    "HumanEvalPlus": DATA_ROOT / "HumanEvalPlus" / "test.parquet",
    "MBPPPlus": DATA_ROOT / "MBPPPlus" / "test.parquet",
    "LiveCodeBench": DATA_ROOT / "LiveCodeBench" / "raw" / "test6.jsonl",
    "CRUXEval": DATA_ROOT / "CRUXEval" / "test.parquet",
    "BigCodeBench-Hard": DATA_ROOT / "BigCodeBench-Hard" / "test.parquet",
}

EXPECTED_ROWS = {
    "HumanEvalPlus": 164,
    "MBPPPlus": 378,
    "LiveCodeBench": 175,
    "CRUXEval-I": 800,
    "CRUXEval-O": 800,
    "BigCodeBench-Hard": 148,
}

SYSTEM_CODE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:\n```python\n"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        f.write(payload)
        temp = Path(f.name)
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temp = Path(f.name)
    os.replace(temp, path)


def write_once(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Immutable artifact changed: {path}")
        return
    atomic_json(path, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_files(checkpoint: Path) -> dict[str, str]:
    names = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    names += sorted(p.name for p in checkpoint.glob("model*.safetensors"))
    files = {name: sha256(checkpoint / name) for name in names if (checkpoint / name).is_file()}
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise RuntimeError(f"Incomplete checkpoint: {checkpoint}")
    return files


def verify_inputs() -> None:
    for method, checkpoint in CHECKPOINTS.items():
        complete = read_json(METHODS[method] / "complete.json")
        expected_method = "opd" if method == "opd" else "prr_gamma2"
        if complete.get("method") != expected_method or complete.get("step") != 551:
            raise RuntimeError(f"Wrong completion marker for {method}: {complete}")
        checkpoint_files(checkpoint)
    for name, path in DATASETS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset {name}: {path}")


def chat_prompt(tokenizer, user: str, system: str | None = None, prefill: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if prefill is None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    marker = "-[[]]-PRR-CODE-PREFILL-[[]]-"
    messages.append({"role": "assistant", "content": prefill + marker + "\n```"})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    if rendered.count(marker) != 1:
        raise RuntimeError("Chat template did not preserve the assistant prefill marker")
    return rendered.split(marker, 1)[0]


def lcb_user_prompt(row: dict[str, Any]) -> str:
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += (
            "### Format: You will use the following starter code to write the solution "
            "to the problem and enclose your code within delimiters.\n"
            f"```python\n{row['starter_code']}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin solve the problem and write the answer "
            "to stdout (do not directly test on the sample inputs). Enclose your code "
            "within delimiters as follows. Ensure that when the python program runs, it "
            "reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def crux_output_prompt(code: str, inp: str) -> str:
    return f'''You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({inp}) == ??
[/PYTHON]
[ANSWER]
'''


def crux_input_prompt(code: str, output: str) -> str:
    return f'''You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
'''


def first_function_name(code: str) -> str:
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise ValueError("No function definition found in canonical MBPP solution")


def build_records(tokenizer) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("HumanEvalPlus", "MBPPPlus"):
        rows = pq.read_table(DATASETS[dataset]).to_pylist()
        rendered = []
        for row in rows:
            task_id = str(row["task_id"])
            if dataset == "MBPPPlus":
                task_id = f"Mbpp/{task_id}"
            user = f"{INSTRUCTION_PREFIX}\n```\n{row['prompt'].strip()}\n```"
            rendered.append(
                {
                    "dataset": dataset,
                    "id": task_id,
                    "prompt": chat_prompt(tokenizer, user, prefill=RESPONSE_PREFIX),
                    "stop": "code_fence",
                    "entry_point": row.get("entry_point") or first_function_name(row["code"]),
                }
            )
        records[dataset] = rendered

    lcb_rows = load_jsonl(DATASETS["LiveCodeBench"])
    records["LiveCodeBench"] = [
        {
            "dataset": "LiveCodeBench",
            "id": row["question_id"],
            "prompt": chat_prompt(tokenizer, lcb_user_prompt(row), system=SYSTEM_CODE),
            "stop": "eos",
        }
        for row in lcb_rows
    ]

    crux_rows = pq.read_table(DATASETS["CRUXEval"]).to_pylist()
    records["CRUXEval-I"] = [
        {
            "dataset": "CRUXEval-I",
            "id": row["id"],
            "prompt": chat_prompt(tokenizer, crux_input_prompt(row["code"], row["output"])),
            "stop": "answer_tag",
        }
        for row in crux_rows
    ]
    records["CRUXEval-O"] = [
        {
            "dataset": "CRUXEval-O",
            "id": row["id"],
            "prompt": chat_prompt(tokenizer, crux_output_prompt(row["code"], row["input"])),
            "stop": "answer_tag",
        }
        for row in crux_rows
    ]

    bcb_rows = pq.read_table(DATASETS["BigCodeBench-Hard"]).to_pylist()
    records["BigCodeBench-Hard"] = [
        {
            "dataset": "BigCodeBench-Hard",
            "id": row["task_id"],
            "prompt": chat_prompt(
                tokenizer,
                f"{INSTRUCTION_PREFIX}\n{row['instruct_prompt'].strip()}",
                prefill=RESPONSE_PREFIX,
            ),
            "stop": "code_fence",
            "entry_point": row["entry_point"],
        }
        for row in bcb_rows
    ]

    if {name: len(rows) for name, rows in records.items()} != EXPECTED_ROWS:
        raise RuntimeError("Dataset row counts changed")
    for name, rows in records.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError(f"Duplicate task ids in {name}")
    return records


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_fenced(text: str) -> str:
    text = strip_thinking(text)
    blocks = re.findall(r"```(?:python|Python)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    return (blocks[-1] if blocks else text).strip().replace("\t", "    ")


def extract_lcb(text: str) -> str:
    text = strip_thinking(text)
    lines = text.splitlines()
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]]).strip().replace("\t", "    ")


def extract_crux(text: str, mode: str) -> str:
    text = strip_thinking(text)
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]", 1)[1]
    if "[/ANSWER]" in text:
        text = text.split("[/ANSWER]", 1)[0]
    text = text.strip()
    if mode == "input":
        if "==" in text:
            text = text.split("==", 1)[0].strip()
        if "assert f" in text:
            text = "f" + text.split("assert f", 1)[1].strip()
    else:
        if "==" in text:
            text = text.split("==", 1)[1].strip()
    return text.splitlines()[0].strip() if text else ""


def extract_solution(dataset: str, text: str) -> str:
    if dataset == "LiveCodeBench":
        return extract_lcb(text)
    if dataset == "CRUXEval-I":
        return extract_crux(text, "input")
    if dataset == "CRUXEval-O":
        return extract_crux(text, "output")
    return extract_fenced(text)
