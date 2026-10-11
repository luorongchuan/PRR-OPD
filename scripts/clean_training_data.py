#!/usr/bin/env python3
"""CPU-only exact question decontamination; never edits source Parquet files."""

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq


RULE_VERSION = "dapo_exact_question_nfkc_fixed_suffix_v1"
# Verified on all input rows: double braces in DAPO, single braces in both tests.
SUFFIXES = (
    r"Please reason step by step, and put your final answer within \boxed{{}}.",
    r"Please reason step by step, and put your final answer within \boxed{}.",
)
RULES = {
    "version": RULE_VERSION,
    "question_field": "prompt[0].content; exactly one message with role=user",
    "steps": ["Unicode NFKC", "collapse whitespace using str.split/join and strip",
              "remove at most one exact trailing suffix at a whitespace boundary; strip"],
    "trailing_suffixes": list(SUFFIXES),
    "case_sensitive": True,
    "matching": "exact normalized question string; no answer or score access",
}
DATASETS = ("MATH-500", "AMC23")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path):
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def question(row):
    messages = row.get("prompt")
    require(isinstance(messages, list) and len(messages) == 1,
            "Unsupported prompt structure: expected exactly one user message")
    message = messages[0]
    require(isinstance(message, dict) and message.get("role") == "user"
            and isinstance(message.get("content"), str), "Invalid user question")
    return message["content"]


def normalize(text):
    value = " ".join(unicodedata.normalize("NFKC", text).split())
    removed = None
    for suffix in SUFFIXES:
        if value == suffix or value.endswith(" " + suffix):
            value = value[:-len(suffix)].strip()
            removed = suffix
            break
    require(bool(value), "Empty question after normalization")
    return value, removed


def record_ids(row):
    ids = {}
    if isinstance(row.get("extra_info"), dict) and row["extra_info"].get("index") is not None:
        ids["extra_info.index"] = row["extra_info"]["index"]
    if row.get("id") is not None:
        ids["id"] = row["id"]
    return ids


def primary_id(row):
    ids = record_ids(row)
    return ids.get("extra_info.index", ids.get("id"))


def load_inputs(paths):
    tables, info = {}, {}
    for name, path in paths.items():
        before = file_info(path)
        tables[name] = pq.read_table(path)
        require(file_info(path) == before, f"Input changed while reading: {path}")
        schema = tables[name].schema
        info[name] = dict(before, rows=tables[name].num_rows,
                          schema=str(schema),
                          schema_serialized_sha256=hashlib.sha256(schema.serialize().to_pybytes()).hexdigest(),
                          schema_metadata_base64={base64.b64encode(k).decode(): base64.b64encode(v).decode()
                                                  for k, v in (schema.metadata or {}).items()})
    return tables, info


def unchanged_inputs(paths, info):
    for name, path in paths.items():
        require(file_info(path) == {key: info[name][key] for key in ("path", "size_bytes", "sha256")},
                f"Input changed: {path}")


def analyze(tables, paths):
    rows = {name: table.to_pylist() for name, table in tables.items()}
    texts, suffix_counts = {}, {}
    for name, values in rows.items():
        normalized = [normalize(question(row)) for row in values]
        texts[name] = [item[0] for item in normalized]
        suffix_counts[name] = dict(Counter(item[1] or "<none>" for item in normalized))
    test_index = defaultdict(list)
    for dataset in DATASETS:
        for index, (row, text) in enumerate(zip(rows[dataset], texts[dataset])):
            test_index[text].append({
                "dataset": dataset, "test_file": str(paths[dataset]),
                "test_source_row_index": index, "test_problem_id": primary_id(row),
                "test_original_ids": record_ids(row), "raw_question": question(row),
                "normalized_question": text,
            })
    keep, exclusions = [], []
    for index, (row, text) in enumerate(zip(rows["train"], texts["train"])):
        matches = test_index.get(text, [])
        keep.append(not matches)
        if matches:
            exclusions.append({
                "source_row_index": index, "source_row_number_1based": index + 1,
                "source_record_id": primary_id(row), "source_record_ids": record_ids(row),
                "raw_question": question(row), "normalized_question": text,
                "matches": matches, "original_record": row,
            })
    retained = tables["train"].filter(pa.array(keep, type=pa.bool_()))
    train_set = set(texts["train"])
    kept_set = {text for text, retained_flag in zip(texts["train"], keep) if retained_flag}
    removed_set = {row["normalized_question"] for row in exclusions}
    per_dataset = {}
    for dataset in DATASETS:
        test_set = set(texts[dataset])
        hit = test_set & train_set
        per_dataset[dataset] = {
            "test_rows": len(rows[dataset]), "test_unique_normalized_questions": len(test_set),
            "matched_test_rows": sum(text in train_set for text in texts[dataset]),
            "matched_unique_normalized_questions": len(hit),
            "deleted_training_rows": sum(text in test_set for text in texts["train"]),
            "remaining_overlap_unique_questions": len(kept_set & test_set),
        }
    stats = {
        "training_rows_before": len(keep), "training_rows_removed": len(exclusions),
        "training_rows_retained": retained.num_rows,
        "training_unique_questions_before": len(train_set),
        "training_unique_questions_after": len(kept_set),
        "training_duplicate_excess_rows_before": len(keep) - len(train_set),
        "training_duplicate_excess_rows_after": retained.num_rows - len(kept_set),
        "per_dataset": per_dataset,
        "union": {
            "matched_unique_normalized_questions": len(removed_set),
            "matched_test_records": sum(d["matched_test_rows"] for d in per_dataset.values()),
            "deleted_training_rows": len(exclusions),
            "training_to_test_match_relations": sum(len(row["matches"]) for row in exclusions),
            "deleted_rows_matching_multiple_datasets": sum(
                len({m["dataset"] for m in row["matches"]}) > 1 for row in exclusions),
            "test_cross_dataset_unique_question_overlap": len(set(texts[DATASETS[0]]) & set(texts[DATASETS[1]])),
        },
        "removed_suffix_counts": suffix_counts,
    }
    return {"retained": retained, "exclusions": exclusions, "stats": stats, "keep": keep}


def compare_review(path, analysis):
    if path is None:
        return None
    prior = json.loads(path.read_text())
    stats = analysis["stats"]
    same = (prior["pool_size"] == stats["training_rows_before"]
            and prior["unique_normalized_questions"] == stats["training_unique_questions_before"])
    checks = {}
    for dataset in DATASETS:
        observed = defaultdict(list)
        for excluded in analysis["exclusions"]:
            for match in excluded["matches"]:
                if match["dataset"] == dataset:
                    observed[(match["test_problem_id"], excluded["normalized_question"])].append(
                        str(excluded["source_record_id"]))
        expected = {(r["problem_id"], r["question"]): sorted(r["training_ids"])
                    for r in prior["benchmarks"][dataset]["matches"]}
        checks[dataset] = (expected == {k: sorted(v) for k, v in observed.items()}
                           and prior["benchmarks"][dataset]["total"] == stats["per_dataset"][dataset]["test_rows"]
                           and prior["benchmarks"][dataset]["unique_normalized_questions"]
                           == stats["per_dataset"][dataset]["test_unique_normalized_questions"])
    return {**file_info(path), "pool_counts_match": same, "per_dataset_matches_identical": checks,
            "agrees_with_review": same and all(checks.values())}


def verify_data(directory, tables, analysis):
    actual = pq.read_table(directory / "train.parquet")
    original = tables["train"]
    expected = analysis["retained"]
    require(actual.schema.equals(original.schema, check_metadata=True), "Schema/metadata changed")
    require(actual.equals(expected, check_metadata=True), "Retained Arrow values or order changed")
    # Independently verify every field of every retained row against its source row.
    originals = original.to_pylist()
    source_indices = [i for i, keep in enumerate(analysis["keep"]) if keep]
    require(actual.to_pylist() == [originals[i] for i in source_indices], "Per-row field comparison failed")
    with (directory / "excluded_records.jsonl").open(encoding="utf-8") as stream:
        excluded = [json.loads(line) for line in stream]
    require(excluded == analysis["exclusions"], "Exclusion records/match evidence differ")
    removed_indices = [row["source_row_index"] for row in excluded]
    require(removed_indices == [i for i, keep in enumerate(analysis["keep"]) if not keep],
            "Excluded row indices differ")
    require(original.num_rows == actual.num_rows + len(excluded), "Row accounting failed")
    for row in excluded:
        require(bool(row["matches"]), "Deleted row has no test match")
        for match in row["matches"]:
            test = tables[match["dataset"]].slice(match["test_source_row_index"], 1).to_pylist()[0]
            require(normalize(question(test))[0] == row["normalized_question"], "Invalid exact-match link")
    actual_text = {normalize(question(row))[0] for row in actual.to_pylist()}
    for dataset in DATASETS:
        test_text = {normalize(question(row))[0] for row in tables[dataset].to_pylist()}
        require(not actual_text & test_text, f"Remaining overlap: {dataset}")
    return {
        "row_accounting": True, "schema_and_metadata_identical": True,
        "all_retained_fields_and_order_identical": True, "all_exclusion_links_verified": True,
        "internal_training_duplicates_preserved": True,
        "remaining_overlap": {dataset: 0 for dataset in DATASETS},
    }


def verify_bundle(directory, paths, tables, info, analysis, review):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["rules"] == RULES, "Normalization rules differ")
    require(manifest["inputs"] == info, "Input files/schema/hashes differ")
    require(manifest["statistics"] == analysis["stats"], "Statistics differ")
    require(manifest["previous_review_comparison"] == review, "Review comparison differs")
    require(manifest["output_directory"] == str(directory), "Output directory differs")
    require(set(manifest["artifacts"]) == {"clean.py", "test_clean.py", "README.md",
                                            "train.parquet", "excluded_records.jsonl"}, "Artifact inventory differs")
    for name, expected in manifest["artifacts"].items():
        require(file_info(directory / name) == expected, f"Artifact changed: {name}")
    require(sha256(Path(__file__).resolve()) == manifest["artifacts"]["clean.py"]["sha256"],
            "Current cleaning script differs from recorded script")
    checks = verify_data(directory, tables, analysis)
    unchanged_inputs(paths, info)
    checks["source_files_sha256_unchanged"] = True
    require(manifest["acceptance"] == checks, "Acceptance record differs")
    return manifest


def readme(directory, paths, info, analysis, review):
    s = analysis["stats"]
    quote = shlex.quote
    command = (f"python -B {quote(str(directory / 'clean.py'))} \\\n"
               f"  --train {quote(str(paths['train']))} \\\n"
               f"  --math500 {quote(str(paths['MATH-500']))} \\\n"
               f"  --amc23 {quote(str(paths['AMC23']))} \\\n"
               f"  --output-dir {quote(str(directory))}")
    if review:
        command += f" \\\n  --review-report {quote(review['path'])}"
    lines = ["# DAPO 数学训练池：测试题文精确去重 v1", "",
             "## 用途与原因", "",
             "此前实验审查发现，DAPO 训练池与 MATH-500、AMC23 测试集存在规范化后完全相同的题目，部分重叠题已经进入实际训练流。OPD 即使不直接使用测试标准答案，教师在这些题目上的输出分布仍参与了学生参数更新，因此这些题目不能被视为完全未见的测试样本。", "",
             "本数据版本从训练池中排除上述重叠记录，保持测试集不变，用于后续实验建立更可信的泛化比较。该处理不以提高评测分数为目的，也不意味着已有方法排名全部由数据重叠造成。", "",
             "## 输入与输出", ""]
    for name, path in paths.items():
        lines += [f"- {name}：`{path}`", f"  SHA256：`{info[name]['sha256']}`"]
    lines += [f"- 唯一清洗后训练数据：`{directory / 'train.parquet'}`", "",
              "## 规则与实现", "", f"规则版本：`{RULE_VERSION}`，实现位于 `clean.py` 的 `question`、`normalize`、`analyze`。", "",
              "1. 从 `prompt[0].content` 提取题文。已检查全部输入均为一条 `role=user` 消息；脚本遇到其他结构会报错，不会把 messages 对象转成字符串。", 
              "2. 对题文执行 Unicode NFKC，再用 `str.split()` / 空格拼接合并空白并去除首尾空白。",
              "3. 仅当下面的完整固定指令位于末尾、且前面是空白边界时，移除一次该指令。没有匹配则保留题文；不使用宽泛的 `.*` 删除规则。", "", "```text", *SUFFIXES, "```", "",
              "训练集全部记录使用双层花括号版本；两个测试集全部记录使用单层花括号版本。其他指令（包括要求返回 m+n、选项等题目条件）一律保留。",
              "除明确要求的 NFKC 兼容字符规范化及空白处理外，不改写数学符号、数字、变量大小写、LaTeX 或题目条件。匹配保持大小写敏感，不读取答案或评分结果来决定删除。",
              "4. 用规范化后的题文做完全相等匹配。测试重复题保留所有映射；同一题的全部训练行都删除，跨测试集命中的训练行只删除一次。",
              "5. 在原始 PyArrow Table 上应用布尔掩码；不经 pandas 重建。保留行的题文、原始格式指令、答案、其他字段、嵌套类型、顺序和 metadata 均不修改。内部重复不去重。", "",
              "## 实际统计", "", "| 项目 | 数值 |", "| --- | ---: |",
              f"| 原始训练行数 | {s['training_rows_before']} |", f"| 删除训练行数（并集） | {s['training_rows_removed']} |",
              f"| 保留训练行数 | {s['training_rows_retained']} |",
              f"| 清洗前 / 后唯一规范化题文数 | {s['training_unique_questions_before']} / {s['training_unique_questions_after']} |",
              f"| 清洗前 / 后内部重复额外行数 | {s['training_duplicate_excess_rows_before']} / {s['training_duplicate_excess_rows_after']} |", "",
              "| 测试集 | 测试行数 | 唯一题文 | 命中测试行 | 命中唯一题文 | 删除训练行 | 清洗后交集 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, d in s["per_dataset"].items():
        lines.append(f"| {name} | {d['test_rows']} | {d['test_unique_normalized_questions']} | {d['matched_test_rows']} | {d['matched_unique_normalized_questions']} | {d['deleted_training_rows']} | {d['remaining_overlap_unique_questions']} |")
    union = s["union"]
    lines += ["", f"并集命中 {union['matched_unique_normalized_questions']} 个规范化题文、{union['matched_test_records']} 条测试记录，实际删除 {union['deleted_training_rows']} 条训练记录。训练到测试对应关系共 {union['training_to_test_match_relations']} 条；同时匹配多个测试集的删除行共 {union['deleted_rows_matching_multiple_datasets']} 条。两个测试集之间共有 {union['test_cross_dataset_unique_question_overlap']} 个相同规范化题文。",
              "测试题数、唯一题文数和训练删除行数是分别计算的，并非把命中题数直接当作删除行数。"]
    if review:
        lines += ["", f"审查对照文件：`{review['path']}`。",
                  ("此次训练池计数、测试唯一题数、命中题文、测试 ID 和对应训练 ID 均与此前审查一致。当前使用明确的末尾字面量规则，不复用此前的宽泛正则。"
                   if review["agrees_with_review"] else "此次结果与审查记录有差异，详见 manifest.json 的 previous_review_comparison；不可沿用旧计数。")]
    lines += ["", "## 产物", "",
              "| 文件 | 用途 |", "| --- | --- |",
              "| train.parquet | 唯一清洗后训练数据 |", "| clean.py | CPU 清洗、复跑与验收入口 |",
              "| excluded_records.jsonl | 每个被删源行的原记录、原始及规范化题文、所有匹配测试记录 |",
              "| manifest.json | 规则、输入输出绝对路径与 SHA256、schema metadata、统计及验收记录 |",
              "| README.md | 本说明和准确命令 |", "| test_clean.py | 不启动 GPU 的边界与安全回归测试 |", "",
              "排除记录中的 `source_row_index` / `test_source_row_index` 是原始 Parquet 顺序的零起始行号，`source_row_number_1based` 为一基行号；ID 优先使用 `extra_info.index`，同时保留顶层 `id`（存在时）。无 ID 时记 null，以行号定位。`original_record` 包含原行全部字段供复查，答案仅被复制留档，不参与匹配。", "",
              "## 复跑与验收命令", "", "下面命令可从任意目录执行，只使用 CPU。", "", "```bash", command, "```", "",
              "再次执行时，只有输入、规则、代码和所有产物一致且验收通过才复用，绝不覆盖旧产物；冲突时保留旧目录并新建带 `_revision_` 时间戳的同级目录，终端输出实际目录。", "",
              "仅验收，不写任何文件：", "", "```bash", command + " \\\n  --verify-only", "```", "",
              "CPU 回归测试：", "", "```bash",
              f"python -B {quote(str(directory / 'test_clean.py'))}", "```", "",
              "已验收：原始行数 = 保留 + 删除；写出后重新读取的 schema（含 metadata）一致；每条保留记录所有字段与源行逐条相等且顺序不变；所有排除行都有具体测试题对应；清洗后两测试集交集为 0；输入文件 SHA256 在清洗前后完全一致。验收使用显式异常检查，不依赖可被 `python -O` 关闭的 assert。", "",
              "原始 pandas RangeIndex 描述也作为 schema metadata 原样保留，其中历史 stop 值没有改写；请以 Arrow Table / Parquet 实际行数为准。", "",
              "## 使用限制", "",
              "- 未进行模糊或语义近似去重，未验证模型预训练污染，不能声称排除了所有数据污染。",
              "- 不划分开发集，不筛除难题、长题或错误答案，不改变 V1 当前训练 batch 校准定义。",
              "- 现有训练入口、配置、数据流和 checkpoint 均未切换，正在运行的实验保持原样。本轮没有启动训练、生成或 GPU 评测。",
              "- 后续清洁确认实验须明确引用本目录的 train.parquet，并使用未经过这些受污染实验训练的共同起点；不能把训练中途切换数据称为全程清洁实验。",
              "- 本次检查没有验证下载来源、许可或更广泛的数据污染情况；零交集仅指本文件版本和本规范化规则。", ""]
    return "\n".join(lines)


@contextmanager
def directory_lock(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def bundle_source(name):
    return Path(__file__).resolve() if name == "clean.py" else Path(__file__).resolve().with_name(name)


def create_bundle(directory, paths, tables, info, analysis, review, previous=None):
    for name in ("clean.py", "test_clean.py"):
        data = bundle_source(name).read_bytes()
        target = directory / name
        if target.exists():
            require(target.read_bytes() == data, f"Refusing to change {target}")
        else:
            with target.open("xb") as stream:
                stream.write(data)
    with (directory / "train.parquet").open("xb") as stream:
        pq.write_table(analysis["retained"], stream, compression="snappy", store_schema=True)
    with (directory / "excluded_records.jsonl").open("x", encoding="utf-8") as stream:
        for row in analysis["exclusions"]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    checks = verify_data(directory, tables, analysis)
    unchanged_inputs(paths, info)
    checks["source_files_sha256_unchanged"] = True
    with (directory / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(readme(directory, paths, info, analysis, review))
    manifest = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(directory), "rules": RULES, "inputs": info,
        "statistics": analysis["stats"], "acceptance": checks,
        "previous_review_comparison": review, "previous_output_preserved": previous,
        "runtime": {"python": platform.python_version(), "pyarrow": pa.__version__,
                    "unicode_database": unicodedata.unidata_version},
        "artifacts": {name: file_info(directory / name) for name in
                      ("clean.py", "test_clean.py", "train.parquet", "excluded_records.jsonl", "README.md")},
    }
    with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return verify_bundle(directory, paths, tables, info, analysis, review)


def run(paths, output, review_path=None, verify_only=False):
    paths = {name: Path(path).resolve(strict=True) for name, path in paths.items()}
    output = Path(output).absolute()
    require(not output.is_symlink(), "Output directory must not be a symlink")
    output = output.resolve()
    require(len(set(paths.values())) == 3, "Expected three distinct input files")
    require(all(not path.is_relative_to(output) for path in paths.values()), "Inputs cannot be inside output")
    tables, info = load_inputs(paths)
    analysis = analyze(tables, paths)
    review_path = Path(review_path).resolve(strict=True) if review_path else None
    review = compare_review(review_path, analysis)
    if review:
        require(review["agrees_with_review"], "Review differs; investigate input/rule differences before publishing")
    if verify_only:
        manifest = verify_bundle(output, paths, tables, info, analysis, review)
        return "VERIFIED", output, manifest
    output.mkdir(parents=True, exist_ok=True)
    previous = None
    with directory_lock(output):
        if (output / "manifest.json").exists():
            try:
                manifest = verify_bundle(output, paths, tables, info, analysis, review)
                return "REUSED", output, manifest
            except (ValueError, KeyError, OSError) as exc:
                previous = {"path": str(output), "reason": str(exc)}
        elif set(p.name for p in output.iterdir()) - {"clean.py", "test_clean.py"}:
            previous = {"path": str(output), "reason": "Existing or incomplete artifacts preserved"}
        else:
            for name in ("clean.py", "test_clean.py"):
                if (output / name).exists() and (output / name).read_bytes() != bundle_source(name).read_bytes():
                    previous = {"path": str(output), "reason": "Existing source differs"}
        if previous is None:
            manifest = create_bundle(output, paths, tables, info, analysis, review)
            return "CREATED", output, manifest
    target = output.with_name(output.name + "_revision_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    target.mkdir(exist_ok=False)
    with directory_lock(target):
        manifest = create_bundle(target, paths, tables, info, analysis, review, previous)
    return "CREATED_NEW_VERSION", target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--math500", type=Path, required=True)
    parser.add_argument("--amc23", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-report", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    status, output, manifest = run({"train": args.train, "MATH-500": args.math500, "AMC23": args.amc23},
                                   args.output_dir, args.review_report, args.verify_only)
    print(json.dumps({"status": status, "output_directory": str(output), "statistics": manifest["statistics"],
                      "acceptance": manifest["acceptance"], "previous_output_preserved": manifest["previous_output_preserved"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
