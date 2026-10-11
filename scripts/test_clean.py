"""CPU-only regression tests; all generated test files live in temporary directories."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

source = Path(__file__).with_name("clean_training_data.py")
if not source.exists():
    source = Path(__file__).with_name("clean.py")
spec = importlib.util.spec_from_file_location("dapo_clean", source)
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def row(text, index, answer="arbitrary"):
    return {"prompt": [{"role": "user", "content": text}], "id": str(index),
            "extra_info": {"index": str(index)}, "reward_model": {"ground_truth": answer},
            "untouched": {"tokens": [1, 2, None], "enabled": True}}


class CleaningTests(unittest.TestCase):
    def test_exact_suffixes_nfkc_and_whitespace(self):
        for suffix in clean.SUFFIXES:
            self.assertEqual(clean.normalize("  Ａ + １\n = x. \t" + suffix + "\n"), ("A + 1 = x.", suffix))

    def test_keep_case_math_conditions_and_nonliteral_instructions(self):
        question = r"Find x + X, $\frac{a}{b}$; return m+n."
        self.assertEqual(clean.normalize(question)[0], question)
        for text in [question + " " + clean.SUFFIXES[0] + " More conditions.",
                     question + " Please reason step by step, and put your final answer within a circle.",
                     question + "x" + clean.SUFFIXES[1]]:
            self.assertEqual(clean.normalize(text)[0], text)
        self.assertNotEqual(clean.normalize("x = 1")[0], clean.normalize("X = 1")[0])

    def test_reject_unsupported_messages_and_empty_text(self):
        for prompt in ["text", [], [{"role": "assistant", "content": "q"}],
                       [{"role": "user", "content": "q"}] * 2]:
            with self.assertRaises(ValueError):
                clean.question({"prompt": prompt})
        with self.assertRaises(ValueError):
            clean.normalize(clean.SUFFIXES[0])

    def fixture(self):
        train = [row("keep", "a"), row("overlap " + clean.SUFFIXES[0], "b"),
                 row("overlap", "c", "different"), row("keep", "a"), row("KEEP", "e")]
        return {"train": pa.Table.from_pylist(train).replace_schema_metadata({b"test": b"exact metadata"}),
                "MATH-500": pa.Table.from_pylist([row("overlap", "m0"), row("overlap", "m1")]),
                "AMC23": pa.Table.from_pylist([row("overlap " + clean.SUFFIXES[1], "a0")])}

    def test_all_duplicates_all_test_links_and_union_counts(self):
        tables = self.fixture()
        result = clean.analyze(tables, {name: Path("/tmp") / name for name in tables})
        self.assertEqual(result["keep"], [True, False, False, True, True])
        self.assertEqual([len(r["matches"]) for r in result["exclusions"]], [3, 3])
        self.assertEqual(result["stats"]["union"]["deleted_training_rows"], 2)
        self.assertEqual(result["stats"]["union"]["matched_unique_normalized_questions"], 1)
        self.assertEqual(result["stats"]["union"]["matched_test_records"], 3)
        self.assertEqual(result["stats"]["per_dataset"]["MATH-500"]["matched_test_rows"], 2)
        self.assertTrue(result["retained"].schema.equals(tables["train"].schema, check_metadata=True))
        self.assertEqual(result["retained"].to_pylist(), [tables["train"].to_pylist()[i] for i in (0, 3, 4)])

    def test_answer_does_not_affect_removal(self):
        tables = self.fixture()
        paths = {name: Path("/tmp") / name for name in tables}
        expected = clean.analyze(tables, paths)["keep"]
        changed = {}
        for name, table in tables.items():
            values = copy.deepcopy(table.to_pylist())
            for value in values:
                value["reward_model"]["ground_truth"] = "completely changed"
            changed[name] = pa.Table.from_pylist(values)
        self.assertEqual(clean.analyze(changed, paths)["keep"], expected)

    def write_inputs(self, directory):
        paths = {}
        for name, table in self.fixture().items():
            paths[name] = directory / (name + ".parquet")
            pq.write_table(table, paths[name])
        return paths

    def test_roundtrip_reuse_readonly_verification_and_conflict_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = self.write_inputs(directory)
            original_hashes = {name: clean.sha256(path) for name, path in paths.items()}
            output = directory / "out"
            status, target, manifest = clean.run(paths, output)
            self.assertEqual(status, "CREATED")
            self.assertEqual(target, output)
            self.assertTrue(manifest["acceptance"]["source_files_sha256_unchanged"])
            snapshot = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            self.assertEqual(clean.run(paths, output)[0], "REUSED")
            self.assertEqual(clean.run(paths, output, verify_only=True)[0], "VERIFIED")
            self.assertEqual(snapshot, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})
            (output / "README.md").write_text("user content", encoding="utf-8")
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(ValueError):
                clean.run(paths, output, verify_only=True)
            status, new, _ = clean.run(paths, output)
            self.assertEqual(status, "CREATED_NEW_VERSION")
            self.assertNotEqual(new, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
            self.assertEqual(original_hashes, {name: clean.sha256(path) for name, path in paths.items()})

    def test_detect_missing_match_and_reordered_retained_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = self.write_inputs(directory)
            _, output, _ = clean.run(paths, directory / "out")
            tables, _ = clean.load_inputs(paths)
            analysis = clean.analyze(tables, paths)
            path = output / "excluded_records.jsonl"
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            lines[0]["matches"].pop()
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            with self.assertRaisesRegex(ValueError, "Exclusion"):
                clean.verify_data(output, tables, analysis)
            actual = pq.read_table(output / "train.parquet")
            pq.write_table(actual.take(pa.array([2, 1, 0])), output / "train.parquet")
            with self.assertRaisesRegex(ValueError, "values or order"):
                clean.verify_data(output, tables, analysis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
