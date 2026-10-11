"""Toy scoring and concurrency tests; no actual benchmark/model evaluation."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import eval_common as c
import eval_scoring
import run
from memory_guard import GIB, supervise

sys.path.insert(0, str(c.EVAL / "vendor"))
from math_scoring.scoring import correct, robust_answer


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


local = module("local_bigcode", c.HERE / "scorers/local_bigcode/score_local.py")
docker = module("docker_bigcode", c.HERE / "scorers/local_bigcode/docker_sandbox.py")


class ScorerTests(unittest.TestCase):
    def test_resource_changes_invalidate_canonical_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "dataset"
            data.write_bytes(b"synthetic")
            assets = root / "resources"
            assets.mkdir()
            resource = assets / "stopwords"
            resource.write_text("first")
            rows = {"toy": {"complete_prompt": "", "canonical_solution": "pass", "test": ""}}
            with patch.dict("os.environ", {"NLTK_DATA": str(assets)}):
                first = local.environment(rows, data, root)["fingerprint"]
                resource.write_text("changed")
                self.assertNotEqual(first, local.environment(rows, data, root)["fingerprint"])

    def test_copied_sources_are_unchanged(self):
        self.assertGreater(len(c.archived_sources()), 100)
        self.assertTrue(str(c.EVAL).startswith(str(c.HERE)))

    def test_math_last_complete_box_and_equivalence(self):
        self.assertEqual(robust_answer(r"first \boxed{9}; final \fbox{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(correct(robust_answer(r"\boxed{\frac{1}{2}}"), "0.5"), 1)
        self.assertEqual(correct(robust_answer(r"\boxed{3}"), "2"), 0)
        self.assertIsNone(robust_answer("some intermediate number 42"))
        self.assertEqual(robust_answer(r"\boxed{7}, unfinished \boxed{99"), "7")

    def test_controls_failure_blocks_predictions_and_missing_dependencies(self):
        rows = {"toy": {"complete_prompt": "def f():", "canonical_solution": "    return 1", "test": "toy"}}
        env = {"missing_modules": [], "fingerprint": "synthetic"}
        failed = local.validate_controls(rows, env, checker=lambda *args: {"time": None})
        self.assertEqual(failed["status"], "FAILED")
        with self.assertRaises(RuntimeError):
            local.score(rows, [], failed)
        with self.assertRaisesRegex(RuntimeError, "lacks modules"):
            local.validate_controls(rows, {**env, "missing_modules": ["not_installed"]})

    def test_official_checker_adapter_and_draw_counts(self):
        row = {"task_id": "toy", "complete_prompt": "def f():", "canonical_solution": "    return 1",
               "test": "toy", "entry_point": "f"}
        rows = {"toy": row}
        controls = local.validate_controls(rows, {"missing_modules": [], "fingerprint": "toy"},
                                         checker=lambda *args: {"time": 0.01})
        arguments = []

        def checker(*args):
            arguments.append(args)
            return "pass", {}

        details = local.score(rows, [{"id": "toy", "solution": "toy source"}], controls, checker=checker)
        self.assertEqual(sum(d["passed"] for d in details), 1)
        self.assertEqual(arguments[0][3:7], (30720, 30720, 10, 1))
        with self.assertRaises(ValueError):
            local.score(rows, [{"id": "wrong", "solution": "x"}], controls, checker=checker)

    def test_local_guard_uses_no_upload_and_bigcode_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = eval_scoring.command(sys.executable, Path(temp), "BigCodeBench-Hard")
        self.assertIn(str(c.HERE / "score_dataset.py"), argv)
        self.assertEqual(argv[argv.index("--per-process-gib") + 1], "30")
        env = run.environment()
        self.assertNotIn("ALLOW_BIGCODE_REMOTE_UPLOAD", env)
        self.assertEqual(env["PRR_BIGCODE_BACKEND"], "host")

    def test_real_official_bigcode_checker_with_toy_code_in_verl(self):
        with tempfile.TemporaryDirectory() as temp:
            code = f"""
import os,sys
os.environ['MPLCONFIGDIR']={temp!r}
sys.path.insert(0,{str(c.HERE / 'scorers/bigcodebench')!r})
from bigcodebench.eval import untrusted_check
from bigcodebench.gen.util import trusted_check
test='import unittest\\nclass TestCases(unittest.TestCase):\\n    def test_value(self):\\n        self.assertEqual(f(), 1)\\n'
gt=trusted_check('def f():\\n    return 1',test,'toy',30720,30720,10,1)
assert gt['time'] is not None
assert untrusted_check('def f():\\n    return 1',test,'f',30720,30720,10,1,gt['time'])[0]=='pass'
assert untrusted_check('def f():\\n    return 2',test,'f',30720,30720,10,1,gt['time'])[0]=='fail'
print('OFFICIAL_TOY_CHECKER_OK')
"""
            self.assertEqual(supervise([sys.executable, "-B", "-c", code], Path(temp) / "guard.json",
                                      process_limit=30 * GIB, min_available=0), 0)

    def test_same_gpu_refused_different_gpu_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with run.gpu_slot(6, folder):
                with self.assertRaisesRegex(RuntimeError, "reserved"):
                    with run.gpu_slot(6, folder):
                        pass
                with run.gpu_slot(5, folder):
                    pass
            with run.gpu_slot(6, folder):
                pass

    def test_docker_is_offline_bounded_and_cache_read_only_for_model_code(self):
        argv = docker.command("sha256:toy", "score", "/problems", "/predictions", "/cache", "/output", "/cid", "toy")
        self.assertEqual(argv[argv.index("--network") + 1], "none")
        self.assertEqual(argv[argv.index("--memory") + 1], "24g")
        self.assertEqual(argv[argv.index("--memory-swap") + 1], "24g")
        self.assertIn("/cache:/controls:ro", argv)
        self.assertNotIn("--gpus", argv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
