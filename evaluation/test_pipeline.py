"""Synthetic pipeline tests. No model loading, benchmark execution or GPU use."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

import eval_common as c
import eval_scoring
import memory_guard as guard
import run


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.datasets = ["AMC23", "AIME24"]
        self.models = {"sgd": {"path": "synthetic_sgd"}, "opd": {"path": "synthetic_opd"}}
        self.plan = {"models": self.models, "datasets": self.datasets, "gpu": 99,
                     "python": sys.executable, "allow_bigcode_remote": False,
                     "profile": "comparable", "max_tokens": 2048, "seed": 42,
                     "gpu_memory_utilization": 0.875}

    def tearDown(self):
        self.temp.cleanup()

    def fake_generation(self, plan, model, output):
        code = f"""
import json,os,time
from pathlib import Path
import sys
sys.path.insert(0, {str(c.HERE)!r})
from eval_common import digest,write,POLICY
root=Path({str(output)!r})
names={self.datasets!r}
p={{'generation':{{'request_batch_size':1,'n':16}},'datasets':{{d:{{'rows':1}} for d in names}},'resource_policy':POLICY}}
p['protocol_sha256']=digest(p)
write(root/'protocol.json',p)
for d in names:
    write(root/'generation'/d/'batch_000000.json',{{'start':0,'protocol_sha256':p['protocol_sha256'],
      'predictions':[{{'id':d,'samples':[{{'sample_index':i,'text':'synthetic'}} for i in range(16)]}}]}})
    write(root/(d+'.generated.json'),{{'time':time.time()}})
    time.sleep(0.5)
write(root/'generation_complete.json',{{'status':'COMPLETE','protocol_sha256':p['protocol_sha256'],'datasets':{{d:1 for d in names}}}})
"""
        return [sys.executable, "-B", "-c", code]

    def fake_score(self, python, output, dataset):
        code = f"""
import sys,time,os
from pathlib import Path
sys.path.insert(0,{str(c.HERE)!r})
from eval_common import write,read
root=Path({str(output)!r});d={dataset!r}
write(root/(d+'.score_start.json'),{{'time':time.time(),'cuda':os.environ['CUDA_VISIBLE_DEVICES']}})
time.sleep(1.2)
p=read(root/'protocol.json')
write(root/'scores'/d/'summary.json',{{'protocol_sha256':p['protocol_sha256'],'total_questions':1,
    'samples_per_question':16,'correct_samples':8,'sampled_pass_at_1':0.5}})
write(root/'scores'/d/'resource_guard.json',{{'completed':True}})
write(root/(d+'.score_end.json'),{{'time':time.time()}})
"""
        return [sys.executable, "-B", "-c", code]

    def test_generation_overlaps_scoring_and_next_model(self):
        def overlap_score(python, output, dataset):
            command = self.fake_score(python, output, dataset)
            if output.name == "sgd" and dataset == "AMC23":
                # Explicit synchronization avoids scheduler-dependent 1.2-second races.
                generated = self.root / "models/opd/AMC23.generated.json"
                command[-1] = command[-1].replace("time.sleep(1.2)", f"""
deadline=time.monotonic()+10
while not Path({str(generated)!r}).exists():
    if time.monotonic()>deadline:
        raise RuntimeError('Next model waited for scoring instead of overlapping')
    time.sleep(0.02)
""")
            return command

        with patch.dict(c.COUNTS, {d: 1 for d in self.datasets}), patch.dict(run.COUNTS, {d: 1 for d in self.datasets}):
            result = run.run_pipeline(self.root, self.plan, self.fake_generation, overlap_score, held=lambda: False)
        self.assertEqual(result, 0)
        first = self.root / "models/sgd"
        self.assertLess(c.read(first / "AMC23.score_start.json")["time"], c.read(first / "AIME24.generated.json")["time"])
        self.assertLess(c.read(first / "AIME24.generated.json")["time"], c.read(first / "AMC23.score_end.json")["time"])
        self.assertLess(c.read(self.root / "models/opd/AMC23.generated.json")["time"], c.read(first / "AMC23.score_end.json")["time"])
        self.assertEqual(c.read(first / "AMC23.score_start.json")["cuda"], "")
        self.assertLessEqual(c.read(first / "AMC23.score_end.json")["time"], c.read(first / "AIME24.score_start.json")["time"])
        self.assertEqual(c.read(self.root / "status.json")["completed"], 4)

    def test_failure_does_not_stop_generation_or_become_zero(self):
        def fail(python, output, dataset):
            if dataset == "AMC23":
                return [sys.executable, "-c", "raise SystemExit(137)"]
            return self.fake_score(python, output, dataset)
        self.plan["models"] = {"sgd": self.models["sgd"]}
        with patch.dict(c.COUNTS, {d: 1 for d in self.datasets}), patch.dict(run.COUNTS, {d: 1 for d in self.datasets}):
            result = run.run_pipeline(self.root, self.plan, self.fake_generation, fail, held=lambda: False)
        self.assertEqual(result, 1)
        status = c.read(self.root / "status.json")
        self.assertEqual(status["completed"], 1)
        self.assertIn("sgd/AMC23", status["failures"])
        self.assertNotIn("AMC23", (self.root / "results.csv").read_text())

    def test_holding_pipeline_does_not_launch(self):
        with self.assertRaises(InterruptedError):
            run.run_pipeline(self.root, self.plan, held=lambda: True)
        self.assertEqual(c.read(self.root / "status.json")["stage"], "paused")

    def test_bad_score_is_quarantined_without_stopping_generation(self):
        original = self.fake_score

        def malformed(python, output, dataset):
            command = original(python, output, dataset)
            if dataset == "AMC23":
                command[-1] = command[-1].replace("'correct_samples':8", "'correct_samples':999")
            return command

        self.plan["models"] = {"sgd": self.models["sgd"]}
        with patch.dict(c.COUNTS, {d: 1 for d in self.datasets}), patch.dict(run.COUNTS, {d: 1 for d in self.datasets}):
            result = run.run_pipeline(self.root, self.plan, self.fake_generation, malformed, held=lambda: False)
        self.assertEqual(result, 1)
        score_dir = self.root / "models/sgd/scores/AMC23"
        self.assertFalse((score_dir / "summary.json").exists())
        self.assertEqual(len(list(score_dir.glob("summary.failed-*.json"))), 1)
        self.assertEqual(c.read(self.root / "status.json")["completed"], 1)

    def test_partial_generation_not_ready(self):
        root = self.root / "output"
        value = {"generation": {"request_batch_size": 4, "n": 16}, "resource_policy": c.POLICY,
                 "datasets": {"AMC23": {"rows": 83}}}
        value["protocol_sha256"] = c.digest(value)
        c.write(root / "protocol.json", value)
        c.write(root / "generation/AMC23/batch_000000.json", {})
        self.assertFalse(c.ready(root, "AMC23"))

    def test_selection_and_fast_profile_are_explicit(self):
        self.assertEqual(len(c.selected("all")), 14)
        self.assertEqual(c.selected("AMC23"), ["AMC23"])
        for text in ("AMC23,AMC23", "fake", ""):
            with self.assertRaises(ValueError):
                c.selected(text)
        command = run.generation_command(self.plan, self.models["sgd"], self.root)
        self.assertEqual(command[command.index("--request-batch-size") + 1], "4")
        self.plan["profile"] = "fast"
        command = run.generation_command(self.plan, self.models["sgd"], self.root)
        self.assertEqual(command[command.index("--request-batch-size") + 1], "16")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "256")
        self.assertEqual(command[command.index("--n-samples") + 1], "16")

    def test_scoring_cpu_only_environment_and_limits(self):
        command = eval_scoring.command(sys.executable, self.root, "LiveCodeBench")
        self.assertIn("--wait-lock", command)
        self.assertEqual(command[command.index("--workers") + 1], "2")
        self.assertEqual(command[command.index("--per-process-gib") + 1], "4")
        env = run.environment()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("ALLOW_BIGCODE_REMOTE_UPLOAD", env)

    def guard_command(self, report, code):
        boot = f"import sys;sys.path.insert(0,{str(c.HERE)!r});import memory_guard as g;from pathlib import Path;g.HERE=Path({str(self.root)!r});g.main()"
        return [sys.executable, "-B", "-c", boot, "--wait-lock", "--min-free-gib", "0.001",
                "--report", str(report), "--", sys.executable, "-c", code]

    def test_global_scoring_lock_and_hard_limit(self):
        processes = []
        try:
            for index in range(2):
                code = f"import time,resource,os,json;from pathlib import Path;assert resource.getrlimit(resource.RLIMIT_AS)==(4*1024**3,4*1024**3);assert os.environ['CUDA_VISIBLE_DEVICES']=='';p=Path({str(self.root / f'timing{index}.json')!r});a=time.time();time.sleep(0.5);p.write_text(json.dumps([a,time.time()]))"
                processes.append(subprocess.Popen(self.guard_command(self.root / f"guard{index}.json", code)))
            for process in processes:
                self.assertEqual(process.wait(timeout=10), 0)
            spans = sorted(c.read(self.root / f"timing{i}.json") for i in range(2))
            self.assertLessEqual(spans[0][1], spans[1][0])
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)

    def test_cancel_guard_cleans_child(self):
        pids = self.root / "child.pid"
        code = f"import os,time;from pathlib import Path;Path({str(pids)!r}).write_text(str(os.getpid()));time.sleep(30)"
        process = subprocess.Popen(self.guard_command(self.root / "cancel.json", code), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while not pids.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(pids.exists())
            pid = int(pids.read_text())
            process.terminate()
            process.wait(timeout=8)
            try:
                self.assertEqual(psutil.Process(pid).status(), psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                pass
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main(verbosity=2)
