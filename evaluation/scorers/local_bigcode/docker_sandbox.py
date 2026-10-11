"""Isolated local official-image backend; never send predictions to a service."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import uuid


def command(image, mode, problems, predictions, cache, output, cidfile, nonce):
    here = Path(__file__).resolve().parent
    cache_mode = "ro" if mode == "score" else "rw"
    return ["docker", "run", "--rm", "--init", "--cidfile", str(cidfile),
            "--label", f"prr-evaluation-owner={os.getuid()}", "--label", f"prr-evaluation-job={nonce}",
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "24g", "--memory-swap", "24g",
            "--cpus", "2", "--pids-limit", "128", "--shm-size", "2g",
            "--ulimit", "as=32212254720:32212254720", "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=8g,mode=1777", "--workdir", "/tmp",
            "-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void",
            "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
            "-e", "HOME=/tmp", "-e", "XDG_CONFIG_HOME=/tmp/config", "-e", "XDG_CACHE_HOME=/tmp/cache",
            "-e", "MPLCONFIGDIR=/tmp/matplotlib", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "PYTHONPATH=/vendor/bigcodebench:/work",
            "-e", "PRR_BIGCODE_DOCKER_SANDBOX=1",
            "-v", f"{here}:/work:ro", "-v", f"{here.parent / 'bigcodebench'}:/vendor/bigcodebench:ro",
            "-v", f"{problems}:/data/problems.parquet:ro", "-v", f"{predictions}:/input/predictions.jsonl:ro",
            "-v", f"{cache}:/controls:{cache_mode}", "-v", f"{output}:/out:rw",
            "--entrypoint", "python", image, "-B", "/work/score_local.py", "--mode", mode,
            "--problems", "/data/problems.parquet", "--predictions", "/input/predictions.jsonl",
            "--cache", "/controls", "--output", "/out", "--vendor", "/vendor/bigcodebench"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "controls", "score"))
    for name in ("problems", "predictions", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("python")
    args = parser.parse_args()
    image = os.environ.get("PRR_BIGCODE_IMAGE", "bigcodebench/bigcodebench-evaluate:latest")
    # Never automatically pull an unpinned image during model scoring.
    data = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    image = data["Id"]
    for path in (args.cache, args.output):
        path.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    with tempfile.TemporaryDirectory(prefix="container-", dir=args.output) as temp:
        cid = Path(temp) / "id"
        argv = command(image, args.mode, args.problems.resolve(), args.predictions.resolve(),
                       args.cache.resolve(), args.output.resolve(), cid, nonce)
        (args.output / "container_command.json").write_text(json.dumps(argv, indent=2) + "\n")
        try:
            subprocess.run(argv, check=True)
        finally:
            if cid.exists():
                identity = cid.read_text().strip()
                result = subprocess.run(["docker", "inspect", identity], capture_output=True, text=True)
                if result.returncode == 0:
                    labels = json.loads(result.stdout)[0]["Config"].get("Labels", {})
                    if labels.get("prr-evaluation-job") == nonce and labels.get("prr-evaluation-owner") == str(os.getuid()):
                        subprocess.run(["docker", "rm", "-f", identity], check=True)


if __name__ == "__main__":
    main()
