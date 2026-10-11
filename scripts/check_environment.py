"""Check imports, versions and namespace prerequisites without allocating a GPU."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['training', 'evaluation'], required=True)
    parser.add_argument('--check-sandbox', action='store_true')
    args = parser.parse_args()
    names = (['torch', 'transformers', 'ray', 'sglang', 'datasets', 'pyarrow', 'flash_attn', 'ring_flash_attn']
             if args.mode == 'training' else ['torch', 'transformers', 'vllm', 'datasets', 'pyarrow', 'math_verify', 'psutil'])
    results, failures = {}, {}
    for name in names:
        try:
            importlib.import_module(name)
            distributions = importlib.metadata.packages_distributions().get(name, [name.replace('_', '-')])
            results[name] = importlib.metadata.version(distributions[0])
        except Exception as exc:
            failures[name] = str(exc)
    if args.check_sandbox:
        result = subprocess.run(['unshare', '--user', '--map-root-user', '--net', '--mount', '--pid',
                                 '--fork', 'true'], capture_output=True, text=True)
        if result.returncode:
            failures['user_namespace'] = result.stderr.strip()
    print(json.dumps({'python': sys.version, 'mode': args.mode, 'versions': results, 'failures': failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
