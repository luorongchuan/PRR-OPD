"""Record resolved arguments and enter the unchanged KDFlow training framework."""
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))


def main():
    from kdflow.arguments import init_args
    args = init_args('on_policy_kd')
    output = Path(args.ckpt.save_path)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'arguments': dataclasses.asdict(args), 'method': os.environ.get('PRR_TRAIN_METHOD'),
                'python': sys.version, 'environment': {name: importlib.metadata.version(name)
                    for name in ('torch', 'transformers', 'ray', 'sglang')}}
    (output / 'arguments.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from kdflow.cli.train_kd_on_policy import train
    import ray
    try:
        train(args)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    main()
