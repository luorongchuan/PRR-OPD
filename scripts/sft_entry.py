"""Record SFT arguments and call the unchanged distributed training entry."""
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))


def main():
    from kdflow.arguments import init_args
    args = init_args('sft')
    if int(os.environ.get('RANK', '0')) == 0:
        output = Path(args.ckpt.save_path)
        output.mkdir(parents=True, exist_ok=True)
        receipt = {'arguments': dataclasses.asdict(args), 'method': 'sft', 'python': sys.version,
                   'environment': {name: importlib.metadata.version(name)
                                   for name in ('torch', 'transformers')}}
        (output / 'arguments.json').write_text(json.dumps(receipt, indent=2) + '\n')
    from kdflow.cli.train_sft import train
    train(args)


if __name__ == '__main__':
    main()
