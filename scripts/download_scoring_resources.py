"""Fetch hash-pinned public task resources; never substitute changed resources."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'evaluation/repairs/scoring_v4/assets'
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, row in manifest['resources'].items():
        path = folder / name
        content = path.read_bytes() if path.is_file() else urllib.request.urlopen(row['source'], timeout=120).read()
        if hashlib.sha256(content).hexdigest() != row['sha256']:
            raise RuntimeError('Resource bytes changed: ' + name + '; obtain the exact frozen resource separately')
        if not path.exists():
            path.write_bytes(content)
        print('VERIFIED ' + name)


if __name__ == '__main__':
    main()
