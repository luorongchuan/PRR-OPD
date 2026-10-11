"""Download exact official EvalPlus releases separately from the Git repository."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = Path(os.environ.get('PRR_DATA_ROOT', ROOT / 'data/evaluation')) / 'EvalPlusOfficial'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'configs/evalplus_releases.json').read_text())
    for name, info in manifest['files'].items():
        destination = folder / name
        data = destination.read_bytes() if destination.exists() else urllib.request.urlopen(info['source_url'], timeout=120).read()
        if hashlib.sha256(data).hexdigest() != info['sha256']:
            raise ValueError('Official release checksum mismatch: ' + name)
        if not destination.exists():
            destination.write_bytes(data)
        print('VERIFIED ' + str(destination))


if __name__ == '__main__':
    main()
