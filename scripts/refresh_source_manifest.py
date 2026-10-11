"""Record published scorer bytes after an intentional source update."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'evaluation'
    files = []
    for path in sorted((folder / 'scorers').rglob('*')):
        if path.is_file() and path.name != 'SOURCE_MANIFEST.json' and '__pycache__' not in path.parts:
            files.append({'target': path.relative_to(folder).as_posix(),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder / 'scorers/SOURCE_MANIFEST.json').write_text(json.dumps({'release': 'PRR-OPD code-only public snapshot',
        'upstream_commits': json.loads((ROOT / 'configs/third_party_commits.json').read_text()), 'files': files}, indent=2) + '\n')
    print('Recorded ' + str(len(files)) + ' scorer files')


if __name__ == '__main__':
    main()
