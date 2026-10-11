"""Apply the recorded prompt order without redistributing training examples."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '45011c3651109088a1132fc41d7c5a91e41b9c2fdcaa106dc68ef9b54b074234'
FROZEN_SHA = '2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/train_frozen_17629_seed42.parquet')
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Cleaned source parquet differs from the recorded source')
    order = json.loads((ROOT / 'configs/training_row_order.json').read_text())
    table = pq.read_table(args.input)
    if (table.num_rows != 17902 or len(order) != 17629 or len(set(order)) != 17629
            or any(not isinstance(index, int) or index < 0 or index >= table.num_rows for index in order)):
        raise ValueError('Invalid frozen row-order metadata')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.parquet.pending')
    pq.write_table(table.take(pa.array(order)), temporary)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != FROZEN_SHA:
        raise RuntimeError('Reconstructed bytes differ (PyArrow serialization may differ); retained ' +
                           str(temporary) + '. Use the exact snapshot for paper reproduction; do not bypass its hash.')
    temporary.replace(args.output)
    print(str(args.output))


if __name__ == '__main__':
    main()
