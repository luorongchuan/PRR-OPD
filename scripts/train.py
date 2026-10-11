"""Launch the paper's two-GPU recipes from portable, recorded configurations."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = '2aac8e297d507e66adfb8cb42e64651883965d9aac665b72efe832ca0e85fc76'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_complete(path):
    if not (path / 'config.json').is_file():
        return False
    index = path / 'model.safetensors.index.json'
    names = (set(json.loads(index.read_text())['weight_map'].values()) if index.is_file()
             else {'model.safetensors'})
    return bool(names) and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in names)


def configuration(args):
    values = json.loads((ROOT / 'configs/training' / (args.size + '.json')).read_text())
    for section, key in [('data', 'train_dataset_path'), ('model', 'student_name_or_path'),
                         ('model', 'teacher_name_or_path')]:
        values[section][key] = str(Path(values[section][key].replace('${REPO_ROOT}', str(ROOT))).resolve())
    if args.data:
        values['data']['train_dataset_path'] = str(args.data.resolve())
    if args.student:
        values['model']['student_name_or_path'] = str(args.student.resolve())
    if args.teacher:
        values['model']['teacher_name_or_path'] = str(args.teacher.resolve())
    output = args.output or ROOT / 'outputs' / (args.size + '_' + args.method + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    values['ckpt']['save_path'] = str(output.resolve())
    if args.resume:
        values['ckpt'].update(resume_from=str(args.resume.resolve()), resume_training=True)
    kd = values['kd']
    kd.update(prr_candidate=args.candidate, prr_feedback='shuffled' if args.shuffle_weights else 'aligned',
              prr_gamma=args.gamma, prr_interval=args.interval,
              prr_method='prr_gamma2' if args.method == 'prr' else 'opd')
    if args.method == 'entropy_opd':
        kd['prr_method'] = 'entropy_opd'
    if args.method == 'sft':
        if args.data is None:
            raise ValueError('SFT requires --data pointing to a prepared teacher-response parquet')
        values['model']['teacher_name_or_path'] = None
        values['data']['output_key'] = 'teacher_response'
        kd['kd_algorithm'] = 'vanilla_kd'
    return values


def command(values, method, python):
    if method == 'sft':
        argv = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                str(ROOT / 'scripts/sft_entry.py')]
    else:
        argv = [python, '-B', str(ROOT / 'scripts/train_entry.py')]
    for section in values.values():
        for key, value in section.items():
            if value is None:
                continue
            if key == 'adam_betas':
                value = str(tuple(value))
            if key in ('eval_steps', 'save_steps') and value == 'inf':
                value = -1
            argv.extend(['--' + key, str(value)])
    return argv


def final_checkpoint(output, method, expected_steps):
    if method == 'sft':
        candidates = [path for path in output.glob('epoch_1_end_global_step_*')
                      if re.fullmatch(r'epoch_1_end_global_step_[1-9][0-9]*', path.name)]
        if len(candidates) != 1:
            raise RuntimeError('SFT must produce one completed final-epoch checkpoint')
        final = candidates[0]
        steps = int(final.name.rsplit('_', 1)[1])
        if not 0 < steps <= expected_steps:
            raise RuntimeError('SFT step count exceeds the input-row budget')
    else:
        steps = expected_steps
        final = output / ('epoch_1_end_global_step_' + str(steps))
    if not model_complete(final / 'model') or any(
            not (final / 'training_state' / ('rank_' + str(rank) + '.pt')).is_file() for rank in (0, 1)):
        raise RuntimeError('Training returned without a complete final checkpoint')
    return final, steps


def run(args):
    if not re.fullmatch(r'\d+,\d+', args.gpus) or len(set(args.gpus.split(','))) != 2:
        raise ValueError('--gpus must contain two distinct indices, e.g. 0,1')
    if not args.gamma >= 0 or not args.gamma < float('inf') or args.interval < 1:
        raise ValueError('Invalid feedback parameters')
    values = configuration(args)
    output = Path(values['ckpt']['save_path'])
    argv = command(values, args.method, sys.executable)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpus,
           'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PRR_TRAIN_METHOD': args.method,
           'IW_OPD_ENABLE': '1' if args.method == 'iw_opd' else '0',
           'IW_OPD_IMPL': str(ROOT / 'baselines/iw_opd_full_kl.py'),
           'IW_GAMMA': '0.5', 'IW_NORMALIZATION': 'none', 'IW_EPS': '1e-8',
           'PYTHONPATH': str(ROOT / 'training') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    if args.dry_run:
        print(json.dumps({'gpus': args.gpus, 'method': args.method, 'configuration': values,
                          'command': argv, 'shell_command': shlex.join(argv)}, indent=2))
        return
    data = Path(values['data']['train_dataset_path'])
    import pyarrow.parquet as pq
    rows = pq.ParquetFile(data).metadata.num_rows
    if args.method != 'sft' and (rows != 17629 or sha(data) != DATA_SHA256):
        raise ValueError('Frozen training parquet must have 17,629 rows and the recorded SHA256')
    if rows < 1:
        raise ValueError('Empty training data')
    if args.method == 'sft':
        values['data']['max_samples'] = rows
        argv = command(values, args.method, sys.executable)
    for key in ['student_name_or_path'] + ([] if args.method == 'sft' else ['teacher_name_or_path']):
        if not model_complete(Path(values['model'][key])):
            raise ValueError('Incomplete Hugging Face model: ' + values['model'][key])
    if args.resume:
        if args.resume.resolve().parent != output:
            raise ValueError('--resume must be a checkpoint inside --output')
        if not model_complete(args.resume.resolve() / 'model') or any(
                not (args.resume / 'training_state' / ('rank_' + str(rank) + '.pt')).is_file() for rank in (0, 1)):
            raise ValueError('Resume requires both model and two ranks of training state')
        previous = json.loads((output / 'launch.json').read_text())
        old, new = previous['configuration'], json.loads(json.dumps(values))
        for item in (old, new):
            item['ckpt'].update(resume_from=None, resume_training=False)
        if old != new or previous['data_sha256'] != sha(data):
            raise ValueError('Resume configuration or training data changed')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; choose another path or explicitly resume')
    output.mkdir(parents=True, exist_ok=True)
    lock_root = ROOT / 'runtime/training_locks'
    lock_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for gpu in sorted(args.gpus.split(','), key=int):
            handle = stack.enter_context((lock_root / ('gpu_' + gpu + '.lock')).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True).strip())
            if used > 512:
                raise RuntimeError('GPU ' + gpu + ' is occupied (' + str(used) + ' MiB)')
        receipt = {'method': args.method, 'gpus': args.gpus, 'configuration': values, 'command': argv,
                   'data_sha256': sha(data), 'rows': rows, 'expected_steps': (rows + 31) // 32,
                   'framework_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'training/kdflow').rglob('*.py')}}
        (output / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        with (output / 'train.log').open('a') as log:
            print('Training log: ' + str(output / 'train.log'), flush=True)
            process = subprocess.Popen(argv, cwd=ROOT / 'training', env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            previous_handlers = {}

            def stop(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)

            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, stop)
            try:
                status = process.wait()
            except BaseException:
                stop(signal.SIGTERM, None)
                process.wait()
                raise
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
            if status:
                raise RuntimeError('Training exited ' + str(status) + '; inspect train.log')
        final, steps = final_checkpoint(output, args.method, receipt['expected_steps'])
        (output / 'train_complete.json').write_text(json.dumps({'model': str(final / 'model'),
                  'steps': steps, 'data_sha256': receipt['data_sha256']}, indent=2) + '\n')
        print('Completed: ' + str(final / 'model'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['prr', 'opd', 'iw_opd', 'entropy_opd', 'sft'], default='prr')
    parser.add_argument('--size', choices=['0.6b', '1.7b'], default='0.6b')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES') or '0,1')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--student', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--candidate', choices=['sgd', 'adamw'], default='sgd')
    parser.add_argument('--shuffle-weights', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for key in ('data', 'student', 'teacher', 'output', 'resume'):
        path = getattr(args, key)
        if path is not None:
            path = path.expanduser()
            setattr(args, key, (path if path.is_absolute() else ROOT / path).resolve())
    run(args)


if __name__ == '__main__':
    main()
