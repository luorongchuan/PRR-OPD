"""Validate paper settings and launcher commands without allocating GPUs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))


@pytest.mark.parametrize('size,micro', [('0.6b', 8), ('1.7b', 4)])
@pytest.mark.parametrize('method', ['prr', 'opd', 'iw_opd', 'entropy_opd'])
def test_main_recipe(size, micro, method):
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train.py'), '--size', size,
                             '--method', method, '--dry-run'], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    config = plan['configuration']
    assert config['train']['micro_train_batch_size'] == micro
    assert config['train']['train_batch_size'] == config['rollout']['rollout_batch_size'] == 32
    assert config['train']['learning_rate'] == config['train']['min_lr'] == 3e-6
    assert config['train']['adam_betas'] == [0.9, 0.95]
    assert config['rollout']['generate_max_len'] == 4096
    assert config['data']['max_len'] == 5120 and config['data']['prompt_max_len'] == 1024
    assert config['kd']['prr_candidate'] == 'sgd'
    assert config['kd']['prr_method'] == {'prr': 'prr_gamma2', 'entropy_opd': 'entropy_opd'}.get(method, 'opd')
    assert '--prr_candidate' in plan['command']
    assert config['data']['train_shuffle'] is False and config['data']['train_drop_last'] is False


@pytest.mark.parametrize('size', ['0.6b', '1.7b'])
@pytest.mark.parametrize('method', ['prr', 'opd', 'iw_opd', 'entropy_opd', 'sft'])
def test_generated_arguments_are_accepted_by_actual_framework(size, method, monkeypatch):
    from kdflow.arguments import init_args
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train.py'), '--size', size,
                             '--method', method, '--data', 'data/teacher_targets.parquet', '--dry-run'],
                            capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    start = plan['command'].index('--train_dataset_path')
    monkeypatch.setattr(sys, 'argv', ['train_entry.py', *plan['command'][start:]])
    args = init_args('sft' if method == 'sft' else 'on_policy_kd')
    assert args.train.adam_betas == (0.9, 0.95)
    assert args.train.eval_steps == float('inf')
    assert args.train.micro_train_batch_size == (8 if size == '0.6b' else 4)
    assert Path(args.ckpt.save_path).is_absolute()


def test_sft_uses_fixed_responses_without_teacher():
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train.py'), '--method', 'sft',
        '--data', '/tmp/teacher_targets.parquet', '--dry-run'], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert 'torch.distributed.run' in plan['command']
    assert plan['configuration']['data']['output_key'] == 'teacher_response'
    assert plan['configuration']['model']['teacher_name_or_path'] is None


@pytest.mark.parametrize('method,steps,valid', [('sft', 2, True), ('prr', 2, False), ('prr', 3, True)])
def test_final_checkpoint_verification_accounts_for_sft_filtering(tmp_path, method, steps, valid):
    spec = importlib.util.spec_from_file_location('public_train', ROOT / 'scripts/train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    final = tmp_path / ('epoch_1_end_global_step_' + str(steps))
    (final / 'model').mkdir(parents=True)
    (final / 'model/config.json').write_text('{}')
    (final / 'model/model.safetensors').write_bytes(b'synthetic')
    (final / 'training_state').mkdir()
    for rank in (0, 1):
        (final / 'training_state' / ('rank_' + str(rank) + '.pt')).write_bytes(b'synthetic')
    if valid:
        assert module.final_checkpoint(tmp_path, method, 3) == (final, steps)
    else:
        with pytest.raises(RuntimeError, match='complete final checkpoint'):
            module.final_checkpoint(tmp_path, method, 3)


def test_no_weights_datasets_or_local_run_artifacts():
    prohibited = {'.safetensors', '.pt', '.pth', '.ckpt', '.parquet', '.arrow', '.pkl', '.whl', '.log', '.jsonl', '.gz'}
    for directory in ['training', 'configs', 'baselines', 'evaluation', 'docs', 'scripts', 'data', 'models']:
        for path in (ROOT / directory).rglob('*'):
            assert not path.is_symlink()
            assert path.suffix not in prohibited, path


def test_frozen_order_selects_distinct_clean_source_rows():
    order = json.loads((ROOT / 'configs/training_row_order.json').read_text())
    assert len(order) == len(set(order)) == 17629
    assert all(isinstance(index, int) and 0 <= index < 17902 for index in order)


def test_only_two_student_sizes_are_configured():
    assert sorted(path.stem for path in (ROOT / 'configs/training').glob('*.json')) == ['0.6b', '1.7b']


def test_relative_training_paths_resolve_against_checkout_not_cwd(tmp_path):
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train.py'), '--dry-run',
                             '--output', 'outputs/path_test', '--student', 'models/Qwen3-1.7B'],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    config = json.loads(result.stdout)['configuration']
    assert config['ckpt']['save_path'] == str(ROOT / 'outputs/path_test')
    assert config['model']['student_name_or_path'] == str(ROOT / 'models/Qwen3-1.7B')
    for section, key in [('data', 'train_dataset_path'), ('model', 'teacher_name_or_path')]:
        assert Path(config[section][key]).is_absolute()


def test_public_sources_have_no_private_home_paths():
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.sh', '.md', '.json', '.txt', '.toml', '.yml'}:
            if any(part in {'runtime', '__pycache__', '.pytest_cache'} for part in path.parts):
                continue
            assert ('/' + 'home' + '/') not in path.read_text(), path


def test_eval_loads_only_requested_math_data(monkeypatch):
    path = ROOT / 'evaluation/scorers/pinned14/eval_14_pass1.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('public_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('PRR_EVAL_DATASETS', 'AMC23')
    # An unrelated empty math test directory must not trigger any file reads.
    monkeypatch.setenv('PRR_EVAL_DATASETS', 'HumanEvalPlus')
    assert module.load_math(ROOT / 'evaluation/scorers/pinned14/vendor/other-test',
                            Path('/nonexistent-datasets'), None, 2048, 8192) == {}
