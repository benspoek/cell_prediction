import json
from pathlib import Path

import pytest

from cell_property_prediction.cli import parser
from cell_property_prediction.config import atomic_json, load_config
from cell_property_prediction.jobs import compute_fingerprint, create_jobs, estimate_jobs, read_job


def test_config_and_command_contract():
    root = Path(__file__).parents[1]
    config = load_config(root / 'configs/benchmark.yaml')
    assert config['cluster']['account'] == 'thes2301'
    assert len(config['evaluation']['representations']) == 6
    parsed = parser().parse_args(['evaluate', '--protocol', 'cell_kfold', '--representation',
        'authors25', '--predictor', 'kumo', '--fold', '0'])
    assert parsed.fold == 0


def test_job_matrix_and_budget(tmp_path):
    root = Path(__file__).parents[1]
    config = load_config(root / 'configs/benchmark.yaml')
    for scheme in config['evaluation']['protocols']:
        atomic_json(tmp_path / config['paths']['splits_dir'] / f'{scheme}.json', {'folds': [{}] * 10})
    jobs = create_jobs(config, tmp_path, 'simple', tmp_path / 'jobs.tsv')
    assert len(jobs) == 60
    assert read_job(tmp_path / 'jobs.tsv', 59)['fold'] == 9
    pilot = tmp_path / 'pilot.json'
    atomic_json(pilot, {'fingerprint': compute_fingerprint(config, tmp_path), 'runs': [{'representation': r, 'predictor': 'tabpfn',
        'elapsed_seconds': 1, 'targets': 2, 'device': 'cuda'} for r in ['pca128', 'ion_channels', 'authors25']]})
    estimate = estimate_jobs(config, jobs, pilot, root=tmp_path)
    assert estimate['estimated_gpu_hours_with_2x_margin'] > 0
    assert not estimate['within_budget']
    config['cluster']['remaining_gpu_hours'] = 100
    assert estimate_jobs(config, jobs, pilot, root=tmp_path)['within_budget']
    config['models']['tabpfn']['n_estimators'] = 4
    with pytest.raises(ValueError, match='provenance'):
        estimate_jobs(config, jobs, pilot, root=tmp_path)
    with pytest.raises(IndexError):
        read_job(tmp_path / 'jobs.tsv', 60)


def test_rrr_job_dispatch(monkeypatch):
    from cell_property_prediction import cli, rrr
    expected = {'status': 'synthetic'}
    monkeypatch.setattr(rrr, 'run_rrr_fold', lambda config, **kwargs: expected)
    assert cli.evaluate_job({}, Path('.'), protocol='cell_kfold', representation='raw1000',
                            predictor='sparse_rrr', fold=0, device='cpu') == expected


def test_pilot_preserves_original_compute_time_after_resume(tmp_path, monkeypatch):
    import yaml
    from cell_property_prediction import cli
    config = load_config(Path(__file__).parents[1] / 'configs/benchmark.yaml')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def resumed_evaluation(*args, **kwargs):
        calls.append(kwargs)
        return {'targets': [{'status': 'complete', 'elapsed_seconds': 90.},
                            {'status': 'complete', 'elapsed_seconds': 100.}]}

    monkeypatch.setattr(cli, 'evaluate_job', resumed_evaluation)
    argv = ['--root', str(tmp_path), '--config', str(config_path), 'pilot', '--predictors',
            'tabpfn', '--representations', 'authors25']
    assert cli.main(argv) == 0
    timing = json.loads((tmp_path / 'results/pilot/timings.json').read_text())
    assert timing['runs'][0]['elapsed_seconds'] >= 190.
    assert cli.main(argv) == 0
    assert len(calls) == 1
