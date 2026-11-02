"""Dataset integrity, official split isolation and five-fold report aggregation."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_benchmark_protocol import report
from pyboost_benchmark_suite import delicious
from pyboost_benchmark_suite.protocol import load_suite
from pyboost_benchmark_suite.run import cv_summary
from pyboost_benchmark_suite import run as runner


def test_paper_profile_has_all_folds_and_outputs():
    suite = load_suite()
    assert suite['benchmark_id'] == 'pyboost-delicious-paper-v1-k5'
    assert len(suite['cases']) == 25
    for method in ['full', 'topk', 'rand', 'proj', 'filter']:
        cases = [c for c in suite['cases'] if c['group'] == f'delicious-{method}']
        assert {c['fold'] for c in cases} == set(range(5))
        assert all(c['outputs'] == 983 for c in cases)
    params = suite['common_params']
    assert (params['ntrees'], params['es'], params['lr'], params['max_depth']) == (20000, 500, .015, 12)
    assert params['use_hess'] is False
    assert suite['repeats'] == 1  # Different folds, rather than five repeats per fold.


def test_corrupted_archive_rejected_before_parsing(tmp_path):
    archive = tmp_path / 'Delicious.zip'
    archive.write_bytes(b'not the official archive')
    with pytest.raises(ValueError, match='archive SHA-256 mismatch'):
        delicious.prepare_data(tmp_path, archive)
    assert not (tmp_path / 'delicious.npz').exists()


def test_official_test_set_never_enters_folds_and_validation_covers_train(monkeypatch, tmp_path):
    # Tiny checked arrays avoid downloading a dataset during unit tests.
    X = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.zeros((20, 2), dtype=np.float32)
    train, test = np.arange(15), np.arange(15, 20)
    np.savez(tmp_path / 'delicious.npz', X=X, y=y, train=train, test=test)
    monkeypatch.setattr(delicious, 'NORMALIZED_SHA256', delicious.array_hash(X, y, train, test))
    validations = []
    hashes = []
    for fold in range(5):
        case = dict(fold=fold, train=12, valid=3, test=5)
        a, b, valid, tx, ty, digest = delicious.load_fold(case, 42, tmp_path)
        np.testing.assert_array_equal(tx, X[test])
        np.testing.assert_array_equal(ty, y[test])
        fit_ids = set(a[:, 0])
        valid_ids = set(valid[0]['X'][:, 0])
        assert fit_ids.isdisjoint(valid_ids)
        assert (fit_ids | valid_ids) == set(X[train, 0])
        validations.extend(valid_ids)
        hashes.append(digest)
        assert digest == delicious.load_fold(case, 42, tmp_path)[-1]
    assert sorted(validations) == sorted(X[train, 0])
    assert len(set(hashes)) == 5
    y[0, 0] = 1
    np.savez(tmp_path / 'delicious.npz', X=X, y=y, train=train, test=test)
    with pytest.raises(ValueError, match='checksum mismatch'):
        delicious.load_fold(case, 42, tmp_path)


def test_cv_uses_mean_not_median_and_maximum_memory():
    base = report()
    sample = base['cases']['test']
    base['cases'] = {}
    base['case_ids'] = []
    for fold, seconds in enumerate([1., 1., 1., 1., 6.]):
        key = f'fold{fold}'
        value = deepcopy(sample)
        value['spec'] = dict(group='delicious-proj', fold=fold, params=dict(sketch_outputs=5))
        value['timing']['fit_seconds']['median'] = seconds
        value['memory']['fit_cupy_peak_bytes'] = (fold + 1) * 2**20
        base['cases'][key] = value
        base['case_ids'].append(key)
    candidate = deepcopy(base)
    text = '\n'.join(cv_summary(base, candidate))
    assert '| delicious-proj | 983 | 5 | 2 ± 2 → 2 ± 2 |' in text
    assert '| 5.00 → 5.00 |' in text
    base['case_ids'].pop()
    assert 'Incomplete CV group' in '\n'.join(cv_summary(base, base))


def test_resume_skips_completed_folds_and_rejects_changed_sources(monkeypatch, tmp_path):
    suite = load_suite()
    suite.pop('dataset')
    suite['cases'] = suite['cases'][:1]
    manifest = tmp_path / 'suite.json'
    manifest.write_text(json.dumps(suite))
    calls = []
    source = dict(commit='123456789abc', path='checkout', library_hash='original')
    monkeypatch.setattr(runner, 'source_info', lambda path: source.copy())
    monkeypatch.setattr(runner, 'resolve_gpu', lambda selector: 'GPU-test')
    def invoke(python, path, job, timeout):
        calls.append(job['phase'])
        return dict(status='ok', environment={'gpu': 'test'})
    monkeypatch.setattr(runner, 'invoke', invoke)
    args = SimpleNamespace(suite=manifest, source=['checkout'], repeats=None,
                           timeout=None, case=None, device='1', python='python',
                           output=tmp_path / 'out', data_dir=tmp_path, resume=False)
    assert runner.run(args) == 0
    args.resume = True
    assert runner.run(args) == 0
    assert calls == ['metadata', 'case', 'metadata']
    source['library_hash'] = 'changed'
    with pytest.raises(ValueError, match='different source'):
        runner.run(args)
