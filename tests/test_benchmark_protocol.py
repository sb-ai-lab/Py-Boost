"""The result protocol must reject misleading cross-version comparisons."""

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

# The CPU wheel workflow runs tests outside the source root. Load only the
# benchmark package, without putting the library checkout ahead of the wheel.
_spec = importlib.util.spec_from_file_location(
    'pyboost_benchmark_suite', Path(__file__).resolve().parents[1] / 'benchmarks' / '__init__.py')
_package = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _package
_spec.loader.exec_module(_package)

from pyboost_benchmark_suite.gpu import memory_tracker
from pyboost_benchmark_suite.protocol import comparison_errors, load_suite, make_data, quality
from pyboost_benchmark_suite.run import render_comparison


def small_case(task='regression'):
    return dict(id='test', train=64, valid=16, test=16, features=32,
                outputs=8, validation_sets=2, task=task)


@pytest.mark.parametrize('task', ['regression', 'multilabel', 'multiclass'])
def test_datasets_are_reproducible_and_separated(task):
    case = small_case(task)
    a, b = make_data(case, 42), make_data(case, 42)
    assert a[-1] == b[-1]
    assert a[-1] != make_data(case, 43)[-1]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[0].shape == (64, 32)
    assert [len(v['y']) for v in a[2]] == [8, 8]
    assert not np.shares_memory(a[0], a[3])
    if task == 'multiclass':
        np.testing.assert_array_equal(np.unique(a[1]), np.arange(8))
    elif task == 'multilabel':
        assert set(np.unique(a[1])) == {0, 1}
    else:
        assert quality(task, a[1], a[1])['value'] == 0


def test_same_multilabel_data_for_different_sketch_methods():
    suite = load_suite()
    rand, filt = [next(c for c in suite['cases'] if c['id'] == name)
                  for name in ['bce-rand-1024', 'bce-filter-1024']]
    for case in [rand, filt]:
        case.update(train=64, valid=16, test=16, outputs=8)
    assert make_data(rand, suite['data_seed'])[-1] == make_data(filt, suite['data_seed'])[-1]


def test_holdout_metrics_and_invalid_predictions():
    assert quality('multilabel', np.array([[0, 1]], dtype=np.float32),
                   np.array([[.5, .5]], dtype=np.float32))['value'] == pytest.approx(np.log(2))
    assert quality('multiclass', np.array([1]), np.array([[.25, .75]]))['value'] == pytest.approx(-np.log(.75))
    with pytest.raises(ValueError, match='Non-finite'):
        quality('regression', np.zeros((1, 1)), np.array([[np.nan]]))
    with pytest.raises(ValueError, match='output count'):
        quality('regression', np.zeros((1, 2)), np.zeros((1, 1)))


def report():
    timing = dict(fit_seconds={'median': 1.}, predict_seconds={'median': .1},
                  quality={'median': .2}, measurements=[{'retained_trees': 32}])
    return dict(schema_version=1, benchmark_id='pyboost-gpu-v1', implementation_hash='abc',
                suite_hash='def', case_ids=['test'], repeats=5, warmups=1, complete=True,
                environment={'gpu': 'A100', 'packages': {'numpy': '2'}},
                source={'commit': 'abc'}, cases={'test': dict(status='ok', data_hash='123',
                spec={'outputs': 1024}, timing=timing, memory={'fit_cupy_peak_bytes': 2**20})})


@pytest.mark.parametrize('key,value', [
    ('implementation_hash', 'changed'), ('suite_hash', 'changed'),
    ('environment', {'gpu': 'T4'}), ('repeats', 1), ('complete', False),
])
def test_incompatible_measurements_are_not_compared(key, value):
    base, candidate = report(), report()
    candidate[key] = value
    assert comparison_errors(base, candidate)
    text = render_comparison(base, candidate)
    assert 'Not comparable' in text
    assert '| Case |' not in text


@pytest.mark.parametrize('change', ['dataset', 'error', 'missing_case', 'spec'])
def test_incomplete_or_different_workloads_are_rejected(change):
    base, candidate = report(), report()
    if change == 'dataset':
        candidate['cases']['test']['data_hash'] = 'different'
    elif change == 'error':
        candidate['cases']['test']['status'] = 'error'
    elif change == 'missing_case':
        candidate['cases'].clear()
    else:
        candidate['cases']['test']['spec']['outputs'] = 1
    assert comparison_errors(base, candidate)


def test_comparison_preserves_direction_and_discloses_tree_count_changes():
    base, candidate = report(), report()
    candidate['source']['commit'] = 'new'
    candidate['cases']['test']['timing']['fit_seconds']['median'] = 1.5
    candidate['cases']['test']['memory']['fit_cupy_peak_bytes'] = 2**19
    candidate['cases']['test']['timing']['measurements'][0]['retained_trees'] = 31
    assert comparison_errors(base, candidate) == []
    text = render_comparison(base, candidate)
    assert '+50.0%' in text
    assert '-50.0%' in text
    assert 'Retained tree counts differ' in text


def test_live_memory_tracking_handles_reuse_and_preexisting_allocations():
    tracker = memory_tracker(SimpleNamespace(cuda=SimpleNamespace(MemoryHook=object)))
    tracker.free_postprocess(0, 512, 1, 999)  # Allocated before the tracked scope.
    tracker.malloc_postprocess(0, 100, 512, 1, 1)
    tracker.malloc_postprocess(0, 800, 1024, 2, 2)
    assert tracker.current == tracker.peak == 1536
    tracker.free_postprocess(0, 512, 1, 1)
    tracker.malloc_postprocess(0, 100, 512, 1, 3)  # Reused pool block.
    assert tracker.current == tracker.peak == 1536
    tracker.free_postprocess(0, 1024, 2, 2)
    tracker.reset_peak()
    assert tracker.current == tracker.peak == 512
    tracker.free_postprocess(0, 512, 1, 3)
    assert tracker.current == 0


def test_real_hook_sees_private_cupy_pool():
    cp = pytest.importorskip('cupy')
    if not cp.is_available():
        pytest.skip('Requires a CUDA device')
    pool = cp.cuda.MemoryPool()
    with memory_tracker(cp) as tracker:
        with cp.cuda.using_allocator(pool.malloc):
            a = cp.empty(1024, dtype=cp.float32)
            b = cp.empty(1024, dtype=cp.float64)
            assert tracker.current == tracker.peak == a.nbytes + b.nbytes
            del a, b
            assert tracker.current == 0
    pool.free_all_blocks()
