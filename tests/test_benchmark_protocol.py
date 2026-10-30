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
from pyboost_benchmark_suite.protocol import ROOT, comparison_errors, implementation_hash, load_suite, make_data, quality
from pyboost_benchmark_suite.run import environment, render_comparison
from pyboost_benchmark_suite import run as runner


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


def test_k5_profile_changes_only_sketch_width_and_identity():
    from copy import deepcopy
    original = load_suite()
    k5 = load_suite(ROOT / 'suite-v1-k5.json')
    assert k5['benchmark_id'] == 'pyboost-gpu-v1-k5'
    expected = deepcopy(original)
    expected['benchmark_id'] = k5['benchmark_id']
    for case in expected['cases']:
        if case['model'] == 'SketchBoost':
            case['params']['sketch_outputs'] = 5
    assert k5 == expected
    assert implementation_hash(k5) != implementation_hash(original)
    assert all(c['params']['sketch_outputs'] == 8 for c in original['cases']
               if c['model'] == 'SketchBoost')
    before, after = small_case(), small_case()
    before['params'] = dict(sketch_outputs=8)
    after['params'] = dict(sketch_outputs=5)
    assert make_data(before, 42)[-1] == make_data(after, 42)[-1]


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
                environment={'gpu_count': 1, 'gpu': {'name': 'NVIDIA A100 80GB PCIe',
                'uuid': 'GPU-test', 'vram_bytes': 80 * 2**30, 'compute_capability': [8, 0],
                'physical_vram_bytes': 80 * 2**30, 'power_limit_watts': '300',
                'max_sm_clock_mhz': '1410',
                'pci_bus_id': '00000000:14:00.0', 'multiprocessors': 108,
                'memory_bus_width_bits': 5120, 'driver_version': '550.163.01'},
                'cuda_driver': 12040, 'cuda_runtime': 12090, 'packages': {'numpy': '2'}},
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


def test_report_displays_single_gpu_characteristics():
    text = render_comparison(report(), report())
    for value in ['GPU count: 1', 'NVIDIA A100 80GB PCIe', '80.0 GiB',
                  'GPU-test', '8.0', '108', '5120', '550.163.01', '12.4', '12.9',
                  '300 W', '1410 MHz']:
        assert value in text


@pytest.mark.parametrize('method', ['proj', 'rand', 'filter', 'topk'])
def test_report_distinguishes_target_dimension_from_sketch_width(method):
    base, candidate = report(), report()
    for result in [base, candidate]:
        result['cases']['test']['spec'] = dict(outputs=10000, params=dict(
            sketch_method=method, sketch_outputs=8))
    text = render_comparison(base, candidate)
    assert '| test | 10000 | 8 (' + method + ') |' in text
    assert 'at most k distinct outputs' in text
    assert 'Predictions always have K outputs' in text


def test_report_shows_full_output_scoring_without_sketch():
    text = render_comparison(report(), report())
    assert '| test | 1024 | all 1024 |' in text


@pytest.mark.parametrize('count', [0, 2])
def test_preflight_rejects_any_visible_gpu_count_other_than_one(count):
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: count)))
    with pytest.raises(RuntimeError, match='Exactly one visible CUDA device'):
        environment(cp, np, None, '0')


def test_cli_rejects_multiple_gpu_ids_before_starting_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'source_info', lambda path: dict(commit='abc', path=str(path)))
    monkeypatch.setattr(runner, 'invoke', lambda *args: pytest.fail('Worker must not start'))
    output = tmp_path / 'results'
    args = SimpleNamespace(source=['checkout'], suite=None, case=None, repeats=5, timeout=900,
                           device='0,1', output=str(output))
    with pytest.raises(ValueError, match='exactly one GPU'):
        runner.run(args)
    assert not output.exists()


def test_gpu_index_is_resolved_to_physical_uuid(monkeypatch):
    commands = []
    def query(command, **kwargs):
        commands.append(command)
        return 'GPU-selected\n'
    monkeypatch.setattr(runner.subprocess, 'check_output', query)
    assert runner.resolve_gpu('1') == 'GPU-selected'
    assert commands == [['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader']]


@pytest.mark.parametrize('response', ['', 'GPU-a\nGPU-b\n', 'MIG-instance\n'])
def test_gpu_selector_must_resolve_to_one_physical_card(monkeypatch, response):
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match='exactly one physical GPU'):
        runner.resolve_gpu('selector')


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
