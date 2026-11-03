"""GPU transfer recovery must not retain live arrays after a failed upload."""

import pickle
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA device unavailable: {exc}')
    if not count:
        pytest.skip('A CUDA device is required')
    return cupy


@pytest.fixture(scope='module', params=[False, True], ids=['compact', 'debug'])
def fitted(request, cp):
    from py_boost import GradientBoosting

    rng = np.random.default_rng(12)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    X[::11, 0] = np.nan
    clean = np.nan_to_num(X)
    y = np.column_stack([clean[:, 0] + .5 * clean[:, 1], clean[:, 2] - clean[:, 3]])
    model = GradientBoosting('mse', ntrees=3, max_depth=2, max_bin=16,
                             min_data_in_leaf=2, es=0, verbose=0, debug=request.param).fit(X, y)
    expected = model.predict(X, batch_size=32)
    model.to_cpu()
    return X, expected, pickle.dumps(model)


@pytest.fixture
def model_data(fitted, cp):
    X, expected, serialized = fitted
    model = pickle.loads(serialized)
    yield model, X, expected
    # Also clean up when these regressions are run against the broken version.
    for tree in model.models:
        tree.to_cpu()
    if isinstance(model.base_score, cp.ndarray):
        model.base_score = model.base_score.get()
    cp.get_default_memory_pool().free_all_blocks()


def gpu_array_counts(model, cp):
    return [sum(isinstance(value, cp.ndarray) for value in vars(tree).values())
            for tree in model.models]


def assert_cpu(model, cp):
    assert not model._on_device
    assert isinstance(model.base_score, np.ndarray)
    assert not any(gpu_array_counts(model, cp))


@pytest.mark.parametrize('stage', ['first-tree', 'second-tree', 'base-score'])
@pytest.mark.parametrize('nondefault_stream', [False, True], ids=['default-stream', 'nondefault-stream'])
def test_upload_oom_rolls_back_and_allows_retry(model_data, cp, monkeypatch, stage, nondefault_stream):
    model, X, expected = model_data
    # Measure real pool allocation sizes instead of assuming allocator rounding.
    probe = cp.cuda.MemoryPool()
    sizes = []
    with cp.cuda.using_allocator(probe.malloc):
        for tree in model.models:
            tree.to_device()
            sizes.append(probe.used_bytes())
    full_counts = gpu_array_counts(model, cp)
    for tree in model.models:
        tree.to_cpu()
    probe.free_all_blocks()

    pool = cp.cuda.MemoryPool()
    # One small allocation fits, then the next attribute fails. For the other
    # stages, allow one whole tree or all trees before the failing allocation.
    limit = {'first-tree': 512, 'second-tree': sizes[0] + 512, 'base-score': sizes[-1]}[stage]
    pool.set_limit(size=limit)
    asarray = cp.asarray
    observed = {}

    def upload(array, *args, **kwargs):
        try:
            return asarray(array, *args, **kwargs)
        except cp.cuda.memory.OutOfMemoryError as exc:
            observed['counts'] = gpu_array_counts(model, cp)
            observed['base_score'] = array is model.base_score
            observed['error'] = exc
            raise

    stream = cp.cuda.Stream(non_blocking=True) if nondefault_stream else cp.cuda.Stream.null
    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', upload)
        with stream, cp.cuda.using_allocator(pool.malloc):
            with pytest.raises(cp.cuda.memory.OutOfMemoryError) as caught:
                model.to_device()
        stream.synchronize()

    assert caught.value is observed['error']
    if stage == 'first-tree':
        assert 0 < observed['counts'][0] < full_counts[0]
        assert observed['counts'][1:] == [0, 0]
    elif stage == 'second-tree':
        assert observed['counts'][0] == full_counts[0]
        assert 0 < observed['counts'][1] < full_counts[1]
        assert observed['counts'][2] == 0
    else:
        assert observed['counts'] == full_counts
        assert observed['base_score']

    # Check automatic rollback before any explicit to_cpu or garbage collection.
    assert_cpu(model, cp)
    assert pool.used_bytes() == 0
    pool.free_all_blocks()
    assert pool.total_bytes() == 0
    model.to_cpu()
    restored = pickle.loads(pickle.dumps(model))
    assert_cpu(restored, cp)
    try:
        np.testing.assert_array_equal(model.predict(X, batch_size=32), expected)
        np.testing.assert_array_equal(restored.predict(X, batch_size=32), expected)
    finally:
        model.to_cpu()
        restored.to_cpu()


@pytest.mark.parametrize('base_on_gpu', [False, True])
@pytest.mark.parametrize('flag', [False, True])
def test_to_cpu_recovers_mixed_state(model_data, cp, base_on_gpu, flag):
    model, X, expected = model_data
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        model.models[0].values = cp.asarray(model.models[0].values)
        if base_on_gpu:
            model.base_score = cp.asarray(model.base_score)
    model._on_device = flag
    assert pool.used_bytes() > 0
    model.to_cpu()
    model.to_cpu()
    assert_cpu(model, cp)
    assert pool.used_bytes() == 0
    pool.free_all_blocks()
    np.testing.assert_array_equal(model.predict(X, batch_size=32), expected)


def test_pickle_recovers_partial_state(model_data, cp):
    model, X, expected = model_data
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        model.models[0].values = cp.asarray(model.models[0].values)
        model.base_score = cp.asarray(model.base_score)
    assert not model._on_device
    restored = pickle.loads(pickle.dumps(model))
    assert_cpu(model, cp)
    assert_cpu(restored, cp)
    assert pool.used_bytes() == 0
    pool.free_all_blocks()
    try:
        np.testing.assert_array_equal(restored.predict(X, batch_size=32), expected)
    finally:
        restored.to_cpu()


def test_upload_preserves_other_exceptions(model_data, cp, monkeypatch):
    model, _, _ = model_data
    failure = RuntimeError('transfer failed')
    asarray = cp.asarray

    def upload(array, *args, **kwargs):
        if array is model.base_score:
            raise failure
        return asarray(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', upload)
        with pytest.raises(RuntimeError) as caught:
            model.to_device()
    assert caught.value is failure
    assert_cpu(model, cp)


def test_cpu_state_remains_serializable_without_cupy():
    # The CPU export/inference workflow can load an ensemble without CuPy.
    # Deny imports in a fresh process so an installed CuPy cannot hide a regression.
    script = r'''
import importlib.abc
import pickle
import sys
class NoCuPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise ModuleNotFoundError('CuPy is unavailable')
sys.meta_path.insert(0, NoCuPy())
import numpy as np
from py_boost.gpu.base import Ensemble
from py_boost.gpu.tree import Tree
for trees in (None, [], [Tree(max_nodes=1, nout=1, ngroups=1)]):
    model = Ensemble()
    model.models = trees
    if trees is not None:
        model.base_score = np.array([.25], dtype=np.float32)
    model.to_cpu()
    restored = pickle.loads(pickle.dumps(model))
    restored.to_cpu()
    assert not restored._on_device
    np.testing.assert_equal(restored.base_score, model.base_score)
assert 'cupy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
