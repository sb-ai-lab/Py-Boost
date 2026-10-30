"""End-to-end parity between trained GPU models, GTIL and compiled CPU inference."""

import shutil
import sys

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
if not cp.is_available():
    pytest.skip('Training parity tests require a CUDA device', allow_module_level=True)

from py_boost import GradientBoosting, TLCompiledPredictor, TLPredictor


@pytest.mark.parametrize('loss,nout', [
    ('mse', 1), ('mse', 3), ('msle', 1), ('msle', 3),
    ('bce', 1), ('multilabel', 3), ('multiclass', 3),
])
@pytest.mark.parametrize('constant_tree', [False, True], ids=['splits', 'constant'])
def test_trained_model_cpu_parity(loss, nout, constant_tree, tmp_path):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(192, 4)).astype(np.float32)
    X[::11, 0] = np.nan
    X[::13, 1] = np.nan
    clean = np.nan_to_num(X)
    scores = np.column_stack([
        clean[:, 0] + clean[:, 1], clean[:, 2] - clean[:, 1], clean[:, 0] * clean[:, 3],
    ])
    if loss == 'multiclass':
        y = scores.argmax(axis=1).astype(np.float32)
    elif loss in ('bce', 'multilabel'):
        y = (scores[:, :nout] > 0).astype(np.float32)
    elif loss == 'msle':
        y = np.exp(scores[:, :nout] / 3)
    else:
        y = scores[:, :nout]

    params = dict(ntrees=4, max_depth=3, min_data_in_leaf=2, max_bin=32,
                  es=0, verbose=1000)
    if constant_tree:
        params['min_gain_to_split'] = 1e10
    model = GradientBoosting(loss, **params).fit(X[:160], y[:160])
    expected = model.predict(X)
    assert expected.shape == (len(X), nout)
    if constant_tree:
        np.testing.assert_allclose(expected, np.broadcast_to(expected[:1], expected.shape))
    else:
        assert np.std(expected, axis=0).max() > .001

    predictor = TLPredictor(model)
    path = tmp_path / 'gtil'
    predictor.dump(path)
    restored = TLPredictor.load(path)
    for instance in (predictor, restored):
        actual = instance.predict(X, nthread=2)
        assert type(actual) is np.ndarray
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    toolchain = 'msvc' if sys.platform == 'win32' else 'gcc'
    executable = 'cl' if sys.platform == 'win32' else toolchain
    if shutil.which(executable) is None:
        pytest.skip(f'Compilation parity requires {executable}')
    suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    compiled = predictor.compile(toolchain, tmp_path / ('model' + suffix),
                                 params={'parallel_comp': 2}, nthread=2)
    compiled.dump(tmp_path / 'compiled.pkl')
    restored = TLCompiledPredictor.load(tmp_path / 'compiled.pkl')
    for instance in (compiled, restored):
        actual = instance.predict(X)
        assert type(actual) is np.ndarray
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
