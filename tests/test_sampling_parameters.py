"""Regression coverage for numeric sampling rates and custom samplers (#27)."""

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
if not cp.is_available():
    pytest.skip('Sampling tests require a CUDA device', allow_module_level=True)

from py_boost import Callback, GradientBoosting, SketchBoost
from py_boost.sampling.bagging import BaseSampler


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 8)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1]).astype(np.float32)
    return X, y


@pytest.mark.parametrize('name', ['colsample', 'subsample'])
@pytest.mark.parametrize('rate', [
    pytest.param(0.5, id='python-float'),
    pytest.param(1, id='python-int'),
    pytest.param(1.0, id='full-sample-float'),
    pytest.param(np.float16(0.5), id='float16'),
    pytest.param(np.float32(0.5), id='float32'),
    pytest.param(np.float64(0.5), id='float64'),
    pytest.param(np.int32(1), id='int32'),
    pytest.param(np.int64(1), id='int64'),
])
def test_numpy_sampling_rates_train_like_python(regression_data, name, rate):
    X, y = regression_data
    params = dict(loss='mse', ntrees=3, max_depth=2, es=0, verbose=0)
    model = GradientBoosting(**params, **{name: rate}).fit(X, y)
    reference = GradientBoosting(**params, **{name: float(rate)}).fit(X, y)

    sampler = getattr(model, name)
    assert isinstance(sampler, BaseSampler)
    assert sampler.axis == (1 if name == 'colsample' else 0)
    assert len(model.models) == 3
    actual = model.predict(X)
    assert np.isfinite(actual).all()
    assert np.std(actual) > 0
    np.testing.assert_allclose(actual, reference.predict(X), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', ['colsample', 'subsample'])
@pytest.mark.parametrize('rate', [
    0, -0.1, 1.1, np.nan, np.inf, -np.inf,
    np.float32(0), np.float64(np.nan), np.int64(2),
])
def test_invalid_sampling_rates_raise_clear_errors(regression_data, name, rate):
    X, y = regression_data
    model = GradientBoosting('mse', **{name: rate})
    with pytest.raises(ValueError, match=rf'{name}.*\(0, 1\]'):
        model.fit(X, y)


@pytest.mark.parametrize('name', ['colsample', 'subsample'])
@pytest.mark.parametrize('value', [
    True, False, np.bool_(True), '0.5', None, object(), 0.5j,
    np.array(0.5), np.array([0.5]), lambda: None, Callback(),
])
def test_invalid_sampler_types_raise_clear_errors(regression_data, name, value):
    X, y = regression_data
    model = GradientBoosting('mse', **{name: value})
    with pytest.raises(TypeError, match=name):
        model.fit(X, y)


class CustomSampler:
    """A sampler implementing the callback protocol without inheriting Callback."""

    def __init__(self, axis):
        self.sampler = BaseSampler(0.5, axis=axis)
        self.calls = []

    def before_train(self, build_info):
        self.calls.append('before_train')
        self.sampler.before_train(build_info)

    def before_iteration(self, build_info):
        self.calls.append('before_iteration')
        self.sampler.before_iteration(build_info)

    def __call__(self):
        self.calls.append('sample')
        return self.sampler()

    def after_iteration(self, build_info):
        self.calls.append('after_iteration')
        return False

    def after_train(self, build_info):
        self.calls.append('after_train')
        self.sampler.after_train(build_info)


class CallbackSampler(CustomSampler, Callback):
    """The same protocol implemented by a Callback subclass."""


@pytest.mark.parametrize('name, axis', [('colsample', 1), ('subsample', 0)])
@pytest.mark.parametrize('sampler_type', [CustomSampler, CallbackSampler])
def test_custom_sampler_lifecycle_is_preserved(regression_data, name, axis, sampler_type):
    X, y = regression_data
    sampler = sampler_type(axis)
    params = dict(loss='mse', ntrees=3, max_depth=2, es=0, verbose=0)
    model = GradientBoosting(**params, **{name: sampler}).fit(X, y)
    reference = GradientBoosting(**params, **{name: 0.5}).fit(X, y)

    assert getattr(model, name) is sampler
    assert sampler.calls[0] == 'before_train'
    assert sampler.calls[-1] == 'after_train'
    assert sampler.calls.count('before_iteration') == 3
    assert sampler.calls.count('after_iteration') == 3
    assert sampler.calls.count('sample') == 3
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('model_type', [GradientBoosting, SketchBoost])
def test_numpy_rates_work_together(regression_data, model_type):
    X, y = regression_data
    y = np.column_stack([y, -y])
    params = dict(loss='mse', ntrees=3, max_depth=2, es=0, verbose=0)
    model = model_type(**params, colsample=np.float32(0.5), subsample=np.float64(0.5)).fit(X, y)
    reference = model_type(**params, colsample=0.5, subsample=0.5).fit(X, y)
    np.testing.assert_allclose(model.predict(X), reference.predict(X), rtol=1e-5, atol=1e-6)
