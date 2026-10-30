"""Regression coverage for disabled progress logging (#33)."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
if not cp.is_available():
    pytest.skip('Training tests require a CUDA device', allow_module_level=True)

from py_boost import GradientBoosting
from py_boost.callbacks import callback


@pytest.mark.parametrize('verbose, expected', [
    (0, []), (-1, []), (1, [0, 1, 2, 3, 4, 5]), (2, [0, 2, 4, 5]), (10, [0, 5]),
])
def test_iteration_logging_intervals(monkeypatch, verbose, expected):
    messages = []
    monkeypatch.setattr(callback.logger, 'info', messages.append)
    history = callback.EvalHistory([], verbose=verbose)
    history.metric = SimpleNamespace(alias='RMSE')
    history.ntrees = 6
    valid = dict(target=[], ensemble=[], sample_weight=[])

    for iteration in range(6):
        history.after_iteration(dict(num_iter=iteration, data=dict(valid=valid)))

    assert messages == [f'Iter {iteration}; ' for iteration in expected]


@pytest.mark.parametrize('verbose', [pytest.param(0, id='zero'), pytest.param(-1, id='negative')])
@pytest.mark.parametrize('with_eval', [False, True])
def test_training_without_progress_logging(verbose, with_eval):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 8)).astype(np.float32)
    y = (2 * X[:, 0] - X[:, 1]).astype(np.float32)
    eval_sets = [dict(X=X[:64], y=y[:64]), dict(X=X[64:], y=y[64:])] if with_eval else None
    params = dict(loss='mse', ntrees=4, max_depth=2, es=0)
    model = GradientBoosting(**params, verbose=verbose).fit(X, y, eval_sets=eval_sets)
    reference = GradientBoosting(**params, verbose=2).fit(X, y, eval_sets=eval_sets)

    assert len(model.models) == 4
    actual = model.predict(X)
    assert np.isfinite(actual).all()
    assert np.std(actual) > 0
    np.testing.assert_allclose(actual, reference.predict(X), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.history, reference.history, rtol=1e-5, atol=1e-6)
    assert len(model.history) == (4 if with_eval else 0)
    if with_eval:
        assert all(len(scores) == 2 for scores in model.history)


@pytest.mark.parametrize('verbose', [0, -1])
def test_early_stopping_without_progress_logging(verbose):
    X = np.random.default_rng(42).normal(size=(128, 8)).astype(np.float32)
    y = np.zeros(len(X), dtype=np.float32)
    model = GradientBoosting('mse', ntrees=10, max_depth=2, es=2, verbose=verbose)
    model.fit(X, y, eval_sets=[dict(X=X[:64], y=y[:64])])

    assert len(model.models) == 1
    assert model.best_round == 1
    assert model.history == [[0.0], [0.0], [0.0]]
    np.testing.assert_array_equal(model.predict(X), y[:, None])
