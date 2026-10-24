"""Numerical and training checks for binary losses with missing labels."""

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score


@pytest.fixture(scope='module')
def losses():
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA device unavailable: {exc}')
    if not count:
        pytest.skip('A CUDA device is required')
    from py_boost.gpu import losses
    return losses


@pytest.fixture
def data():
    y = np.array([
        [0, 1, np.nan], [1, np.nan, 0], [0, 0, 1],
        [1, 1, 0], [np.nan, 0, 1], [1, 0, np.nan],
    ], dtype=np.float64)
    p = np.array([
        [.1, .8, np.nan], [.3, np.nan, .9], [.3, .7, .8],
        [.8, .5, .4], [np.nan, .1, .3], [.7, .5, np.nan],
    ])
    return y, p, np.array([1, 5, 2, 0, 3, 4], dtype=np.float64)


def test_gradients_and_hessians_match_finite_differences(losses, data):
    cp = losses.cp
    y, _, _ = data
    logits = np.linspace(-2, 2, y.size).reshape(y.shape)
    loss = losses.BCEWithNaNLoss()
    grad, hess = loss.get_grad_hess(cp.asarray(y), cp.asarray(logits))

    def objective(raw):
        return np.where(np.isnan(y), 0, np.logaddexp(0, raw) - np.nan_to_num(y) * raw)

    step = 1e-4
    plus, center, minus = objective(logits + step), objective(logits), objective(logits - step)
    np.testing.assert_allclose(grad.get(), (plus - minus) / (2 * step), atol=1e-8)
    np.testing.assert_allclose(hess.get(), (plus - 2 * center + minus) / step ** 2, atol=1e-7)


@pytest.mark.parametrize('constant', [None, 0, 1])
def test_without_missing_labels_matches_bce(losses, constant):
    cp = losses.cp
    y = cp.asarray([[0, 1], [1, 0], [1, 1]], dtype=cp.float32)
    if constant is not None:
        y[:] = constant
    raw = cp.asarray([[-2, 1], [0, 2], [3, -1]], dtype=cp.float32)
    old, new = losses.BCELoss(), losses.BCEWithNaNLoss()
    np.testing.assert_allclose(old.base_score(y).get(), new.base_score(y).get(), rtol=1e-6)
    for left, right in zip(old(y, raw), new(y, raw)):
        np.testing.assert_array_equal(left.get(), right.get())
    prediction = new.postprocess_output(raw)
    np.testing.assert_allclose(
        float(losses.NaNBCEMetric()(y, prediction)), float(losses.BCEMetric()(y, prediction)), rtol=1e-6,
    )


def test_missing_output_has_neutral_prior_and_zero_derivatives(losses):
    cp = losses.cp
    y = cp.asarray([[1, cp.nan, 0], [0, cp.nan, 0], [cp.nan, cp.nan, 1]], dtype=cp.float32)
    loss = losses.BCEWithNaNLoss()
    expected = np.array([0, 0, np.log(0.5)], dtype=np.float32)
    np.testing.assert_allclose(loss.base_score(y).get(), expected, atol=1e-6)
    raw = cp.where(cp.isnan(y), cp.nan, 0).astype(cp.float32)
    grad, hess = loss(y, raw)
    assert bool(cp.isfinite(grad).all()) and bool(cp.isfinite(hess).all())
    assert bool((grad[cp.isnan(y)] == 0).all())
    assert bool((hess[cp.isnan(y)] == 0).all())


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('column_weights', [False, True])
def test_bce_matches_observed_label_reference(losses, data, weighted, column_weights):
    y, p, w = data
    weights = w if weighted else np.ones(len(y))
    passed_weights = None if not weighted else weights[:, None] if column_weights else weights
    observed = ~np.isnan(y)
    rows, cols = np.nonzero(observed)
    expected = log_loss(y[rows, cols], p[rows, cols], sample_weight=weights[rows], labels=[0, 1])
    metric = losses.NaNBCEMetric()
    assert metric(y, p, passed_weights) == pytest.approx(expected)
    per_output = metric(y, p, passed_weights, return_list=True).get()
    expected_outputs = [
        log_loss(y[observed[:, i], i], p[observed[:, i], i],
                 sample_weight=weights[observed[:, i]], labels=[0, 1])
        for i in range(y.shape[1])
    ]
    np.testing.assert_allclose(per_output, expected_outputs)
    output_weights = np.array([1, 3, 2])
    weighted_metric = losses.BCEWithNaNMetricWeighted(output_weights)
    assert weighted_metric(y, p, passed_weights) == pytest.approx(np.average(expected_outputs, weights=output_weights))
    assert weighted_metric.compare(0.1, 0.2)


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('column_weights', [False, True])
def test_auc_and_gini_match_sklearn(losses, data, weighted, column_weights):
    y, p, w = data
    weights = w if weighted else np.ones(len(y))
    passed_weights = None if not weighted else weights[:, None] if column_weights else weights
    expected = []
    for i in range(y.shape[1]):
        observed = ~np.isnan(y[:, i]) & (weights > 0)
        expected.append(roc_auc_score(y[observed, i], p[observed, i], sample_weight=weights[observed]))
    expected = np.array(expected)
    metric = losses.NaNAucMetric()
    assert metric(y, p, passed_weights) == pytest.approx(expected.mean())
    np.testing.assert_allclose(metric(y, p, passed_weights, return_list=True), expected)
    np.testing.assert_allclose(losses.NaNAucMetricByClass()(y, p, passed_weights), expected)
    np.testing.assert_allclose(losses.NaNGiniMetricByClass()(y, p, passed_weights), 2 * expected - 1)
    gini = losses.NaNGiniMetricAtN()
    assert gini(y, p, passed_weights) == pytest.approx(2 * expected.mean() - 1)
    assert gini(y, p, passed_weights, prod_weights=[1, 3, 2], n=2) == pytest.approx(
        np.average(2 * expected[:2] - 1, weights=[1, 3]),
    )
    assert metric.compare(0.8, 0.7) and gini.compare(0.8, 0.7)


def test_undefined_outputs_and_zero_weight_rows(losses):
    y = np.array([[np.nan, 1, 0, 0], [np.nan, 1, 1, 1], [np.nan, 1, 1, 0]])
    p = np.array([[np.nan, .5, .5, .1], [np.inf, .5, .5, .8], [-np.inf, .5, .5, np.nan]])
    weights = np.array([1, 1, 0])
    expected_auc = [np.nan, np.nan, 0.5, 1]
    np.testing.assert_allclose(losses.NaNAucMetricByClass()(y, p, weights), expected_auc, equal_nan=True)
    assert losses.NaNAucMetric()(y, p, weights) == pytest.approx(0.75)
    assert losses.NaNGiniMetricAtN()(y, p, weights, prod_weights=[100, 100, 1, 3]) == pytest.approx(0.75)
    assert np.isnan(losses.NaNGiniMetricAtN()(y, p, weights, n=2))
    errors = losses.NaNBCEMetric()(y, p, weights, return_list=True).get()
    assert np.isnan(errors[0]) and np.isfinite(errors[1:]).all()
    assert losses.BCEWithNaNMetricWeighted([100, 0, 1, 3])(y, p, weights) == pytest.approx(
        np.average(errors[2:], weights=[1, 3]),
    )
    assert np.isnan(losses.NaNBCEMetric()(y, p, np.zeros(3)))
    assert np.isnan(losses.NaNAucMetric()(y, p, np.zeros(3)))
    assert np.isnan(losses.BCEWithNaNMetricWeighted([1, 0, 0, 0])(y, p, weights))
    assert np.isinf(losses.NaNBCEMetric()([1], [0]))


@pytest.mark.parametrize('metric', ['NaNBCEMetric', 'NaNAucMetric', 'NaNGiniMetricAtN'])
def test_empty_and_single_output_inputs(losses, metric):
    metric = getattr(losses, metric)()
    assert np.isnan(metric(np.empty((0, 2)), np.empty((0, 2))))
    assert np.isnan(metric([np.nan], [np.nan]))
    assert np.isfinite(metric([0, 1], [.1, .9]))


@pytest.mark.parametrize('metric', ['NaNBCEMetric', 'NaNAucMetric'])
@pytest.mark.parametrize('prediction', [np.nan, np.inf, -np.inf])
def test_invalid_observed_predictions_are_not_hidden(losses, metric, prediction):
    with pytest.raises(ValueError, match='Predictions must be finite'):
        getattr(losses, metric)()([[0, 1], [1, 0]], [[prediction, .8], [.7, .2]])


@pytest.mark.parametrize('label', [-1, 2, np.inf])
def test_invalid_binary_targets(losses, label):
    with pytest.raises(ValueError, match='binary targets'):
        losses.NaNAucMetric()([0, label], [.1, .8])


@pytest.mark.parametrize('prediction', [-.1, 1.1])
def test_bce_requires_probabilities(losses, prediction):
    with pytest.raises(ValueError, match='probabilities'):
        losses.NaNBCEMetric()([0, 1], [.1, prediction])


@pytest.mark.parametrize('weights', [[1, -1], [0, 0], [1, np.nan], [1, np.inf], [[1, 2]]])
def test_invalid_output_weights(losses, weights):
    with pytest.raises(ValueError, match='weights'):
        losses.BCEWithNaNMetricWeighted(weights)
    with pytest.raises(ValueError, match='weights'):
        losses.NaNGiniMetricAtN()([[0, 1], [1, 0]], [[.1, .9], [.9, .1]], prod_weights=weights)


@pytest.mark.parametrize('metric', ['NaNBCEMetric', 'NaNAucMetric'])
@pytest.mark.parametrize('weights', [[1, -1], [1, np.nan], [1, np.inf], [1], [[1, 1], [1, 1]]])
def test_invalid_sample_weights(losses, metric, weights):
    with pytest.raises(ValueError, match='sample_weight'):
        getattr(losses, metric)()([[0], [1]], [[.1], [.9]], weights)


@pytest.mark.parametrize('n', [0, -1, 3, 1.5, True])
def test_invalid_output_count(losses, n):
    with pytest.raises(ValueError, match='n must'):
        losses.NaNGiniMetricAtN()([[0, 1], [1, 0]], [[.1, .9], [.9, .1]], n=n)


def test_shapes_and_aliases(losses):
    with pytest.raises(ValueError, match='same nonempty'):
        losses.NaNBCEMetric()(np.zeros((2, 2)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match='one output weight'):
        losses.BCEWithNaNMetricWeighted([1, 2])([0, 1], [.1, .9])
    with pytest.raises(ValueError, match='one output weight'):
        losses.NaNGiniMetricAtN()([0, 1], [.1, .9], prod_weights=[1, 2])
    assert losses.BCEwithNaNMetricWeighted is losses.BCEWithNaNMetricWeighted
    assert isinstance(losses.loss_alias['bce_nan'], losses.BCEWithNaNLoss)
    assert isinstance(losses.loss_alias['multilabel_nan'], losses.BCEWithNaNLoss)
    assert type(losses.loss_alias['bce']) is losses.BCELoss
    assert type(losses.metric_alias['bce']) is losses.BCEMetric
    assert isinstance(losses.metric_alias['gini_nan'], losses.NaNGiniMetricAtN)
    for name in [None, 'bce', 'logloss', 'bce_nan']:
        assert isinstance(losses.BCEWithNaNLoss().get_metric_from_string(name), losses.NaNBCEMetric)
    for name in ['auc', 'roc', 'auc_nan']:
        assert isinstance(losses.BCEWithNaNLoss().get_metric_from_string(name), losses.NaNAucMetric)
    for name in ['BCEWithNaNLoss', 'NaNBCEMetric', 'NaNAucMetric', 'NaNAucMetricByClass',
                 'NaNGiniMetricByClass', 'NaNGiniMetricAtN', 'BCEWithNaNMetricWeighted', 'BCEwithNaNMetricWeighted']:
        assert name in losses.__all__


@pytest.mark.parametrize('sketch', [None, 1, 2])
@pytest.mark.parametrize('metric', [None, 'auc', 'gini_nan'])
def test_training_missing_targets(losses, sketch, metric):
    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import RandomProjectionSketch

    rng = np.random.default_rng(12)
    X = rng.normal(size=(128, 6)).astype(np.float32)
    y = (X[:, :3] > 0).astype(np.float32)
    y[rng.random(y.shape) < 0.25] = np.nan
    y[:, 2] = np.nan  # No labels in an entire output must not poison the ensemble.
    model = GradientBoosting(
        'bce_nan', metric=metric, ntrees=5, max_depth=2, min_data_in_leaf=2,
        max_bin=16, es=2, verbose=100,
        multioutput_sketch=None if sketch is None else RandomProjectionSketch(sketch),
    )
    model.fit(X[:96], y[:96], eval_sets=[{'X': X[96:], 'y': y[96:]}])
    prediction = model.predict(X[96:])
    assert prediction.shape == y[96:].shape
    assert np.isfinite(prediction).all() and np.isfinite(model.history).all()
    assert ((prediction > 0) & (prediction < 1)).all()
    np.testing.assert_array_equal(prediction[:, 2], np.full(32, 0.5))
    assert np.std(prediction[:, 0]) > 0


def test_cross_validation_missing_targets(losses):
    from py_boost import GradientBoosting
    from py_boost.cv import CrossValidation

    rng = np.random.default_rng(12)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = (X[:, :2] > 0).astype(np.float32)
    y[rng.random(y.shape) < .25] = np.nan
    model = GradientBoosting('bce_nan', ntrees=3, max_depth=2, max_bin=16, verbose=100)
    cv = CrossValidation(model)
    predictions = cv.fit_predict(X, y, cv=np.arange(len(y)) % 2)
    assert predictions.shape == y.shape and np.isfinite(predictions).all()
    assert np.isfinite(cv.predict(X)).all()
    assert len(cv.models) == 2
