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


def test_missing_labels_have_zero_derivatives(losses):
    cp = losses.cp
    y = cp.asarray([[1, 1, 0], [0, 0, 0], [cp.nan, cp.nan, 1]], dtype=cp.float32)
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
                 'NaNGiniMetricByClass', 'NaNGiniMetricAtN', 'BCEWithNaNMetricWeighted']:
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
    y[96:, 2] = np.nan  # Entirely missing evaluation outputs remain valid.
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


@pytest.fixture
def training_data():
    rng = np.random.default_rng(67)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    y = (X[:, :2] > 0).astype(np.float32)
    return X, y


@pytest.mark.parametrize('with_eval', [False, True])
def test_reject_unobserved_training_columns(losses, training_data, with_eval):
    from py_boost import GradientBoosting

    X, y = training_data
    y = np.tile(y, (1, 2))
    valid = [dict(X=X, y=y.copy())] if with_eval else None
    y[:, [1, 3]] = np.nan
    model = GradientBoosting('bce_nan', ntrees=2, max_bin=16, verbose=100)
    with pytest.raises(ValueError, match=r'no observed labels.*\[1, 3\]'):
        model.fit(X, y, eval_sets=valid)


@pytest.mark.parametrize('bad_label', [-1, 2, .5, np.inf, -np.inf])
@pytest.mark.parametrize('with_eval', [False, True])
def test_reject_invalid_training_labels(losses, training_data, bad_label, with_eval):
    from py_boost import GradientBoosting

    X, y = training_data
    valid = [dict(X=X, y=y.copy())] if with_eval else None
    y[0, 0] = bad_label
    weights = np.ones(len(y), dtype=np.float32)
    weights[0] = 0  # Label validation applies even to zero-weight rows.
    with pytest.raises(ValueError, match='Observed binary targets must be 0 or 1'):
        GradientBoosting('bce_nan', ntrees=2, max_bin=16, verbose=100).fit(
            X, y, sample_weight=weights, eval_sets=valid)


@pytest.mark.parametrize('bad_label', [-1, 2, .5, np.inf, -np.inf])
def test_reject_invalid_evaluation_labels(losses, training_data, bad_label, monkeypatch):
    from py_boost import GradientBoosting

    X, y = training_data
    y_valid = y.copy()
    y_valid[0, 1] = bad_label
    loss = losses.BCEWithNaNLoss()

    def no_training(*args):
        pytest.fail('Invalid evaluation labels must be rejected before computing gradients')

    monkeypatch.setattr(loss, 'get_grad_hess', no_training)
    with pytest.raises(ValueError, match='Observed binary targets must be 0 or 1'):
        GradientBoosting(loss, ntrees=2, max_bin=16, es=0, verbose=100).fit(
            X, y, eval_sets=[dict(X=X, y=y_valid)])


def test_targets_validated_once_before_training(losses, training_data):
    from py_boost import GradientBoosting

    class CountingLoss(losses.BCEWithNaNLoss):
        calls = 0

        def preprocess_input(self, y_true):
            self.calls += 1
            return super().preprocess_input(y_true)

    X, y = training_data
    y[::4, 0] = np.nan
    loss = CountingLoss()
    model = GradientBoosting(loss, ntrees=3, max_depth=2, max_bin=16, es=0, verbose=100)
    model.fit(X, y, eval_sets=[dict(X=X, y=y.copy())])
    assert loss.calls == 2
    assert len(model.models) == 3
    assert np.isnan(y[::4, 0]).all()


@pytest.mark.parametrize('es', [0, 2])
@pytest.mark.parametrize('case', ['missing_labels', 'single_class_auc', 'zero_weights'])
def test_undefined_early_stopping_metric(losses, training_data, es, case):
    from py_boost import GradientBoosting

    X, y = training_data
    y_valid = y.copy()
    metric = 'auc' if case == 'single_class_auc' else 'bce'
    valid = dict(X=X, y=y_valid)
    if case == 'missing_labels':
        y_valid[:] = np.nan
    elif case == 'single_class_auc':
        y_valid[:] = 1
    else:
        valid['sample_weight'] = np.zeros(len(y), dtype=np.float32)
    model = GradientBoosting('bce_nan', metric=metric, ntrees=4, max_depth=2,
                             max_bin=16, es=es, verbose=100)
    if es:
        with pytest.raises(ValueError, match=r'Early stopping metric.*undefined.*evaluation set 0'):
            model.fit(X, y, eval_sets=[valid])
    else:
        model.fit(X, y, eval_sets=[valid])
        assert len(model.models) == 4
        assert np.isnan(model.history).all()
        assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize('undefined_last', [False, True])
def test_only_selected_early_stopping_set_must_be_defined(losses, training_data, undefined_last):
    from py_boost import GradientBoosting

    X, y = training_data
    valid = [dict(X=X, y=np.full_like(y, np.nan)), dict(X=X, y=y)]
    if undefined_last:
        valid.reverse()
    model = GradientBoosting('bce_nan', ntrees=3, max_depth=2, max_bin=16, es=2, verbose=100)
    if undefined_last:
        with pytest.raises(ValueError, match='undefined.*evaluation set 1'):
            model.fit(X, y, eval_sets=valid)
    else:
        model.fit(X, y, eval_sets=valid)
        assert np.isnan(np.array(model.history)[:, 0]).all()
        assert np.isfinite(np.array(model.history)[:, 1]).all()


@pytest.mark.parametrize('scale', [1e-308, 1e-200, 1., 1e200, 1e307])
def test_row_weight_scale_invariance(losses, data, scale):
    y, p, w = data
    for metric in [losses.NaNBCEMetric(), losses.BCEWithNaNMetricWeighted([1, 3, 2]),
                   losses.NaNAucMetric(), losses.NaNAucMetricByClass(),
                   losses.NaNGiniMetricByClass(), losses.NaNGiniMetricAtN()]:
        np.testing.assert_allclose(metric(y, p, w * scale), metric(y, p, w), rtol=1e-12)
    np.testing.assert_allclose(
        losses.NaNBCEMetric()(y, p, w * scale, return_list=True).get(),
        losses.NaNBCEMetric()(y, p, w, return_list=True).get(), rtol=1e-12,
    )


@pytest.mark.parametrize('weight', [1e308, 1e-308, 1e-320])
def test_extreme_equal_row_weights(losses, weight):
    y, p, w = [0, 1], [.1, .9], [weight, weight]
    assert losses.NaNBCEMetric()(y, p, w) == pytest.approx(-np.log(.9))
    assert losses.NaNAucMetric()(y, p, w) == pytest.approx(1.)
    cp = losses.cp
    # The shared ROC AUC helper remains usable by the original binary metric.
    assert losses.RocAucMetric()(cp.asarray(y)[:, None], cp.asarray(p)[:, None], cp.asarray(w)[:, None]) == 1.


def test_extreme_weights_with_missing_outputs_and_classes(losses):
    y = [[0, np.nan], [1, np.nan], [np.nan, 0], [np.nan, 1]]
    p = [[.1, np.nan], [.9, np.nan], [np.nan, .2], [np.nan, .8]]
    w = [1e-300, 1e-300, 1e300, 1e300]
    np.testing.assert_allclose(losses.NaNBCEMetric()(y, p, w, return_list=True).get(),
                               [-np.log(.9), -np.log(.8)])
    np.testing.assert_allclose(losses.NaNAucMetricByClass()(y, p, w), [1., 1.])
    assert losses.NaNAucMetric()([0, 1], [.1, .9], [1e-300, 1e300]) == 1.
    assert losses.NaNAucMetric()([0, 1], [.9, .1], [1e-300, 1e300]) == 0.
    assert np.isinf(losses.NaNBCEMetric()([1, 0], [0, .1], [1e-300, 1e300]))


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('smooth', [0., .1, 1.])
def test_filter_sketch_ignores_zero_weight_training_outputs(losses, training_data, sample, smooth):
    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import FilterSketch

    class RecordingSketch(FilterSketch):
        def _select(self, pi):
            probabilities.append(pi.get())
            return super()._select(pi)

    X, y = training_data
    weights = (np.arange(len(y)) % 2).astype(np.float32)
    y[weights > 0, 1] = np.nan
    probabilities = []
    model = GradientBoosting('bce_nan', ntrees=3, max_depth=2, max_bin=16, es=0, verbose=100,
                             use_hess=True, multioutput_sketch=RecordingSketch(k=2, sample=sample, smooth=smooth))
    model.fit(X, y, sample_weight=weights, eval_sets=[dict(X=X, y=y)])
    np.testing.assert_allclose(probabilities, [[1., 0.]] * 3)
    assert np.isfinite(model.predict(X)).all()
    assert np.isfinite(model.history).all()


def test_filter_sketch_rejects_all_zero_training_weights(losses, training_data):
    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import FilterSketch

    X, y = training_data
    model = GradientBoosting('bce_nan', ntrees=2, max_bin=16, verbose=100,
                             use_hess=True, multioutput_sketch=FilterSketch())
    with pytest.raises(ValueError, match='FilterSketch cannot select.*all Hessian sums are zero'):
        model.fit(X, y, sample_weight=np.zeros(len(y), dtype=np.float32))


@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('shared_hessian', [False, True])
@pytest.mark.parametrize('sample', [False, True])
def test_filter_sketch_zero_importance_fallback(losses, queued, shared_hessian, sample):
    from types import SimpleNamespace
    from py_boost.multioutput.sketching import FilterSketch

    cp = losses.cp
    grad = cp.zeros((4, 3), dtype=cp.float32)
    hess = cp.ones((4, 1), dtype=cp.float32) if shared_hessian else cp.asarray([[1., 0., 1.]] * 4)
    sketch = FilterSketch(k=2, sample=sample, smooth=0.)
    info = dict(data=dict(train=dict(features_gpu=grad, last_tree=dict(leaves=cp.zeros((4, 1), dtype=cp.int32)))),
                builder=SimpleNamespace(params=dict(lambda_l2=0.)), num_iter=1)
    sketch.before_train(info)
    if queued:
        sketch.before_iteration(info)
        assert sketch.max_nodes == 1  # A tree with one leaf still needs one accumulator.
    probabilities = []
    select = sketch._select

    def record(pi):
        probabilities.append(pi.get())
        return select(pi)

    sketch._select = record
    g, h = sketch(grad, hess)
    expected = [1 / 3] * 3 if shared_hessian else [.5, 0., .5]
    np.testing.assert_allclose(probabilities, [expected])
    assert bool(cp.isfinite(g).all()) and bool((h > 0).all())


def test_filter_sketch_keeps_distinct_leaf_accumulators(losses):
    from types import SimpleNamespace
    from py_boost.multioutput.sketching import FilterSketch

    cp = losses.cp
    grad = cp.asarray([[1., 2.], [3., 4.]], dtype=cp.float32)
    hess = cp.ones_like(grad)
    sketch = FilterSketch(smooth=0.)
    info = dict(data=dict(train=dict(features_gpu=grad, last_tree=dict(leaves=cp.asarray([[0], [1]])))),
                builder=SimpleNamespace(params=dict(lambda_l2=1.)), num_iter=1)
    sketch.before_train(info)
    sketch.before_iteration(info)
    np.testing.assert_allclose(sketch._calc_weights(grad, hess).get(), [5., 10.])


def test_cross_validation_rejects_unobserved_training_fold(losses, training_data):
    from py_boost import GradientBoosting
    from py_boost.cv import CrossValidation

    X, y = training_data
    folds = np.arange(len(y)) % 2
    y[folds == 1, 1] = np.nan
    model = GradientBoosting('bce_nan', ntrees=2, max_depth=2, max_bin=16, verbose=100)
    with pytest.raises(ValueError, match=r'no observed labels.*\[1\]'):
        CrossValidation(model).fit_predict(X, y, cv=folds)
