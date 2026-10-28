"""CUDA regressions for weighted NaN-aware training and shared loss interfaces."""
import numpy as np
import pytest


@pytest.fixture(scope='module')
def gpu():
    cp = pytest.importorskip('cupy')
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f'CUDA device unavailable: {exc}')
    if count == 0:
        pytest.skip('CUDA required')
    return cp


@pytest.fixture
def data():
    X = np.random.default_rng(12).normal(size=(64, 6)).astype(np.float32)
    return X, (X[:, :2] > 0).astype(np.float64)


def make_model(**kwargs):
    from py_boost import GradientBoosting
    params = dict(loss='bce_nan', ntrees=1, max_depth=2, max_bin=16, es=0, verbose=100)
    params.update(kwargs)
    return GradientBoosting(**params)


@pytest.mark.parametrize('sketch', [None, 'filter'])
def test_strict_rejects_zero_effective_labels(gpu, data, sketch):
    from py_boost.multioutput.sketching import FilterSketch
    X, y = data
    y = np.column_stack([y, np.full(len(y), np.nan)])
    y[:2, 2] = [0, 1]
    weights = np.ones(len(y))
    weights[:2] = 0
    model = make_model(multioutput_sketch=FilterSketch(1) if sketch else None, use_hess=True)
    with pytest.raises(ValueError) as exc:
        model.fit(X, y, sample_weight=weights)
    assert '2' in str(exc.value)


@pytest.mark.parametrize('positive_weight, expected', [(9., .9), (0., 1e-7)])
def test_weighted_initial_frequency(gpu, data, positive_weight, expected):
    X, _ = data
    y = np.repeat([0., 1.], len(X) // 2)[:, None]
    weights = np.where(y[:, 0] == 1, positive_weight, 1.)
    model = make_model().fit(X, y, sample_weight=weights)
    initial = model.loss.postprocess_output(model.base_score)
    np.testing.assert_allclose(initial, [expected], rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('where', ['training', 'evaluation'])
def test_validate_labels_before_float32_conversion(gpu, data, where):
    X, y = data
    invalid = y.copy()
    invalid[0, 0] = 1.00000001
    with pytest.raises(ValueError, match='binary targets'):
        if where == 'training':
            make_model().fit(X, invalid)
        else:
            make_model().fit(X, y, eval_sets=[{'X': X, 'y': invalid}])


@pytest.mark.parametrize('weight', [-1., np.nan, np.inf, 1e-308, 1e308])
@pytest.mark.parametrize('evaluate', [False, True])
def test_training_weight_validation_and_representation(gpu, data, weight, evaluate):
    X, y = data
    with pytest.raises(ValueError, match='(?i)(weight|finite|nonnegative|represent|float32)'):
        make_model().fit(X, y, sample_weight=np.full(len(y), weight),
                         eval_sets=[{'X': X, 'y': y}] if evaluate else None)


@pytest.mark.parametrize('weight', [1e-308, 1e308])
@pytest.mark.parametrize('metric', [None, 'auc', 'gini_nan'])
def test_extreme_evaluation_weights_through_fit(gpu, data, weight, metric):
    X, y = data
    model = make_model(es=2, metric=metric)
    model.fit(X, y, eval_sets=[{'X': X, 'y': y, 'sample_weight': np.full(len(y), weight)}])
    assert np.isfinite(model.history).all()


@pytest.mark.parametrize('stage', ['initial', 'queued'])
@pytest.mark.parametrize('factor', [1., 1e-30, 1e20])
def test_filter_importance_precision(gpu, stage, factor):
    from py_boost.multioutput.sketching import FilterSketch
    cp = gpu
    sketch = FilterSketch(1, smooth=0)
    sketch.queue = [] if stage == 'initial' else [cp.zeros((2, 1), dtype=cp.int32)]
    sketch.max_trees = 1
    sketch.max_nodes = 1
    sketch.lambda_l2 = 1
    probabilities = []
    def select(pi):
        probabilities.append(pi.get())
        return cp.asarray([0])
    sketch._select = select
    grad = cp.asarray([[1., 2.], [1., 2.]], dtype=cp.float32) * factor
    hess = cp.ones_like(grad) * factor
    assert bool(cp.isfinite(grad).all() & cp.isfinite(hess).all())
    sketch(grad, hess)
    np.testing.assert_allclose(probabilities[0], [.2, .8], rtol=1e-5)


@pytest.mark.parametrize('output_weights, expected', [
    ([1e-308, 1e308], np.inf), ([0, 1], -np.log(.9)),
])
def test_output_weights_preserve_infinite_errors_and_zero_exclusions(gpu, output_weights, expected):
    from py_boost.gpu.losses import BCEWithNaNMetricWeighted
    with np.errstate(invalid='ignore'):
        score = BCEWithNaNMetricWeighted(output_weights)(
            [[1., 0.], [1., 0.]], [[0., .1], [0., .1]])
    assert score == pytest.approx(expected)


@pytest.mark.parametrize('loss', ['mse', 'msle', 'bce', 'multiclass'])
def test_existing_losses_keep_initialization(gpu, data, loss):
    X, _ = data
    if loss == 'mse':
        y = np.linspace(-1, 1, len(X))[:, None]
    elif loss == 'msle':
        y = np.linspace(0, 2, len(X))[:, None]
    elif loss == 'bce':
        y = np.repeat([.2, .8], len(X) // 2)[:, None]
    else:
        y = (np.arange(len(X)) % 3)[:, None]
    weights = np.repeat([1., 9.], len(X) // 2)
    model = make_model(loss=loss).fit(X, y, sample_weight=weights)
    expected = y.astype(np.float32).mean(axis=0)
    if loss == 'bce':
        expected = np.log(expected / (1 - expected))
    elif loss == 'msle':
        expected = np.log1p(y.astype(np.float32)).mean(axis=0)
    elif loss == 'multiclass':
        expected = np.zeros(3)
    np.testing.assert_allclose(model.base_score, expected, atol=1e-6)
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('smooth', [0, .1, 1])
def test_shared_filter_with_zero_gradient_mse(gpu, data, sample, smooth):
    from py_boost.multioutput.sketching import FilterSketch
    X, _ = data
    y = np.zeros((len(X), 2), dtype=np.float32)
    model = make_model(loss='mse', ntrees=2, multioutput_sketch=FilterSketch(1, sample=sample, smooth=smooth), use_hess=True)
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize('loss', ['mse', 'bce'])
def test_shared_early_stopping_reports_nan(gpu, data, loss):
    class UndefinedMetric:
        alias = 'undefined'
        def __call__(self, *args):
            return float('nan')
        def compare(self, left, right):
            return left < right
    X, y = data
    model = make_model(loss=loss, metric=UndefinedMetric(), es=2)
    with pytest.raises(ValueError, match='Early stopping metric.*undefined'):
        model.fit(X, y, eval_sets=[{'X': X, 'y': y}])


def test_cross_validation_weighted_initialization(gpu):
    from py_boost.cv import CrossValidation

    X = np.random.default_rng(12).normal(size=(96, 6)).astype(np.float32)
    y = (X[:, :3] > 0).astype(np.float64)
    folds = np.arange(len(X)) % 3
    y[folds == 0, 2] = np.nan
    weights = np.linspace(.25, 2, len(X))
    cv = CrossValidation(make_model())
    prediction = cv.fit_predict(X, y, sample_weight=weights, cv=folds)
    assert np.isfinite(prediction).all()
    for fold, model in enumerate(cv.models):
        train = folds != fold
        observed = ~np.isnan(y[train])
        eligible_weights = observed * weights[train, None]
        expected = (np.where(observed, y[train], 0) * eligible_weights).sum(axis=0) / eligible_weights.sum(axis=0)
        np.testing.assert_allclose(model.postprocess_fn(model.base_score), expected, rtol=1e-5)


def test_cross_validation_rejects_zero_effective_training_labels(gpu, data):
    from py_boost.cv import CrossValidation

    X, y = data
    y[32:, 1] = np.nan
    y[32:34, 1] = [0, 1]
    weights = np.ones(len(y))
    weights[32:34] = 0
    cv = CrossValidation(make_model())
    with pytest.raises(ValueError, match=r'no observed labels.*\[1\]'):
        cv.fit_predict(X, y, sample_weight=weights, cv=np.arange(len(y)) // 32)


def test_cross_validation_validates_original_label_dtype(gpu, data):
    from py_boost.cv import CrossValidation

    X, y = data
    y[0, 0] = 1.00000001
    cv = CrossValidation(make_model())
    with pytest.raises(ValueError, match='Observed binary targets must be 0 or 1'):
        cv.fit_predict(X, y, cv=np.arange(len(y)) % 2)


@pytest.mark.parametrize('weight', [-1., np.nan, np.inf])
def test_evaluation_weights_are_validated_before_building_trees(gpu, data, monkeypatch, weight):
    from py_boost.gpu.tree import DepthwiseTreeBuilder

    def unexpected_tree(*args, **kwargs):
        pytest.fail('Invalid evaluation weights must be rejected before building trees')

    monkeypatch.setattr(DepthwiseTreeBuilder, 'build_tree', unexpected_tree)
    X, y = data
    with pytest.raises(ValueError, match='sample_weight must be finite and nonnegative'):
        make_model().fit(X, y, eval_sets=[{'X': X, 'y': y, 'sample_weight': np.full(len(y), weight)}])


@pytest.mark.parametrize('weight', [1e-308, 1e308])
def test_cross_validation_reports_unrepresentable_training_weights(gpu, data, weight):
    from py_boost.cv import CrossValidation

    X, y = data
    cv = CrossValidation(make_model())
    with pytest.raises(ValueError, match='Training sample_weight cannot be represented in float32'):
        cv.fit_predict(X, y, sample_weight=np.full(len(y), weight), cv=np.arange(len(y)) % 2)


@pytest.mark.parametrize('scale', [1e-308, 1e308])
def test_initial_frequency_rescales_large_and_small_weights(gpu, scale):
    from py_boost.gpu.losses import BCEWithNaNLoss

    cp = gpu
    loss = BCEWithNaNLoss()
    score = loss.base_score(cp.asarray([[0.], [1.]]), np.array([.25, .75]) * scale)
    np.testing.assert_allclose(loss.postprocess_output(score).get(), [.75])


def test_initial_frequency_excludes_missing_and_zero_weight_labels(gpu):
    from py_boost.gpu.losses import BCEWithNaNLoss

    cp = gpu
    y = cp.asarray([[cp.nan, 1], [0, cp.nan], [1, cp.nan], [1, 0]], dtype=cp.float64)
    weights = [1e308, 1e-308, 2e-308, 0]
    loss = BCEWithNaNLoss()
    probabilities = loss.postprocess_output(loss.base_score(y, weights)).get()
    np.testing.assert_allclose(probabilities, [2 / 3, 1 - loss.clip_value])


@pytest.mark.parametrize('empty', [False, True])
def test_filter_zero_hessians_on_cuda(gpu, empty):
    from py_boost.multioutput.sketching import FilterSketch

    cp = gpu
    grad = cp.asarray([[0., 1.], [0., 2.]], dtype=cp.float32)
    hess = cp.asarray([[0., 1.], [0., 1.]], dtype=cp.float32)
    if empty:
        grad[:] = 0
        hess[:] = 0
    sketch = FilterSketch(1, smooth=0)
    sketch.queue = []
    probabilities = []
    def select(pi):
        probabilities.append(pi.get())
        return cp.asarray([0])
    sketch._select = select
    if empty:
        with pytest.raises(ValueError, match='all Hessian sums are zero'):
            sketch(grad, hess)
        assert probabilities == []
        return
    result = sketch(grad, hess)
    np.testing.assert_allclose(probabilities[0], [0, 1])
    assert all(bool(cp.isfinite(array).all()) for array in result)


def test_training_weight_scale_retains_regularization_effect(gpu):
    X = np.linspace(-1, 1, 64, dtype=np.float32)[:, None]
    y = (X > 0).astype(np.float32)
    ordinary = make_model(max_depth=1, lambda_l2=1).fit(X, y).predict(X)
    weighted = make_model(max_depth=1, lambda_l2=1).fit(X, y, sample_weight=np.full(len(X), 5.)).predict(X)
    equivalent = make_model(max_depth=1, lambda_l2=.2).fit(X, y).predict(X)
    np.testing.assert_allclose(weighted, equivalent, rtol=1e-6, atol=1e-7)
    assert not np.allclose(weighted, ordinary)


def test_loss_preprocessing_runs_once_per_dataset(gpu, data):
    from py_boost.gpu.losses import BCEWithNaNLoss

    class CountingLoss(BCEWithNaNLoss):
        calls = 0
        def preprocess_input(self, y_true):
            self.calls += 1
            return super().preprocess_input(y_true)

    X, y = data
    loss = CountingLoss()
    make_model(loss=loss).fit(X, y, eval_sets=[{'X': X, 'y': y}])
    assert loss.calls == 2


def test_legacy_custom_loss_interface(gpu, data):
    from py_boost.gpu.losses import BCELoss

    class LegacyLoss:
        def __init__(self):
            self.impl = BCELoss()
        def __getattr__(self, name):
            if name in ('prepare_input', 'initial_score'):
                raise AttributeError(name)
            return getattr(self.impl, name)
        def __call__(self, y_true, y_pred):
            return self.impl(y_true, y_pred)

    X, y = data
    model = make_model(loss=LegacyLoss()).fit(X, y)
    assert np.isfinite(model.predict(X)).all()
