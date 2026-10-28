"""GPU regression tests for FilterSketch leaf aggregation (issue #25)."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
if not cp.is_available():
    pytest.skip('FilterSketch tests require a CUDA device', allow_module_level=True)

from py_boost import SketchBoost
from py_boost.multioutput.sketching import FilterSketch


def make_sketch(ntrees=1):
    sketch = FilterSketch(ntrees=ntrees)
    sketch.before_train({
        'data': {'train': {'features_gpu': cp.zeros((4, 1), dtype=cp.float32)}},
        'builder': SimpleNamespace(params={'lambda_l2': 0.5}),
    })
    return sketch


def add_previous_tree(sketch, leaves, iteration=1):
    sketch.before_iteration({
        'num_iter': iteration,
        'data': {'train': {'last_tree': {
            'leaves': cp.asarray(leaves, dtype=cp.int32),
        }}},
    })


def reference_weights(history, grad, hess, lambda_l2):
    weights = np.zeros(grad.shape[1], dtype=np.float32)
    for leaves in history:
        for group in range(leaves.shape[1]):
            for leaf in np.unique(leaves[:, group]):
                rows = leaves[:, group] == leaf
                grad_sum = grad[rows].sum(axis=0)
                hess_sum = hess[rows].sum(axis=0)
                weights += grad_sum ** 2 / (hess_sum + lambda_l2)
    return weights


@pytest.mark.parametrize('nleaves', [1, 2, 3])
@pytest.mark.parametrize('hess_outputs', [1, 2])
def test_leaf_aggregation_matches_reference(nleaves, hess_outputs):
    grad = np.array([[1, 2], [3, -4], [-2, 1], [4, -3]], dtype=np.float32)
    hess = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.float32)[:, :hess_outputs]
    leaves = (np.arange(4, dtype=np.int32) % nleaves)[:, None]
    sketch = make_sketch()
    add_previous_tree(sketch, leaves)

    # Fail before launching scatter_add with an empty buffer on unfixed code.
    assert sketch.max_nodes > 0
    actual = sketch._calc_weights(cp.asarray(grad), cp.asarray(hess))
    expected = reference_weights([leaves], grad, hess, sketch.lambda_l2)
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-6)


def test_leaf_aggregation_with_multiple_groups_and_tree_history():
    grad = np.array([[1, 2], [3, -4], [-2, 1], [4, -3]], dtype=np.float32)
    hess = np.ones((4, 1), dtype=np.float32)
    history = [
        np.array([[0, 0], [1, 0], [2, 1], [0, 2]], dtype=np.int32),
        np.zeros((4, 1), dtype=np.int32),
        np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32),
    ]
    sketch = make_sketch(ntrees=2)
    for iteration, leaves in enumerate(history, start=1):
        add_previous_tree(sketch, leaves, iteration)
        actual = sketch._calc_weights(cp.asarray(grad), cp.asarray(hess))
        expected = reference_weights(history[max(0, iteration - 2):iteration],
                                     grad, hess, sketch.lambda_l2)
        np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-6)


def test_training_after_a_single_leaf_tree():
    rng = np.random.RandomState(42)
    features = rng.normal(size=(32, 4)).astype(np.float32)
    target = rng.randint(0, 2, size=(32, 3)).astype(np.float32)
    target[:, -1] = 0
    model = SketchBoost(loss='multilabel', ntrees=3, max_depth=1,
                        min_gain_to_split=1e6, sketch_method='filter', es=0)
    model.fit(features, target)
    prediction = model.predict(features)

    assert len(model.models) == 3
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()
