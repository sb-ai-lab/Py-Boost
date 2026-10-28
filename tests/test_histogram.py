"""GPU regression tests for histogram batching."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gpu_utils():
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if device_count == 0:
        pytest.skip("A CUDA device is required")

    from py_boost.gpu import utils

    return utils


@pytest.mark.parametrize("nout", [1, 3])
@pytest.mark.parametrize(
    "nfeatures, nselected, nnodes, nbins",
    [
        pytest.param(8, 1, 2047, 256, id="below-budget"),
        pytest.param(8, 1, 2048, 256, id="at-budget"),
        pytest.param(8, 1, 2049, 256, id="above-budget"),
        pytest.param(64, 8, 512, 256, id="sampled-columns"),
        pytest.param(65, 9, 256, 256, id="padded-columns"),
        pytest.param(4, 1, 4096, 256, id="single-feature-tuple"),
        pytest.param(64, 64, 64, 256, id="all-columns"),
        pytest.param(8, 2, 1, 256, id="batched-outputs"),
        pytest.param(8, 2, 512, 256, id="partial-output-batch"),
    ],
)
def test_histogram_matches_numpy(gpu_utils, nfeatures, nselected, nnodes, nbins, nout):
    """Check every bin, including sampled rows/columns and reordered outputs."""
    cp = gpu_utils.cp
    rng = np.random.default_rng(42)
    nrows = 257
    features = rng.integers(0, nbins, size=(nrows, nfeatures), dtype=np.uint8)
    # Integer-valued floats keep accumulation exact, regardless of atomic order.
    target = rng.integers(-8, 9, size=(nrows, nout)).astype(np.float32)
    nodes = rng.integers(0, nnodes, size=nrows, dtype=np.int32)
    nodes[[0, 2]] = [0, nnodes - 1]
    columns = np.concatenate((
        [nfeatures - 1], rng.choice(nfeatures - 1, size=nselected - 1, replace=False),
    )).astype(np.uint64)
    rows = np.arange(0, nrows, 2, dtype=np.uint64)
    outputs = np.arange(nout, dtype=np.uint64)[::-1].copy()

    expected = np.zeros((nout, nnodes, nselected, nbins), dtype=np.float32)
    for out_pos, output in enumerate(outputs):
        for col_pos, column in enumerate(columns):
            np.add.at(
                expected[out_pos, :, col_pos, :],
                (nodes[rows], features[rows, column]),
                target[rows, output],
            )

    actual = gpu_utils.histogram(
        gpu_utils.pad_and_move(features),
        cp.asarray(target),
        cp.asarray(nodes),
        cp.asarray(columns),
        cp.asarray(rows),
        cp.asarray(outputs),
        nnodes,
        nbins,
    )

    np.testing.assert_array_equal(actual.get(), expected.cumsum(axis=-1))


@pytest.mark.parametrize("cross_validation", [False, True], ids=["fit", "cv"])
def test_training_with_column_sampling(gpu_utils, cross_validation):
    """Exercise the failing batch size through the public training APIs."""
    from py_boost import GradientBoosting
    from py_boost.cv import CrossValidation

    # At depth 3: 8 nodes * 512 sampled columns * 256 bins > 2**19.
    # 4096 source columns give 1024 packed tuples and a batch of 512.
    # Identical informative columns keep the splits balanced for any sample.
    y = np.linspace(-1, 1, 1024, dtype=np.float32)
    X = np.broadcast_to(y[:, None], (1024, 4096)).copy()
    model = GradientBoosting(
        "mse", ntrees=1, max_depth=4, colsample=0.125,
        min_data_in_leaf=1, max_bin=256, es=0,
    )

    if cross_validation:
        cv = CrossValidation(model)
        folds = np.arange(X.shape[0]) % 2
        prediction = cv.fit_predict(X, y, cv=folds)
        assert len(cv.models) == 2
        assert np.isfinite(cv.predict(X[:16])).all()
    else:
        model.fit(X, y)
        prediction = model.predict(X)
        assert len(model.models) == 1

    assert prediction.shape == (X.shape[0], 1)
    assert np.isfinite(prediction).all()
    assert np.std(prediction) > 0
