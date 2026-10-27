"""GPU regressions for histogram correctness and temporary memory usage."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gpu_utils():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if count == 0:
        pytest.skip("A CUDA device is required")

    from py_boost.gpu import utils

    return utils


@pytest.mark.parametrize("nout", [1, 3])
@pytest.mark.parametrize("subtract", [False, True])
@pytest.mark.parametrize("nbins", [32, 256])
def test_cumulative_histogram(gpu_utils, nout, subtract, nbins):
    """Cover sampled rows/columns, missing bins and parent subtraction."""
    cp = gpu_utils.cp
    rng = np.random.default_rng(17)
    nrows, nfeatures, nnodes = 1025, 17, 8
    X = rng.integers(0, nbins, (nrows, nfeatures), dtype=np.uint8)
    gh = rng.integers(-8, 9, (nrows, nout)).astype(np.float32)
    nodes = rng.integers(0, nnodes, nrows, dtype=np.int32)
    rows = np.arange(0, nrows, 2, dtype=np.uint64)
    cols = np.array([16, 5, 0, 8, 3], dtype=np.uint64)
    outs = np.arange(nout, dtype=np.uint64)[::-1].copy()
    expected = np.zeros((nout, nnodes, len(cols), nbins), dtype=np.float32)
    for i, out in enumerate(outs):
        for j, col in enumerate(cols):
            np.add.at(expected[i, :, j], (nodes[rows], X[rows, col]), gh[rows, out])
    expected = expected.cumsum(axis=-1)

    kwargs = {}
    if subtract:
        # Include both even and odd siblings in the subtraction path.
        big = np.array([0, 3, 4, 7], dtype=np.int64)
        small = big ^ 1
        parent = expected[:, big] + expected[:, small]
        kwargs = dict(prev_hist=cp.asarray(parent), small_index=cp.asarray(small),
                      big_index=cp.asarray(big))

    # This also exercises allocation reuse on a non-default CUDA stream.
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True) as stream:
        actual = gpu_utils.histogram(
            gpu_utils.pad_and_move(X), cp.asarray(gh), cp.asarray(nodes),
            cp.asarray(cols), cp.asarray(rows), cp.asarray(outs), nnodes, nbins,
            **kwargs,
        )
        stream.synchronize()
    np.testing.assert_array_equal(actual.get(), expected)
    if subtract:
        np.testing.assert_array_equal(kwargs["prev_hist"].get(), parent)


@pytest.mark.parametrize("nout", [1, 3])
def test_histogram_fits_without_second_full_buffer(gpu_utils, nout):
    """A 1.5-histogram pool cannot hold a separate cumulative output."""
    cp = gpu_utils.cp
    nnodes, nfeatures, nbins, nrows = 64, 64, 256, 257
    X = np.ones((nrows, nfeatures), dtype=np.uint8)
    args = (
        gpu_utils.pad_and_move(X), cp.ones((nrows, nout), dtype=cp.float32),
        cp.zeros(nrows, dtype=cp.int32), cp.arange(nfeatures, dtype=cp.uint64),
        cp.arange(nrows, dtype=cp.uint64), cp.arange(nout, dtype=cp.uint64),
        nnodes, nbins,
    )
    histogram_bytes = nout * nnodes * nfeatures * nbins * 4
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=histogram_bytes * 3 // 2)
    with cp.cuda.using_allocator(pool.malloc):
        histogram = gpu_utils.histogram(*args)
        actual = histogram.get()

    assert histogram.nbytes == histogram_bytes
    np.testing.assert_array_equal(actual[:, 0, :, 0], 0)
    np.testing.assert_array_equal(actual[:, 0, :, 1:], nrows)
    np.testing.assert_array_equal(actual[:, 1:], 0)


def test_training_with_bounded_histogram_memory(gpu_utils):
    """Keep predictions unchanged when training under a small pool limit."""
    from py_boost import GradientBoosting

    class LimitedBoost(GradientBoosting):
        def _create_build_info(self, mempool, *args, **kwargs):
            mempool.set_limit(size=32 * 2**20)
            return super()._create_build_info(mempool, *args, **kwargs)

    y = np.linspace(-1, 1, 4096, dtype=np.float32)
    X = np.broadcast_to(y[:, None], (len(y), 64)).copy()
    params = dict(loss="mse", ntrees=2, max_depth=8, max_bin=256,
                  lambda_l2=.01, min_data_in_leaf=1, es=0, verbose=1000)
    reference = GradientBoosting(**params).fit(X, y)
    expected = reference.predict(X)
    reference.to_cpu()

    model = LimitedBoost(**params).fit(X, y)
    actual = model.predict(X)
    assert len(model.models) == 2
    assert np.isfinite(actual).all()
    assert np.std(actual) > 0
    # Histogram atomics can differ slightly between repeated fits.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_training_stops_when_no_node_can_split(gpu_utils):
    """The early exit must still produce leaf predictions and stop normally."""
    from py_boost import GradientBoosting

    X = np.random.default_rng(19).normal(size=(1024, 8)).astype(np.float32)
    y = np.zeros((len(X), 1), dtype=np.float32)
    model = GradientBoosting("mse", ntrees=10, max_depth=4, es=2, verbose=1000)
    model.fit(X, y, eval_sets=[dict(X=X[:128], y=y[:128])])
    assert len(model.models) == 1
    assert model.history == [[0.0], [0.0], [0.0]]
    np.testing.assert_array_equal(model.predict(X), y)
