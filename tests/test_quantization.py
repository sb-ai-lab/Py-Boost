"""CPU coverage for the Numba kernels used by all training paths."""

import numpy as np
import pytest

from py_boost.quantization.base import (
    QuantileQuantizer,
    UniformQuantizer,
    UniquantQuantizer,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("quantizer_cls", [
    QuantileQuantizer, UniformQuantizer, UniquantQuantizer,
])
def test_quantization_matches_borders(quantizer_cls, dtype):
    # More distinct values than bins exercises each quantizer's binning kernel.
    X = np.linspace(-10, 10, 128, dtype=dtype).reshape(32, 4)
    X[0] = np.nan
    quantizer = quantizer_cls(max_bin=8)

    encoded = quantizer.fit_transform(X)
    borders = quantizer.get_borders()

    assert encoded.dtype == np.uint8
    assert encoded.shape == X.shape
    assert quantizer.get_max_bin() <= 8
    for column, column_borders in enumerate(borders):
        assert np.isneginf(column_borders[0])
        assert np.isposinf(column_borders[-1])
        assert np.all(np.diff(column_borders) > 0)
        valid = ~np.isnan(X[:, column])
        expected = np.zeros(X.shape[0], dtype=np.uint8)
        expected[valid] = np.searchsorted(column_borders, X[valid, column])
        np.testing.assert_array_equal(encoded[:, column], expected)

    np.testing.assert_array_equal(quantizer.transform(X), encoded)
