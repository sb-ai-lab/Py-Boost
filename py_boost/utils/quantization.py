"""Functions for features quantization"""

import numba
import numpy as np
from numba import float32, float64, uint8, prange, njit, int64

numba.config.THREADING_LAYER = 'threadsafe'


def quantize_1d(X, X_sampled, max_bins, X_enc, i, borders, neg_inf_clip_value):
    """Quantize and fill encoded features and borders

    Args:
        X: np.ndarray, raw features to transform
        X_sampled: np.ndarray, raw features to split
        max_bins: int, maximum number of bin counts
        X_enc: np.ndarray, placeholder for quantized data
        i: int, index of feature to handle
        borders: np.ndarray, placeholder for borders
        neg_inf_clip_value: float, minimum value to clip

    Returns:

    """
    x_sample = X_sampled[:, i]
    x_sample = x_sample[~np.isnan(x_sample)]
    x_sample[x_sample < neg_inf_clip_value] = neg_inf_clip_value
    x_sample = np.sort(x_sample)
    # get unique values ...
    bins = np.sort(np.unique(x_sample))[:-1]
    if len(bins) > (max_bins - 1):
        # quantile with higher interpolation ...
        # first bins is reserved for NaN values
        grid = (np.linspace(0, 1, max_bins)[1:-1] * x_sample.shape[0]).astype(np.int64)
        bins = x_sample[grid]
        bins = np.sort(np.unique(bins))

    borders[i, 1:len(bins) + 1] = bins
    borders[i, 0] = -np.inf

    # encode raw values
    x_raw = X[:, i]
    x_raw[x_raw < neg_inf_clip_value] = neg_inf_clip_value
    x_raw[np.isnan(x_raw)] = -np.inf
    X_enc[:, i] = np.searchsorted(borders[i, :len(bins) + 1], x_raw)

    return


sign = [(float64[:, :], float64[:, :], int64,
         uint8[:, :], int64, float64[:, :], float64),
        (float32[:, :], float32[:, :], int64,
         uint8[:, :], int64, float32[:, :], float64),
        ]

numba_quantize_1d = njit(sign, parallel=False)(quantize_1d)


def _quantize_features(X, X_enc, max_bins, sample, neg_inf_clip_value):
    """

    Args:
        X:
        X_enc:
        max_bins:
        sample:
        neg_inf_clip_value:

    Returns:

    """
    borders = np.empty((X.shape[1], max_bins), dtype=X.dtype)
    borders[:] = np.nan

    if sample < X.shape[0]:
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        idx = idx[:sample]
        X_sampled = X[idx]
    else:
        X_sampled = X

    for i in prange(X.shape[1]):
        numba_quantize_1d(X, X_sampled, max_bins, X_enc, i, borders, neg_inf_clip_value)

    return borders


sign = [(float64[:, :], uint8[:, :], int64, int64, float64),
        (float32[:, :], uint8[:, :], int64, int64, float64),
        ]

numba_quantize_features = njit(sign, parallel=True)(_quantize_features)


def quantize_features(X, max_bins=255, sample=None, random_state=42):
    """
    Perform feature quantization

    Args:
        X: np.ndarray, raw features
        max_bins: int, maximum number of bins, <= 255
        sample: int, sample size for bins construction
        random_state: int, random state to sample bins construction sample

    Returns:

    """
    assert 0 < max_bins <= 255, 'Max bins should be between 0 and 255'

    if sample is None:
        sample = X.shape[0]

    neg_inf_clip_value = np.finfo(X.dtype).min

    np.random.seed(random_state)
    X_enc = np.empty_like(X, dtype=np.uint8, order='C')
    borders_ = numba_quantize_features(X, X_enc, max_bins, sample, neg_inf_clip_value)

    borders = []

    for i in range(X_enc.shape[1]):
        for j in range(max_bins):
            val = borders_[i, j]
            if np.isnan(val):
                break
        borders_[i, j] = np.inf
        borders.append(borders_[i, :j + 1])

    return X_enc, borders


def apply_borders(X, borders):
    X_enc = np.empty_like(X, dtype=np.uint8, order='C')
    for i in range(X.shape[1]):
        X_enc[:, i] = np.searchsorted(borders[i], X[:, i])

    np.place(X_enc, np.isnan(X), 0)

    return X_enc
