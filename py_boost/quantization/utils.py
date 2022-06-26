"""Quantization utilities"""

import numba
import numpy as np
from numba import float32, float64, uint8, prange, njit, int64

numba.config.THREADING_LAYER = 'threadsafe'


def _apply_borders_1d(x_raw, x_enc, borders):
    # encode raw values
    sl = ~np.isnan(x_raw)
    x_enc[sl] = np.searchsorted(borders, x_raw[sl])

    return


sign = [(float64[:], uint8[:], float64[:]),
        (float32[:], uint8[:], float32[:]),
        ]

numba_apply_borders_1d = njit(sign, parallel=False)(_apply_borders_1d)


def _apply_borders(X, X_enc, borders):
    for i in prange(X.shape[1]):
        numba_apply_borders_1d(X[:, i], X_enc[:, i], borders[i])

    return


numba_apply_borders = njit(parallel=True)(_apply_borders)


def apply_borders(X, borders):
    X_enc = np.zeros_like(X, dtype=np.uint8, order='C')
    numba_apply_borders(X, X_enc, numba.typed.List(borders))

    return X_enc


def _preprocess_1d(x_sample):
    x_sample = x_sample[~np.isnan(x_sample)].copy()
    neg_inf_clip_value = np.finfo(x_sample).min
    x_sample[x_sample < neg_inf_clip_value] = neg_inf_clip_value
    x_sample = np.sort(x_sample)

    return x_sample


sign = [(float64[:],), (float32[:],), ]
numba_preprocess_1d = njit(sign, parallel=False)(_preprocess_1d)


def _quantile_1d(x_sample, max_bins, min_data_in_bin):
    x_sample = numba_preprocess_1d(x_sample)
    bins = np.unique(x_sample)[:-1]

    if len(bins) > (max_bins - 1):
        # get quantiles
        grid = (np.linspace(0, 1, max_bins + 1) * x_sample.shape[0])[1:-1].astype(np.int64)
        bins = x_sample[grid]
        bins = np.unique(bins)

    return bins


q1d_sign = [(float64[:], int64, int64),
            (float32[:], int64, int64),
            ]

numba_quantile_1d = njit(q1d_sign, parallel=False)(_quantile_1d)


def _quantize_features(fn, X, max_bins, min_data_in_bin, borders):
    """
    Args:
        X:

    Returns:
    """
    for i in prange(X.shape[1]):
        bins = fn(X[:, i], max_bins, min_data_in_bin)
        borders[i, 1: len(bins) + 1] = bins

    return borders


numba_quantize_features = njit(parallel=True)(_quantize_features)


def quantize_features(fn, X, max_bins=255, min_data_in_bin=3):
    """
    Perform feature quantization
    Args:
        fn: JIT compiled function for 1d quantization
        X: np.ndarray, raw features
        max_bins: int, maximum number of bins, <= 255
        min_data_in_bin: int, sample size for bins construction
    Returns:
    """
    assert 0 < max_bins <= 255, 'Max bins should be between 0 and 255'

    borders_ = np.empty((X.shape[1], max_bins + 1), dtype=X.dtype)
    borders_[:] = np.nan
    borders_[:, 0] = -np.inf

    numba_quantize_features(fn, X, max_bins, min_data_in_bin, borders_)
    borders = []

    for i in range(X.shape[1]):
        j = 0
        for j in range(max_bins + 1):
            val = borders_[i, j]
            if np.isnan(val):
                break
        borders_[i, j] = np.inf
        borders.append(borders_[i, :j + 1])

    return borders


def _uniform_1d(x_sample, max_bins, min_data_in_bin):
    x_sample = numba_preprocess_1d(x_sample)
    bins = np.unique(x_sample)[:-1]

    if len(bins) > (max_bins - 1):
        # get uniform
        bins = np.linspace(x_sample[0], x_sample[-1], max_bins + 1)[1:-1].astype(x_sample.dtype)

    return bins


numba_uniform_1d = njit(q1d_sign, parallel=False)(_uniform_1d)


def _uniquant_1d(x_sample, max_bins, min_data_in_bin):
    x_sample = numba_preprocess_1d(x_sample)
    bins = np.unique(x_sample)[:-1]

    if len(bins) > (max_bins - 1):
        # get uniform
        max_bins_u = max_bins // 2
        bins_u = np.linspace(x_sample[0], x_sample[-1], max_bins_u + 1)[1:-1].astype(x_sample.dtype)
        # get quantile
        max_bins_q = max_bins - max_bins_u
        grid = (np.linspace(0, 1, max_bins_q + 1) * x_sample.shape[0])[1:-1].astype(np.int64)
        bins_q = x_sample[grid]
        # merge
        bins = np.unique(np.concatenate((bins_u, bins_q)))

    return bins


numba_uniquant_1d = njit(q1d_sign, parallel=False)(_uniquant_1d)
