"""Basic quantizer implementations"""

import numpy as np
from .utils import apply_borders, quantize_features, numba_quantile_1d, numba_uniform_1d, numba_uniquant_1d


class Quantizer:
    """
    General class for all quantizers
    """

    def __init__(self, sample=None, max_bin=256, min_data_in_bin=3, random_state=42):
        """

        Args:
            sample: None or int, subsample size for quantizers
            max_bin: int, max bins
            min_data_in_bin: int, min bin size
            random_state: int
        """
        self.sample = sample
        # actual nbins eq max_bin - 1, zero bin is always reserved for NaNs
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin
        self.random_state = random_state

        self.borders = None

    def _sample(self, X):
        """Sample train set

        Args:
            X: np.ndarray

        Returns:

        """
        if self.sample is not None and self.sample < X.shape[0]:
            np.random.seed(self.random_state)

            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            idx = idx[:self.sample]
            return X[idx]

        return X

    def transform(self, X):
        """Apply borders is similar for all quantizers

        Args:
            X: np.ndarray

        Returns:

        """
        return apply_borders(X, self.borders)

    def fit(self, X):
        """Fit quantizer

        Args:
            X: np.ndarray

        Returns:

        """
        return self

    def fit_transform(self, X):
        """Fit quantizer and transform

        Args:
            X:

        Returns:

        """
        self.fit(X)

        return self.transform(X)

    def get_borders(self):
        """Get fitted borders

        Returns:

        """
        assert self.borders is not None, 'Should be fitted first'

        return self.borders

    def get_max_bin(self):
        """Get actual max bins

        Returns:

        """
        return max(map(len, self.get_borders()))


class QuantileQuantizer(Quantizer):
    """
    Quantization by quantiles
    """

    def fit(self, X):
        self.borders = quantize_features(

            numba_quantile_1d,
            self._sample(X),
            max_bins=self.max_bin - 1,
            min_data_in_bin=self.min_data_in_bin

        )

        return self


class UniformQuantizer(Quantizer):
    """
    Uniform quantization
    """

    def fit(self, X):
        self.borders = quantize_features(

            numba_uniform_1d,
            self._sample(X),
            max_bins=self.max_bin - 1,
            min_data_in_bin=self.min_data_in_bin

        )

        return self


class UniquantQuantizer(Quantizer):
    """
    Mix of uniform and quantile bins
    """

    def fit(self, X):
        self.borders = quantize_features(

            numba_uniquant_1d,
            self._sample(X),
            max_bins=self.max_bin - 1,
            min_data_in_bin=self.min_data_in_bin

        )

        return self
