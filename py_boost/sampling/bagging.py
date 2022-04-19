"""Basic sampling strategy"""

import cupy as cp

from ..callbacks.callback import Callback


class BaseSampler(Callback):
    """Random uniform rows/columns sampler"""

    def __init__(self, sample=1, axis=0):
        """

        Args:
            sample: subsample to select at the each iteration
            axis: int, 0 for rows, 1 for columns
        """
        self.sample = sample
        self.axis = axis
        self.length = None
        self.valid_sl = None
        self.indexer = None

    def before_train(self, build_info):
        """Create indexers

        Args:
            build_info: dict

        Returns:

        """
        self.length = build_info['data']['train']['features_gpu'].shape[self.axis]
        self.indexer = cp.arange(self.length, dtype=cp.uint64)
        if self.sample < 1:
            self.valid_sl = cp.zeros(self.length, dtype=cp.bool_)
            self.valid_sl[:max(1, int(self.length * self.sample))] = True

    def before_iteration(self, build_info):
        """Shuffle indexers

        Args:
            build_info: dict

        Returns:

        """
        if self.sample < 1:
            cp.random.shuffle(self.valid_sl)

    def __call__(self):
        """Get the last actual indexer

        Returns:

        """
        if self.sample == 1:
            return self.indexer

        return self.indexer[self.valid_sl]

    def after_train(self, build_info):
        """Clean the state

        Args:
            build_info:

        Returns:

        """
        self.__init__()
