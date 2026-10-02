"""Strategies to splitting multiple outputs by different trees"""

import cupy as cp

from ..callbacks.callback import Callback


class SingleSplitter(Callback):
    """Basic splitter, means no split. Single tree will be built at each boosting step"""

    def __init__(self):
        self.ensemble_indexer = None
        self.indexer = None

    def before_iteration(self, build_info):
        """Initialize indexers

        Args:
            build_info: dict

        Returns:

        """
        if build_info['num_iter'] == 0:
            nout = build_info['data']['train']['grad'].shape[1]
            self.indexer = cp.arange(nout, dtype=cp.uint64)

    def __call__(self):
        """Get list of indexers for each group

        Returns:
            list of cp.ndarrays of indexers
        """
        return [self.indexer]

    def after_train(self, build_info):
        """Clean state not to keep the indexer in trained model

        Args:
            build_info:

        Returns:

        """
        self.__init__()


class RandomGroupsSplitter(SingleSplitter):
    """Random Groups Splitter, means all outputs will be randomly grouped at the each iteration.
    Single tree will be created for the each group.
    """

    def __init__(self, ngroups=2):
        """

        Args:
            ngroups: int, maximum number of groups to split outputs
        """
        super().__init__()
        self.ngroups = ngroups
        self._ngroups = None

    def before_iteration(self, build_info):
        """Update groups count with the actual target shape if needed

        Args:
            build_info: dict

        Returns:

        """
        super().before_iteration(build_info)
        if build_info['num_iter'] == 0:
            self._ngroups = min(self.ngroups, build_info['data']['train']['grad'].shape[1])

    def __call__(self):
        """

        Returns:
            list of cp.ndarrays of indexers
        """
        cp.random.shuffle(self.indexer)
        return cp.array_split(self.indexer, self._ngroups)


class OneVsAllSplitter(SingleSplitter):
    """One-Vs-All splitter, means build separate tree for each output"""

    def __call__(self):
        """

        Returns:
            list of cp.ndarrays of indexers
        """
        return cp.array_split(self.indexer, self.indexer.shape[0])
