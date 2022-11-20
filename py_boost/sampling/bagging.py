"""Basic sampling strategy"""

import cupy as cp
import numpy as np

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
        self.__init__(sample=self.sample, axis=self.axis)


class MVSSampler(Callback):
    """
    MVS rows sampler proposed in
    https://proceedings.neurips.cc/paper/2019/file/5c8cb735a1ce65dac514233cbd5576d6-Paper.pdf
    """

    def __init__(self, sample=0.1, lmbd='auto', grid_search_steps=100, grid_multiplier=100):
        """

        Args:
            sample: float, subsample
            lmbd: float or 'auto', lambda hyperparameter
            grid_search_steps: float, cut off search steps
            grid_multiplier: float, cut off search multiplier
        """
        self.sample = sample
        self.lmbd = lmbd
        self.grid_search_steps = grid_search_steps
        self.grid_multiplier = grid_multiplier
        self.indexer = None

    def get_probs(self, reg_grad):

        min_ = reg_grad.min()

        grid = cp.linspace(min_, min_ * self.grid_multiplier, self.grid_search_steps, dtype=cp.float32)[cp.newaxis, :]

        probs = cp.clip(reg_grad[:, cp.newaxis] / grid, 0, 1)
        sample_rates = probs.mean(axis=0)
        best_idx = cp.abs(sample_rates - self.sample).argmin()

        return probs[:, best_idx]

    def before_train(self, build_info):

        return

    def before_iteration(self, build_info):

        train = build_info['data']['train']
        grad, hess = train['grad'], train['hess']

        if self.lmbd == 'auto':
            lmbd = ((grad.sum() / hess.sum()) ** 2).sum()
        else:
            lmbd = self.lmbd

        mult = grad.shape[1] / hess.shape[1]

        reg_grad = cp.sqrt((grad ** 2).sum(axis=1) + lmbd * (hess ** 2).sum(axis=1) * mult)

        probs = self.get_probs(reg_grad)

        build_info['data']['train']['grad'] = grad / probs[:, cp.newaxis]
        sl = probs >= cp.random.rand(grad.shape[0], dtype=cp.float32)
        self.indexer = cp.arange(grad.shape[0], dtype=cp.uint64)[sl]

    def __call__(self, *args, **kwargs):

        return self.indexer

    def after_train(self, build_info):

        self.indexer = None


class ColumnImportanceSampler(Callback):
    """
    This class implements a sampling strategy,
    that sample columns in proportion to thier importance at each step
    """

    def __init__(self, rate=0.5, smooth=0.1,
                 update_freq=10, inverse=False, n_force=None, imp_type='split'):
        """

        Args:
            rate: float, sampling rate
            smooth: float, smoothing parameter
            update_freq: int importance update frequency
            inverse: inverse the probability of sampling
            n_force: int or None, number of feats to ignore by sample (always select), counts from the end of data
            imp_type: str, importance type

        Returns:

        """
        self.rate = rate
        self.smooth = smooth
        self.update_freq = update_freq
        self.inverse = inverse
        self.n_force = n_force
        self.imp_type = imp_type
        self.p = None
        self.imp = None

    def update_importance(self, model):

        if self.imp is None:
            self.imp = model.get_feature_importance(self.imp_type)
            return self.imp

        for tree in model.models[-self.update_freq:]:
            sl = tree.feats >= 0
            acc_val = 1 if self.imp_type == 'split' else tree.gains[sl]
            np.add.at(self.imp, tree.feats[sl], acc_val)

        return self.imp

    def before_iteration(self, build_info):
        """
        Define what should be doe before each iteration
        """
        # Update feature importance
        num_iter = build_info['num_iter']

        if (num_iter % self.update_freq) == 0:
            # update probabilities with actual importance
            p = self.update_importance(build_info['model']) + 1e-3

            if self.n_force is not None:
                p = p[:-self.n_force]

            p = cp.asarray(p) / (p.sum())
            # inverse if needed
            if self.inverse:
                p = 1 - p
                p = p / p.sum()
            # apply smoothing
            self.p = p * (1 - self.smooth) + cp.ones_like(p) * self.smooth / p.shape[0]

    def __call__(self):
        """
        Method should return the array of indices, that will be used
        to grow the tree at the current step
        """
        # Sample rows
        n = self.p.shape[0]
        index = cp.random.choice(cp.arange(n, dtype=cp.uint64),
                                 size=int(self.rate * n), p=self.p)

        if self.n_force is not None:
            index = cp.concatenate([index, cp.arange(n, n + self.n_force, dtype=cp.uint64)])

        return index

    def after_train(self, build_info):

        self.p = None
        self.imp = None
