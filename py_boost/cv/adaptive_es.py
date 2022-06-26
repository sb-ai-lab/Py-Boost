"""Adaptive early stopping"""

import numpy as np
import cupy as cp
from copy import deepcopy
from numba import njit

from ..gpu.losses import MSELoss, CrossEntropyLoss, BCELoss, loss_alias
from ..gpu.utils import validate_input

from .base import CrossValidation


def check_input(y_true, sample_weight):
    if len(y_true.shape) == 1:
        y_true = y_true[:, np.newaxis]

    y_true = y_true[np.newaxis, :, :]

    if sample_weight is not None and len(sample_weight.shape) == 1:
        sample_weight = sample_weight[:, np.newaxis]

    return y_true, sample_weight


def bce_scorer(y_true, y_pred, sample_weight=None):
    """

    Args:
        y_true: (nobj, nout)
        y_pred: (niter, nobj, nout)

    Returns:

    """
    y_true, sample_weight = check_input(y_true, sample_weight)

    path = -np.log(y_true * y_pred + (1 - y_true) * (1 - y_pred))
    path = path.sum(axis=-1).T

    if sample_weight is not None:
        path *= sample_weight

    return path


def mse_scorer(y_true, y_pred, sample_weight=None):
    """

    Args:
        y_true: (nobj, nout)
        y_pred: (niter, nobj, nout)

    Returns:

    """
    y_true, sample_weight = check_input(y_true, sample_weight)

    path = (y_true - y_pred) ** 2
    path = path.sum(axis=-1).T

    if sample_weight is not None:
        path *= sample_weight

    return path


def cent_scorer(y_true, y_pred, sample_weight=None):
    """

    Args:
        y_true: (nobj, nout)
        y_pred: (niter, nobj, nout)

    Returns:

    """
    y_true, sample_weight = check_input(y_true, sample_weight)

    path = -np.log(np.take_along_axis(y_pred, y_true, axis=2)[..., 0].T)

    if sample_weight is not None:
        path *= sample_weight

    return path


@njit
def select_preds(arr, leaves, order):
    """Select corresponding to cluster prediction

    Args:
        arr: np.ndarray, predictions
        leaves: np.ndarray, clusters
        order: np.ndarray, maps cluster label with position in prediction array

    Returns:
        np.ndarray, pruned prediction
    """
    res = np.empty(arr.shape[1:], dtype=arr.dtype)

    for i in range(leaves.shape[0]):
        res[i] = arr[order[leaves[i]], i, :]

    return res


class AdaptiveESCV(CrossValidation):
    """
    Cross validation wrapper with built in adaptive early stopping
    """

    def __init__(self, base_learner, cluster, iters_to_fit, metric=None, random_state=42, batch_size=10000):
        super().__init__(deepcopy(base_learner), random_state)
        self._base_learner.params['es'] = 0
        self.cluster = cluster
        self.iters_to_fit = iters_to_fit
        self.metric = metric
        self.batch_size = batch_size

        self.best_split = None
        self.best_trees = None
        self.best_oof_trees = None

    def get_es_metric(self):

        if self.metric:
            return self.metric

        loss = self._base_learner.params['loss']
        if type(loss) is str:
            loss = loss_alias[loss]

        if type(loss) is MSELoss:
            return mse_scorer

        if type(loss) is BCELoss:
            return bce_scorer

        if type(loss) is CrossEntropyLoss:
            return cent_scorer

        raise ValueError('Unknown loss func. Please specify metric manually')

    def fit_predict(self, X, y, sample_weight=None, cv=5, stratify=False, random_state=42):
        """

        Args:
            X:
            y:
            sample_weight:
            cv:

        Returns:

        """
        assert self.models is None, 'Is already trained'

        self.models = []

        X, y, sample_weight, eval_sets = validate_input(X, y, sample_weight, {})
        self._base_learner._infer_params()
        X_enc, max_bin, borders, eval_enc = self._base_learner.quantize(X, eval_sets)

        # create validation
        cv_iter = self.get_cv_iter(cv, stratify, random_state)

        # fit and free memory
        mempool = cp.cuda.MemoryPool()

        oof_pred, folds = self._fit_predict(mempool, X, X_enc, y, sample_weight, max_bin, borders, cv_iter)
        self.fit_cluster_tree(X, X_enc, y, sample_weight, max_bin, borders, folds)
        self.search_for_best_cluster(X, y, sample_weight, folds)

        # create out of fold pruned prediction

        for f in range(folds.max() + 1):
            idx = np.arange(X_enc.shape[0])[folds == f]
            X_test = X[idx]
            pred = self._get_stages([self.models[f]], self.best_oof_trees[f], X_test, batch_size=self.batch_size)
            oof_pred[idx] = self._prune_preds(self.best_oof_trees[f], X_test, pred, batch_size=self.batch_size)

        return oof_pred

    def fit_cluster_tree(self, X, X_enc, y, sample_weight, max_bin, borders, folds):
        """Fit cluster tree

        Args:
            X:
            X_enc:
            y:
            sample_weight:
            max_bin:
            borders:
            folds:

        Returns:

        """
        paths = np.zeros((X_enc.shape[0], len(self.iters_to_fit)), dtype=np.float32)
        scorer = self.get_es_metric()

        for f in range(folds.max() + 1):
            idx = np.arange(X_enc.shape[0])[folds == f]
            val_pred = self.models[f].predict_staged(X[idx], iterations=self.iters_to_fit)
            paths[idx] = scorer(y[idx], val_pred, None if sample_weight is None else sample_weight[idx])

        self.cluster.fit_quantized(X_enc, paths, max_bin, borders)
        self.cluster.to_cpu()

    def search_for_best_cluster(self, X, y, sample_weight, folds):
        """Search for the best cluster tree

        Args:
            X:
            y:
            sample_weight:
            folds:

        Returns:

        """
        # predict cluster trees
        cl_ = self.cluster.predict(X)
        # zero clustering is a simple early stopping
        clusters = np.zeros((cl_.shape[0], cl_.shape[1] + 1), dtype=np.uint32)
        clusters[:, 1:] = cl_

        scorer = self.get_es_metric()
        n_cand = clusters.shape[1]
        clust_per_split = clusters.max(axis=0) + 1
        nfolds = folds.max() + 1
        max_clust = clust_per_split.max()
        iter_num = self._base_learner.params['ntrees']
        batch_size = 1000

        folds_stats = np.zeros((nfolds, n_cand, max_clust, iter_num), dtype=np.float32)

        # calculate oof errors
        for f in range(nfolds):
            idx = np.arange(X.shape[0])[folds == f]
            X_test, y_test, cl_test = X[idx], y[idx], clusters[idx]

            for i in range(0, X_test.shape[0], batch_size):

                val_pred = self.models[f].predict_staged(X_test[i:i + batch_size])
                err = scorer(y_test[i:i + batch_size], val_pred,
                             None if sample_weight is None else sample_weight[i:i + batch_size])

                for j in range(n_cand):
                    np.add.at(folds_stats[f, j], (cl_test[i:i + batch_size, j],), err)

        # select best by oof
        stats = folds_stats.sum(axis=0)  # shape (nsplits, max_clust, niters)
        oof_stats = stats[np.newaxis, ...] - folds_stats  # shape (nfolds, nsplits, max_clust, niters)

        best_iters = oof_stats.argmin(axis=-1)  # shape (nfolds, nsplits, max_clust)
        best_errs = np.take_along_axis(folds_stats, best_iters[..., np.newaxis], axis=3)[..., 0].sum(
            axis=0)  # shape  (nsplits, max_clust)
        self.best_split = best_errs.sum(axis=1).argmin()  # scalar
        best_oof_trees = best_iters[:, self.best_split]  # shape (nfolds, max_clust)
        self.best_oof_trees = best_oof_trees[:, :clust_per_split[self.best_split]]

        # select best in total
        best_trees = stats[self.best_split].argmin(axis=-1)  # shape (max_clust, )
        self.best_trees = best_trees[:clust_per_split[self.best_split]]

    def _get_stages(self, models, iters, X, batch_size=100000):
        """

        Args:
            models:
            iters:
            X:
            batch_size:

        Returns:

        """
        sorted_iters = np.sort(np.unique(iters))
        pred = models[0].predict_staged(X, iterations=sorted_iters, batch_size=batch_size)

        for i in range(1, len(models)):
            pred += models[i].predict_staged(X, iterations=sorted_iters, batch_size=batch_size)

        pred /= len(models)

        return pred

    def _prune_preds(self, iters, X, pred, batch_size=100000):
        """

        Args:
            iters:
            X:
            pred:
            batch_size:

        Returns:

        """
        if self.best_split == 0:
            cluster = np.zeros((X.shape[0],), dtype=np.uint32)
        else:
            cluster = self.cluster.predict(X, iterations=[self.best_split - 1], batch_size=batch_size)[:, 0]

        sorted_iters = np.sort(np.unique(iters))
        order = np.searchsorted(sorted_iters, iters)

        return select_preds(pred, cluster, order)

    def predict(self, X, batch_size=100000):
        """

        Args:
            X:
            batch_size:

        Returns:

        """
        pred = self._get_stages(self.models, self.best_trees, X, batch_size=batch_size)
        return self._prune_preds(self.best_trees, X, pred, batch_size=batch_size)
