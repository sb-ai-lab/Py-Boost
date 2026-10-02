import cupy as cp
import logging
import numpy as np

from numba import njit

from .callback import Callback
from ..gpu.tree import DepthwiseTreeBuilder
from ..multioutput.sketching import TopOutputsSketch

logger = logging.getLogger(__name__)


def get_grouped_stats(clusters, error, leaves_count, sample_weight=None):
    """Get error statistincs among groups

    Args:
        clusters: cp.ndarray, cluster labels for each split strategy
        error: cp.ndarray, errors
        leaves_count: list of int, number of cluster for each split strategy
        sample_weight: cp.ndarray, sample_weight

    Returns:
        cp.ndarray, error sum among cluster
        cp.ndarray, weights sum among cluster
    """
    ntrees = len(leaves_count)
    nclust = max(leaves_count)

    esum = cp.zeros((ntrees, nclust,), dtype=cp.float32)
    ecount = cp.zeros((ntrees, nclust,), dtype=cp.float32)

    for i in range(ntrees):
        acc_val = error if len(error.shape) == 1 else error[:, i]
        esum[i].scatter_add(clusters[:, i], acc_val)
        acc_val = 1 if sample_weight is None else sample_weight[:, 0]
        ecount[i].scatter_add(clusters[:, i], acc_val)

    ecount[ecount == 0] = 1

    return esum, ecount


def get_lvo_error(clusters, error, esum, ecount):
    """Get Leave-One-Out error amonng cluster for each data point

    Args:
        clusters: cp.ndarray, cluster labels for each split strategy
        error: cp.ndarray, errors
        esum: cp.ndarray, cluster error
        ecount: cp.ndarray, cluster sum of weights

    Returns:
        cp.ndarray, LVO error for each point/split strategy
    """
    nrows, ntrees = clusters.shape

    lvo_error = cp.empty((nrows, ntrees), dtype=cp.float32)
    for i in range(ntrees):
        lvo_error[:, i] = esum[i][clusters[:, i]]
        lvo_error[:, i] -= error

        lvo_error[:, i] /= (ecount[i] - 1)[clusters[:, i]]

    return lvo_error


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


class AdvancedES(Callback):
    """Advanced early stopping. Search for optimal ensemble prunning point separately among cluster
    Cluster means the leaf id of Decision Tree.
    Decision tree of multiple depth are candidate strategies to split points into the clusters. S
    Strategy 0 means the single cluster (corresponds to the common early stopping strategy).

    """

    def __init__(self, num_rounds=100, freq=10, topk=5, max_depths=(1, 2, 3, 4, 5, 6), min_data_in_leaf=100):
        self.num_rounds = num_rounds
        self.freq = freq
        self.topk = topk
        self.max_depths = max_depths
        self.min_data_in_leaf = min_data_in_leaf
        self.verbose = None
        self.model = None
        self.best_cluster_strategy = None

    def _initialize(self, build_info):
        """Train candidate decision trees and calc initial statistics

        Args:
            build_info: dict

        Returns:

        """
        self.model = build_info['model']
        self.verbose = self.model.verbose
        train = build_info['data']['train']
        X, grad, hess, sample_weight = train['features_gpu'], train['grad'], train['hess'], train['sample_weight']
        valid = build_info['data']['valid']
        X_val = valid['features_gpu'][-1]
        self.sample_weight = valid['sample_weight'][-1]

        sketch = None
        if grad.shape[1] > self.topk:
            sketch = TopOutputsSketch(self.topk)

        builder = DepthwiseTreeBuilder(build_info['borders'],
                                       multioutput_sketch=sketch,
                                       min_gain_to_split=self.model.min_gain_to_split,
                                       min_data_in_leaf=self.min_data_in_leaf,
                                       lambda_l2=self.model.lambda_l2,
                                       max_bin=max((len(x) for x in build_info['borders']))
                                       )

        self.trees = []
        self.leaves_count = [1]
        self.clusters = cp.zeros((X_val.shape[0], len(self.max_depths) + 1), dtype=cp.uint32)
        for n, max_depth in enumerate(self.max_depths):
            builder.params['max_depth'] = max_depth
            tree, train_nodes, __, val_nodes, ___ = builder.build_tree(X, grad, hess, sample_weight, X_val)
            val_nodes = val_nodes[0]
            tree.to_device()
            self.trees.append(tree)
            val_leafs = tree.predict_leaf_from_nodes(val_nodes)
            self.clusters[:, n + 1] = val_leafs[:, 0]
            train_leafs = tree.predict_leaf_from_nodes(train_nodes)
            self.leaves_count.append(int(train_leafs.max() + 1))

        ntrees = len(self.leaves_count)
        nclust = max(self.leaves_count)

        self.best_iters = cp.zeros((ntrees, nclust), dtype=np.uint32)
        error = self.get_error(build_info)
        esum, ecount = get_grouped_stats(self.clusters, error, self.leaves_count, sample_weight=self.sample_weight)
        self.ecount = ecount.get()

        self.best_errors = esum / ecount
        # set lvo estimation
        self.best_lvo_error = get_lvo_error(self.clusters, error, esum, ecount)
        self.best_by_lvo_preds = [valid['ensemble'][-1]] * self.clusters.shape[1]
        metric_val = float(self.model.metric(valid['target'][-1], self.model.postprocess_fn(valid['ensemble'][-1]),
                                             valid['sample_weight'][-1]))
        self.best_lvo_metrics = [metric_val] * self.clusters.shape[1]
        self.no_upd_rounds = 0

    def before_iteration(self, build_info):
        """Initialize before first iteration

        Args:
            build_info: dict

        Returns:

        """
        num_iter = build_info['num_iter']
        if num_iter == 0:
            self._initialize(build_info)

    def get_error(self, build_info):
        """Calc error

        Args:
            build_info: dict

        Returns:

        """
        metric = self.model.metric
        valid_data = build_info['data']['valid']
        y = valid_data['target'][-1]
        sample_weight = valid_data['sample_weight'][-1]
        pred = valid_data['ensemble'][-1]

        error = metric.error(y, self.model.postprocess_fn(pred))
        if len(error.shape) > 1:
            error = error.mean(axis=1)

        if sample_weight is not None:
            error = error * sample_weight[:, 0]

        return error

    def after_iteration(self, build_info):
        """Update clusters statistics and check for early stopping condition

        Args:
            build_info: dict

        Returns:
            bool, it training should be terminated
        """
        num_iter = build_info['num_iter']
        # update info only each freq iteration
        if (((num_iter % self.freq) != 0) or (num_iter == 0)) and (num_iter != (self.model.ntrees - 1)):
            return False

        error = self.get_error(build_info)
        esum, ecount = get_grouped_stats(self.clusters, error, self.leaves_count, sample_weight=self.sample_weight)

        grp_error = esum / ecount
        better = self.model.metric.compare(grp_error, self.best_errors)

        # get best iter among all possible splits
        self.best_errors = grp_error * better + self.best_errors * (1 - better)
        self.best_iters = better * num_iter + (1 - better) * self.best_iters

        # estimate lvo to select best possible split after training
        lvo_error = get_lvo_error(self.clusters, error, esum, ecount)
        better = self.model.metric.compare(lvo_error, self.best_lvo_error)
        self.best_lvo_error = lvo_error * better + self.best_lvo_error * (1 - better)

        valid = build_info['data']['valid']

        flg_update = False
        for i in range(ecount.shape[0]):
            bet = better[:, [i]]
            self.best_by_lvo_preds[i] = valid['ensemble'][-1] * bet + \
                                        self.best_by_lvo_preds[i] * (1 - bet)
            metric_val = float(
                self.model.metric(valid['target'][-1], self.model.postprocess_fn(self.best_by_lvo_preds[i]),
                                  valid['sample_weight'][-1]))

            if self.model.metric.compare(metric_val, self.best_lvo_metrics[i]):
                flg_update = True
                self.best_lvo_metrics[i] = metric_val

        curr_best = self.model.metric.argmax(self.best_lvo_metrics)

        msg = 'Advanced ES: Best strategy {0}, best LVO metric {1}'.format(
            curr_best, self.best_lvo_metrics[curr_best])

        if (((num_iter) % self.verbose) == 0) or (num_iter == (self.model.ntrees - 1)):
            logger.info(msg)

        if flg_update:
            self.no_upd_rounds = 0
        else:
            self.no_upd_rounds += self.freq

        return self.no_upd_rounds >= self.num_rounds

    def after_train(self, build_info):
        """Select best strategy as default and clean state

        Args:
            build_info:

        Returns:

        """
        self.best_cluster_strategy = self.model.metric.argmax(self.best_lvo_metrics)

        msg = 'Best prunning stategy ID: {0}. Best LVO estimation: {1}'.format(
            self.best_cluster_strategy, self.best_lvo_metrics[self.best_cluster_strategy])
        logger.info(msg)

        del self.sample_weight, self.clusters, self.best_lvo_error, self.best_by_lvo_preds
        self.best_iters = self.best_iters.get()
        self.best_errors = self.best_errors.get()
        [x.to_cpu() for x in self.trees]

    def __getstate__(self):
        """Move clustering trees to CPU to save

        Returns:

        """
        [x.to_cpu() for x in self.trees]
        return self.__dict__

    def predict(self, X, batch_size=100000, strategy=None):
        """Make prediction

        Args:
            X: np.ndarray, feature matrix
            batch_size: inner batch size to avoid OOM
            strategy: strategy id to prune. 0 stands for the common early sptopping

        Returns:
            np.ndarray, prediction
        """
        n = self.best_cluster_strategy if strategy is None else strategy

        if n == 0:
            iters = [self.best_iters[0, 0]]
            return self.model.predict_staged(X, batch_size=batch_size, iterations=iters)[0]

        iters = self.best_iters[n][:self.leaves_count[n]]
        un_iters = np.sort(np.unique(iters))
        order = np.searchsorted(un_iters, iters)

        preds = self.model.predict_staged(X, batch_size=batch_size, iterations=un_iters)

        tree = self.trees[n - 1]
        tree.to_device()
        cp.cuda.get_current_stream().synchronize()
        cluster = np.empty((X.shape[0],), dtype=np.uint32)

        for i in range(0, X.shape[0], batch_size):
            cluster[i: i + batch_size] = tree.predict_leaf(cp.asarray(X[i: i + batch_size].astype(np.float32))
                                                           ).get()[:, 0]

        return select_preds(preds, cluster, order)
