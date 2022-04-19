"""Gradient boosting builder"""

import cupy as cp
import numpy as np

from .base import Ensemble
from .losses import loss_alias
from .tree import DepthwiseTreeBuilder
from .utils import pad_and_move
from ..callbacks.callback import EarlyStopping, EvalHistory, CallbackPipeline
from ..multioutput.sketching import GradSketch
from ..sampling.bagging import BaseSampler
from ..multioutput.target_splitter import SingleSplitter, OneVsAllSplitter
from ..utils.quantization import quantize_features, apply_borders


class GradientBoosting(Ensemble):
    """Basic Gradient Boosting on depthwise trees"""

    def __init__(self, loss,
                 metric=None,
                 ntrees=100,
                 lr=0.05,
                 min_gain_to_split=0,
                 lambda_l2=1,
                 max_bin=256,
                 max_depth=6,
                 min_data_in_leaf=10,
                 colsample=1.,
                 subsample=1.,
                 quant_sample=100000,
                 target_splitter='Single',
                 multioutput_sketch=None,
                 use_hess=True,
                 es=100,
                 seed=42,
                 verbose=10,
                 callbacks=None
                 ):
        """

        Args:
            loss: str or Loss, loss function
            metric: None or str or Metric, metric
            ntrees: int, maximum number of trees
            lr: float, learning rate
            min_gain_to_split: float >=0, minimal gain to split
            lambda_l2: float > 0, l2 leaf regularization
            max_bin: int in [2, 256] maximum number of bins to quantize features
            max_depth: int > 0, maximum tree depth. Setting it to large values (>12) may cause OOM for wide datasets
            min_data_in_leaf: int, minimal leaf size. Note - for some loss fn leaf size is approximated
                with hessian values to speed up training
            colsample: float or Callable, sumsample of columns to construct trees or callable - custom sampling
            subsample: float or Callable, sumsample of rows to construct trees or callable - custom sampling
            quant_sample: int, subsample to quantize features
            target_splitter: str or Callable, target splitter, defined multioutput strategy:
                'Single', 'OneVsAll' or custom
            multioutput_sketch: None or Callable. Defines the sketching strategy to simplify scoring function
                in multioutput case. If None full scoring function is used
            use_hess: If True hessians will be used in tree structure search
            es: int, early stopping rounds. If 0, no early stopping
            seed: int, random state
            verbose: int, verbosity freq
            callbacks: list of Callback, callbacks to customize training are passed here
        """
        super().__init__()

        self.ntrees = ntrees
        self.lr = lr
        self.min_gain_to_split = min_gain_to_split
        self.lambda_l2 = lambda_l2
        self.max_bin = max_bin
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.use_hess = use_hess

        self.colsample = colsample
        if type(colsample) in [float, int]:
            self.colsample = BaseSampler(colsample, axis=1)

        self.subsample = subsample
        if type(subsample) in [float, int]:
            self.subsample = BaseSampler(subsample, axis=0)

        if target_splitter == 'Single':
            splitter = SingleSplitter()
        elif target_splitter == 'OneVsAll':
            splitter = OneVsAllSplitter()
        else:
            splitter = target_splitter

        self.target_splitter = splitter

        self.multioutput_sketch = multioutput_sketch
        if multioutput_sketch is None:
            self.multioutput_sketch = GradSketch()

        self.quant_sample = quant_sample
        self.es = es
        self.verbose = verbose

        self.loss = loss
        if type(loss) is str:
            self.loss = loss_alias[loss]

        self.metric = metric
        if metric is None or type(metric) is str:
            self.metric = self.loss.get_metric_from_string(metric)
        self.seed = seed

        self.history = []

        self.callbacks = CallbackPipeline(

            EvalHistory(self.history, verbose=verbose),
            EarlyStopping(es),
            self.subsample,
            self.colsample,
            self.target_splitter,
            self.multioutput_sketch,
            *([] if callbacks is None else callbacks)
        )

    def _fit(self, builder, build_info):
        """Fit with tree builder and build info

        Args:
            builder: DepthwiseTreeBuilder
            build_info: build info state dict

        Returns:

        """
        train, valid = build_info['data']['train'], build_info['data']['valid']
        self.callbacks.before_train(build_info)

        for i in range(self.ntrees):

            build_info['num_iter'] = i
            train['grad'], train['hess'] = self.loss(train['target'], train['ensemble'])

            self.callbacks.before_iteration(build_info)

            tree, nodes, preds, val_nodes, val_preds = \
                builder.build_tree(train['features_gpu'],
                                   train['grad'],
                                   train['hess'],
                                   train['sample_weight'],
                                   *valid['features_gpu'])

            # update ensemble
            train['ensemble'] += preds
            for vp, tp in zip(valid['ensemble'], val_preds):
                vp += tp

            train['last_tree'] = {

                'nodes': nodes,
                'preds': preds

            }
            valid['last_tree'] = {

                'nodes': val_nodes,
                'preds': val_preds

            }

            self.models.append(tree)
            # check exit info
            if self.callbacks.after_iteration(build_info):
                break

        self.callbacks.after_train(build_info)
        self.base_score = self.base_score.get()

    def fit(self, X, y, sample_weight=None, eval_sets=None):
        """Fit model

        Args:
            X: np.ndarray feature matrix
            y: np.ndarray of target
            sample_weight: np.ndarray of sample weights
            eval_sets: list of dict of eval sets. Ex [{'X':X0, 'y': y0, 'sample_weight': w0}, ...}]

        Returns:
            trained instance
        """
        assert self.models is None, 'Is already trained'

        if eval_sets is None:
            eval_sets = []

        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        if (sample_weight is not None) and (len(sample_weight.shape) == 1):
            sample_weight = sample_weight[:, np.newaxis]

        eval_sets = list(eval_sets)
        for val_arr in eval_sets:
            if len(val_arr['y'].shape) == 1:
                val_arr['y'] = val_arr['y'][:, np.newaxis]

            if 'sample_weight' not in val_arr:
                val_arr['sample_weight'] = None

            if (val_arr['sample_weight'] is not None) and (len(val_arr['sample_weight'].shape) == 1):
                val_arr['sample_weight'] = val_arr['sample_weight'][:, np.newaxis]

        cp.random.seed(self.seed)
        # fit and free memory
        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            builder, build_info = self._create_build_info(mempool, X, y, sample_weight, eval_sets)
            self._fit(builder, build_info)
        mempool.free_all_blocks()

        return self

    def _create_build_info(self, mempool, X, y, sample_weight, eval_sets):
        """Quantize data, create tree builder and build_info

        Args:
            mempool: cp.cuda.MemoryPool, memory pool to use
            X: np.ndarray feature matrix
            y: np.ndarray of target
            sample_weight: np.ndarray of sample weights
            eval_sets: list of dict of eval sets. Ex [{'X':X0, 'y': y0, 'sample_weight': w0}, ...}]

        Returns:
            DepthwiseTreeBuilder, build_info
        """
        # quantization
        y = cp.array(y, order='C', dtype=cp.float32)

        if sample_weight is not None:
            sample_weight = cp.array(sample_weight, order='C', dtype=cp.float32)

        max_bin = min(self.max_bin - 1, self.quant_sample, X.shape[0])
        X_enc, borders = quantize_features(X, max_bin - 1, min(self.quant_sample, X.shape[0]))
        max_bin = max((len(x) for x in borders))

        X_cp = pad_and_move(X_enc)
        # save nfeatures for the feature importances
        self.nfeats = X_cp.shape[1]
        self.postprocess_fn = self.loss.postprocess_output

        # apply quantization to valid data
        X_val = [cp.array(apply_borders(x['X'], borders), order='C') for x in eval_sets]
        y_val = [cp.array(x['y'], order='C', dtype=cp.float32) for x in eval_sets]
        w_val = [None if x['sample_weight'] is None else cp.array(x['sample_weight'], order='C', dtype=cp.float32)
                 for x in eval_sets]

        builder = DepthwiseTreeBuilder(borders,
                                       use_hess=self.use_hess,
                                       colsampler=self.colsample,
                                       subsampler=self.subsample,
                                       target_splitter=self.target_splitter,
                                       multioutput_sketch=self.multioutput_sketch,
                                       lr=self.lr,
                                       min_gain_to_split=self.min_gain_to_split,
                                       min_data_in_leaf=self.min_data_in_leaf,
                                       lambda_l2=self.lambda_l2,
                                       max_depth=self.max_depth,
                                       max_bin=max_bin
                                       )

        y = self.loss.preprocess_input(y)
        y_val = [self.loss.preprocess_input(x) for x in y_val]
        self.base_score = self.loss.base_score(y)

        # init ensembles
        ens = cp.empty((y.shape[0], self.base_score.shape[0]), order='C', dtype=cp.float32)
        ens[:] = self.base_score
        # init val ens
        val_ens = [cp.empty((x.shape[0], self.base_score.shape[0]), order='C') for x in y_val]
        for ve in val_ens:
            ve[:] = self.base_score

        self.models = []

        build_info = {
            'data': {
                'train': {
                    'features_cpu': X,
                    'features_gpu': X_cp,
                    'target': y,
                    'sample_weight': sample_weight,
                    'ensemble': ens,
                    'grad': None,
                    'hess': None
                },
                'valid': {
                    'features_cpu': [dat['X'] for dat in eval_sets],
                    'features_gpu': X_val,
                    'target': y_val,
                    'sample_weight': w_val,
                    'ensemble': val_ens,
                }
            },
            'borders': borders,
            'model': self,
            'mempool': mempool,
            'builder': builder
        }

        return builder, build_info
