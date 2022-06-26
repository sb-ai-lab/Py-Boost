"""Gradient boosting builder"""

import cupy as cp

from .base import Ensemble
from .losses import loss_alias
from .tree import DepthwiseTreeBuilder
from .utils import pad_and_move, validate_input
from ..callbacks.callback import EarlyStopping, EvalHistory, CallbackPipeline
from ..multioutput.sketching import GradSketch
from ..sampling.bagging import BaseSampler
from ..multioutput.target_splitter import SingleSplitter, OneVsAllSplitter


class GradientBoosting(Ensemble):
    """Basic Gradient Boosting on depthwise trees"""

    def __init__(self, loss,
                 metric=None,
                 ntrees=100,
                 lr=0.05,
                 min_gain_to_split=0,
                 lambda_l2=1,
                 gd_steps=1,

                 max_depth=6,
                 min_data_in_leaf=10,
                 colsample=1.,
                 subsample=1.,
                 target_splitter='Single',
                 multioutput_sketch=None,
                 use_hess=True,

                 quantization='Quantile',
                 quant_sample=2000000,
                 max_bin=256,
                 min_data_in_bin=3,

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
            gd_steps: int > 0, number of gradient steps while computing leaf values

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

            quantization: str or Quantizer, method for quantizatrion. One of 'Quantile', 'Uniform',
                'Uniquant' or custom implementation
            quant_sample: int, subsample to quantize features
            max_bin: int in [2, 256] maximum number of bins to quantize features
            min_data_in_bin: int in [2, 256] minimal bin size. NOTE: currently ignored

            es: int, early stopping rounds. If 0, no early stopping
            seed: int, random state
            verbose: int, verbosity freq
            callbacks: list of Callback, callbacks to customize training are passed here
        """

        super().__init__()

        self.params = {

            'loss': loss,
            'metric': metric,
            'ntrees': ntrees,
            'lr': lr,
            'min_gain_to_split': min_gain_to_split,
            'lambda_l2': lambda_l2,
            'gd_steps': gd_steps,

            'max_depth': max_depth,
            'min_data_in_leaf': min_data_in_leaf,
            'colsample': colsample,
            'subsample': subsample,

            'target_splitter': target_splitter,
            'multioutput_sketch': multioutput_sketch,
            'use_hess': use_hess,

            'quantization': quantization,
            'quant_sample': quant_sample,
            'max_bin': max_bin,
            'min_data_in_bin': min_data_in_bin,

            'es': es,
            'seed': seed,
            'verbose': verbose,
            'callbacks': callbacks

        }

    def _infer_params(self):

        self.ntrees = self.params['ntrees']
        self.lr = self.params['lr']
        self.min_gain_to_split = self.params['min_gain_to_split']
        self.lambda_l2 = self.params['lambda_l2']
        self.gd_steps = self.params['gd_steps']

        self.max_depth = self.params['max_depth']
        self.min_data_in_leaf = self.params['min_data_in_leaf']
        self.use_hess = self.params['use_hess']

        self.colsample = self.params['colsample']
        if type(self.params['colsample']) in [float, int]:
            self.colsample = BaseSampler(self.params['colsample'], axis=1)

        self.subsample = self.params['subsample']
        if type(self.params['subsample']) in [float, int]:
            self.subsample = BaseSampler(self.params['subsample'], axis=0)

        if self.params['target_splitter'] == 'Single':
            splitter = SingleSplitter()
        elif self.params['target_splitter'] == 'OneVsAll':
            splitter = OneVsAllSplitter()
        else:
            splitter = self.params['target_splitter']

        self.target_splitter = splitter

        self.multioutput_sketch = self.params['multioutput_sketch']
        if self.params['multioutput_sketch'] is None:
            self.multioutput_sketch = GradSketch()

        self.quantization = self.params['quantization']
        self.quant_sample = self.params['quant_sample']
        self.max_bin = self.params['max_bin']
        self.min_data_in_bin = self.params['min_data_in_bin']

        self.es = self.params['es']
        self.verbose = self.params['verbose']

        self.loss = self.params['loss']
        if type(self.params['loss']) is str:
            self.loss = loss_alias[self.params['loss']]

        self.metric = self.params['metric']
        if self.params['metric'] is None or type(self.params['metric']) is str:
            self.metric = self.loss.get_metric_from_string(self.params['metric'])
        self.seed = self.params['seed']

        self.history = []

        self.callbacks = CallbackPipeline(

            self.subsample,
            self.colsample,
            self.target_splitter,
            self.multioutput_sketch,
            *([] if self.params['callbacks'] is None else self.params['callbacks']),
            EvalHistory(self.history, verbose=self.params['verbose']),
            EarlyStopping(self.params['es']),
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

            tree, leaves, preds, val_leaves, val_preds = \
                builder.build_tree(train['features_gpu'],
                                   train['grad'],
                                   train['hess'],
                                   train['sample_weight'],
                                   lambda x: self.loss(train['target'], train['ensemble'] + x),
                                   *valid['features_gpu'])

            # update ensemble
            train['ensemble'] += preds
            for vp, tp in zip(valid['ensemble'], val_preds):
                vp += tp

            train['last_tree'] = {

                'leaves': leaves,
                'preds': preds

            }
            valid['last_tree'] = {

                'leaves': val_leaves,
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
        self._infer_params()

        X, y, sample_weight, eval_sets = validate_input(X, y, sample_weight, eval_sets)
        # fit and free memory
        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            # quantize
            X_enc, max_bin, borders, eval_enc = self.quantize(X, eval_sets)
            # create build info
            builder, build_info = self._create_build_info(mempool, X, X_enc, y, sample_weight,
                                                          max_bin, borders, eval_sets, eval_enc)
            self._fit(builder, build_info)
        mempool.free_all_blocks()

        return self

    def _create_build_info(self, mempool, X, X_enc, y, sample_weight, max_bin, borders, eval_sets, eval_enc):
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

        X_cp = pad_and_move(X_enc)

        X_val = [cp.array(x, order='C') for x in eval_enc]
        y_val = [cp.array(x['y'], order='C', dtype=cp.float32) for x in eval_sets]
        w_val = [None if x['sample_weight'] is None else cp.array(x['sample_weight'], order='C', dtype=cp.float32)
                 for x in eval_sets]

        # save nfeatures for the feature importances
        self.nfeats = X.shape[1]
        self.postprocess_fn = self.loss.postprocess_output

        builder = DepthwiseTreeBuilder(borders,
                                       use_hess=self.use_hess,
                                       colsampler=self.colsample,
                                       subsampler=self.subsample,
                                       target_splitter=self.target_splitter,
                                       multioutput_sketch=self.multioutput_sketch,
                                       gd_steps=self.gd_steps,
                                       lr=self.lr,
                                       min_gain_to_split=self.min_gain_to_split,
                                       min_data_in_leaf=self.min_data_in_leaf,
                                       lambda_l2=self.lambda_l2,
                                       max_depth=self.max_depth,
                                       max_bin=max_bin,
                                       )
        cp.random.seed(self.seed)

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
