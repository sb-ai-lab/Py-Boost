"""Default callbacks"""
import logging
from ..utils.logging import verbosity_to_loglevel, set_stdout_level

logger = logging.getLogger(__name__)


class Callback:
    """Abstract class for callback. All Callback methods define the actions, should be done between training stages
    There are 4 methods, that could be redefined:
        - before_train - outputs None
        - before_iteration - outputs None
        - after_train - outputs None
        - after_iteration - outputs bool - if training should be stopped after iteration

    Methods received build_info - the state dict, that could be accessed and modifier

    Basic build info structure:

    build_info = {
            'data': {
                'train': {
                    'features_cpu': np.ndarray - raw feature matrix,
                    'features_gpu': cp.ndarray - uint8 quantized feature matrix on GPU,
                    'target': y - cp.ndarray - processed target variable on GPU,
                    'sample_weight': cp.ndarray - processed sample_weight on GPU or None,
                    'ensemble': cp.ndarray - current model prediction (with no postprocessing,
                        ex. before sigmoid for logloss) on GPU,
                    'grad': cp.ndarray of gradients on GPU, before first iteration - None,
                    'hess': cp.ndarray of hessians on GPU, before first iteration - None,

                    'last_tree': {
                        'nodes': cp.ndarray - nodes indices of the last trained tree,
                        'preds': cp.ndarray - predictions of the last trained tree,
                    }

                },
                'valid': {
                    'features_cpu' the same as train, but list, each element corresponds each validation sample,
                    'features_gpu': ...,
                    'target': ...,
                    'sample_weight': ...,
                    'ensemble': ...,

                    'last_tree': {
                        'nodes': ...,
                        'preds': ...,
                    }

                }
            },
            'borders': list of np.ndarray - list or quantization borders,
            'model': GradientBoosting - model, that is trained,
            'mempool': cp.cuda.MemoryPool - memory pool used for train, could be used to clean memory to prevent OOM,
            'builder': DepthwiseTreeBuilder - the instance of tree builder, contains training params,

            'num_iter': int, current number of iteration,
            'iter_scores': list of float - list of metric values for all validation sets for the last iteration,
        }

    """

    def before_train(self, build_info):
        """Actions to be made before train starts

        Args:
            build_info: dict

        Returns:

        """
        return

    def before_iteration(self, build_info):
        """Actions to be made before each iteration starts

        Args:
            build_info: dict

        Returns:

        """
        return

    def after_iteration(self, build_info):
        """Actions to be made after each iteration finishes

        Args:
            build_info: dict

        Returns:
            bool, if train process should be terminated
        """
        return False

    def after_train(self, build_info):
        """Actions to be made before train finishes

        Args:
            build_info:

        Returns:

        """
        return


class CallbackPipeline:
    """Sequential pipeline of callbacks"""

    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def after_iteration(self, build_info):
        stop = False

        for callback in self.callbacks:
            stop = stop or callback.after_iteration(build_info)

        return stop

    def after_train(self, build_info):

        for callback in self.callbacks:
            callback.after_train(build_info)

    def before_train(self, build_info):

        for callback in self.callbacks:
            callback.before_train(build_info)

    def before_iteration(self, build_info):

        for callback in self.callbacks:
            callback.before_iteration(build_info)


class EvalHistory(Callback):
    """Callback for history evaluation"""

    def __init__(self, history, verbose=0):

        self.history = history
        self.verbose = verbose
        self.metric = None
        self.postprocess_fn = None
        self.ntrees = None

    def before_train(self, build_info):
        """Init params and logger

        Args:
            build_info: dict

        Returns:

        """
        self.metric = build_info['model'].metric
        self.postprocess_fn = build_info['model'].loss.postprocess_output
        self.ntrees = build_info['model'].ntrees

        self.set_verbosity_level(int(self.verbose > 0) * 1)

        msg = 'GDBT train starts. Max iter {0}, early stopping rounds {1}'.format(
            build_info['model'].ntrees, build_info['model'].es)

        logger.info(msg)

    def after_iteration(self, build_info):
        """Save the iteration results and output log

        Args:
            build_info: dict

        Returns:

        """
        valid = build_info['data']['valid']
        y_val, val_ens, w_val = valid['target'], valid['ensemble'], valid['sample_weight']

        num_iter = build_info['num_iter']

        msg = 'Iter {0}; '.format(num_iter)

        if self.metric is None:
            return

        alias = self.metric.alias

        if len(y_val) > 0:
            val_metric = [float(self.metric(y, self.postprocess_fn(x), w)) for (y, x, w) in zip(y_val, val_ens, w_val)]
            self.history.append(val_metric)

            msg += ' '.join(['Sample {0}, {1} = {2}; '.format(n, alias, x) for (n, x) in enumerate(val_metric)])

            build_info['iter_score'] = val_metric

        if (((num_iter) % self.verbose) == 0) or (num_iter == (self.ntrees - 1)):
            logger.info(msg)

    @staticmethod
    def set_verbosity_level(verbose):
        """Verbosity level setter.

        Args:
            verbose: Controls the verbosity: the higher, the more messages.
                <1  : messages are not displayed;
                >=1 : the computation process for layers is displayed;
                >=2 : the information about folds processing is also displayed;
                >=3 : the hyperparameters optimization process is also displayed;
                >=4 : the training process for every algorithm is displayed;
        """
        level = verbosity_to_loglevel(verbose)
        set_stdout_level(level)

        logger.info(f"Stdout logging level is {logging._levelToName[level]}.")


class EarlyStopping(Callback):
    """Callback for early stopping"""

    def __init__(self, num_rounds=100):

        self.num_rounds = num_rounds
        self.best_round = 1
        self.no_upd_rounds = 0
        self.best_score = None
        self.metric = None

    def before_train(self, build_info):
        """Init params

        Args:
            build_info: dict

        Returns:

        """
        self.metric = build_info['model'].metric

    def after_iteration(self, build_info):
        """Check early stopping condition and update the state

        Args:
            build_info: dict

        Returns:
            bool, if early stopping condition was met
        """
        if ('iter_score' not in build_info) or (self.num_rounds == 0):
            return False

        num_iter = build_info['num_iter']
        # if multiple valid sets passed - use the last one
        score = build_info['iter_score'][-1]

        if num_iter == 0:
            self.best_score = score
            return False

        if self.metric.compare(score, self.best_score):
            self.best_score = score
            self.best_round = num_iter + 1
            self.no_upd_rounds = 0
            return False

        self.no_upd_rounds += 1

        stop = self.no_upd_rounds >= self.num_rounds

        if stop:
            msg = 'Early stopping at iter {0}, best iter {1}, best_score {2}'.format(
                num_iter + 1, self.best_round, self.best_score)
            logger.info(msg)

        return stop

    def after_train(self, build_info):
        """Prune the model to the best iteration

        Args:
            build_info: dict

        Returns:

        """
        if self.best_score is not None:
            model = build_info['model']
            model.models = model.models[:self.best_round]
            model.best_round = self.best_round
