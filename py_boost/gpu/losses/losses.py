"""Common losses"""

import numpy as np
try:
    import cupy as cp
    CUDA_FOUND = True
except Exception:
    CUDA_FOUND = False

from .metrics import metric_alias, RMSEMetric, RMSLEMetric, BCEMetric, NaNBCEMetric, NaNAucMetric
from .metrics import _row_weights, _rescale_row_weights
from .multiclass_metrics import multiclass_metric_alias, CrossEntropyMetric


class Loss:
    """Base class to define loss function"""

    def get_grad_hess(self, y_true, y_pred):
        """Method implements how to calculate gradients and hessians.
        Output gradient should have the shape (n_samples, n_outputs)
        Output hessian should have the shape (n_samples, n_outputs) or (n_samples, 1)
            if the same hess used for all outputs (for ex. MSELoss)

        Definition don't use sample_weight, because it is applied later at the tree building stage

        Args:
            y_true: cp.ndarray, target
            y_pred: cp.ndarray, current prediction

        Returns:

        """
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        grad, hess = self.get_grad_hess(y_true, y_pred)
        return grad, hess

    def preprocess_input(self, y_true):
        """Method defines how raw input target variable should be processed before train starts
            (ex. applying log1p for MSLELoss)

        Args:
            y_true: cp.ndarray, raw target

        Returns:

        """
        return y_true

    def prepare_input(self, y_true, sample_weight=None, training=False):
        """Prepare a dataset using the existing float32 preprocessing contract.

        Losses may override this to validate values before conversion or
        retain higher precision for evaluation weights.
        """
        y_true = cp.array(y_true, order='C', dtype=cp.float32)
        if sample_weight is not None:
            sample_weight = cp.array(sample_weight, order='C', dtype=cp.float32)
        return self.preprocess_input(y_true), sample_weight

    def initial_score(self, y_true, sample_weight=None):
        """Initialize while preserving existing base_score(y_true) overrides."""
        return self.base_score(y_true)

    def postprocess_output(self, y_pred):
        """Method defines how to postprocess sum of trees to output prediction (ex. apply sigmoid for BCELoss)

        Args:
            y_pred: cp.ndarray, predictions

        Returns:

        """
        return y_pred

    def base_score(self, y_true):
        """Method defines how to initialize an empty ensemble (ex. initialize with mean values for MSELoss)

        Args:
            y_true: cp.ndarray, processed target (after applying preprocess_input)

        Returns:

        """
        raise NotImplementedError

    def get_metric_from_string(self, name=None):
        """Method defines how to interpret eval metric given in str format or None.
        For ex. you can define the default metric to use here if name is None

        Args:
            name:

        Returns:

        """
        return metric_alias['name']


class MSELoss(Loss):
    """MSE Loss function for regression/multiregression"""

    def get_grad_hess(self, y_true, y_pred):
        return (y_pred - y_true), cp.ones((y_true.shape[0], 1), dtype=cp.float32)

    def base_score(self, y_true):
        return y_true.mean(axis=0)

    def get_metric_from_string(self, name=None):
        if name is None:
            return RMSEMetric()
        return metric_alias[name]


class MSLELoss(Loss):
    """MSLE Loss function for regression/multiregression"""

    def preprocess_input(self, y_true):
        assert (y_true >= 0).all(), 'Inputs for msle should be non negative'

        return y_true

    def get_grad_hess(self, y_true, y_pred):
        y_true = cp.log1p(y_true)

        return (y_pred - y_true), cp.ones((y_true.shape[0], 1), dtype=cp.float32)

    def postprocess_output(self, y_pred):
        return cp.expm1(y_pred)

    def base_score(self, y_true):
        y_true = cp.log1p(y_true)
        return y_true.mean(axis=0)

    def get_metric_from_string(self, name=None):
        if name is None:
            return RMSLEMetric()
        return metric_alias[name]


class BCELoss(Loss):
    """LogLoss for binary/multilabel classification"""

    def __init__(self, clip_value=1e-7):
        self.clip_value = clip_value

    def base_score(self, y_true):
        means = cp.clip(y_true.mean(axis=0), self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        pred = 1 / (1 + cp.exp(-y_pred))
        pred = cp.clip(pred, self.clip_value, 1 - self.clip_value)
        grad = pred - y_true
        hess = pred * (1 - pred)

        return grad, hess

    def postprocess_output(self, y_pred):
        xp = np if type(y_pred) is np.ndarray else cp
        pred = 1 / (1 + xp.exp(-y_pred))
        pred = xp.clip(pred, self.clip_value, 1 - self.clip_value)

        return pred

    def get_metric_from_string(self, name=None):
        if name is None:
            return BCEMetric()
        return metric_alias[name]


class BCEWithNaNLoss(BCELoss):
    """Binary/multilabel log loss with missing (NaN) targets.

    Missing labels contribute zero gradient and Hessian. Initial log odds
    use weighted observed-label frequencies. Every training output must have
    an observed label with positive row weight. Evaluation outputs may be
    entirely missing. Use positive lambda_l2 for leaves with no observed
    labels. The default evaluation metric also ignores missing labels.
    """

    def preprocess_input(self, y_true):
        observed = ~cp.isnan(y_true)
        if bool((observed & (y_true != 0) & (y_true != 1)).any()):
            raise ValueError('Observed binary targets must be 0 or 1; use NaN for missing labels')
        return y_true

    def prepare_input(self, y_true, sample_weight=None, training=False):
        # Validate labels in their original dtype, before rounding to float32.
        y_true = self.preprocess_input(cp.asarray(y_true))
        if sample_weight is not None:
            sample_weight = _row_weights(sample_weight, y_true.shape[0])
            if training:
                converted = sample_weight.astype(cp.float32)
                invalid = ~cp.isfinite(converted) | ((sample_weight > 0) & (converted == 0))
                if bool(invalid.any()):
                    rows = cp.flatnonzero(invalid[:, 0]).get().tolist()
                    raise ValueError(
                        'Training sample_weight cannot be represented in float32 '
                        'without overflow or loss of positive weights; affected rows: '
                        f'{rows}'
                    )
                sample_weight = converted
            # Evaluation weights stay float64 for stable metric aggregation.
            sample_weight = cp.array(sample_weight, order='C')
        return cp.array(y_true, order='C', dtype=cp.float32), sample_weight

    def initial_score(self, y_true, sample_weight=None):
        return self.base_score(y_true, sample_weight)

    def base_score(self, y_true, sample_weight=None):
        y_true = cp.asarray(y_true)
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        observed = ~cp.isnan(y_true)
        if sample_weight is not None:
            weights = _row_weights(sample_weight, y_true.shape[0])
            observed &= weights > 0
        counts = observed.sum(axis=0)
        missing_outputs = cp.flatnonzero(counts == 0)
        if missing_outputs.size:
            raise ValueError(
                'Training target columns have no observed labels '
                f'(zero-based indices): {missing_outputs.get().tolist()} '
                '(positive sample weight required)'
            )
        if sample_weight is None:
            means = cp.where(observed, y_true, 0).sum(axis=0) / counts
        else:
            weights = _rescale_row_weights(cp.where(observed, weights, 0))
            means = (cp.where(observed, y_true, 0) * weights).sum(axis=0) / weights.sum(axis=0)
        means = means.astype(cp.result_type(y_true.dtype, cp.float32))
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        missing = cp.isnan(y_true)
        grad, hess = super().get_grad_hess(cp.where(missing, 0, y_true), y_pred)
        return cp.where(missing, 0, grad), cp.where(missing, 0, hess)

    def get_metric_from_string(self, name=None):
        if name is None or name in ('bce', 'logloss'):
            return NaNBCEMetric()
        if name in ('auc', 'roc'):
            return NaNAucMetric()
        return metric_alias[name]


def softmax(x, clip_val=1e-5):
    
    xp = np if type(x) is np.ndarray else cp
    exp_p = xp.exp(x - x.max(axis=1, keepdims=True))

    return xp.clip(exp_p / exp_p.sum(axis=1, keepdims=True), clip_val, 1 - clip_val)


# multiclass losses

ce_grad_kernel = cp.ElementwiseKernel(
    'T pred, raw S label, raw S nlabels, T factor',
    'T grad, T hess',

    """
    int y_pr = i % nlabels;
    int y_tr = label[i / nlabels];

    grad = pred - (float) (y_pr == y_tr);
    hess = pred * (1. - pred) * factor;

    """,
    "ce_grad_kernel"
) if CUDA_FOUND else None


def ce_grad(y_true, y_pred):
    factor = y_pred.shape[1] / (y_pred.shape[1] - 1)
    grad, hess = ce_grad_kernel(y_pred, y_true, y_pred.shape[1], factor)

    return grad, hess


class CrossEntropyLoss(Loss):
    """CrossEntropy for multiclass classification"""

    def __init__(self, clip_value=1e-6):
        self.clip_value = clip_value

    def base_score(self, y_true):
        num_classes = int(y_true.max() + 1)
        hist = cp.zeros((num_classes,), dtype=cp.float32)

        return hist

    def get_grad_hess(self, y_true, y_pred):
        pred = softmax(y_pred, self.clip_value)
        grad, hess = ce_grad(y_true, pred)
        return grad, hess

    def postprocess_output(self, y_pred):
        
        return softmax(y_pred, self.clip_value)

    def preprocess_input(self, y_true):
        return y_true[:, 0].astype(cp.int32)

    def get_metric_from_string(self, name=None):
        if name is None:
            return CrossEntropyMetric()
        return multiclass_metric_alias[name]


loss_alias = {

    # for bce
    'binary': BCELoss(),
    'bce': BCELoss(),
    'multilabel': BCELoss(),
    'logloss': BCELoss(),

    'bce_nan': BCEWithNaNLoss(),
    'multilabel_nan': BCEWithNaNLoss(),

    # for multiclass
    'multiclass': CrossEntropyLoss(),
    'crossentropy': CrossEntropyLoss(),

    # for regression
    'mse': MSELoss(),
    'regression': MSELoss(),
    'l2': MSELoss(),
    'multitask': MSELoss(),
    'msle': MSLELoss()

}
