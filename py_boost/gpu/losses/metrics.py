"""Common metrics"""

import numpy as np

try:
    import cupy as cp
except Exception:
    pass


class Metric:
    """Base class to define eval metric function.
    Metric could be defined in 2 ways:

        - redefine .error method. Preferred if possible. Simplified metric definition by calculating error function
            for each point (ex. see RMSEMetric). If metric is defined via .error it also could be used with AdvancedES

        - redefine __call__ method. Used for more complex functions, like ROC-AUC. Handling sample_weight
            should be done manually here if needed


    """
    alias = 'score'  # defines how metric will be named in the output log

    def error(self, y_true, y_pred):
        """Simplified metric definition via individual objects error

        Args:
            y_true: cp.array, targets
            y_pred: cp.array, predictions

        Returns:
            float, metric value
        """
        raise ValueError('Pointwise error is not implemented for this metric')

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Full metric definition. Default is just weighted aggregation of pointwise errors

        Args:
            y_true: cp.array, targets
            y_pred: cp.array, predictions
            sample_weight: None or cp.ndarray, weights

        Returns:
            float, metric value
        """
        err = self.error(y_true, y_pred)
        shape = err.shape
        assert shape[0] == y_true.shape[0], 'Error shape should match target shape at first dim'

        if len(shape) == 1:
            err = err[:, cp.newaxis]

        if sample_weight is None:
            return err.mean()

        err = (err.mean(axis=1, keepdims=True) * sample_weight).sum() / sample_weight.sum()
        return err

    def compare(self, v0, v1):
        """Method defines how to decide if metric was improved. Commonly it should be one of 'v0 > v1' or 'v0 < v1 '

        Args:
            v0: float, metric value
            v1: float, metric value

        Returns:
            bool, if v0 improves score against v1
        """
        raise NotImplementedError

    def argmax(self, arr):
        """Select best metric from list of metrics based on .compare method

        Args:
            arr: list of float, metric values

        Returns:
            int, position of the best metric value
        """
        best = arr[0]
        best_n = 0

        for n, val in enumerate(arr[1:], 1):
            if self.compare(val, best):
                best = val
                best_n = n

        return best_n


class RMSEMetric(Metric):
    """RMSE Metric for the regression/multiregression task"""
    alias = 'rmse'

    def error(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight) ** .5

    def compare(self, v0, v1):
        return v0 < v1


class R2Score(RMSEMetric):
    """R2 Score Metric for the regression/multiregression task"""
    alias = 'R2_score'

    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            err = ((y_true - y_pred) ** 2 * sample_weight).sum(axis=0) / sample_weight.sum()
            std = ((y_true - y_true.mean(axis=0)) ** 2 * sample_weight).sum(axis=0) / sample_weight.sum()
        else:
            err = ((y_true - y_pred) ** 2).mean(axis=0)
            std = y_true.var(axis=0)

        return (1 - err / std).mean()

    def compare(self, v0, v1):
        return v0 > v1


class RMSLEMetric(RMSEMetric):
    """RMSLE Metric for the regression/multiregression classification task"""
    alias = 'rmsle'

    def error(self, y_true, y_pred):
        return super().error(cp.log1p(y_true), cp.log1p(y_pred))


class BCEMetric(Metric):
    """LogLoss for the binary/multilabel classification task"""
    alias = 'BCE'

    def error(self, y_true, y_pred):
        return -cp.log(y_true * y_pred + (1 - y_pred) * (1 - y_true))

    def compare(self, v0, v1):
        return v0 < v1


def auc(y, x, sample_weight=None):
    """Roc-auc score via cupy

    Args:
        y: cp.ndarray, 1d prediction
        x: cp.ndarray, 1d binary target
        sample_weight: optional 1d array of sample weights

    Returns:
        float, roc-auc metric value
    """
    unique_x = cp.unique(x)

    if unique_x.shape[0] <= 1:
        return 0.5

    if sample_weight is None:
        sample_weight = cp.ones(y.shape, dtype=cp.float64)

    # AUC is invariant to separate rescaling of the two classes. Normalize
    # before summation to avoid overflow/underflow, even for imbalanced weights.
    positive_weight = cp.asarray(sample_weight, dtype=cp.float64) * y
    negative_weight = cp.asarray(sample_weight, dtype=cp.float64) * (1 - y)
    positive_scale, negative_scale = positive_weight.max(), negative_weight.max()
    positive_weight /= cp.where(positive_scale > 0, positive_scale, 1)
    negative_weight /= cp.where(negative_scale > 0, negative_scale, 1)

    rank_x = cp.searchsorted(unique_x, x)

    sum_1 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_1.scatter_add(rank_x, positive_weight)

    sum_0 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_0.scatter_add(rank_x, negative_weight)

    sum_0 /= sum_0.sum()
    sum_1 /= sum_1.sum()
    return float(((sum_0.cumsum() - sum_0 / 2) * sum_1).sum())


class RocAucMetric(Metric):
    """Roc-auc metric for the binary classification task"""
    alias = 'AUC'

    def __call__(self, y_true, y_pred, sample_weight=None):
        """

        Args:
            y_true: cp.ndarray of targets
            y_pred: cp.ndarray of predictions
            sample_weight: None or cp.ndarray of sample_weights

        Returns:

        """
        assert y_pred.shape[1] == 1, 'Multioutput is not supported'

        if sample_weight is not None:
            sample_weight = sample_weight[:, 0]

        return auc(y_true[:, 0], y_pred[:, 0], sample_weight)

    def compare(self, v0, v1):
        return v0 > v1


def _row_weights(sample_weight, n_samples):
    """Validate row weights without losing their original floating-point range."""
    if sample_weight is None:
        return cp.ones((n_samples, 1), dtype=cp.float64)
    weights = cp.asarray(sample_weight, dtype=cp.float64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if weights.shape != (n_samples, 1):
        raise ValueError('sample_weight must have shape (n_samples,) or (n_samples, 1)')
    if not bool(cp.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('sample_weight must be finite and nonnegative')
    return weights


def _rescale_row_weights(weights):
    """Rescale each output independently after excluding its missing labels."""
    if weights.shape[0] == 0:
        return weights
    scale = weights.max(axis=0, keepdims=True)
    return weights / cp.where(scale > 0, scale, 1)


def _nan_metric_inputs(y_true, y_pred, sample_weight):
    """Validate shapes and prepare row weights for metrics with missing labels."""
    y_true, y_pred = cp.asarray(y_true), cp.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or y_true.shape[1] == 0:
        raise ValueError('Targets and predictions must have the same nonempty output dimension')

    weights = _row_weights(sample_weight, y_true.shape[0])
    observed = ~cp.isnan(y_true) & (weights > 0)
    if bool((observed & (y_true != 0) & (y_true != 1)).any()):
        raise ValueError('Observed binary targets must be 0 or 1')
    if bool((observed & ~cp.isfinite(y_pred)).any()):
        raise ValueError('Predictions must be finite at observed, positive-weight labels')
    return y_true, y_pred, weights


def _output_weights(weights):
    weights = np.array(weights, dtype=np.float64, copy=True)
    if (weights.ndim != 1 or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()):
        raise ValueError('Output weights must be a finite, nonnegative vector with positive total weight')
    return weights


def _nan_weighted_mean(values, weights=None):
    """Average defined outputs, preserving infinite errors for incorrect predictions."""
    if weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError('There must be one output weight per target column')
    valid = ~np.isnan(values) & (weights > 0)
    if not valid.any():
        return float('nan')
    selected = values[valid]
    # Positive original weights retain infinite errors even if their
    # normalized values would underflow to zero.
    if np.isposinf(selected).any():
        return float('nan') if np.isneginf(selected).any() else float('inf')
    if np.isneginf(selected).any():
        return float('-inf')
    # Rescale first to avoid overflow for large finite output weights.
    weights = weights[valid] / weights[valid].max()
    return float(np.sum(selected * weights) / weights.sum())


class NaNBCEMetric(BCEMetric):
    """Binary log loss averaged over observed labels, ignoring NaN targets.

    Row weights may have shape (n_samples,) or (n_samples, 1). With
    return_list=True, return a CuPy vector of per-output losses. An output
    with no positively weighted labels has NaN loss; the scalar result is
    NaN only when there are no observed labels with positive weight.
    """
    alias = 'BCE_nan'

    def __call__(self, y_true, y_pred, sample_weight=None, return_list=False):
        y_true, y_pred, weights = _nan_metric_inputs(y_true, y_pred, sample_weight)
        observed = ~cp.isnan(y_true) & (weights > 0)
        if bool((observed & ((y_pred < 0) | (y_pred > 1))).any()):
            raise ValueError('BCE predictions must be probabilities between 0 and 1')
        # Mask before evaluating the loss: NaN/inf at missing labels must not
        # contaminate the result, and multiplying an infinite error by zero
        # would still produce NaN.
        err = self.error(cp.where(observed, y_true, 0), cp.where(observed, y_pred, 0.5))
        # Exclude missing rows before selecting the scale. Per-output metrics
        # need independent scales so large weights in other outputs cannot
        # underflow all weights in an otherwise well-defined output.
        weights = cp.where(observed, weights, 0)
        axis = 0 if return_list else None
        if weights.shape[0]:
            scale = weights.max(axis=axis)
            weights /= cp.where(scale > 0, scale, 1)
        # An impossible prediction still has infinite loss at any positive
        # original weight, even if rescaling underflows that weight to zero.
        weighted_error = cp.where(observed & cp.isinf(err), cp.inf, err * weights)
        numerator = weighted_error.sum(axis=axis)
        denominator = weights.sum(axis=axis)
        result = numerator / cp.where(denominator > 0, denominator, cp.nan)
        return result if return_list else float(result)


class BCEWithNaNMetricWeighted(NaNBCEMetric):
    """Weighted mean of per-output NaN-aware binary log losses.

    Output weights are normalized over outputs with observed labels and
    positive weight. If no such output remains, return NaN.
    """

    def __init__(self, output_weights):
        self.output_weights = _output_weights(output_weights)

    def __call__(self, y_true, y_pred, sample_weight=None):
        errors = super().__call__(y_true, y_pred, sample_weight, return_list=True).get()
        return _nan_weighted_mean(errors, self.output_weights)


class NaNAucMetric(Metric):
    """Macro ROC AUC over binary outputs with missing targets.

    NaN labels and zero-weight rows are excluded independently per output.
    Outputs without positive weight in both classes have undefined AUC
    (NaN) and are excluded from the macro average. Return NaN when every
    output is undefined. return_list=True returns a NumPy vector, including
    NaNs for undefined outputs. Sample weights are passed to each ROC AUC.
    """
    alias = 'AUC_nan'

    def __call__(self, y_true, y_pred, sample_weight=None, return_list=False):
        y_true, y_pred, weights = _nan_metric_inputs(y_true, y_pred, sample_weight)
        scores = np.full(y_true.shape[1], np.nan, dtype=np.float64)
        for i in range(y_true.shape[1]):
            observed = ~cp.isnan(y_true[:, i]) & (weights[:, 0] > 0)
            labels = y_true[observed, i]
            w = weights[observed, 0]
            if not bool((labels == 0).any()) or not bool((labels == 1).any()):
                continue
            scores[i] = auc(labels, y_pred[observed, i], w)
        return scores if return_list else _nan_weighted_mean(scores)

    def compare(self, v0, v1):
        return v0 > v1


class NaNAucMetricByClass(NaNAucMetric):
    """Per-output ROC AUC as a NumPy vector; for reporting, not early stopping."""

    def __call__(self, y_true, y_pred, sample_weight=None):
        return super().__call__(y_true, y_pred, sample_weight, return_list=True)


class NaNGiniMetricByClass(NaNAucMetricByClass):
    """Per-output Gini (2 * AUC - 1); for reporting, not early stopping."""
    alias = 'Gini_nan'

    def __call__(self, y_true, y_pred, sample_weight=None):
        return 2 * super().__call__(y_true, y_pred, sample_weight) - 1


class NaNGiniMetricAtN(NaNAucMetric):
    """Weighted macro Gini over the first n outputs (all outputs by default).

    prod_weights has one finite, nonnegative weight per target column and
    is normalized over selected outputs with defined AUC and positive
    weight. Return NaN if none remain. n selects by column order, not score.
    """
    alias = 'Gini_nan'

    def __call__(self, y_true, y_pred, sample_weight=None, prod_weights=None, n=None):
        scores = super().__call__(y_true, y_pred, sample_weight, return_list=True)
        if n is None:
            n = len(scores)
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 1 <= n <= len(scores):
            raise ValueError('n must be an integer between 1 and the number of outputs')
        weights = None if prod_weights is None else _output_weights(prod_weights)
        if weights is not None:
            if weights.shape != scores.shape:
                raise ValueError('There must be one output weight per target column')
            weights = weights[:n]
        return _nan_weighted_mean(2 * scores[:n] - 1, weights)


class ThresholdMetrics(Metric):
    """Basic class to handle metrics, that accept class label prediction as input"""

    def __init__(self, threshold=0.5, q=None):
        """Define binarization rule. If quantile is given, threshold defined with quantile

        Args:
            threshold: float, threshold value
            q: None or float, quantile threshold
        """
        self.threshold = threshold
        self.q = q

    def get_label(self, y_pred):
        """Get labels from probabilities

        Args:
            y_pred: cp.ndarray, predictions

        Returns:
            cp.ndarray, predicted class labels
        """
        threshold = self.threshold
        if self.q is not None:
            threshold = cp.quantile(y_pred, self.q, axis=0, interpolation='higher')

        return y_pred >= threshold

    def get_stats(self, y_true, y_pred, sample_weight=None, mode='f1'):
        """Helpful utils to calc Precision/Recall/F1

        Args:
            y_true: cp.ndarray, target
            y_pred: cp.ndarray, predicted class label
            sample_weight: None or cp.ndarray, weights
            mode:

        Returns:

        """

        y_pred = self.get_label(y_pred)
        true = y_pred == y_true

        tp = true * y_pred
        if sample_weight is not None:
            tp = tp * sample_weight
        tp = tp.sum(axis=0)

        if mode == 'p':
            if sample_weight is not None:
                return tp, (y_pred * sample_weight).sum(axis=0)
            return tp, y_pred.sum(axis=0)

        if sample_weight is not None:
            tot = (y_true * sample_weight).sum(axis=0)
        else:
            tot = y_true.sum(axis=0)
        if mode == 'r':
            return tp, tot

        if sample_weight is not None:
            tot_p = (y_pred * sample_weight).sum(axis=0)
        else:
            tot_p = y_pred.sum(axis=0)

        return tp, tot, tot_p

    def compare(self, v0, v1):
        return v0 > v1


class AccuracyMetric(ThresholdMetrics):
    """Accuracy Metric for the binary/multilabel classification task"""
    alias = 'Accuracy'

    def error(self, y_true, y_pred):
        y_pred = self.get_label(y_pred)
        return (y_true == y_pred).mean(axis=1)


class Precision(ThresholdMetrics):
    """Precision Metric for the binary/multilabel classification task"""
    alias = 'Precision'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight, mode='p')
        tot = cp.clip(tot, 1e-5, None)
        return (tp / tot).mean()


class Recall(ThresholdMetrics):
    """Recall Metric for the binary/multilabel classification task"""
    alias = 'Recall'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight, mode='r')
        tot = cp.clip(tot, 1e-5, None)
        return (tp / tot).mean()


class F1Score(ThresholdMetrics):
    """F1 Score Metric for the binary/multilabel classification task"""
    alias = 'F1_score'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot, tot_p = self.get_stats(y_true, y_pred, sample_weight, mode='f1')
        precision = tp / cp.clip(tot_p, 1e-5, None)
        recall = tp / cp.clip(tot, 1e-5, None)

        return (2 * (precision * recall) / cp.clip(precision + recall, 1e-5, None)).mean()


metric_alias = {

    # for bce
    'bce': BCEMetric(),
    'logloss': BCEMetric(),

    'bce_nan': NaNBCEMetric(),
    'auc_nan': NaNAucMetric(),
    'gini_nan': NaNGiniMetricAtN(),

    'precision': Precision(),
    'recall': Recall(),
    'f1_score': F1Score(),
    'f1': F1Score(),

    'accuracy': AccuracyMetric(),
    'acc': AccuracyMetric(),

    'auc': RocAucMetric(),
    'roc': RocAucMetric(),

    # for regression
    'rmse': RMSEMetric(),
    'l2': RMSEMetric(),
    'rmsle': RMSLEMetric(),
    'r2': R2Score(),
    'r2_score': R2Score(),

}
