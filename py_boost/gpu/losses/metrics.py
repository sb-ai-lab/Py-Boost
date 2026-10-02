"""Common metrics"""

import cupy as cp


class Metric:
    """Base class to define eval metric function.
    Metric could be defined in 2 ways:

        - redefine .error method. Preffered if possible. Simplified metric definition by calculating error function
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
            int, position of best metric value
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
        sample_weight = cp.ones_like(y)

    rank_x = cp.searchsorted(unique_x, x)

    sum_1 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_1.scatter_add(rank_x, sample_weight * y)

    sum_0 = cp.zeros_like(unique_x, dtype=cp.float64)
    sum_0.scatter_add(rank_x, sample_weight * (1 - y))

    cs_0 = sum_0.cumsum()
    auc_ = (cs_0 - sum_0 / 2) * sum_1

    tot = cs_0[-1] * sum_1.sum()

    return float(auc_.sum() / tot)


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
