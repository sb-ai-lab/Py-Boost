"""Common multiclass metrics"""

import cupy as cp

from .metrics import Metric, metric_alias


class CrossEntropyMetric(Metric):
    """CrossEntropy Metric for the multiclassification task"""
    alias = 'Crossentropy'

    def error(self, y_true, y_pred):
        return -cp.log(cp.take_along_axis(y_pred, y_true[:, cp.newaxis], axis=1))

    def compare(self, v0, v1):
        return v0 < v1


class MultiAccuracyMetric(Metric):
    """Accuracy Metric for the multiclassification task"""
    alias = 'Accuracy'

    def error(self, y_true, y_pred):
        cl_pred = y_pred.argmax(axis=1)
        return (cl_pred == y_true).astype(cp.float32)

    def compare(self, v0, v1):
        return v0 > v1


class MultiMetric(Metric):
    """Basic class to handle metrics, that accept class label prediction as input for the multiclassificationn task"""

    def __init__(self, average='macro'):
        """

        Args:
            average: str, one of 'micro' / 'macro'
        """
        self.average = average

    @staticmethod
    def get_stats(y_true, y_pred, sample_weight=None, mode='f1'):
        """Helpful utils to calc Precision/Recall/F1

        Args:
            y_true: cp.ndarray, target
            y_pred: cp.ndarray, predicted class label
            sample_weight: None or cp.ndarray, weights
            mode:

        Returns:

        """

        if sample_weight is None:
            sample_weight = cp.ones(y_true.shape, dtype=cp.float32)
        else:
            sample_weight = sample_weight[:, 0]

        cl_pred = y_pred.argmax(axis=1)
        true = y_true == cl_pred

        tp = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        tp.scatter_add(cl_pred, true * sample_weight)

        tot = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        if mode == 'p':
            tot.scatter_add(cl_pred, sample_weight)
            return tp, tot

        tot.scatter_add(y_true, sample_weight)
        if mode == 'r':
            return tp, tot

        tot_p = cp.zeros(y_pred.shape[1], dtype=cp.float64)
        tot_p.scatter_add(cl_pred, sample_weight)

        return tp, tot, tot_p

    def get_metric(self, tp, tot):

        tot = cp.clip(tot, 1e-5, None)

        if self.average == 'micro':
            return float(tp.sum() / tot.sum())

        return float((tp / tot).mean())

    def compare(self, v0, v1):
        return v0 > v1


class MultiPrecision(MultiMetric):
    """Precision Metric for the multiclassification classification task"""
    alias = 'Precision'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='p')
        return self.get_metric(tp, tot)


class MultiRecall(MultiMetric):
    """Recall Metric for the multiclassification classification task"""
    alias = 'Recall'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='r')
        return self.get_metric(tp, tot)


class MultiF1Score(MultiMetric):
    """F1 Score Metric for the multiclassification classification task"""
    alias = 'F1_score'

    def __call__(self, y_true, y_pred, sample_weight=None):
        tp, tot, tot_p = self.get_stats(y_true, y_pred, sample_weight=sample_weight, mode='f1')
        precision = self.get_metric(tp, tot_p)
        recall = self.get_metric(tp, tot)
        return 2 * (precision * recall) / (precision + recall)


multiclass_metric_alias = {**metric_alias, **{

    'crossentropy': CrossEntropyMetric(),

    'precision': MultiPrecision(),
    'recall': MultiRecall(),
    'f1_score': MultiF1Score(),
    'f1': MultiF1Score(),

    'accuracy': MultiAccuracyMetric(),
    'acc': MultiAccuracyMetric(),

}}
