"""Gradient Boosting with built-in cross validation"""

import numpy as np
import cupy as cp
from copy import deepcopy

from sklearn.model_selection import KFold, StratifiedKFold
from ..gpu.utils import validate_input


class CustomFolds:
    """
    Class to imitate sklearn cv for custom folds
    """

    def __init__(self, folds):
        self.folds = folds

    def split(self, *args, **kwargs):
        nfolds = int(self.folds.max() + 1)
        idx = np.arange(len(self.folds))

        splits = []

        for i in range(nfolds):
            splits.append((idx[self.folds != i], idx[self.folds == i]))

        return splits


class CrossValidation:
    """
    Cross validation wrapper for gradient boosting
    """

    def __init__(self, base_learner, random_state=42):
        """

        Args:
            base_learner:
            random_state:
        """
        self._base_learner = base_learner
        self.random_state = random_state
        self.models = None

    def _fit_predict(self, mempool, X, X_enc, y, sample_weight, max_bin, borders, cv_iter):

        oof_pred = None
        folds = np.zeros(X.shape[0], dtype=np.int32)

        with cp.cuda.using_allocator(allocator=mempool.malloc):

            for n, (f0, f1) in enumerate(cv_iter.split(X, y)):

                # split data

                X_tr, X_enc_tr, y_tr, = X[f0], X_enc[f0], y[f0]

                sample_weight_tr = None
                if sample_weight is not None:
                    sample_weight_tr = sample_weight[f0]

                eval_sets = [{

                    'X': X[f1],
                    'y': y[f1],
                    'sample_weight': None if sample_weight is None else sample_weight[f1]

                }]

                eval_enc = [X_enc[f1]]

                # fit model
                model = deepcopy(self._base_learner)
                model._infer_params()
                builder, build_info = model._create_build_info(mempool, X_tr, X_enc_tr, y_tr, sample_weight_tr,
                                                               max_bin, borders, eval_sets, eval_enc)
                model._fit(builder, build_info)

                # predict

                val_pred = model.predict(eval_sets[0]['X'])
                model.to_cpu()

                if oof_pred is None:
                    oof_pred = np.zeros((X.shape[0], val_pred.shape[1]), dtype=np.float32)

                oof_pred[f1] = val_pred
                folds[f1] = n
                self.models.append(model)

                mempool.free_all_blocks()

        return oof_pred, folds

    def get_cv_iter(self, cv, stratify, random_state):

        if type(cv) in [int, float]:
            cv = int(cv)
            if stratify:
                folds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
            else:
                folds = KFold(cv, shuffle=True)

        else:
            folds = CustomFolds(cv)

        return folds

    def fit_predict(self, X, y, sample_weight=None, cv=5, stratify=False, random_state=42):
        """

        Args:
            X:
            y:
            sample_weight:
            cv:
            stratify:
            random_state:

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

        return oof_pred

    def predict(self, X):
        """

        Args:
            X:

        Returns:

        """
        res = None

        for model in self.models:

            pred = model.predict(X)
            if res is None:
                res = pred
            else:
                res += pred

        res /= len(self.models)

        return res
