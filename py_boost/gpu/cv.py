"""Gradient Boosting with built-in cross validation"""

import numpy as np
import cupy as cp
from copy import deepcopy

from sklearn.model_selection import KFold, StratifiedKFold
from .utils import validate_input, quantize_train_valid


class CustomFolds:

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

    def __init__(self, base_learner, random_state=42):
        """

        Args:
            base_learner:
            random_state:
        """
        self._base_learner = base_learner
        self.random_state = random_state
        self.models = None

    def fit_predict(self, X, y, sample_weight=None, cv=5, stratify=True, random_state=42):
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
        X_enc, max_bin, borders, eval_enc = quantize_train_valid(X, eval_sets, self._base_learner.max_bin,
                                                                 self._base_learner.quant_sample)

        # create validation

        if type(cv) in [int, float]:
            cv = int(cv)
            if stratify:
                folds = StratifiedKFold(cv, shuffle=True, random_state=random_state)
            else:
                folds = KFold(cv, shuffle=True)

        else:
            folds = CustomFolds(cv)

        # fit and free memory
        mempool = cp.cuda.MemoryPool()

        oof_pred = None

        with cp.cuda.using_allocator(allocator=mempool.malloc):

            for n, (f0, f1) in enumerate(folds.split(X, y)):

                # split data

                X_tr, X_enc_tr, y_tr, = X[f0], X_enc[f0], y[f0]

                sample_weight_tr = None
                if sample_weight is not None:
                    sample_weight_tr = sample_weight[f0]

                eval_sets = [{}]
                eval_sets[0]['X'] = X[f1]
                eval_sets[0]['y'] = y[f1]
                eval_sets[0]['sample_weight'] = None
                if sample_weight is not None:
                    eval_sets[0]['sample_weight'] = sample_weight[f1]

                eval_enc = [X_enc[f1]]

                # fit model
                model = deepcopy(self._base_learner)
                builder, build_info = model._create_build_info(mempool, X_tr, X_enc_tr, y_tr, sample_weight_tr,
                                                               max_bin, borders, eval_sets, eval_enc)
                model._fit(builder, build_info)

                # predict

                val_pred = model.predict(eval_sets[0]['X'])

                if oof_pred is None:
                    oof_pred = np.zeros((X.shape[0], val_pred.shape[1]), dtype=np.float32)

                oof_pred[f1] = val_pred
                self.models.append(model)

                mempool.free_all_blocks()

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
