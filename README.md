# Py-boost: a research tool for exploring GBDTs

Modern gradient boosting toolkits are very complex and are written in low-level programming languages. As a result,

* It is hard to customize them to suit one’s needs
* New ideas and methods are not easy to implement
* It is difficult to understand how they work

Py-boost is a Python-based gradient boosting library which aims at overcoming the aforementioned problems.

**Authors**: [Anton Vakhrushev](https://kaggle.com/btbpanda), [Leonid Iosipoi](http://iosipoi.com/)
, [Sergey Kupriyanov](https://www.linkedin.com/in/sergeykupriyanov).

## Py-boost Key Features

**Simple**. Py-boost is a simplified gradient boosting library, but it supports all main features and hyperparameters
available in other implementations.

**Fast with GPU**. Despite the fact that Py-boost is written in Python, it works only on GPU and uses Python GPU
libraries such as `CuPy` and `Numba`.

**Efficient inference**. Since v0.4 Py-Boost is able to perform the efficient inference of tree ensembles on GPU.
Moreover, ones your model is trained on GPU, it could be converted to perform the inference on CPU only machine via
converting to the [treelite](https://treelite.readthedocs.io/) format with build-in wrapper (limitation - model should
be trained with `target_splitter='Single'`, which is the default).

**ONNX compatible** Since v0.5 Py-Boost is compatible with ONNX format that allows more options the CPU inference and
model deployment.

**Easy to customize**. Py-boost can be easily customized even if one is not familiar with GPU programming (just replace
np with cp). What can be customized? Almost everything via custom callbacks. Examples: Row/Col sampling strategy,
Training control, Losses/metrics, Multioutput handling strategy, Anything via custom callbacks

## SketchBoost [paper](https://openreview.net/forum?id=WSxarC8t-T)

**Multioutput training**. Current state-of-atr boosting toolkits provide very limited support of multioutput training.
And even if this option is available, training time for such tasks as multiclass/multilabel classification and multitask
regression is quite slow because of the training complexity that scales linearly with the number of outputs. To overcome
the existing limitations we create **SketchBoost** algorithm that uses approximate tree structure search. As we show
in [paper](https://openreview.net/forum?id=WSxarC8t-T) that strategy at least does not lead to performance decrease and
often is able to improve the accuracy

**SketchBoost**. You can try our sketching strategies by using `SketchBoost` class or if you want you can implement your
own and pass to the `GradientBoosting` constructor as `multioutput_sketch` parameter. For the details please
see [Tutorial_2_Advanced_multioutput](https://github.com/AILab-MLTools/Py-Boost/blob/master/tutorials/Tutorial_2_Advanced_multioutput.ipynb)

## Installation

Before installing py-boost via pip you should have cupy installed. You can use:

`pip install -U cupy-cuda110 py-boost`

**Note**: replace with your cuda version! For the details see [this guide](https://docs.cupy.dev/en/stable/install.html)

## Quick tour

Py-boost is easy to use since it has similar to scikit-learn interface. For usage example please see:

* [Tutorial_1_Basics](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_1_Basics.ipynb) for simple
  usage examples
* [Tutorial_2_Advanced_multioutput](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_2_Advanced_multioutput.ipynb)
  for advanced multioutput features
* [Tutorial_3_Custom_features](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_3_Custom_features.ipynb)
  for examples of customization
* [Tutorial_4_Handle_null_targets](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_4_Handle_null_targets.ipynb)
  for the case when multioutput target contains NaNs
* [Tutorial_5_ONNX_inference](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_5_ONNX_inference.ipynb)
  examples of parsing and inference on CPU with ONNX

More examples are coming soon

## Missing binary targets

For binary or multilabel targets, use `NaN` for unknown labels and `0` for
observed negatives. `BCEWithNaNLoss` excludes missing labels from gradients,
Hessians and initial class frequencies. An entirely unobserved output starts
at probability 0.5. Keep `lambda_l2 > 0` so leaves without observed labels
have finite updates.

```python
from py_boost import GradientBoosting
from py_boost.gpu.losses import BCEWithNaNLoss, NaNAucMetric

model = GradientBoosting(BCEWithNaNLoss(), metric=NaNAucMetric(), lambda_l2=1)
model.fit(X_train, y_train, eval_sets=[{"X": X_valid, "y": y_valid}])
```

The loss also accepts aliases `bce_nan` and `multilabel_nan`. Its default
metric is `NaNBCEMetric`; metric names `bce`/`logloss` and `auc`/`roc` select
the corresponding NaN-aware metrics when used with this loss. Explicit metric
aliases `bce_nan`, `auc_nan` and `gini_nan` are also available. Existing losses
and their default metrics are unchanged.

Import these metrics from `py_boost.gpu.losses`:

| Metric | Result |
| --- | --- |
| `NaNBCEMetric` | Mean BCE over observed labels; `return_list=True` gives per-output losses as a CuPy array. |
| `BCEWithNaNMetricWeighted(output_weights)` | Weighted mean of per-output BCE. `BCEwithNaNMetricWeighted` is an alias. |
| `NaNAucMetric` | Macro ROC AUC; `return_list=True` gives a NumPy array. |
| `NaNAucMetricByClass` | Per-output ROC AUC as a NumPy array, for reporting. |
| `NaNGiniMetricByClass` | Per-output Gini (`2 * AUC - 1`) as a NumPy array, for reporting. |
| `NaNGiniMetricAtN` | Weighted macro Gini; call with `prod_weights=` and `n=` to restrict to the first `n` target columns. |

All metrics support nonnegative row weights of shape `(n_samples,)` or
`(n_samples, 1)`. NaN labels and zero-weight rows are excluded independently
for each target. ROC AUC/Gini are undefined for outputs without positive
weight in both classes; BCE is undefined for outputs without any positively
weighted labels. Per-output results contain NaN for these outputs, and macro
metrics exclude them. A scalar metric is NaN if no eligible output remains.
Validate that your evaluation set has enough observed labels for early stopping;
the per-class vector metrics are for reporting only.

Observed targets must be binary. Predictions must be finite wherever a label
has positive row weight; invalid predictions are reported as errors rather
than treated as missing labels. BCE additionally requires probabilities in
`[0, 1]`, while ROC AUC/Gini also accept finite decision scores.

Output/product weights must follow target column order and are normalized
over eligible, positively weighted outputs. `n` selects leading columns,
not the outputs with the highest scores. Compared with custom implementations
that return an unnormalized weighted sum, non-unit-sum weights therefore have
different semantics. Standard `CrossValidation` is supported; `AdaptiveESCV`
requires an explicit scorer for missing labels.

GPU regression tests can be run with `python -m pytest -q tests/test_nan_losses.py`
after installing Py-Boost, a compatible CuPy build, and pytest.
