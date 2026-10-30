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

**Fast with GPU**. Despite the fact that Py-boost is written in Python, it works only on a GPU and uses Python GPU
libraries such as `CuPy` and `Numba`.

**Efficient inference**. Since v0.4, Py-Boost has been able to perform efficient inference of tree ensembles on a GPU.
Moreover, once your model is trained on a GPU, it can be used for inference on a CPU-only machine
after conversion to the [treelite](https://treelite.readthedocs.io/) format using the built-in wrapper (limitation: the model should
be trained with `target_splitter='Single'`, which is the default).

**ONNX compatible**. Since v0.5, Py-Boost has been compatible with the ONNX format, which provides more options
for CPU inference and model deployment.

**Easy to customize**. Py-boost can be easily customized even if one is not familiar with GPU programming (just replace
np with cp). What can be customized? Almost everything via custom callbacks. Examples: Row/Col sampling strategy,
Training control, Losses/metrics, Multioutput handling strategy, Anything via custom callbacks.

## SketchBoost [paper](https://openreview.net/forum?id=WSxarC8t-T)

**Multioutput training**. Current state-of-the-art boosting toolkits offer limited support for multioutput training.
Even when this option is available, training times for tasks such as multiclass/multilabel classification and multitask
regression are quite long due to the training complexity, which scales linearly with the number of outputs. To overcome
the existing limitations, we introduce the **SketchBoost** algorithm, which uses approximate tree structure search. As we show
in [the paper](https://openreview.net/forum?id=WSxarC8t-T), this strategy does not lead to a decrease in performance and
can often improve accuracy.

**SketchBoost**. You can try our sketching strategies by using the `SketchBoost` class, or you can implement your own
strategies and pass them to the `GradientBoosting` constructor as the `multioutput_sketch` parameter. For details, please
see [Tutorial_2_Advanced_multioutput](https://github.com/AILab-MLTools/Py-Boost/blob/master/tutorials/Tutorial_2_Advanced_multioutput.ipynb)

## Installation

Py-Boost supports Python 3.8–3.14. Install the CuPy distribution compatible
with your Python and CUDA versions before installing Py-Boost.

On Python 3.14, use Numba 0.63 or later and CuPy 14 or later. For example,
with CUDA 12.x:

`pip install -U "cupy-cuda12x>=14" py-boost`

For other Python or CUDA versions, choose the appropriate CuPy package from the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

## CPU inference with Treelite and TL2cgen

Py-Boost uses Treelite 4 (`>=4.1.2,<5`) and TL2cgen 1 (`>=1,<2`) for CPU
inference. Both dependencies are installed with Py-Boost; `treelite_runtime`
is no longer used. NumPy 1.x and 2.x are supported, subject to your Python
version and the other dependencies.

Convert a fitted model trained with `target_splitter='Single'` (the default):

```python
from py_boost import TLPredictor, TLCompiledPredictor

tl_model = TLPredictor(model)
prediction = tl_model.predict(X_test, nthread=4)
tl_model.dump('cpu_model')
restored = TLPredictor.load('cpu_model')
```

Single-output and multi-output regression, binary/multilabel classification,
and multiclass classification keep the original model's postprocessing.
Predictions have shape `(n_samples, n_outputs)`, including single-output models.
The underlying Treelite model stores raw scores as regression targets; the
wrapper applies `model.postprocess_fn` once, after restoring this output shape.
A custom CPU postprocessor can be supplied with `postprocess_fn=...`.
Both predictor wrappers support shallow copying with `copy.copy()`; saving a
predictor leaves the original and its copies usable.

Compile the model with TL2cgen for faster CPU inference. Compilation requires a
C compiler; prediction with the compiled library does not require a compiler:

```python
# Linux: GCC must be installed and on PATH.
compiled = tl_model.compile('gcc', 'model.so', nthread=4)
# Windows: run from a Visual Studio developer shell, with MSVC on PATH.
# compiled = tl_model.compile('msvc', 'model.dll', nthread=4)

prediction = compiled.predict(X_test)
compiled.dump('compiled.pkl')
restored_compiled = TLCompiledPredictor.load('compiled.pkl')
```

`TLCompiledPredictor.predict()` also accepts SciPy sparse matrices (CSR, CSC
and COO) without converting them to dense arrays. Unstored entries and stored
NaNs are treated as missing values; explicitly stored zeros remain zero.

Compiled predictor dumps contain metadata and the library path. Deploy the
`.so`/`.dll` alongside the dump. If its path changes, load it with
`TLCompiledPredictor.load('compiled.pkl', libpath=new_path)`. Use
`set_libpath(new_path)` to relocate an already loaded predictor.
Use a library built for the target operating system and architecture. The
legacy `compiler` argument accepts only its default value, `'ast_native'`.
See the [TL2cgen migration guide](https://tl2cgen.readthedocs.io/en/latest/treelite-migration.html)
for the underlying API changes.

Treelite 3 model checkpoints and libraries compiled with `treelite_runtime`
should be re-exported from the original Py-Boost model with this version.

## Quick tour

Py-boost is easy to use since it has an interface similar to scikit-learn's. For usage examples, please see:

* [Tutorial_1_Basics](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_1_Basics.ipynb) for simple
  usage examples
* [Tutorial_2_Advanced_multioutput](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_2_Advanced_multioutput.ipynb)
  for advanced multioutput features
* [Tutorial_3_Custom_features](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_3_Custom_features.ipynb)
  for examples of customization
* [Tutorial_4_Handle_null_targets](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_4_Handle_null_targets.ipynb)
  for the case when the multioutput target contains NaNs
* [Tutorial_5_ONNX_inference](https://github.com/sb-ai-lab/Py-Boost/blob/master/tutorials/Tutorial_5_ONNX_inference.ipynb)
  examples of parsing and inference on CPU with ONNX

More examples are coming soon.
