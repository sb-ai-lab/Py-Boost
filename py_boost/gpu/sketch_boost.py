"""Implements SketchBoost for multioutput class"""

from .boosting import GradientBoosting
from ..multioutput.sketching import FilterSketch, TopOutputsSketch, SVDSketch, RandomSamplingSketch, \
    RandomProjectionSketch


class SketchBoost(GradientBoosting):
    """
    Gradient Boosting with built in FilterSketch to handle multioutput tasks. If single output is passed,
    it is handled as usual
    """

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

                 quantization='Quantile',
                 quant_sample=2000000,
                 max_bin=256,
                 min_data_in_bin=3,

                 es=100,
                 seed=42,
                 verbose=10,

                 sketch_outputs=1,
                 sketch_method='proj',
                 use_hess=True,

                 callbacks=None,
                 sketch_params=None
                 ):
        """

        Args:
            loss: str or Loss, loss function
            metric: None or str or Metric, metric
            ntrees: int, maximum number of trees
            lr: float, learning rate
            min_gain_to_split: float >=0, minimal gain to split
            lambda_l2: float > 0, l2 leaf regularization
            gd_steps: int > 0, number of gradient steps
            max_depth: int > 0, maximum tree depth. Setting it to large values (>12) may cause OOM for wide datasets
            min_data_in_leaf: int, minimal leaf size. Note - for some loss fn leaf size is approximated
                with hessian values to speed up training
            colsample: float or Callable, sumsample of columns to construct trees or callable - custom sampling
            subsample: float or Callable, sumsample of rows to construct trees or callable - custom sampling

            quantization: str or Quantizer, method for quantizatrion. One of 'Quantile', 'Uniform',
                'Uniquant' or custom implementation
            quant_sample: int, subsample to quantize features
            max_bin: int in [2, 256] maximum number of bins to quantize features
            min_data_in_bin: int in [2, 256] minimal bin size. NOTE: currently ignored

            es: int, early stopping rounds. If 0, no early stopping
            seed: int, random state
            verbose: int, verbosity freq
            sketch_outputs: int, number of outputs to keep
            sketch_method: str, name of the sketching strategy
            use_hess: bool, use hessians in multioutput training
            callbacks: list of Callback, callbacks to customize training are passed here
            sketch_params: dict, optional kwargs for sketching strategy
        """
        if sketch_params is None:
            sketch_params = {}

        if sketch_method == 'filter':
            sketch = FilterSketch(sketch_outputs, **sketch_params)

        elif sketch_method == 'svd':
            sketch = SVDSketch(sketch_outputs, **sketch_params)

        elif sketch_method == 'topk':
            sketch = TopOutputsSketch(sketch_outputs, **sketch_params)

        elif sketch_method == 'rand':
            sketch = RandomSamplingSketch(sketch_outputs, **sketch_params)

        elif sketch_method == 'proj':
            sketch = RandomProjectionSketch(sketch_outputs, **sketch_params)

        elif sketch_method is None:
            sketch = None

        else:
            raise ValueError('Unknown sketching strategy')

        super().__init__(loss=loss,
                         metric=metric,
                         ntrees=ntrees,
                         lr=lr,
                         min_gain_to_split=min_gain_to_split,
                         lambda_l2=lambda_l2,
                         gd_steps=gd_steps,
                         max_depth=max_depth,
                         min_data_in_leaf=min_data_in_leaf,
                         colsample=colsample,
                         subsample=subsample,

                         quantization=quantization,
                         quant_sample=quant_sample,
                         max_bin=max_bin,
                         min_data_in_bin=min_data_in_bin,

                         target_splitter='Single',
                         multioutput_sketch=sketch,
                         use_hess=use_hess,
                         es=es,
                         seed=seed,
                         verbose=verbose,
                         callbacks=callbacks)
