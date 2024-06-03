import logging
import subprocess
import sys
import warnings

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False

try:
    subprocess.check_output('nvidia-smi')
    CUDA_FOUND = True
except Exception:
    CUDA_FOUND = False

from .utils.tl_wrapper import TLPredictor, TLCompiledPredictor
from .utils.onnx_wrapper import pb_to_onnx, ONNXPredictor

if CUDA_FOUND:
    from .gpu.boosting import GradientBoosting
    from .gpu.sketch_boost import SketchBoost
    from .gpu.losses.losses import Loss
    from .gpu.losses.metrics import Metric
    from .callbacks.callback import Callback

    __all__ = [

        'GradientBoosting',
        'SketchBoost',
        'Callback',
        'Loss',
        'Metric',
        'callbacks',
        'gpu',
        'multioutput',
        'sampling',
        'utils',
        'pb_to_onnx',

    ]

else:
    warnings.warn('No Nvidia GPU detected! Only treelite inference on CPU is available')
    __all__ = []

__all__.extend([

    'TLPredictor',
    'TLCompiledPredictor',
    'ONNXPredictor'

])

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)
