import logging
import sys
import subprocess
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
        'utils'

    ]
    
except Exception:
    warnings.warn('No Nvidia GPU detected! Only treelite inference on CPU is available')
    __all__ = []

from .utils.tl_wrapper import TLPredictor, TLCompiledPredictor

__all__.extend([

    'TLPredictor',
    'TLCompiledPredictor',

])

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

__version__ = importlib_metadata.version(__name__)
