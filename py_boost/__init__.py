import logging
import sys

_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

# if root logger has handlers, propagate messages up and let root logger process them
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler(sys.stdout))
    _logger.propagate = False

from .gpu.boosting import GradientBoosting
from .gpu.sketch_boost import SketchBoost
from .utils.tl_wrapper import TLPredictor, TLCompiledPredictor
from .callbacks.callback import Callback
from .gpu.losses.losses import Loss
from .gpu.losses.metrics import Metric

__all__ = [

    'GradientBoosting',
    'SketchBoost',
    'TLPredictor',
    'TLCompiledPredictor',
    'Callback',
    'Loss',
    'Metric',
    'callbacks',
    'gpu',
    'multioutput',
    'sampling',
    'utils'

]

# try:
#     import importlib.metadata as importlib_metadata
# except ModuleNotFoundError:
#     import importlib_metadata
#
# __version__ = importlib_metadata.version(__name__)
