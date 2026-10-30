"""Import TL2cgen on Windows Python installations without a Conda DLL directory."""

import os
import sys
from importlib import import_module


def _import_tl2cgen():
    conda_bins = {
        os.path.normcase(os.path.join(os.path.normpath(prefix), 'Library', 'bin'))
        for prefix in (sys.prefix, sys.base_prefix)
    }
    if sys.platform != 'win32' or all(os.path.isdir(path) for path in conda_bins):
        return import_module('tl2cgen')

    # TL2cgen 1.0 and older Treelite 4 versions register optional Conda directories.
    # CPython and venv installations do not have it, so import otherwise fails
    # before the packaged DLL can be loaded. Ignore only that missing directory
    # during import; all real DLL-loading errors and other paths still propagate.
    add_dll_directory = os.add_dll_directory

    def add_directory(path):
        if (os.path.normcase(os.path.normpath(path)) in conda_bins
                and not os.path.isdir(path)):
            return None
        return add_dll_directory(path)

    try:
        os.add_dll_directory = add_directory
        return import_module('tl2cgen')
    finally:
        os.add_dll_directory = add_dll_directory


tl2cgen = _import_tl2cgen()
# TL2cgen imports Treelite inside the guarded scope above.
treelite = import_module('treelite')
