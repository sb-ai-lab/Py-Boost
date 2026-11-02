"""Import TL2cgen on Windows Python installations without a Conda DLL directory."""

import builtins
import os
import sys
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import PathFinder


class _DllDirectoryOs:
    """An os view used only inside the dependencies' native-library loaders."""

    def __init__(self, conda_bins):
        self._conda_bins = conda_bins

    def __getattr__(self, name):
        return getattr(os, name)

    def add_dll_directory(self, path):
        if (os.path.normcase(os.path.normpath(path)) in self._conda_bins
                and not os.path.isdir(path)):
            return None
        return os.add_dll_directory(path)


class _DllDirectoryLoader(Loader):
    def __init__(self, loader, conda_bins):
        self._loader = loader
        self._os = _DllDirectoryOs(conda_bins)

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        original_import = builtins.__import__

        def module_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'os' and level == 0:
                return self._os
            return original_import(name, globals, locals, fromlist, level)

        # Give only this module a guarded os import. The process-wide os module
        # and builtins stay unchanged, including during concurrent imports.
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=module_import)
        self._loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _DllDirectoryFinder(MetaPathFinder):
    def __init__(self, conda_bins):
        self._conda_bins = conda_bins

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('treelite.core', 'tl2cgen.libloader'):
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _DllDirectoryLoader(spec.loader, self._conda_bins)
        return spec


def _import_tl2cgen():
    conda_bins = {
        os.path.normcase(os.path.join(os.path.normpath(prefix), 'Library', 'bin'))
        for prefix in (sys.prefix, sys.base_prefix)
    }
    if sys.platform != 'win32' or all(os.path.isdir(path) for path in conda_bins):
        return import_module('tl2cgen')

    # TL2cgen 1.0 and older Treelite 4 versions register optional Conda directories.
    # CPython and venv installations do not have them, so import otherwise fails
    # before the packaged DLL can be loaded. Ignore only that missing directory
    # inside those two modules; real DLL-loading errors still propagate.
    finder = _DllDirectoryFinder(conda_bins)
    sys.meta_path.insert(0, finder)
    try:
        return import_module('tl2cgen')
    finally:
        sys.meta_path.remove(finder)


tl2cgen = _import_tl2cgen()
# TL2cgen imports Treelite inside the guarded scope above.
treelite = import_module('treelite')
