"""Windows loader compatibility must leave process-wide import and os APIs intact."""

import builtins
import os
import subprocess
import sys
from importlib.util import module_from_spec

import pytest

from py_boost.utils import tl2cgen_compat


@pytest.mark.parametrize('module_name,filename', [
    ('treelite.core', 'core.py'), ('tl2cgen.libloader', 'libloader.py'),
])
def test_loader_uses_private_os_view(tmp_path, monkeypatch, module_name, filename):
    conda_bin = str(tmp_path / 'Library' / 'bin')
    other_missing = str(tmp_path / 'unrelated')
    existing = str(tmp_path)
    cookie = object()
    calls = []

    def add_directory(path):
        calls.append(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        return cookie

    # This function belongs to the test module, outside the guarded dependency.
    # It runs on another thread while the dependency's import is still executing.
    def other_thread():
        assert os.add_dll_directory is add_directory
        with pytest.raises(FileNotFoundError):
            os.add_dll_directory(conda_bin)
        return True

    source = tmp_path / filename
    source.write_text("""
import os
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
private_os = os
missing_result = os.add_dll_directory(conda_bin)
existing_result = os.add_dll_directory(existing)
try:
    os.add_dll_directory(other_missing)
except FileNotFoundError:
    unrelated_error = True
assert import_module('os').add_dll_directory is original
with ThreadPoolExecutor(max_workers=1) as pool:
    thread_ok = pool.submit(other_thread).result(timeout=10)
""")
    monkeypatch.setattr(os, 'add_dll_directory', add_directory, raising=False)
    finder = tl2cgen_compat._DllDirectoryFinder({os.path.normcase(conda_bin)})
    spec = finder.find_spec(module_name, [str(tmp_path)])
    module = module_from_spec(spec)
    module.__dict__.update(conda_bin=conda_bin, other_missing=other_missing,
                           existing=existing, original=add_directory, other_thread=other_thread)
    original_import = builtins.__import__
    spec.loader.exec_module(module)
    assert module.private_os is not os
    assert module.private_os.path is os.path
    assert module.missing_result is None
    assert module.existing_result is cookie
    assert module.unrelated_error
    assert module.thread_ok
    assert os.add_dll_directory is add_directory
    assert builtins.__import__ is original_import
    assert calls == [existing, other_missing, conda_bin]
    assert not os.path.exists(conda_bin)
    assert finder.find_spec('unrelated.module', [str(tmp_path)]) is None


@pytest.mark.parametrize('fail', [False, True])
def test_windows_import_removes_finder(tmp_path, monkeypatch, fail):
    before = list(sys.meta_path)
    original = getattr(os, 'add_dll_directory', None)
    imported = object()

    def import_module(name):
        assert name == 'tl2cgen'
        assert getattr(os, 'add_dll_directory', None) is original
        assert any(isinstance(finder, tl2cgen_compat._DllDirectoryFinder)
                   for finder in sys.meta_path)
        if fail:
            raise OSError('DLL dependency is missing')
        return imported

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'platform', 'win32')
        patch.setattr(sys, 'prefix', str(tmp_path))
        patch.setattr(sys, 'base_prefix', str(tmp_path / 'base_python'))
        patch.setattr(tl2cgen_compat, 'import_module', import_module)
        if fail:
            with pytest.raises(OSError, match='DLL dependency is missing'):
                tl2cgen_compat._import_tl2cgen()
        else:
            assert tl2cgen_compat._import_tl2cgen() is imported
    assert sys.meta_path == before
    assert getattr(os, 'add_dll_directory', None) is original


def test_windows_conda_import_uses_normal_loader(tmp_path, monkeypatch):
    (tmp_path / 'Library' / 'bin').mkdir(parents=True)
    before = list(sys.meta_path)
    imported = object()

    def import_module(name):
        assert sys.meta_path == before
        return imported

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'platform', 'win32')
        patch.setattr(sys, 'prefix', str(tmp_path))
        patch.setattr(sys, 'base_prefix', str(tmp_path))
        patch.setattr(tl2cgen_compat, 'import_module', import_module)
        assert tl2cgen_compat._import_tl2cgen() is imported


@pytest.mark.skipif(sys.platform != 'win32', reason='Requires the actual Windows DLL loader')
def test_windows_fresh_import_preserves_dll_loader():
    subprocess.run([sys.executable, '-c', """
import builtins
import os
import sys
original = os.add_dll_directory
original_import = builtins.__import__
conda_bins = {os.path.join(prefix, 'Library', 'bin') for prefix in (sys.prefix, sys.base_prefix)}
existed = {path: os.path.exists(path) for path in conda_bins}
import py_boost
import tl2cgen
assert os.add_dll_directory is original
assert builtins.__import__ is original_import
assert all(type(finder).__module__ != 'py_boost.utils.tl2cgen_compat'
           for finder in sys.meta_path)
assert {path: os.path.exists(path) for path in conda_bins} == existed
print(tl2cgen.__version__)
"""], check=True)
