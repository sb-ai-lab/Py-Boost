"""Regression coverage for the TL2cgen 1.0 Windows DLL-directory import failure."""

import os
import subprocess
import sys

import pytest

from py_boost.utils import tl2cgen_compat


def test_windows_import_ignores_only_missing_conda_directory(tmp_path, monkeypatch):
    conda_bin = os.path.join(str(tmp_path), 'Library', 'bin')
    base_prefix = str(tmp_path / 'base_python')
    base_conda_bin = os.path.join(base_prefix, 'Library', 'bin')
    other_missing = os.path.join(str(tmp_path), 'unrelated')
    calls = []
    imported = object()

    def add_directory(path):
        calls.append(path)
        raise FileNotFoundError(path)

    def import_module(name):
        assert name == 'tl2cgen'
        os.add_dll_directory(conda_bin)
        os.add_dll_directory(base_conda_bin)
        with pytest.raises(FileNotFoundError, match='unrelated'):
            os.add_dll_directory(other_missing)
        return imported

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'platform', 'win32')
        patch.setattr(sys, 'prefix', str(tmp_path))
        patch.setattr(sys, 'base_prefix', base_prefix)
        patch.setattr(os, 'add_dll_directory', add_directory, raising=False)
        patch.setattr(tl2cgen_compat, 'import_module', import_module)
        assert tl2cgen_compat._import_tl2cgen() is imported
        assert os.add_dll_directory is add_directory
    assert calls == [other_missing]
    assert not os.path.exists(conda_bin)
    assert not os.path.exists(base_conda_bin)


def test_windows_import_restores_dll_search_function_on_failure(tmp_path, monkeypatch):
    def add_directory(path):
        raise AssertionError('An unrelated DLL-directory call must not be made')

    def import_module(name):
        raise OSError('DLL dependency is missing')

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'platform', 'win32')
        patch.setattr(sys, 'prefix', str(tmp_path))
        patch.setattr(os, 'add_dll_directory', add_directory, raising=False)
        patch.setattr(tl2cgen_compat, 'import_module', import_module)
        with pytest.raises(OSError, match='DLL dependency is missing'):
            tl2cgen_compat._import_tl2cgen()
        assert os.add_dll_directory is add_directory


def test_windows_conda_import_keeps_existing_dll_directory(tmp_path, monkeypatch):
    conda_bin = tmp_path / 'Library' / 'bin'
    conda_bin.mkdir(parents=True)
    calls = []

    def add_directory(path):
        calls.append(path)

    def import_module(name):
        assert os.add_dll_directory is add_directory
        os.add_dll_directory(str(conda_bin))

    with monkeypatch.context() as patch:
        patch.setattr(sys, 'platform', 'win32')
        patch.setattr(sys, 'prefix', str(tmp_path))
        patch.setattr(sys, 'base_prefix', str(tmp_path))
        patch.setattr(os, 'add_dll_directory', add_directory, raising=False)
        patch.setattr(tl2cgen_compat, 'import_module', import_module)
        tl2cgen_compat._import_tl2cgen()
    assert calls == [str(conda_bin)]


@pytest.mark.skipif(sys.platform != 'win32', reason='Requires the actual Windows DLL loader')
def test_windows_fresh_import_preserves_dll_loader():
    subprocess.run([sys.executable, '-c', '''
import os
import sys
original = os.add_dll_directory
conda_bin = os.path.join(sys.prefix, 'Library', 'bin')
existed = os.path.exists(conda_bin)
import py_boost
import tl2cgen
assert os.add_dll_directory is original
assert os.path.exists(conda_bin) == existed
print(tl2cgen.__version__)
'''], check=True)
