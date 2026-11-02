"""CPU integration tests using real Treelite models and compiled TL2cgen libraries."""

import copy
import os
import shutil
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from scipy import sparse

from py_boost import TLCompiledPredictor, TLPredictor


def identity(prediction):
    return prediction


def sigmoid(prediction):
    return 1 / (1 + np.exp(-prediction))


def softmax(prediction):
    exp = np.exp(prediction - prediction.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class CPUEnsemble:
    """Small, CPU-resident ensemble with independently specified expected outputs."""

    def __init__(self, nout, postprocess_fn=identity):
        self.nfeats = 2
        self.base_score = np.array([0.75, -1.5, 3], dtype=np.float32)[:nout]
        self.postprocess_fn = postprocess_fn
        self.models = [
            SimpleNamespace(
                ngroups=1,
                # x0 <= .5 or missing: test x1 <= -.25 (missing goes right).
                # Otherwise, use leaf 2. Negative children encode leaf indices.
                test_format=np.array([-1, .5, 1, -3, 2, -.25, -1, -2], dtype=np.float32),
                values=np.array([[1, -2, .5], [3, 4, -1], [-5, 6, 2]],
                                dtype=np.float32)[:, :nout],
            ),
            SimpleNamespace(
                ngroups=1,
                # Py-Boost represents a constant tree as a split with equal leaves.
                test_format=np.array([1, 0, -1, -1], dtype=np.float32),
                values=np.array([[.25, -.5, 2]], dtype=np.float32)[:, :nout],
            ),
        ]

    def to_cpu(self):
        pass


@pytest.fixture(scope='module', params=[1, 3], ids=['scalar', 'vector'])
def ensemble(request):
    return CPUEnsemble(request.param)


@pytest.fixture
def examples():
    X = np.array([
        [0, -1], [0, 0], [1, -1], [np.nan, -1], [np.nan, np.nan],
        [.5, -.25], [-np.inf, np.inf], [np.inf, np.nan], [0, np.nan],
    ], dtype=np.float32)
    # Computed from the tree's decision rules, including equality and NaN routing.
    expected = np.array([
        [2, -4, 5.5], [4, 2, 4], [-4, 4, 7], [2, -4, 5.5], [4, 2, 4],
        [2, -4, 5.5], [4, 2, 4], [-4, 4, 7], [4, 2, 4],
    ], dtype=np.float32)
    return X, expected


@pytest.fixture(scope='module')
def toolchain():
    if sys.platform == 'win32':
        name, suffix = 'msvc', '.dll'
        executable = 'cl'
    else:
        name, suffix = 'gcc', '.dylib' if sys.platform == 'darwin' else '.so'
        executable = 'gcc'
    if shutil.which(executable) is None:
        if os.environ.get('CI'):
            pytest.fail(f'Compiled inference tests require {executable} in CI')
        pytest.skip(f'Compiled inference tests require {executable}')
    return name, suffix


@pytest.fixture(scope='module')
def compiled(ensemble, tmp_path_factory, toolchain):
    name, suffix = toolchain
    tmp_path = tmp_path_factory.mktemp('compiled')
    return TLPredictor(ensemble).compile(
        name, tmp_path / ('model' + suffix), params={'parallel_comp': 2},
        nthread=2, predictor_params={'nthread': 1},
    )


@pytest.fixture(scope='module')
def sparse_compiled(ensemble, tmp_path_factory, toolchain):
    model = CPUEnsemble(len(ensemble.base_score))
    # Explicit zero goes left; a missing sparse entry goes right at this node.
    model.models[0].test_format[5] = .25
    name, suffix = toolchain
    path = tmp_path_factory.mktemp('sparse_compiled') / ('model' + suffix)
    return TLPredictor(model).compile(name, path, params={'parallel_comp': 1}, nthread=2)


@pytest.mark.parametrize('format', ['csr', 'csc', 'coo'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_compiled_sparse_inputs(ensemble, sparse_compiled, format, dtype, index_dtype,
                               monkeypatch):
    # Keep explicit zeros. Unstored entries represent missing values, not zeros.
    X = sparse.csr_matrix((np.array([0, 0, 1, -1, 0, -1, 0, np.nan, np.nan, -1], dtype=dtype),
                           [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                           [0, 1, 2, 3, 4, 4, 6, 8, 10]), shape=(8, 2)).asformat(format)
    if format == 'coo':
        X.row = X.row.astype(index_dtype)
        X.col = X.col.astype(index_dtype)
    else:
        X.indices = X.indices.astype(index_dtype)
        X.indptr = X.indptr.astype(index_dtype)
    original = copy.deepcopy(X)

    def no_dense(*args, **kwargs):
        pytest.fail('Sparse prediction must not densify its input')

    for cls in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
        monkeypatch.setattr(cls, 'toarray', no_dense)

    expected = np.array([
        [2, -4, 5.5], [4, 2, 4], [-4, 4, 7], [2, -4, 5.5],
        [4, 2, 4], [2, -4, 5.5], [4, 2, 4], [2, -4, 5.5],
    ], dtype=np.float32)[:, :len(ensemble.base_score)]
    actual = sparse_compiled.predict(X)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(X.data, original.data)
    fields = ('row', 'col') if format == 'coo' else ('indices', 'indptr')
    for field in fields:
        actual_buffer, original_buffer = getattr(X, field), getattr(original, field)
        assert actual_buffer.dtype == original_buffer.dtype
        np.testing.assert_array_equal(actual_buffer, original_buffer)


def test_shallow_copy_keeps_native_predictors(ensemble, compiled, examples):
    X, expected = examples
    expected = expected[:, :len(ensemble.base_score)]
    for predictor in (TLPredictor(ensemble), compiled):
        cloned = copy.copy(predictor)
        cloned.postprocess_fn = sigmoid
        np.testing.assert_allclose(cloned.predict(X), sigmoid(expected))
        np.testing.assert_allclose(predictor.predict(X), expected)


@pytest.mark.parametrize('batch_size', [0, 1])
def test_sparse_batch_keeps_output_shape(ensemble, sparse_compiled, batch_size):
    X = sparse.csr_matrix((np.zeros(2, dtype=np.float32), [0, 1], [0, 2]), shape=(1, 2))
    actual = sparse_compiled.predict(X[:batch_size])
    expected = np.array([[2, -4, 5.5]], dtype=np.float32)[:batch_size, :len(ensemble.base_score)]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('layout', ['float32', 'float64', 'fortran', 'strided', 'list'])
def test_gtil_and_compiled_predictions(ensemble, compiled, examples, layout):
    X, expected = examples
    if layout == 'float64':
        X = X.astype(np.float64)
    elif layout == 'fortran':
        X = np.asfortranarray(X)
    elif layout == 'strided':
        buffer = np.zeros((len(X), 4), dtype=np.float32)
        buffer[:, ::2] = X
        X = buffer[:, ::2]
    elif layout == 'list':
        X = X.tolist()
    expected = expected[:, :len(ensemble.base_score)]
    for predictor in (TLPredictor(ensemble), compiled):
        actual = predictor.predict(X)
        assert type(actual) is np.ndarray
        assert actual.shape == expected.shape
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('batch_size', [0, 1])
def test_prediction_keeps_batch_and_output_axes(ensemble, compiled, examples, batch_size):
    X, expected = examples
    expected = expected[:batch_size, :len(ensemble.base_score)]
    for predictor in (TLPredictor(ensemble), compiled):
        actual = predictor.predict(X[:batch_size])
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('postprocess_fn', [sigmoid, softmax])
def test_postprocess_receives_2d_raw_scores(examples, tmp_path, toolchain, postprocess_fn):
    X, raw_expected = examples
    model = CPUEnsemble(3, postprocess_fn=postprocess_fn)
    predictor = TLPredictor(model)
    name, suffix = toolchain
    compiled = predictor.compile(name, tmp_path / ('model' + suffix),
                                 params={'parallel_comp': 1}, nthread=2)
    expected = postprocess_fn(raw_expected)
    for instance in (predictor, compiled):
        np.testing.assert_allclose(instance.predict(X), expected, rtol=1e-6, atol=1e-6)
        assert instance.predict(X[:1]).shape == (1, 3)


@pytest.mark.parametrize('loss_name,transform', [
    ('MSELoss', identity), ('MSLELoss', np.expm1),
    ('BCELoss', sigmoid), ('CrossEntropyLoss', softmax),
])
def test_builtin_postprocessors_on_cpu(examples, tmp_path, toolchain, loss_name, transform):
    # Exercise the actual bound methods stored by GPU-trained model dumps. These
    # must also import, unpickle and run in an environment without CuPy installed.
    from py_boost.gpu.losses import losses

    postprocess_fn = getattr(losses, loss_name)().postprocess_output
    predictor = TLPredictor(CPUEnsemble(3, postprocess_fn=postprocess_fn))
    predictor.dump(tmp_path / 'gtil')
    restored = TLPredictor.load(tmp_path / 'gtil')
    name, suffix = toolchain
    compiled = predictor.compile(name, tmp_path / ('model' + suffix),
                                 params={'parallel_comp': 1}, nthread=2)
    compiled.dump(tmp_path / 'compiled.pkl')
    restored_compiled = TLCompiledPredictor.load(tmp_path / 'compiled.pkl')
    X, expected = examples
    for instance in (predictor, restored, compiled, restored_compiled):
        actual = instance.predict(X)
        assert type(actual) is np.ndarray
        np.testing.assert_allclose(actual, transform(expected), rtol=1e-6, atol=1e-6)


def test_custom_postprocess_overrides_model(ensemble, examples):
    X, raw_expected = examples
    predictor = TLPredictor(ensemble, postprocess_fn=sigmoid)
    np.testing.assert_allclose(predictor.predict(X),
                               sigmoid(raw_expected[:, :len(ensemble.base_score)]))


def test_compile_with_default_options(examples, tmp_path, toolchain):
    X, expected = examples
    name, suffix = toolchain
    compiled = TLPredictor(CPUEnsemble(3)).compile(name, tmp_path / ('model' + suffix))
    np.testing.assert_allclose(compiled.predict(X), expected)


def test_gtil_serialization(ensemble, examples, tmp_path):
    X, expected = examples
    predictor = TLPredictor(ensemble, postprocess_fn=sigmoid)
    path = tmp_path / 'saved_model'
    predictor.dump(path)
    with pytest.raises(FileExistsError):
        predictor.dump(path)
    predictor.dump(path, rewrite=True)
    loaded = TLPredictor.load(path)
    for instance in (predictor, loaded):
        np.testing.assert_allclose(instance.predict(X),
                                   sigmoid(expected[:, :len(ensemble.base_score)]))


def test_compiled_serialization_and_relocation(compiled, examples, tmp_path, toolchain):
    X, expected = examples
    nout = compiled.predict(X).shape[1]
    filename = tmp_path / 'compiled.pkl'
    compiled.dump(filename)
    # Dump must leave the original predictor usable.
    np.testing.assert_allclose(compiled.predict(X), expected[:, :nout])
    loaded = TLCompiledPredictor.load(filename)
    assert loaded.nthread == 1  # predictor_params overrides compile's nthread.
    np.testing.assert_allclose(loaded.predict(X), expected[:, :nout])
    _, suffix = toolchain
    relocated = tmp_path / ('relocated' + suffix)
    # Copy rather than rename: Windows holds loaded DLLs open.
    shutil.copy2(loaded.libpath, relocated)
    loaded.set_libpath(relocated, nthread=2)
    loaded.dump(filename)
    loaded = TLCompiledPredictor.load(filename)
    assert loaded.nthread == 2
    assert loaded.libpath == os.path.abspath(relocated)
    np.testing.assert_allclose(loaded.predict(X), expected[:, :nout])

    # A deployment may have no file at the original machine's absolute path.
    metadata = joblib.load(filename)
    metadata.libpath = str(tmp_path / 'missing_library')
    metadata.dump(filename)
    loaded = TLCompiledPredictor.load(filename, libpath=relocated, nthread=1)
    assert loaded.nthread == 1
    np.testing.assert_allclose(loaded.predict(X), expected[:, :nout])


@pytest.mark.parametrize('compiled_predictor', [False, True])
def test_failed_dump_leaves_predictor_usable(ensemble, examples, tmp_path, toolchain,
                                           monkeypatch, compiled_predictor):
    predictor = TLPredictor(ensemble)
    if compiled_predictor:
        name, suffix = toolchain
        predictor = predictor.compile(name, tmp_path / ('model' + suffix),
                                      params={'parallel_comp': 1}, nthread=2)

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr('py_boost.utils.tl_wrapper.joblib.dump', fail)
    with pytest.raises(OSError, match='disk full'):
        predictor.dump(tmp_path / 'failed_dump')
    X, expected = examples
    np.testing.assert_allclose(predictor.predict(X), expected[:, :len(ensemble.base_score)])


def test_rejects_unsupported_compiler(ensemble, tmp_path):
    with pytest.raises(ValueError, match='ast_native'):
        TLPredictor(ensemble).compile('gcc', tmp_path / 'unused.so', compiler='failsafe')


def test_rejects_multiple_tree_groups():
    ensemble = CPUEnsemble(3)
    ensemble.models[0].ngroups = 2
    with pytest.raises(ValueError, match="target_splitter='Single'"):
        TLPredictor(ensemble)


def test_rejects_unfitted_model():
    ensemble = CPUEnsemble(1)
    ensemble.models = None
    with pytest.raises(ValueError, match='without trees'):
        TLPredictor(ensemble)
