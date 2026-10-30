import os
from collections import deque
from itertools import count

import joblib
import numpy as np
from tqdm import tqdm

from .tl2cgen_compat import tl2cgen, treelite
from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation


def parse_pb_tree(tree, builder):
    """Append a single-group Py-Boost tree to a Treelite 4 model builder."""
    if tree.ngroups != 1:
        raise ValueError("Treelite conversion requires target_splitter='Single'")

    # Py-Boost uses negative child indices for leaves. Assign fresh, nonnegative
    # Treelite keys to both split and leaf nodes, including repeated leaf indices.
    node_ids = count(1)
    nodes = deque([(0, 0)])
    builder.start_tree()
    while nodes:
        old_id, new_id = nodes.popleft()
        builder.start_node(new_id)
        if old_id < 0:
            values = np.asarray(tree.values[-old_id - 1]).reshape(-1).tolist()
            builder.leaf(values[0] if len(values) == 1 else values)
        else:
            feature, threshold, left, right = tree.test_format[old_id * 4:old_id * 4 + 4]
            left_id, right_id = next(node_ids), next(node_ids)
            builder.numerical_test(
                feature_id=abs(int(feature)) - 1,
                threshold=float(threshold),
                default_left=bool(feature < 0),
                opname='<=',
                left_child_key=left_id,
                right_child_key=right_id,
            )
            nodes.extend([(int(left), left_id), (int(right), right_id)])
        builder.end_node()
    builder.end_tree()


def convert_pb_to_treelite(model):
    """Return a Treelite 4 builder containing the ensemble's raw scores.

    All outputs are represented as regression targets. The original Py-Boost
    postprocess_fn handles the loss-specific transform, including classification,
    so custom losses and postprocessors retain their existing behavior.
    """
    if not model.models:
        raise ValueError('Cannot convert an ensemble without trees')
    nout = model.models[0].values.shape[1]
    num_tree = len(model.models) + 1
    builder = ModelBuilder(
        threshold_type='float32',
        leaf_output_type='float32',
        metadata=Metadata(
            num_feature=int(model.nfeats),
            task_type='kRegressor',
            average_tree_output=False,
            num_target=nout,
            num_class=[1] * nout,
            leaf_vector_shape=(nout, 1),
        ),
        tree_annotation=TreeAnnotation(
            num_tree=num_tree,
            target_id=[-1 if nout > 1 else 0] * num_tree,
            class_id=[0] * num_tree,
        ),
        postprocessor=PostProcessorFunc(name='identity'),
        base_scores=[0.0] * nout,
    )
    # TL2cgen 1.0 applies base_scores[0] to every regression target. Store the
    # vector intercept as a constant tree instead, before the boosting trees,
    # matching Py-Boost's accumulation order in both GTIL and compiled inference.
    bias = np.asarray(model.base_score).reshape(-1).tolist()
    builder.start_tree()
    builder.start_node(0)
    builder.leaf(bias[0] if nout == 1 else bias)
    builder.end_node()
    builder.end_tree()
    for tree in tqdm(model.models):
        parse_pb_tree(tree, builder)
    return builder


def _reshape_prediction(pred):
    """Restore Py-Boost's (n_samples, n_outputs) contract, including empty batches."""
    return pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2])


class TLCompiledPredictor:
    """
    Compiled TL2cgen model for CPU inference
    """

    @staticmethod
    def _default_postprocess_fn(x):
        return x

    def __init__(self, libpath, nthread=None, verbose=False, postprocess_fn=None):
        """

        Args:
            libpath: str, path to compiled model
            nthread: int or None, number of threads to use
            verbose: bool, verbosity mode
            postprocess_fn: Callable or None, prediction postprocessing function
        """
        self.verbose = verbose
        self.nthread = nthread
        self.libpath = None
        self.set_libpath(libpath)

        self.postprocess_fn = self._default_postprocess_fn
        if postprocess_fn is not None:
            self.postprocess_fn = postprocess_fn

    def predict(self, X):
        """Make prediction

        Args:
            X: np.ndarray

        Returns:
            np.ndarray
        """
        X = np.asarray(X, dtype=np.float32, order="C")
        pred = self.predictor.predict(tl2cgen.DMatrix(X))
        return self.postprocess_fn(_reshape_prediction(pred))

    def set_libpath(self, libpath=None, nthread=None):
        """Update library path

        Args:
            libpath:
            nthread: int, num threads

        Returns:

        """
        if libpath is None:
            libpath = self.libpath
        if nthread is None:
            nthread = self.nthread
        self.libpath = os.path.abspath(libpath)
        self.nthread = nthread
        self.predictor = tl2cgen.Predictor(self.libpath, nthread=nthread, verbose=self.verbose)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["predictor"] = None
        return state

    def dump(self, filename):
        """Dump instance

        Args:
            filename: str, path to save

        Returns:

        """
        joblib.dump(self, filename)

    @staticmethod
    def load(filename, libpath=None, nthread=None):
        """Load instance

        Args:
            filename: str, filename
            libpath: str or None, override the saved compiled library path
            nthread: int or None, override the saved number of worker threads

        Returns:
            TLCompiledPredictor
        """
        predictor = joblib.load(filename)
        predictor.set_libpath(libpath=libpath, nthread=nthread)

        return predictor


class TLPredictor:
    """
    Treelite predictor. Could be use for inference via built-in treelite utils
    or to compilation to get TLCompiledPredictor
    """

    def __init__(self, model, postprocess_fn=None):
        """

        Args:
            model: Py-Boost Ensemble
            postprocess_fn: Callable or None, postprocessing function
        """
        model.to_cpu()
        self.tl_model = convert_pb_to_treelite(model).commit()

        self.postprocess_fn = postprocess_fn
        if postprocess_fn is None:
            self.postprocess_fn = model.postprocess_fn

    def set_tl_model(self, tl_model):
        """Update underlying treelite model

        Args:
            tl_model:

        Returns:

        """
        self.tl_model = tl_model

    def compile(
            self,
            toolchain,
            libpath,
            params=None,
            compiler='ast_native',
            verbose=False,
            nthread=None,
            options=None,
            predictor_params=None
    ):
        """Compile a CPU inference library with TL2cgen.

        See https://tl2cgen.readthedocs.io/en/latest/api.html#tl2cgen.export_lib

        Args:
            toolchain:
            libpath:
            params:
            compiler: retained for compatibility; only 'ast_native' is supported
            verbose:
            nthread:
            options:
            predictor_params:

        Returns:

        """

        if compiler != 'ast_native':
            raise ValueError("TL2cgen only supports compiler='ast_native'")
        params = {} if params is None else params
        params = {**{'parallel_comp': os.cpu_count() or 1}, **params}

        tl2cgen.export_lib(
            self.tl_model, toolchain=toolchain, libpath=libpath, params=params,
            verbose=verbose, nthread=nthread, options=options,
        )

        if predictor_params is None:
            predictor_params = {}
        predictor_params = {**{'nthread': nthread}, **predictor_params}

        predictor = TLCompiledPredictor(libpath, postprocess_fn=self.postprocess_fn, **predictor_params)
        return predictor

    def predict(self, X, nthread=None):
        """Make prediction

        Args:
            X: np.ndarray
            nthread: int/None, used for prediction

        Returns:
            np.ndarray
        """
        if nthread is None:
            # Let Treelite respect OpenMP's thread limit (for example in CI).
            nthread = -1
        X = np.asarray(X, dtype=np.float32, order="C")
        pred = treelite.gtil.predict(self.tl_model, X, nthread=nthread)
        return self.postprocess_fn(_reshape_prediction(pred))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tl_model"] = None
        return state

    def dump(self, dirname, rewrite=False):
        """Dump treelite Model and predictor instance

        Args:
            dirname: str, path to save
            rewrite: bool, possible to overwrite

        Returns:

        """
        os.makedirs(dirname, exist_ok=rewrite)
        self.tl_model.serialize(os.path.join(dirname, 'model.mod'))
        joblib.dump(self, os.path.join(dirname, 'predictor.pkl'))

    @staticmethod
    def load(dirname):
        """Load predictor from folder

        Args:
            dirname: str, path

        Returns:
            TLPredictor
        """
        predictor = joblib.load(os.path.join(dirname, 'predictor.pkl'))
        predictor.set_tl_model(treelite.Model.deserialize(os.path.join(dirname, 'model.mod')))

        return predictor
