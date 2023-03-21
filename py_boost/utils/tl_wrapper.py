import os

import joblib
from tqdm import tqdm
import treelite
import treelite_runtime as tl_run


def _create_node_deprecated(tree, node_id):
    """(DEPRECATED) Create a node of treelite tree
    Args:
        tree: Py-Boost Tree, tree to parse
        node_id: int, node index
    Returns:
        dict, args of treelite.ModelBuilder.Tree .set_numerical_test_node or .set_leaf_node
    """
    feature_id = tree.feats[0][node_id]

    if feature_id >= 0:
        left, right = tuple(tree.split[0][node_id])
        node = {

            'feature_id': feature_id,
            'opname': '<=',
            'threshold': tree.val_splits[0][node_id],
            'default_left': tree.nans[0][node_id],
            'left_child_key': left,
            'right_child_key': right,
        }

        return node, left, right

    return {'value': tree.values[tree.leaves[node_id][0]]}, None, None


def create_node(tree, node_id, id_gen):
    """Create a node of treelite tree

    Args:
        tree: Py-Boost Tree, tree to parse
        node_id: int, node index in original format tree
        id_gen: generator, new id generator

    Returns:
        dict, args of treelite.ModelBuilder.Tree .set_numerical_test_node
    """

    assert node_id >= 0

    feature_id = int(tree.test_format[node_id * 4])
    nan_left = feature_id < 0
    feature_id = abs(feature_id) - 1

    left = int(tree.test_format[node_id * 4 + 2])
    right = int(tree.test_format[node_id * 4 + 3])
    new_id_left = next(id_gen)
    new_id_right = next(id_gen)
    node = {
        'feature_id': feature_id,
        'opname': '<=',
        'threshold': tree.test_format[node_id * 4 + 1],
        'default_left': nan_left,
        'left_child_key': new_id_left,
        'right_child_key': new_id_right,
    }

    return node, (left, new_id_left), (right, new_id_right)


def parse_pb_tree(tree):
    """Parse s single Py-Boost Tree to treelite.ModelBuilder.Tree format

    Args:
        tree: Py-Boost tree

    Returns:
        treelite.ModelBuilder.Tree
    """
    assert tree.ngroups == 1, 'Models with more than 1 group are not currently supported'

    def id_generator():
        id_num = 1
        while True:
            yield id_num
            id_num += 1
    id_gen = id_generator()

    tl_tree = treelite.ModelBuilder.Tree()
    curr_nodes = [(0, 0)]  # (old_id, new_id)

    while len(curr_nodes) > 0:
        old_id, new_id = curr_nodes.pop(0)
        curr_node, left, right = create_node(tree, old_id, id_gen)
        tl_tree[new_id].set_numerical_test_node(**curr_node)

        if left[0] >= 0:
            curr_nodes.append(left)
        else:
            tl_tree[left[1]].set_leaf_node(tree.values[abs(left[0]) - 1])
        if right[0] >= 0:
            curr_nodes.append(right)
        else:
            tl_tree[right[1]].set_leaf_node(tree.values[abs(right[0]) - 1])

    tl_tree[0].set_root()
    return tl_tree


def _parse_pb_tree_deprecated(tree):
    """Parse s single Py-Boost Tree to treelite.ModelBuilder.Tree format

    Args:
        tree: Py-Boost tree

    Returns:
        treelite.ModelBuilder.Tree
    """
    assert tree.ngroups == 1, 'Models with more than 1 group are not currently supported'

    tl_tree = treelite.ModelBuilder.Tree()
    curr_nodes = [0]

    while len(curr_nodes) > 0:

        next_nodes = []

        for node_id in curr_nodes:

            curr_node, left, right = _create_node_deprecated(tree, node_id)
            # add node
            tl_tree[node_id]
            if left is not None:
                tl_tree[node_id].set_numerical_test_node(**curr_node)
                next_nodes.extend([left, right])
            else:
                tl_tree[node_id].set_leaf_node(curr_node['value'])

        curr_nodes = next_nodes

    tl_tree[0].set_root()

    return tl_tree


def convert_pb_to_treelite(model):
    """Convert Py-Boost Ensemble instance to the treelite.ModelBuilder.Tree

    Args:
        model: Py-Boost Tree

    Returns:
        treelite.ModelBuilder.Tree
    """
    nfeats = model.nfeats
    ngroups = model.models[0].values.shape[1]

    builder = treelite.ModelBuilder(
        num_feature=nfeats,
        num_class=ngroups,
        pred_transform='identity_multiclass' if ngroups > 1 else 'identity'
    )

    for tree in tqdm(model.models):
        builder.append(parse_pb_tree(tree))

    # add bias tree
    bias_tree = treelite.ModelBuilder.Tree()
    bias_tree[0].set_numerical_test_node(**{

        'feature_id': 0,
        'opname': '<',
        'threshold': 0,
        'default_left': True,
        'left_child_key': 1,
        'right_child_key': 2
    })

    for i in range(1, 3):
        bias_tree[i].set_leaf_node(model.base_score)

    bias_tree[0].set_root()
    builder.append(bias_tree)

    return builder


class TLCompiledPredictor:
    """
    Compiled treelite model saved to predict
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
        pred = self.predictor.predict(tl_run.DMatrix(X))
        return self.postprocess_fn(pred)

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
        self.predictor = tl_run.Predictor(self.libpath, nthread=nthread, verbose=self.verbose)

    def dump(self, filename):
        """Dump instance

        Args:
            filename: str, path to save

        Returns:

        """
        self.predictor = None
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        """Load instance

        Args:
            filename: str, filename

        Returns:
            TLCompiledPredictor
        """
        predictor = joblib.load(filename)
        predictor.set_libpath()

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
        """Compile model for faster inference. For the details please see
        https://treelite.readthedocs.io/en/latest/tutorials/first.html

        Args:
            toolchain:
            libpath:
            params:
            compiler:
            verbose:
            nthread:
            options:
            predictor_params:

        Returns:

        """

        params = {} if params is None else params
        params = {**{'parallel_comp': os.cpu_count(), }, **params}

        self.tl_model.export_lib(toolchain, libpath,
                                 params, compiler, verbose, nthread, options)

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
            nthread = os.cpu_count()
        pred = treelite.gtil.predict(self.tl_model, X, nthread=nthread)
        return self.postprocess_fn(pred)

    def dump(self, dirname, rewrite=False):
        """Dump treelite Model and predictor instance

        Args:
            dirname: str, path to save
            rewrite: bool, possible to overwrite

        Returns:

        """
        os.makedirs(dirname, exist_ok=rewrite)
        temp = self.tl_model
        self.tl_model = None
        temp.serialize(os.path.join(dirname, 'model.mod'))
        joblib.dump(self, os.path.join(dirname, 'predictor.pkl'))
        self.tl_model = temp

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
