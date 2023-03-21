"""Decision trees building and inference"""

try:
    import cupy as cp
except Exception:
    pass
import numpy as np

from .utils import apply_values, depthwise_grow_tree, get_tree_node, set_leaf_values, calc_node_values
from .utils import tree_prediction_leaves_typed_kernels, tree_prediction_leaves_typed_kernels_f
from .utils import tree_prediction_values_kernel


class Tree:
    """This class initializes an empty tree structure, implements methods to set tree values and single tree inference.
    The instance of this object represents the actual boosting step, but not the single tree!
    Actual amount of trees in the instance (at each boosting step) is defined by ngroups argument. What it means:
    Assume you have 5 class classification task, so you model output size equals 5. Possible cases here:
        - Build single decision tree that outputs a vector of 5 values. In this case ngroups eq. 1
        - Build 5 decision trees, each tree predict a value for its own class (one-vs-all).
            In this case ngroups eq. 5
        - Create custom target split strategy. For ex. you can build 2 trees, first will predict [0, 1, 2] classes,
            second - [3, 4]. In this case ngroups eq. 2

    Grouped trees structure is defined by arrays:
        feats, shape (ngroups, max_nodes) - feature index to use for the split in each group/node.
            If -1, the node is terminal (leaf)
        val_splits, shape (ngroups, max_nodes) - threshold to compare when choosing the next node
            if feature value is not NaN
        nans, shape (ngroups, max_nodes) - bool values, if True, NaN feature values objects moves left, else - right
        split, shape (ngroups, max_nodes, 2) - node indices corresponding left/right split for the current node

    Trees structure defines single node id value for each object
    Values assigned to the outputs are defined by arrays:
        group_index, shape (nout, ). Defines the group id for predicting each output
        values, shape (max_nodes, nout). Define output value for each node/output
        leaves, shape (max_leaves, ngroups). Assigns the leaf index to the terminal nodes

    During the fit stage, the format described above is used.
    After fitting, additional reformatting occurs that converts the tree to another format to achieve faster inference:
    - Sub-trees for each group are stored in one array named "test_format":
        [gr0_node0, ..., gr0_nodeN, gr1_node0, ..., gr1_nodeM, gr2_node0, ..., gr2_nodeK, gr3_node0, ...]
    - Each node in new formatted tree consists of 4 fields:
        [feature_index, split_value, left_node_index, right_node_index],
        feature_index - feature index to use for the split in each node.
        split_value - threshold to compare when choosing the next node if feature value is not NaN
        left_node_index - index of the left child in "test_format" array
        right_node_index - index of the right child in "test_format" array
    - The size of "test_format" array equals to the sum of all nodes in all subtrees except leaves multiplied by 4.
        Multiplication by 4 occurs because each node consists of the 4 fields described above.
        Examples:
            test_format[0 * 4] == test_format[0] - yields feature_index for node with index 0.
            test_format[0 * 4 + 1] == test_format[1] - yields split_value for node with index 0.
            test_format[0 * 4 + 2] == test_format[2] - yields left_node_index for node with index 0.
            test_format[0 * 4 + 3] == test_format[3] - yields right_node_index for node with index 0.
            test_format[1 * 4] == test_format[4]  - yields feature_index for node with index 1.
            test_format[1 * 4 + 1] == test_format[5] - yields split_value for node with index 1.
            test_format[1 * 4 + 2] == test_format[6] - yields left_node_index for node with index 1.
            test_format[1 * 4 + 3] == test_format[7] - yields right_node_index for node with index 1.
            test_format[2 * 4] == test_format[8]  - yields feature_index for node with index 2.
            ...
            test_format[79 * 4] == test_format[316]  - yields feature_index for node with index 79.
            test_format[79 * 4 + 1] == test_format[317] - yields split_value for node with index 79.
            test_format[79 * 4 + 2] == test_format[318] - yields left_node_index for node with index 79.
            test_format[79 * 4 + 3] == test_format[319] - yields right_node_index for node with index 79.
            ...
    - The sign of the feature_index value shows the behavior in case of feature == NaN (split to the left/right),
        to the value written in feature_index an extra "1" is added to deal with zero.
        Examples:
            feature_index == 8, positive value means that tree follows to the left in case of NaN in feature,
                the real feature index is calculated as follows: abs(8) - 1 = 7.
            feature_index == -19, negative value means that tree follows to the right in case of NaN in feature,
                the real feature index is calculated as follows: abs(-19) - 1 = 18.
            feature_index == 0, impossible due to construction algorithm.
    - If left_node_index/right_node_index is negative, it means that it shows index in the values array;
        In case of a negative value, an extra "1" is added to deal with zero.
        Examples:
            left_node_index == 8, non-negative value means that left child node is stored in "test_format" with index 8;
            left_node_index == -13, means that left child is a leaf, the index in "values" array for that leaf can
                be calculated as follows: abs(-13) - 1 = 12. Thus, index in "values" array is 12.
    - All subtrees are stored in one array, so an additional array of indexes where each subtree is starting
        is required (index of the subtree roots), array "gr_subtree_offsets" stores these indexes,
        size of "gr_subtree_offsets" equals to number of groups in the tree (number of subtrees).
        Example:
            gr_subtree_offsets == [0, 56, 183], means that tree has 3 subtrees (3 groups).
            The first subtree has its root as node with index 0;
            The second subtree has its root as node with index 56;
            The third subtree has its root as node with index 183.
            Example how to access the values of the root node in the second subtree:
                test_format[56 * 4] == test_format[224]  - yields feature_index for the root of the second subtree;
                test_format[56 * 4 + 1] == test_format[225] - yields split_value for the root of the second subtree;
                test_format[56 * 4 + 2] == test_format[226] - yields left_node_index for the root of the second subtree;
                test_format[56 * 4 + 3] == test_format[227] - yields right_node_index for the root of the second subtree
    - Two fields, 'feature_importance_gain' and 'feature_importance_split', store feature importance arrays
        and describe the fitted tree accordingly.
    """

    def __init__(self, max_nodes, nout, ngroups):
        """Initialize empty tree

        Args:
            max_nodes: int, maximum number of nodes in tree
            nout: int, number of outputs in tree
            ngroups: int, number of groups
        """
        self.nout = nout
        self.ngroups = ngroups
        self.max_nodes = max_nodes

        self.gains = np.zeros((ngroups, max_nodes,), dtype=np.float32)
        self.feats = np.zeros((ngroups, max_nodes,), dtype=np.int64) - 1
        self.bin_splits = np.zeros((ngroups, max_nodes,), dtype=np.int32)
        self.nans = np.zeros((ngroups, max_nodes,), dtype=np.bool_)

        self.split = np.zeros((ngroups, max_nodes, 2), dtype=np.int32)

        self.val_splits = None
        self.values = None
        self.group_index = None
        self.leaves = None
        self.max_leaves = None

        self.feature_importance_gain = None
        self.feature_importance_split = None

        self._debug = None
        self.test_format = None
        self.test_format_offsets = None

    def set_nodes(self, group, unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left):
        """Write info about new nodes

        Args:
            group: int, group id to write
            unique_nodes: np.ndarray, nodes id to set info
            new_nodes_id: np.ndarray, nodes id to left/right split current node
            best_feat: np.ndarray, feature value to perform a split
            best_gain: np.ndarray, gain from the split
            best_split: np.ndarray, quantized threshold to compare when split
            best_nan_left: np.ndarray, bool if True, nans moved in the left node, else right

        Returns:

        """

        self.gains[group, unique_nodes] = best_gain
        self.feats[group, unique_nodes] = best_feat
        self.bin_splits[group, unique_nodes] = best_split
        self.nans[group, unique_nodes] = best_nan_left
        self.split[group, unique_nodes] = new_nodes_id

    def set_node_values(self, values, group_index):
        """Assign output values for each nodes

        Args:
            values: np.ndarray, node values
            group_index: np.ndarray, group id of each output

        Returns:

        """
        self.values = values
        self.group_index = group_index

    def set_borders(self, borders):
        """Assign actual feature values based on quantized

        Args:
            borders: list of np.ndarray, actual node values

        Returns:

        """
        # borders - list of arrays. Array is borderlines
        val_splits = [0 if x == -1 else borders[x][min(y, len(borders[x]) - 1)]
                      for (x, y) in zip(self.feats.ravel(), self.bin_splits.ravel())]
        self.val_splits = np.array(val_splits, dtype=np.float32).reshape(self.feats.shape)

    def set_leaves(self):
        """Assign leaf id to the terminal nodes

        Returns:

        """
        self.leaves, self.max_leaves = set_leaf_values(self.feats, self.split)

    def to_device(self):
        """Move tree data to the current GPU memory

        Returns:

        """
        for attr in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'values', 'group_index', 'leaves',
                     'test_format', 'test_format_offsets', 'feature_importance_gain', 'feature_importance_split']:
            arr = getattr(self, attr)

            if type(arr) is np.ndarray:
                setattr(self, attr, cp.asarray(arr))

    def to_cpu(self):
        """Move tree data to the CPU memory

        Returns:

        """
        for attr in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'values', 'group_index', 'leaves',
                     'test_format', 'test_format_offsets', 'feature_importance_gain', 'feature_importance_split']:
            arr = getattr(self, attr)

            if type(arr) is cp.ndarray:
                setattr(self, attr, arr.get())

    def _predict_node_deprecated(self, X):
        """(DEPRECATED) Predict node id from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:

        """
        if self.feats is None:
            raise Exception('To use _deprecated funcs pass debug=True to .reformat')

        assert type(self.feats) is cp.ndarray, 'Should be moved to GPU first. Call .to_device()'
        nodes = get_tree_node(X, self.feats, self.val_splits, self.split, self.nans)
        return nodes

    def _predict_from_nodes_deprecated(self, nodes):
        """(DEPRECATED) Predict outputs from the nodes indices

        Args:
            nodes: cp.ndarray of predicted nodes

        Returns:
            cp.ndarray of nodes
        """
        return apply_values(nodes, self.group_index, self.values)

    def _predict_leaf_from_nodes_deprecated(self, nodes):
        """Predict leaf indices from the nodes indices (Use predict_leaf() method if you need to predict leaves)

        Args:
            nodes: cp.ndarray of predicted nodes

        Returns:
            cp.ndarray of leaves
        """
        return apply_values(nodes, cp.arange(self.ngroups, dtype=cp.uint64), self.leaves)

    def _predict_deprecated(self, X):
        """(DEPRECATED) Predict from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:
            cp.ndarray of predictions
        """
        return self._predict_from_nodes_deprecated(
            self._predict_leaf_from_nodes_deprecated(self._predict_node_deprecated(X)))

    def _predict_leaf_deprecated(self, X):
        """(DEPRECATED) Predict leaf indices from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:
            cp.ndarray of leaves
        """
        return self._predict_leaf_from_nodes_deprecated(self._predict_node_deprecated(X))

    def predict_leaf(self, X, pred_leaves=None):
        """Predict leaf indexes from the feature matrix X

        Args:
            X: cp.ndarray, array of features
            pred_leaves: cp.ndarray, buffer for predictions

        Returns:
            pred_leaves: leaf predictions

        """
        # check if buffer is None and X on GPU
        assert type(X) is cp.ndarray, "X must be type of cp.ndarray (located on gpu)"

        dt = str(X.dtype)

        assert dt in tree_prediction_leaves_typed_kernels, \
            f"X array must be of type: {list(tree_prediction_leaves_typed_kernels.keys())}"

        if pred_leaves is None:
            pred_leaves = cp.empty((X.shape[0], self.ngroups), dtype=cp.int32)

        # CUDA parameters initialization
        threads = 128  # threads in one CUDA block
        sz = X.shape[0] * self.ngroups
        blocks = sz // threads
        if sz % threads != 0:
            blocks += 1

        if X.flags["C_CONTIGUOUS"]:
            tree_prediction_leaves_typed_kernels[dt]((blocks,), (threads,), ((X,
                                                                              self.test_format,
                                                                              self.test_format_offsets,
                                                                              X.shape[1],
                                                                              X.shape[0],
                                                                              self.ngroups,
                                                                              pred_leaves.shape[1],
                                                                              pred_leaves)))
        elif X.flags["F_CONTIGUOUS"]:
            tree_prediction_leaves_typed_kernels_f[dt]((blocks,), (threads,), ((X,
                                                                                self.test_format,
                                                                                self.test_format_offsets,
                                                                                X.shape[1],
                                                                                X.shape[0],
                                                                                self.ngroups,
                                                                                pred_leaves.shape[1],
                                                                                pred_leaves)))
        else:
            raise Exception("X must be 'C_CONTIGUOUS' or 'F_CONTIGUOUS'")
        return pred_leaves

    def predict(self, X, pred=None, pred_leaves=None):
        """Predict from the feature matrix X

        Args:
            X: cp.ndarray, array of features
            pred: cp.ndarray, buffer for predictions on GPU, if None - created automatically
            pred_leaves: cp.ndarray, buffer for internal leaf predictions on GPU, if None - created automatically

        Returns:
            pred: cp.ndarray, prediction array

        """
        # check if buffers are None
        if pred is None:
            pred = cp.zeros((X.shape[0], self.nout), dtype=cp.float32)
        if pred_leaves is None:
            pred_leaves = cp.empty((X.shape[0], self.ngroups), dtype=cp.int32)

        # first step - leaves predictions, actually prediction of indexes in values
        self.predict_leaf(X, pred_leaves)

        # CUDA parameters initialization
        threads = 128  # threads in one CUDA block
        sz = X.shape[0] * self.nout
        blocks = sz // threads
        if sz % threads != 0:
            blocks += 1

        # second step, prediction of actual values
        tree_prediction_values_kernel((blocks,), (threads,), ((pred_leaves,
                                                               self.group_index,
                                                               self.values,
                                                               self.nout,
                                                               X.shape[0],
                                                               pred_leaves.shape[1],
                                                               pred)))
        return pred

    def reformat(self, nfeats, debug):
        """Creates new internal format of the tree for faster inference
        
        Args:
            nfeats: int, number of features in X (train set)
            debug: bool, if in debug mode

        Returns:

        """
        n_gr = self.ngroups

        # memory allocation for new tree array
        gr_subtree_offsets = np.zeros(n_gr, dtype=np.int32)
        check_empty = []
        total_size = 0
        for i in range(n_gr):
            curr_size = int((self.feats[i] >= 0).sum())
            # add special case handling - single leaf, no splits
            check_empty.append(curr_size == 0)
            curr_size = max(1, curr_size)
            total_size += curr_size

            if i < n_gr - 1:
                gr_subtree_offsets[i + 1] = total_size
        nf = np.zeros(total_size * 4, dtype=np.float32)

        # reformatting the tree
        for i in range(n_gr):
            # handle special case - single leaf, no splits - make a pseudo split node
            if check_empty[i]:
                nf[4 * gr_subtree_offsets[i]] = 1.
                nf[4 * gr_subtree_offsets[i] + 1] = 0.
                nf[4 * gr_subtree_offsets[i] + 2] = -1.
                nf[4 * gr_subtree_offsets[i] + 3] = -1.

                continue

            q = [(0, 0)]

            while len(q) != 0:  # BFS in tree
                n_old, n_new = q[0]
                if not self.nans[i][n_old]:
                    nf[4 * (gr_subtree_offsets[i] + n_new)] = float(self.feats[i][n_old] + 1)
                else:
                    nf[4 * (gr_subtree_offsets[i] + n_new)] = float(-(self.feats[i][n_old] + 1))
                nf[4 * (gr_subtree_offsets[i] + n_new) + 1] = float(self.val_splits[i][n_old])
                ln = self.split[i][n_old][0]
                rn = self.split[i][n_old][1]

                if self.feats[i][ln] < 0:
                    nf[4 * (gr_subtree_offsets[i] + n_new) + 2] = float(-(self.leaves[ln][i] + 1))
                else:
                    new_node_number = q[-1][1] + 1
                    nf[4 * (gr_subtree_offsets[i] + n_new) + 2] = float(new_node_number)
                    q.append((ln, new_node_number))

                if self.feats[i][rn] < 0:
                    nf[4 * (gr_subtree_offsets[i] + n_new) + 3] = float(-(self.leaves[rn][i] + 1))
                else:
                    new_node_number = q[-1][1] + 1
                    nf[4 * (gr_subtree_offsets[i] + n_new) + 3] = float(new_node_number)
                    q.append((rn, new_node_number))
                q.pop(0)

        self.test_format = nf
        self.test_format_offsets = gr_subtree_offsets

        # feature_ importance with gain
        self.feature_importance_gain = np.zeros(nfeats, dtype=np.float32)
        sl = self.feats >= 0
        np.add.at(self.feature_importance_gain, self.feats[sl], self.gains[sl])

        # feature_ importance with split
        self.feature_importance_split = np.zeros(nfeats, dtype=np.float32)
        sl = self.feats >= 0
        np.add.at(self.feature_importance_split, self.feats[sl], 1)

        if not debug:
            for attr in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'leaves']:
                setattr(self, attr, None)


class DepthwiseTreeBuilder:
    """This class builds decision tree with given parameters"""

    def __init__(self, borders,
                 use_hess=True,
                 colsampler=None,
                 subsampler=None,
                 target_splitter=None,
                 multioutput_sketch=None,
                 gd_steps=1,
                 **tree_params
                 ):
        """

        Args:
            borders: list of np.ndarray, actual split borders for quantized features
            colsampler: Callable or None, column sampling strategy
            subsampler: Callable or None, rows sampling strategy
            target_splitter: Callable or None, target grouping strategy
            multioutput_sketch: Callable or None, multioutput sketching strategy
            **tree_params: other tree building parameters
        """
        self.borders = borders
        self.use_hess = use_hess
        self.params = {**{

            'lr': 1.,
            'lambda_l2': .01,
            'max_bin': 256,
            'max_depth': 6,
            'min_data_in_leaf': 10,
            'min_gain_to_split': 0
        }, **tree_params}

        self.colsampler = colsampler
        self.subsampler = subsampler
        self.target_grouper = target_splitter
        self.multioutput_sketch = multioutput_sketch
        self.gd_steps = gd_steps

    def build_tree(self, X, grad, hess, sample_weight=None, grad_fn=None, *val_arrays):
        """Build tree and return nodes/values predictions for train and validation sets

        Args:
            X: cp.ndarray, quantized feature matrix
            grad: cp.ndarray, gradient matrix
            hess: cp.ndarray, hessian matrix
            sample_weight: cp.ndarray or None, sample's weights
            grad_fn: gradient fn
            *val_arrays: list of cp.ndarray, list of quantized features for validation sets

        Returns:
            tree, Tree, constructed tree
            nodes_group, cp.ndarray, nodes id for the train set
            pred, cp.ndarray, prediction for the train set
            valid_nodes_group, list of cp.ndarray, list of predicted nodes for valid sets
            val_preds, list of cp.ndarray, list of predictions for valid sets
        """
        if self.colsampler is None:
            col_indexer = cp.arange(X.shape[1], dtype=cp.uint64)
        else:
            col_indexer = self.colsampler()

        if self.subsampler is None:
            row_indexer = cp.arange(X.shape[0], dtype=cp.uint64)
        else:
            row_indexer = self.subsampler()

        if self.target_grouper is None:
            output_groups = [cp.arange(grad.shape[1], dtype=cp.uint64)]
        else:
            output_groups = self.target_grouper()

        if sample_weight is not None:
            grad = grad * sample_weight
            hess = hess * sample_weight

        max_nodes = int((2 ** np.arange(self.params['max_depth'] + 1)).sum())
        tree = Tree(max_nodes, grad.shape[1], len(output_groups))

        nodes_group = cp.empty((grad.shape[0], len(output_groups)), dtype=cp.int32)
        valid_nodes_group = [cp.empty((x.shape[0], len(output_groups)), dtype=cp.int32) for x in val_arrays]

        group_index = cp.zeros(grad.shape[1], dtype=cp.uint64)

        for n_grp, grp_indexer in enumerate(output_groups):
            G = grad[:, grp_indexer]
            # if output group len eq. 1, we have single output tree, use hess for structure search
            if G.shape[1] == 1:
                H = hess if hess.shape[1] == 1 else hess[:, grp_indexer]
            # else we can decide: should we use hess for tree structure search or
            # assume hess eq. sample weight for all outputs, and then we can use proxy for tree structure search
            else:
                if self.use_hess:
                    H = hess[:, grp_indexer]
                else:
                    H = sample_weight if sample_weight is not None else cp.ones((G.shape[0], 1), dtype=cp.float32)
                if self.multioutput_sketch is not None:
                    G, H = self.multioutput_sketch(G, H)

            group_index[grp_indexer] = n_grp
            # grow single group of the tree and get nodes index
            train_nodes, valid_nodes = depthwise_grow_tree(tree, n_grp, X, G, H,
                                                           row_indexer, col_indexer, self.params,
                                                           valid_arrs=val_arrays)
            # update nodes group
            nodes_group[:, n_grp] = train_nodes
            for vn, vp in zip(valid_nodes_group, valid_nodes):
                vn[:, n_grp] = vp

        # transform nodes to leaves
        tree.set_leaves()
        leaves_idx, max_leaves, leaves_grp = cp.asarray(tree.leaves, dtype=cp.int32), tree.max_leaves, \
                                             cp.arange(len(output_groups), dtype=cp.uint64)

        leaves = apply_values(nodes_group, leaves_grp, leaves_idx)
        val_leaves = [apply_values(x, leaves_grp, leaves_idx) for x in valid_nodes_group]

        # perform multiple grad steps
        values = calc_node_values(grad, hess, leaves, row_indexer, group_index, max_leaves, self.params['lr'],
                                  lambda_l2=self.params['lambda_l2'])
        pred = apply_values(leaves, group_index, values)

        tree.set_borders(self.borders)

        for i in range(1, self.gd_steps):
            grad, hess = grad_fn(pred)
            values += calc_node_values(grad, hess, leaves, row_indexer, group_index, max_leaves, self.params['lr'],
                                       lambda_l2=self.params['lambda_l2'])
            pred = apply_values(leaves, group_index, values)

        # transform leaves to values
        val_preds = [apply_values(x, group_index, values) for x in val_leaves]
        tree.set_node_values(values.get(), group_index.get())

        return tree, leaves, pred, val_leaves, val_preds
