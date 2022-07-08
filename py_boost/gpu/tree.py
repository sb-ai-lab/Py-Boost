"""Decision trees building and inference"""

import cupy as cp
import numpy as np

from .utils import apply_values, depthwise_grow_tree, get_tree_node, set_leaf_values, calc_node_values


class Tree:
    """This class initializes an empty tree structure, implements methods to set tree values and single tree inference.
    The instance of this object represents the actual boosting step, but not the single tree!
    Actual amount of trees in the instance (at each boosting step) is defined by ngroups argument. What does it mean:
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
    Values assigned to the outpus are defined by arrays:
        grouop_index, shape (nout, ). Defines the group id for predicting each output
        values, shape (max_nodes, nout). Define output value for each node/output
        leaves, shape (max_leaves, ngroups). Assigns the leaf index to the terminal nodes

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
        for attr in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'values', 'group_index', 'leaves']:
            arr = getattr(self, attr)
            setattr(self, attr, cp.asarray(arr))

    def to_cpu(self):
        """Move tree data to the CPU memory

        Returns:

        """
        for attr in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'values', 'group_index', 'leaves']:
            arr = getattr(self, attr)
            if type(arr) is not np.ndarray:
                setattr(self, attr, arr.get())

    def predict_node(self, X):
        """Predict node id from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:

        """
        assert type(self.feats) is cp.ndarray, 'Should be moved to GPU first. Call .to_device()'
        nodes = get_tree_node(X, self.feats, self.val_splits, self.split, self.nans)
        return nodes

    def predict_from_nodes(self, nodes):
        """Predict outputs from the nodes indices

        Args:
            nodes: cp.ndarray of predicted nodes

        Returns:
            cp.ndarray of nodes
        """
        return apply_values(nodes, self.group_index, self.values)

    def predict_leaf_from_nodes(self, nodes):
        """Predict leaf indices from the nodes indices

        Args:
            nodes: cp.ndarray of predicted nodes

        Returns:
            cp.ndarray of leaves
        """
        return apply_values(nodes, cp.arange(self.ngroups, dtype=cp.uint64), self.leaves)

    def predict(self, X):
        """Predict from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:
            cp.ndarray of predictions
        """
        return self.predict_from_nodes(self.predict_leaf_from_nodes(self.predict_node(X)))

    def predict_leaf(self, X):
        """Predict leaf indices from the feature matrix X

        Args:
            X: cp.ndarray of features

        Returns:
            cp.ndarray of leaves
        """
        return self.predict_leaf_from_nodes(self.predict_node(X))


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
            sample_weight: cp.ndarray or None, sample weights
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
            # assume hess eq. sample weight for all outputs
            # and then we can use proxy for tree structure search
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
