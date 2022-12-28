from ..gpu.utils import *
from ..gpu.tree import *
from ..gpu.base import Ensemble

from ..quantization.base import QuantileQuantizer


def cluster_grow_tree(tree, group, arr, grad, hess, row_indexer, col_indexer, params):
    """Graw tree for advanced prunning

    Args:
        tree:
        group:
        arr:
        grad:
        hess:
        row_indexer:
        col_indexer:
        params:

    Returns:

    """
    # create gh
    gh = cp.concatenate((grad, hess), axis=1)
    out_indexer = cp.arange(gh.shape[1], dtype=cp.uint64)

    # init nodes with single zero node
    unique_nodes = np.zeros(1, dtype=np.int32)
    # count unique nodes in active rows
    nodes_count = cp.ones(1, dtype=cp.uint64) * row_indexer.shape[0]
    # nodes for all rows
    nodes = cp.zeros(arr.shape[0], dtype=cp.int32)
    # index of node in unique array
    node_indexes = nodes
    prev_hist, small_index, big_index = [None] * 3

    for niter in range(params['max_depth']):

        nnodes = len(unique_nodes)
        gh_hist = histogram(arr, gh, node_indexes,
                            col_indexer=col_indexer,
                            row_indexer=row_indexer,
                            out_indexer=out_indexer,
                            nnodes=nnodes,
                            max_bins=params['max_bin'],
                            prev_hist=prev_hist,
                            small_index=small_index,
                            big_index=big_index)

        # assume hess is the last output

        hist, counts = gh_hist[:-1], gh_hist[-1]
        total = hist[..., :1, -1:]
        curr = total.min(axis=0)
        gain = cp.zeros(hist.shape[1:] + (2,), dtype=cp.float32)

        # NAN to left
        gain[..., 0] = curr - hist.min(axis=0) - (total - hist).min(axis=0)
        gain[..., 0] *= cp.minimum(counts, counts[..., -1:] - counts) >= params['min_data_in_leaf']

        # NAN to right
        gain[..., 1] = curr - (hist - hist[..., :1]).min(axis=0) - (total - hist + hist[..., :1]).min(axis=0)
        gain[..., 1] *= cp.minimum(counts - counts[..., :1:], counts[..., -1:] - counts + counts[..., :1]) >= params[
            'min_data_in_leaf']

        best_feat, best_gain, best_split, best_nan_left = get_best_split(gain, col_indexer)

        # move to CPU and apply min_gain_to_split condition
        unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left, is_valid_node = \
            get_cpu_splitters(unique_nodes, best_feat, best_gain, best_split, best_nan_left,
                              params['min_gain_to_split'])
        # if all nodes are not valid to split - exit
        if len(unique_nodes) == 0:
            break
        # write node info to the Tree
        tree.set_nodes(group, unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left)
        # get args back on gpu
        split_args, unique_nodes = get_gpu_splitters(unique_nodes, new_nodes_id,
                                                     best_feat, best_split, best_nan_left)

        # perform split for train set
        nodes, node_indexes = make_split(nodes, arr, *split_args, return_pos=True)

        # update info for the next step
        if niter < (params['max_depth'] - 1):
            # update counts
            nodes_count = cp.zeros((unique_nodes.shape[0] + 1,), dtype=np.uint64)
            nodes_count.scatter_add(node_indexes[row_indexer], 1)
            nodes_count = nodes_count[:-1]

            cpu_counts = nodes_count.get()

            # remove unused rows from indexer
            if cpu_counts.sum() < row_indexer.shape[0]:
                row_indexer = row_indexer[isin(nodes, split_args[1].ravel(), index=row_indexer)]

            # save histogram for the subs trick
            prev_hist, small_index, big_index = get_prev_hist(cpu_counts,
                                                              gh_hist, cp.asarray(is_valid_node))

    return nodes


class ClusterTreeBuilder:
    """Tree builder for early stopping clusters"""

    def __init__(self, borders,
                 **tree_params
                 ):
        """

        Args:
            borders: list of np.ndarray, actual split borders for quantized features
            **tree_params: other tree building parameters
        """
        self.borders = borders

        self.params = {**{

            'max_bin': 256,
            'max_depth': 6,
            'min_data_in_leaf': 10,
            'min_gain_to_split': 0

        }, **tree_params}

    def build_tree(self, X, y):
        """Build tree

        Args:
            X: cp.ndarray, quantized feature matrix
            y: cp.ndarray, loss path matrix


        Returns:
            tree, Tree, constructed tree
        """

        col_indexer = cp.arange(X.shape[1], dtype=cp.uint64)
        row_indexer = cp.arange(X.shape[0], dtype=cp.uint64)
        max_nodes = int((2 ** np.arange(self.params['max_depth'] + 1)).sum())
        tree = Tree(max_nodes, y.shape[1], 1)
        # grow single group of the tree and get nodes index
        cluster_grow_tree(tree, 0, X, y, cp.ones((y.shape[0], 1), dtype=cp.float32),
                          row_indexer, col_indexer, self.params)

        tree.set_borders(self.borders)
        tree.set_leaves()
        tree.set_node_values(np.zeros((max_nodes, 1), dtype=np.float32), np.zeros((1,), dtype=np.uint64))

        return tree


class ClusterCandidates(Ensemble):
    """
    Ensemble of cluster candidates
    """

    def __init__(self, depth_range=range(1, 7), min_data_in_leaf=100):
        super().__init__()

        self.depth_range = depth_range
        self.min_data_in_leaf = min_data_in_leaf
        self.max_clust = 2 ** max(depth_range)

    def fit(self, X, y):
        X, y, sample_weight, eval_sets = validate_input(X, y, None, [])
        mempool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(allocator=mempool.malloc):
            # TODO: move quantizer to the Ensemble
            quantizer = QuantileQuantizer(sample=self.quant_sample, max_bin=self.max_bin)
            X_enc, max_bin, borders, eval_enc = self.quantize(X, eval_sets)

            self.fit_quantized(X_enc, y, max_bin, borders)
        mempool.free_all_blocks()

        return self

    def fit_quantized(self, X_enc, y, max_bin, borders):
        y = cp.array(y, order='C', dtype=cp.float32)
        X_cp = pad_and_move(X_enc)
        self.models = []

        for d in self.depth_range:
            builder = ClusterTreeBuilder(borders, max_depth=d, min_data_in_leaf=self.min_data_in_leaf, max_bin=max_bin)
            self.models.append(builder.build_tree(X_cp, y))

        self.base_score = np.zeros((1,), dtype=np.float32)

        return self

    def predict(self, X, iterations=None, batch_size=100000):
        return self.predict_leaves(X, iterations=iterations, batch_size=batch_size)[..., 0].T
