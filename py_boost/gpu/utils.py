"""Utilities used for trees building and inference"""

import math
import numba
import cupy as cp
import numpy as np


def validate_input(X, y, sample_weight=None, eval_sets=None):
    if eval_sets is None:
        eval_sets = []

    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    if (sample_weight is not None) and (len(sample_weight.shape) == 1):
        sample_weight = sample_weight[:, np.newaxis]

    eval_sets = list(eval_sets)
    for val_arr in eval_sets:
        if len(val_arr['y'].shape) == 1:
            val_arr['y'] = val_arr['y'][:, np.newaxis]

        if 'sample_weight' not in val_arr:
            val_arr['sample_weight'] = None

        if (val_arr['sample_weight'] is not None) and (len(val_arr['sample_weight'].shape) == 1):
            val_arr['sample_weight'] = val_arr['sample_weight'][:, np.newaxis]

    return X, y, sample_weight, eval_sets


def pad_and_move(arr, pad_size=4):
    """Pad array memory placeholder to make feature size divisible by pad_size and move to GPU.
    Returned array is the same size as input, but it is not contiguous.
    Basic memory placeholder size is divisible by pad_size and could be accesses via arr.base.
    This trick helps to speed up histogram computation for the large arrays.

    Args:
        arr: 2d np.ndarray of uint8 containing quantized features matrix
        pad_size: int, memory placeholder feature size will be divisible by pad_size

    Returns:
        cp.ndarray of quantized features
    """
    assert arr.dtype == np.uint8, 'Array dtype should be unsigned 8bit int'

    pfeats = math.ceil(arr.shape[1] / pad_size) * pad_size
    arr_cpu_padded = np.zeros((arr.shape[0], pfeats), dtype=np.uint8)
    arr_cpu_padded[:, :arr.shape[1]] = arr
    arr_gpu_padded = cp.asarray(arr_cpu_padded)

    return arr_gpu_padded[:, :arr.shape[1]]


isin_code = """
int fnode;
int leaf = 0;
int minval = 0;
int maxval = l - 1;

int node_val = nodes[index];

while (true) {{

    leaf = minval + (maxval - minval) / 2;
    fnode = un[leaf];

    if (fnode == node_val) {{

        isin={0};
        return;

    }} else if (minval >= maxval) {{

        isin={1};
        return ;

    }} else if (fnode < node_val) {{

        minval = leaf + 1;

    }} else  {{

        maxval = leaf - 1;

    }} 
}}

"""

isin_kernel = cp.ElementwiseKernel(
    'Q index, raw E nodes, raw R un, uint64 l',
    'bool isin',

    isin_code.format('true', 'false'),

    'isin_kernel'
)

isin_pos_kernel = cp.ElementwiseKernel(
    'Q index, raw E nodes, raw R un, uint64 l',
    'int32 isin',

    isin_code.format('leaf', -1),

    'isin_pos_kernel'
)


def isin(a, b, index=None, return_pos=False):
    """Check if b contains elements of a. Custom function is faster than build in cupy's

    Args:
        a: cp.ndarray to apply
        b: cp.ndarray lookup array
        index: cp.ndarray indices of a to check. If None, check for all a
        return_pos: bool, if True, return position of a[i] in b, otherwise return bool array if a[i] is in b

    Returns:
        cp.ndarray of bool or int
    """
    if index is None:
        index = cp.arange(a.shape[0], dtype=cp.uint64)

    if return_pos:
        return isin_pos_kernel(index, a, b, b.shape[0])
    else:
        return isin_kernel(index, a, b, b.shape[0])


histogram_kernel_idx = cp.ElementwiseKernel(
    """
    uint64 i_, uint64 j_, uint64 k_, 
    uint64 kk,

    raw uint64 jj,
    raw bool padded_bool_indexer,

    raw float32 target, 
    raw T arr, 
    raw int32 nodes,

    uint64 hlen,
    uint64 flen, 
    uint64 length,
    uint64 feats,
    uint64 nout 
    """,
    'raw float32 hist',

    """
    unsigned int feat_4t = arr[i_ * feats + j_];
    int d;
    int j;
    int val;
    int pos;
    float *x_ptr;
    float y = target[i_ * nout + k_];

    for (d = 0; d < 4; d++) {

        pos = (i_ + d) % 4;

        if (padded_bool_indexer[j_ * 4 + pos]) {

            val = (feat_4t >> (8 * pos)) % 256;
            j = jj[j_ * 4 + pos];
            x_ptr = &hist[0] +  kk * hlen + nodes[i_] * flen + j * length + val;
            atomicAdd(x_ptr, y); 
        }
    }

    """,

    'histogram_kernel_idx')

feature_grouper_kernel = cp.ElementwiseKernel(
    """
    uint64 col_indexer
    """,
    'raw bool padded_bool_indexer, raw bool tuple_indexer, raw uint64 padded_col_indexer',
    """
    int tuple_id = col_indexer / 4;
    padded_bool_indexer[col_indexer] = true;
    tuple_indexer[tuple_id] = true;
    padded_col_indexer[col_indexer] = i;

    """,
    'feature_grouper_kernel')


def fill_histogram(res, arr, target, nodes, col_indexer, row_indexer, out_indexer):
    """Fill the histogram res

    Args:
        res: cp.ndarray, histogram of zeros, shape (n_out, n_nodes, n_features, n_bins)
        arr: cp.ndarray, features array, shape (n_data, n_features)
        target: cp.ndarray, values to accumulate, shape (n_data, n_out)
        nodes: cp.ndarray, tree node indices, shape (n_data, )
        col_indexer: cp.ndarray, indices of features to accumulate
        row_indexer: cp.ndarray, indices of rows to accumulate
        out_indexer: cp.ndarray, indices of outputs to accumulate

    Returns:

    """
    # define data split for kernel launch
    nout, nnodes, nfeats, nbins = res.shape

    # padded array of 4 feature tuple
    arr_4t = arr.base.view(dtype=cp.uint32)
    pfeats = arr_4t.shape[1]

    # create 4 feats tuple indexer
    padded_bool_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.bool_)
    padded_col_indexer = cp.zeros((arr.base.shape[1],), dtype=cp.uint64)
    tuple_indexer = cp.zeros((arr_4t.shape[1],), dtype=cp.bool_)

    feature_grouper_kernel(col_indexer, padded_bool_indexer, tuple_indexer, padded_col_indexer)
    tuple_indexer = cp.arange(arr_4t.shape[1], dtype=cp.uint64)[tuple_indexer]

    fb = nfeats * nbins
    nfb = nnodes * fb

    magic_constant = 2 ** 19  # optimal value for my V100

    # split features
    nsplits = math.ceil(nfb / magic_constant)
    # first split by feats
    feats_batch = math.ceil(pfeats / nsplits)
    # split by features
    if feats_batch == nfeats:
        out_batch = magic_constant // nfb
    else:
        out_batch = 1

    ri = row_indexer[:, cp.newaxis, cp.newaxis]
    ti = tuple_indexer[cp.newaxis, :, cp.newaxis]
    oi = out_indexer[cp.newaxis, cp.newaxis, :]

    oii = cp.arange(oi.shape[2], dtype=cp.uint64)[cp.newaxis, cp.newaxis, :]

    for j in range(0, pfeats, feats_batch):
        ti_ = ti[:, j: j + feats_batch]

        for k in range(0, nout, out_batch):
            oi_ = oi[..., k: k + out_batch]
            oii_ = oii[..., k: k + out_batch]

            histogram_kernel_idx(ri, ti_, oi_,
                                 oii_,
                                 padded_col_indexer,
                                 padded_bool_indexer,
                                 target,
                                 arr_4t,
                                 nodes,
                                 nfb, fb, nbins, arr_4t.shape[1], nout,
                                 res)


def histogram(arr, gh, nodes, col_indexer, row_indexer, out_indexer, nnodes, max_bins,
              prev_hist=None, small_index=None, big_index=None):
    """Compute grad/hess histogram

    Args:
        arr: cp.ndarray, features array, shape (n_data, n_features)
        gh: cp.ndarray of grad/hess, shape (n_data, n_out + 1), assume the last output is hess, others for grads
        nodes: cp.ndarray, tree node indices, shape (n_data, )
        col_indexer: cp.ndarray, indices of features to accumulate
        row_indexer: cp.ndarray, indices of rows to accumulate
        out_indexer: cp.ndarray, indices of outputs to accumulate
        nnodes: int, total number of nodes in the sample
        max_bins: int, maximum number of feature bins
        prev_hist: cp.ndarray or None, histogram from previous step to apply histogram subtraction trick
        small_index: cp.ndarray or None, node indices to accumulate
        big_index: cp.ndarray or None, node indices to compute by subtraction from prev_hist

    Returns:
        cp.ndarray, histogram of shape (n_out + 1, n_nodes, n_features, n_bins)
    """

    nout = out_indexer.shape[0]
    nfeats = col_indexer.shape[0]
    # init union histogram for grad and hess
    # we assume, we have 1 dim hess for one output group

    res = cp.zeros((nout, nnodes, nfeats, max_bins), dtype=cp.float32)

    if prev_hist is not None:
        row_indexer = row_indexer[isin(nodes, small_index, row_indexer)]

    fill_histogram(res, arr, gh, nodes, col_indexer, row_indexer, out_indexer)
    res = res.cumsum(axis=-1)

    if prev_hist is not None:
        # indices are paired.
        res[:, big_index] = prev_hist - res[:, big_index + (1 - 2 * (big_index % 2))]

    return res


loss_kernel = cp.ElementwiseKernel(
    """
    float32 grad, float32 hess, 
    float32 total_grad, float32 total_hess, 
    float32 nan_grad, float32 nan_hess, 

    uint64 nodes_count, 
    float32 lambda_l2, float32 min_data_in_left


    """,
    'raw float32 res',

    """

    float rG;
    float G = grad;
    float H = hess;
    float C = hess / total_hess * nodes_count;


    if (fmin(nodes_count - C, C) > min_data_in_left) {
        rG = total_grad - G;
        res[2 * i] = G * G / (H + lambda_l2) + rG * rG / (total_hess - H + lambda_l2);
    }

    if ((nan_hess > 0) and (hess < total_hess)) {

        C -= nan_hess / total_hess * nodes_count;

        if (fmin(nodes_count - C, C) > min_data_in_left) {

            G -= nan_grad;
            H -= nan_hess;
            rG = total_grad - G;
            res[2 * i + 1] = G * G / (H + lambda_l2) + rG * rG / (total_hess - H + lambda_l2);

        }
    }

    """,

    'loss_kernel')


def calc_loss(grad, hess, nodes_count, lambda_l2=0.1, min_data_in_leaf=10.):
    """Calculate loss function

    Args:
        grad:
        hess:
        nodes_count:
        lambda_l2:
        min_data_in_leaf:

    Returns:

    """
    # fill loss
    res = cp.zeros(grad.shape + (2,), dtype=cp.float32)
    loss_kernel(grad, hess,
                grad[..., -1:], hess[..., -1:],
                grad[..., :1], hess[..., :1],
                nodes_count[cp.newaxis, :, cp.newaxis, cp.newaxis],
                lambda_l2, min_data_in_leaf, res)

    return res


select_among_feature = cp.ElementwiseKernel(
    """
    int64 best_idx, 
    raw float32 loss,
    uint64 binsx2
    """,
    'float32 best_gain, int32 best_split, bool best_nan_left',

    """
    best_gain = loss[i * binsx2 + best_idx];
    best_split = best_idx / 2;
    best_nan_left = (bool) (1 - best_idx % 2);

    """,

    'select_among_feature')

select_total = cp.ElementwiseKernel(
    """
    int64 best_idx,
    raw float32 feat_loss,
    raw int32 feat_split,
    raw bool feat_nan_left,
    raw uint64 col_indexer, 

    uint64 feats
    """,
    'int64 best_feat, float32 best_gain, int32 best_split, bool best_nan_left',

    """
    unsigned long f_ptr = i * feats + best_idx;

    best_gain = feat_loss[f_ptr];
    best_split = feat_split[f_ptr];
    best_nan_left = feat_nan_left[f_ptr];
    best_feat = col_indexer[best_idx];

    """,

    'select_total')


def get_best_split(loss, col_indexer):
    nnodes, nfeats, nbins, _ = loss.shape
    # shape - nnodes * nfeats
    best_loss_among_feat_idx = loss.reshape((nnodes, nfeats, -1)).argmax(axis=-1)

    best_gain, best_split, best_nan_left = \
        select_among_feature(
            best_loss_among_feat_idx, loss, nbins * 2
        )

    best_feat_ = best_gain.argmax(axis=-1)

    best_feat, best_gain, best_split, best_nan_left = \
        select_total(
            best_feat_, best_gain, best_split, best_nan_left, col_indexer, nfeats
        )

    best_gain -= loss[:, 0, -1, 0]

    return best_feat, best_gain, best_split, best_nan_left


split_kernel = cp.ElementwiseKernel(
    """
    int32 nodes_index,
    int32 nodes_old,
    raw uint8 arr,
    raw int32 split_nodes,
    raw int64 feat,
    raw int32 split,
    raw bool nan_left,
    uint64 F
    """,
    'int32 nodes',

    """
    if (nodes_index >= 0) {

        int f = feat[nodes_index];
        int s = split[nodes_index];
        bool nl = nan_left[nodes_index];

        int val = arr[i * F + f];

        int d = 0;
        if ((val > s) or ((val == 0) and (not nl))) {d = 1;}

        nodes = split_nodes[2 * nodes_index + d];

    } else {nodes=nodes_old;}

    """,

    'split_kernel')


def make_split(nodes, arr, unique_nodes, split_nodes, feat, split, nan_left, return_pos=False):
    nodes_index = isin(nodes, unique_nodes, return_pos=True)
    new_nodes = split_kernel(nodes_index, nodes, arr, split_nodes, feat, split, nan_left, arr.shape[1])

    if return_pos:
        nodes_index = isin(new_nodes, split_nodes.ravel(), return_pos=True)
        return new_nodes, nodes_index

    return new_nodes


def get_prev_hist(cpu_counts, prev_hist, is_valid_node):
    nbins = prev_hist.shape[-1]
    magic_constant = 20

    is_max = cpu_counts.reshape((-1, 2)).argsort(axis=1).astype(np.bool_).ravel()
    valid = is_max & (cpu_counts > nbins)

    bigger_pos = np.arange(valid.shape[0])[valid]
    smaller_pos = np.arange(valid.shape[0])[~valid]

    if cpu_counts[valid].sum() > (nbins * magic_constant):
        return None, None, None

    bigger_pos = cp.asarray(bigger_pos)
    smaller_pos = cp.asarray(smaller_pos)

    hist_idx = cp.arange(prev_hist.shape[1])[is_valid_node][bigger_pos // 2]

    prev_hist = prev_hist[:, hist_idx]

    return prev_hist, smaller_pos, bigger_pos


apply_values_kernel = cp.ElementwiseKernel(
    """
    uint64 row_indexer,
    uint64 grp_index,

    raw int32 nodes,
    raw T values,
    uint64 ngroups,
    uint64 nout
    """,
    'T pred',

    """
    int out = i % nout;
    int node = nodes[row_indexer * ngroups + grp_index];
    pred = values[node * nout + out];

    """,

    'apply_values_kernel')


def apply_values(nodes, group_index, values):
    row_indexer = cp.arange(nodes.shape[0], dtype=cp.uint64)
    ngroups = nodes.shape[1]
    nout = group_index.shape[0]

    return apply_values_kernel(row_indexer[:, cp.newaxis], group_index[cp.newaxis, :], nodes,
                               values, ngroups, nout)


node_index_kernel = cp.ElementwiseKernel(
    """
    uint64 i_,
    uint64 j_,

    raw T X,

    raw int64 feats,
    raw float32 val_splits,
    raw int32 splits,
    raw bool nan_left,

    uint64 l
    """,
    'int32 node',

    """
    node = 0;

    int f;
    float x;
    int right;
    unsigned long x_ptr;

    while (true) {

        x_ptr = j_ + node;
        f = feats[x_ptr];

        if (f < 0) {return;}

        x = X[i_ * l + f];

        if (isnan(x)) {

            right = 1 - (int) nan_left[x_ptr];

        } else {

            right = (int) (x > val_splits[x_ptr]);

        }

        node = splits[2 * x_ptr + right];

    }

    """,

    'node_index_kernel')


def get_tree_node(arr, feats, val_splits, split, nan_left):
    n_gr, nf = feats.shape

    row_indexer = cp.arange(arr.shape[0], dtype=cp.uint64)[:, cp.newaxis]
    out_indexer = cp.arange(0, n_gr * nf, nf, dtype=cp.uint64)[cp.newaxis, :]

    nodes = node_index_kernel(row_indexer, out_indexer, arr, feats,
                              val_splits, split, nan_left, arr.shape[1])

    return nodes


def get_cpu_splitters(unique_nodes, best_feat, best_gain, best_split, best_nan_left, min_gain_to_split=0):
    # print(best_gain.shape, best_gain)

    best_gain = best_gain.get()
    vs = best_gain > min_gain_to_split
    n_vs = vs.sum()

    if n_vs == 0:
        return [], None, None, None, None, None, None

    best_feat = best_feat.get()
    best_split = best_split.get()
    best_nan_left = best_nan_left.get()
    last_node = unique_nodes[-1] + 1

    if n_vs < vs.shape[0]:
        unique_nodes = unique_nodes[vs]
        best_gain = best_gain[vs]
        best_feat = best_feat[vs]
        best_split = best_split[vs]
        best_nan_left = best_nan_left[vs]

    new_nodes_id = np.arange(last_node, last_node + unique_nodes.shape[0] * 2,
                             dtype=np.int32).reshape((-1, 2))

    return unique_nodes, new_nodes_id, best_feat, best_gain, best_split, best_nan_left, vs


def get_gpu_splitters(unique_nodes, new_nodes_id, best_feat, best_split, best_nan_left):
    out = []

    for arr in [unique_nodes, new_nodes_id, best_feat, best_split, best_nan_left]:
        gpu_arr = cp.asarray(arr)
        out.append(gpu_arr)

    new_unique_nodes = new_nodes_id.ravel()

    return out, new_unique_nodes


def depthwise_grow_tree(tree, group, arr, grad, hess, row_indexer, col_indexer, params, valid_arrs=None):
    if valid_arrs is None:
        valid_arrs = []

    # create gh
    n_out = grad.shape[1]
    gh = cp.concatenate((grad, hess), axis=1)
    out_indexer = cp.arange(gh.shape[1], dtype=cp.uint64)

    # init nodes with single zero node
    unique_nodes = np.zeros(1, dtype=np.int32)
    # count unique nodes in active rows
    nodes_count = cp.ones(1, dtype=cp.uint64) * row_indexer.shape[0]
    # nodes for all rows
    nodes = cp.zeros(arr.shape[0], dtype=cp.int32)
    # init valid nodes
    valid_nodes = [cp.zeros(x.shape[0], dtype=cp.int32) for x in valid_arrs]
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
        loss = calc_loss(gh_hist[:n_out], gh_hist[n_out:], nodes_count, lambda_l2=params['lambda_l2'],
                         min_data_in_leaf=params['min_data_in_leaf'])

        if loss.shape[0] > 1:
            loss = loss.sum(axis=0)
        else:
            loss = loss[0]

        best_feat, best_gain, best_split, best_nan_left = get_best_split(loss, col_indexer)

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
        # perform split for valid sets
        valid_nodes = [make_split(x, y, *split_args, return_pos=False) for (x, y) in zip(valid_nodes, valid_arrs)]

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

    return nodes, valid_nodes


accumulate_gh_kernel = cp.ElementwiseKernel(
    """
    uint64 row_indexer,
    uint64 group_index, 

    raw float32 grad, 
    raw float32 hess,
    raw int32 nodes,

    uint64 nout,
    uint64 ngroups,
    uint64 hxst,
    uint64 hyst

    """,
    'raw float32 grad_sum, raw float32 hess_sum',

    """
    int out = i % nout;
    int node = nodes[row_indexer * ngroups + group_index];

    int y_ptr =  node * nout + out;

    atomicAdd(&grad_sum[y_ptr], grad[row_indexer * nout + out]); 
    atomicAdd(&hess_sum[y_ptr], hess[row_indexer * hxst + out * hyst]); 

    """,

    'accumulate_gh_kernel')


def calc_node_values(grad, hess, nodes, row_indexer, group_index, max_nodes, lr=1, lambda_l2=0.1):
    """Calculate node values based on grad/hess

    Args:
        grad:
        hess:
        nodes:
        row_indexer:
        group_index:
        max_nodes:
        lr:
        lambda_l2:

    Returns:

    """
    nout = grad.shape[1]
    ngroups = nodes.shape[1]

    grad_sum = cp.zeros((max_nodes, nout), dtype=cp.float32)
    hess_sum = cp.zeros((max_nodes, nout), dtype=cp.float32)

    accumulate_gh_kernel(row_indexer[:, cp.newaxis], group_index[cp.newaxis, :], grad, hess, nodes,
                         nout, ngroups, hess.shape[1], int(hess.shape[1] > 1),
                         grad_sum, hess_sum)

    node_values = - grad_sum * lr / (hess_sum + lambda_l2)

    return node_values


def pinned_array(array):
    """Move cpu array to the pinned memory

    Args:
        array: np.ndarray

    Returns:
        np.ndarray
    """
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


@numba.njit
def set_leaf_values(feats, split):
    """Assign leaf indices to the terminal nodes

    Args:
        feats: np.ndarray of tree's split features, shape (n_groups, max_nodes)

    Returns:
        np.ndarray of leaf indices
    """
    leaves = np.zeros(feats.shape[::-1], dtype=np.int32)
    max_leaves = 0

    for i in range(feats.shape[0]):

        acc = 0

        for j in range(feats.shape[1]):

            if split[i, j, 0] != -1:

                for k in range(2):
                    n = split[i, j, k]
                    if feats[i, n] == -1:
                        leaves[n, i] = acc
                        acc += 1

        max_leaves = max(max_leaves, acc)

    return leaves, max_leaves
