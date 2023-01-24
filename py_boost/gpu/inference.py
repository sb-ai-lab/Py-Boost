import numpy as np
import cupy as cp
from .utils import tree_prediction_kernel_alltogether, pinned_array


class EnsembleInference:
    """Special fast inference class generated from trained ensemble"""

    @staticmethod
    def _default_postprocess_fn(x):
        return x

    def __init__(self, ensemble):
        all_trees = []
        all_tree_offsets = []
        all_values = []
        all_values_offset = []
        all_out_sizes = []
        all_out_indexes = []
        total_tree_offset = 0
        total_value_offset = 0
        ensemble.to_cpu()

        check_grp = np.unique([x.ngroups for x in ensemble.models])
        if check_grp.shape[0] > 1:
            raise ValueError('Different number of groups in trees')

        for n, tree in enumerate(ensemble.models):
            all_trees.append(tree.new_format)
            all_tree_offsets.append(tree.new_format_offsets + total_tree_offset)
            all_values.append(tree.values)
            all_values_offset.append(total_value_offset)
            all_out_indexes.append(tree.new_out_indexes)
            all_out_sizes.append(tree.new_out_sizes)

            total_tree_offset += len(tree.new_format) // 4
            total_value_offset += tree.values.size

        self.all_trees = np.concatenate(all_trees)
        self.all_tree_offsets = np.concatenate(all_tree_offsets)
        self.all_values = np.concatenate(all_values)
        self.all_values_offset = np.array(all_values_offset, dtype=np.int32)
        self.all_out_sizes = np.concatenate(all_out_sizes)
        self.all_out_indexes = np.concatenate(all_out_indexes)

        self.n_models = len(ensemble.models)
        self.n_groups = ensemble.models[0].ngroups
        self.n_feat = ensemble.nfeats
        self.n_out = len(ensemble.base_score)
        self.base_score = ensemble.base_score.copy()
        self.postprocess_fn = ensemble.postprocess_fn
        self._on_device = False

    def to_device(self):
        """Move data to the current GPU memory

        Returns:

        """
        if not self._on_device:
            for attr in ['all_trees', 'all_tree_offsets', 'all_values',
                         'all_values_offset', 'all_out_sizes', 'all_out_indexes', 'base_score']:
                arr = getattr(self, attr)
                setattr(self, attr, cp.asarray(arr))
            self._on_device = True

    def to_cpu(self):
        """Move data to CPU memory

        Returns:

        """
        if self._on_device:
            for attr in ['all_trees', 'all_tree_offsets', 'all_values',
                         'all_values_offset', 'all_out_sizes', 'all_out_indexes', 'base_score']:
                arr = getattr(self, attr)
                if type(arr) is not np.ndarray:
                    setattr(self, attr, arr.get())
            self._on_device = False

    def _predict_kernel(self, X, res):
        """ CUDA kernel call for inference

        Returns:

        """

        threads = 128
        sz = X.shape[0] * self.n_groups
        blocks = sz // threads
        if sz % threads != 0:
            blocks += 1

        tree_prediction_kernel_alltogether((blocks, self.n_models), (threads,), ((X,
                                                                                  self.all_trees,
                                                                                  self.all_tree_offsets,
                                                                                  self.all_values,
                                                                                  self.all_values_offset,
                                                                                  self.all_out_sizes,
                                                                                  self.all_out_indexes,
                                                                                  self.n_feat,
                                                                                  self.n_out,
                                                                                  X.shape[0],
                                                                                  self.n_groups,
                                                                                  res)))

    def predict(self, X, batch_size=100000):
        """Make prediction for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of float32, shape (n_data, n_outputs)
        """
        self.to_device()

        cur_dtype = np.float32
        n_streams = 2  # don't change
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # special case handle if X is already on device
        if type(X) is cp.ndarray:
            cpu_pred = np.empty((X.shape[0], self.n_out), dtype=cur_dtype)
            gpu_pred = cp.empty((X.shape[0], self.n_out), dtype=cur_dtype)

            gpu_pred[:] = self.base_score

            self._predict_kernel(X, gpu_pred)

            cp.cuda.get_current_stream().synchronize()
            self.postprocess_fn(gpu_pred).get(out=cpu_pred)

            return cpu_pred

        # result allocation
        cpu_pred_full = np.empty((X.shape[0], self.n_out), dtype=cur_dtype)
        cpu_pred = [pinned_array(np.empty((batch_size, self.n_out), dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_pred = [cp.empty((batch_size, self.n_out), dtype=cur_dtype) for _ in range(n_streams)]

        # batch allocation
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=cur_dtype)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=cur_dtype) for _ in range(n_streams)]

        cpu_batch_free_event = [None, None]
        cpu_out_ready_event = [None, None]
        last_batch_size = 0
        last_n_stream = 0
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                if k >= 2:
                    cpu_batch_free_event[nst].synchronize()
                cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(cur_dtype)

                if k >= 2:
                    cpu_out_ready_event[nst].synchronize()
                gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                cpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                gpu_pred[nst][:] = self.base_score
                self._predict_kernel(gpu_batch[nst][:real_batch_len], gpu_pred[nst][:real_batch_len])

                if k >= 2:
                    cpu_pred_full[i - 2 * batch_size: i - batch_size] = cpu_pred[nst][:batch_size]

                self.postprocess_fn(gpu_pred[nst][:real_batch_len]).get(out=cpu_pred[nst][:real_batch_len])
                cpu_out_ready_event[nst] = stream.record(cp.cuda.Event(block=True))

                last_batch_size = real_batch_len
                last_n_stream = nst

        # waiting for sync of last two batches
        if int(np.floor(X.shape[0] / batch_size)) == 0:  # only one stream was used
            with map_streams[last_n_stream] as stream:
                stream.synchronize()
                cpu_pred_full[:last_batch_size] = cpu_pred[last_n_stream][:last_batch_size]
        else:
            with map_streams[1 - last_n_stream] as stream:
                stream.synchronize()
                cpu_pred_full[X.shape[0] - batch_size - last_batch_size: X.shape[0] - last_batch_size] = \
                    cpu_pred[1 - last_n_stream][:batch_size]
            with map_streams[last_n_stream] as stream:
                stream.synchronize()
                cpu_pred_full[X.shape[0] - last_batch_size:] = cpu_pred[last_n_stream][:last_batch_size]

        return cpu_pred_full
