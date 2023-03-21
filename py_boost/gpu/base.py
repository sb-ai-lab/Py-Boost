"""Abstracts for the tree ensembles"""
import warnings

try:
    import cupy as cp
except Exception:
    pass
import numpy as np

from .utils import pinned_array
from ..quantization.base import QuantileQuantizer, UniformQuantizer, UniquantQuantizer


class Ensemble:
    """
    Abstract class for tree ensembles.
    Contains prediction, importance and data transfer methods
    """

    @staticmethod
    def _default_postprocess_fn(x):
        return x

    def __init__(self):
        """Initialize ensemble"""
        self.models = None
        self.nfeats = None
        self.postprocess_fn = self._default_postprocess_fn
        self.base_score = None
        self._on_device = False

        self.quantization = 'Quantile'
        self.quant_sample = 200000
        self.max_bin = 256
        self.min_data_in_bin = 3
        self.seed = 42

    def to_device(self):
        """Move trained ensemble data to current GPU device

        Returns:

        """
        if not self._on_device:
            for tree in self.models:
                tree.to_device()
            self.base_score = cp.asarray(self.base_score)

            self._on_device = True

    def to_cpu(self):
        """Move trained ensemble data to CPU memory

        Returns:

        """
        if self._on_device:
            for tree in self.models:
                tree.to_cpu()
            self.base_score = self.base_score.get()

            self._on_device = False

    def __getstate__(self):
        """Get state dict on CPU for picking

        Returns:

        """
        self.to_cpu()
        return self.__dict__

    def quantize(self, X, eval_set):
        """Fit and quantize all sets

        Args:
            X: np.ndarray, train features
            eval_set: list of np.ndarrays, validation features

        Returns:

        """
        quantizer = self.quantization

        if type(quantizer) is str:

            params = {'sample': self.quant_sample, 'max_bin': self.max_bin, 'min_data_in_bin': self.min_data_in_bin,
                      'random_state': self.seed}

            if self.quantization == 'Quantile':
                quantizer = QuantileQuantizer(**params)
            elif self.quantization == 'Uniform':
                quantizer = UniformQuantizer(**params)
            elif self.quantization == 'Uniquant':
                quantizer = UniquantQuantizer(**params)
            else:
                raise ValueError('Unknown quantizer')

        X_enc = quantizer.fit_transform(X)
        eval_enc = [quantizer.transform(x['X']) for x in eval_set]

        return X_enc, quantizer.get_max_bin(), quantizer.get_borders(), eval_enc

    def _predict_deprecated(self, X, batch_size=100000):
        """(DEPRECATED) Make prediction for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of float32, shape (n_data, n_outputs)
        """
        self.to_device()
        prediction = pinned_array(np.empty((X.shape[0], self.base_score.shape[0]), dtype=np.float32))

        n_streams = 2
        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(n_streams)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % n_streams] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                result = cp.zeros((x_batch.shape[0], self.base_score.shape[0]), dtype=np.float32)
                result[:] = self.base_score
                for n, tree in enumerate(self.models):
                    result += tree._predict_deprecated(gpu_batch)

                self.postprocess_fn(result).get(stream=st, out=prediction[i: i + x_batch.shape[0]])

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return prediction

    def _predict_leaves_deprecated(self, X, iterations=None, batch_size=100000):
        """(DEPRECATED) Predict tree leaf indices for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
            for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of uint32, shape (n_iterations, n_data, n_groups).
            For n_groups explanation check Tree class
        """
        if iterations is None:
            iterations = list(range(len(self.models)))

        self.to_device()

        check_grp = np.unique([x.ngroups for x in self.models])
        if check_grp.shape[0] > 1:
            raise ValueError('Different number of groups in trees')

        ngroups = check_grp[0]
        leaves = pinned_array(np.empty((len(iterations), X.shape[0], ngroups), dtype=np.int32))

        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(2)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % 2] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                for j, n in enumerate(iterations):
                    self.models[n]._predict_leaf_deprecated(gpu_batch).get(stream=st, out=leaves[j, i: i + x_batch.shape[0]])

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return leaves

    def _predict_staged_deprecated(self, X, iterations=None, batch_size=100000):
        """(DEPRECATED) Make prediction from different stages for the feature matrix X

        Args:
            X: 2d np.ndarray of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
            for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, 2d np.ndarray of float32, shape (n_iterations, n_data, n_out)
        """

        # Iteration list validation
        if iterations is None:
            iterations = list(range(len(self.models)))
            
        self.to_device()
        prediction = pinned_array(np.empty((len(iterations), X.shape[0], self.base_score.shape[0]), dtype=np.float32))

        map_streams = [cp.cuda.Stream(non_blocking=False) for _ in range(2)]

        stop_events = []

        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            with map_streams[k % 2] as st:
                x_batch = X[i: i + batch_size].astype(np.float32)
                gpu_batch = cp.empty(x_batch.shape, x_batch.dtype)
                x_batch = pinned_array(x_batch)
                gpu_batch.set(x_batch, stream=st)

                result = cp.zeros((x_batch.shape[0], self.base_score.shape[0]), dtype=np.float32)
                result[:] = self.base_score

                next_out = 0
                for n, tree in enumerate(self.models):
                    result += tree._predict_deprecated(gpu_batch)
                    if n == iterations[next_out]:
                        self.postprocess_fn(result).get(
                            stream=st, out=prediction[next_out, i: i + x_batch.shape[0]]
                        )

                        next_out += 1
                        if next_out >= len(iterations):
                            break

                stop_event = st.record()
                stop_events.append(stop_event)

        curr_stream = cp.cuda.get_current_stream()
        for stop_event in stop_events:
            curr_stream.wait_event(stop_event)
        curr_stream.synchronize()
        return prediction

    def _get_feature_importance_deprecated(self, imp_type='split'):
        """(DEPRECATED) Get feature importance

        Args:
            imp_type: str, importance type, 'split' or 'gain'

        Returns:

        """
        self.to_cpu()

        assert imp_type in ['gain', 'split'], "Importance type should be 'gain' or 'split'"
        importance = np.zeros(self.nfeats, dtype=np.float32)

        for tree in self.models:
            sl = tree.feats >= 0
            acc_val = 1 if imp_type == 'split' else tree.gains[sl]
            np.add.at(importance, tree.feats[sl], acc_val)

        return importance

    def get_feature_importance(self, imp_type='split'):
        """Get feature importance

        Args:
            imp_type: str, importance type, 'split' or 'gain'

        Returns:
            importance: np.ndarray 1d of float32, shape (n_features)
        """
        assert imp_type in ['gain', 'split'], "Importance type should be 'gain' or 'split'"
            
        importance = np.zeros(self.nfeats, dtype=np.float32)

        for tree in self.models:
            if imp_type == 'split':
                if type(tree.feature_importance_split) is not np.ndarray:
                    importance += tree.feature_importance_split.get()
                else:
                    importance += tree.feature_importance_split
            else:
                if type(tree.feature_importance_gain) is not np.ndarray:
                    importance += tree.feature_importance_gain.get()
                else:
                    importance += tree.feature_importance_gain

        return importance

    def predict_leaves(self, X, iterations=None, batch_size=100_000):
        """Predict tree leaf indices for the feature matrix X

        Args:
            X: 2d np.ndarray of float32, array of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
                for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, np.ndarray 2d of int32, shape (n_iterations, n_data, n_groups).
            For n_groups explanation check Tree class
        """
        assert batch_size > 0, 'Batch size must be a positive integer'
        
        if iterations is None:
            iterations = np.arange(len(self.models))
        else:
            iterations = np.array(iterations, dtype=np.int64)
            
        assert len(iterations) > 0, 'Iterations are empty sequence'
        assert (max(iterations) < len(self.models)) and (min(iterations) >= 0), 'Invalid stage numbers'
        assert len(np.unique(iterations)) == len(iterations), 'Duplicate values in stages are not allowed'

        check_grp = np.unique([x.ngroups for x in self.models])
        assert len(check_grp) == 1, 'Different number of groups in trees'
        
        ngroups = check_grp[0]

        self.to_device()
        # special case handle, if X is already on device or size of X <= batch_size
        if type(X) is cp.ndarray or X.shape[0] <= batch_size:
            is_on_gpu = True
            if type(X) is not cp.ndarray:
                is_on_gpu = False
                X = cp.array(X, order='C', dtype=cp.float32)
            if not (X.flags['C_CONTIGUOUS'] or X.flags['F_CONTIGUOUS']):
                warnings.warn("X is not 'C_CONTIGUOUS' or 'F_CONTIGUOUS', contiguous copy of array will be created."
                              "To reduce inference time, make sure X is contiguous beforehand")
                X = cp.ascontiguousarray(X)
            gpu_pred = cp.empty((len(iterations), X.shape[0], ngroups), dtype=np.int32)

            for j, n in enumerate(iterations):
                self.models[n].predict_leaf(X, gpu_pred[j])

            cp.cuda.get_current_stream().synchronize()
            if is_on_gpu:
                return gpu_pred
            return gpu_pred.get()

        n_streams = 2  # don't change
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # result allocation
        cpu_leaves_full = np.empty((len(iterations), X.shape[0], ngroups), dtype=np.int32)
        cpu_leaves = [pinned_array(np.empty((len(iterations), batch_size, ngroups), dtype=np.int32)) for _ in range(n_streams)]
        gpu_leaves = [cp.empty((len(iterations), batch_size, ngroups), dtype=cp.int32) for _ in range(n_streams)]

        # batch allocation
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=np.float32)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=cp.float32) for _ in range(n_streams)]

        cpu_batch_free_event = [None, None]
        gpu_batch_free_event = [None, None]
        cpu_out_ready_event = [None, None]
        last_batch_size = 0
        last_n_stream = 0
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                # if cpu_batch is available, copy X batch to pinned memory on H
                if k >= 2:
                    cpu_batch_free_event[nst].synchronize()
                cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(np.float32)

                # copy X batch from pinned memory to D
                if k >= 2:
                    gpu_batch_free_event[nst].synchronize()
                gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                cpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                # when gpu_leaves is available and cpu_out is ready, proceed to the prediction
                if k >= 2:
                    cpu_out_ready_event[nst].synchronize()
                for j, n in enumerate(iterations):
                    self.models[n].predict_leaf(gpu_batch[nst][:real_batch_len], gpu_leaves[nst][j][:real_batch_len])
                gpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                # copy prediction from pinned memory to pageable memory on H (from previous iteration)
                if k >= 2:
                    for j, n in enumerate(iterations):
                        cpu_leaves_full[j][i - 2 * batch_size: i - batch_size] = cpu_leaves[nst][j][:batch_size]

                # copy predictions from device to pinned memory on H
                gpu_leaves[nst][:real_batch_len].get(out=cpu_leaves[nst][:real_batch_len])
                cpu_out_ready_event[nst] = stream.record(cp.cuda.Event(block=True))

                last_batch_size = real_batch_len
                last_n_stream = nst

        with map_streams[1 - last_n_stream]:
            cpu_out_ready_event[1 - last_n_stream].synchronize()
            for j, n in enumerate(iterations):
                cpu_leaves_full[j][X.shape[0] - (batch_size + last_batch_size): X.shape[0] - last_batch_size] = \
                    cpu_leaves[1 - last_n_stream][j][:batch_size]
        with map_streams[last_n_stream]:
            cpu_out_ready_event[last_n_stream].synchronize()
            for j, n in enumerate(iterations):
                cpu_leaves_full[j][X.shape[0] - last_batch_size:] = cpu_leaves[last_n_stream][j][:last_batch_size]

        return cpu_leaves_full

    def predict(self, X, batch_size=100_000):
        """Make prediction for the feature matrix X

        Args:
            X: np.ndarray 2d of float32, array of features
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction: np.ndarray 2d of float32, shape (n_data, n_outputs)
        """
        assert batch_size > 0, 'Batch size must be a positive integer'
        
        ngroups = max((x.ngroups for x in self.models))
        n_out = self.base_score.shape[0]

        self.to_device()
        # special case handle, if X is already on device or size of X <= batch_size
        if type(X) is cp.ndarray or X.shape[0] <= batch_size:
            is_on_gpu = True
            if type(X) is not cp.ndarray:
                is_on_gpu = False
                X = cp.array(X, order='C', dtype=cp.float32)
            if not(X.flags['C_CONTIGUOUS'] or X.flags['F_CONTIGUOUS']):
                warnings.warn("X is not 'C_CONTIGUOUS' or 'F_CONTIGUOUS', contiguous copy of array will be created."
                              "To reduce inference time, make sure X is contiguous beforehand")
                X = cp.ascontiguousarray(X)

            gpu_pred = cp.empty((X.shape[0], n_out), dtype=cp.float32)
            gpu_pred_leaves = cp.empty((X.shape[0], ngroups), dtype=cp.int32)

            gpu_pred[:] = self.base_score

            for tree in self.models:
                tree.predict(X, gpu_pred, gpu_pred_leaves)

            cp.cuda.get_current_stream().synchronize()
            pred = self.postprocess_fn(gpu_pred)
            if is_on_gpu:
                return pred
            return pred.get()

        # cuda stream allocation
        n_streams = 2  # don't change
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # result allocation
        cpu_pred_full = np.empty((X.shape[0], n_out), dtype=np.float32)
        cpu_pred = [pinned_array(np.empty((batch_size, n_out), dtype=np.float32)) for _ in range(n_streams)]
        gpu_pred = [cp.empty((batch_size, n_out), dtype=cp.float32) for _ in range(n_streams)]

        # temp buffer for leaves
        gpu_pred_leaves = [cp.empty((batch_size, ngroups), dtype=cp.int32) for _ in range(n_streams)]

        # batch allocation
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=np.float32)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=cp.float32) for _ in range(n_streams)]

        cpu_batch_free_event = [None, None]
        gpu_batch_free_event = [None, None]
        cpu_out_ready_event = [None, None]
        last_batch_size = 0
        last_n_stream = 0
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                # if cpu_batch is available, copy X batch to pinned memory on H
                if k >= 2:
                    cpu_batch_free_event[nst].synchronize()
                cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(cp.float32)

                # copy X batch from pinned memory to D
                if k >= 2:
                    gpu_batch_free_event[nst].synchronize()
                gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                cpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                # when gpu_pred is available and cpu_out is ready, proceed to the prediction
                if k >= 2:
                    cpu_out_ready_event[nst].synchronize()
                gpu_pred[nst][:] = self.base_score
                for tree in self.models:
                    tree.predict(gpu_batch[nst][:real_batch_len], gpu_pred[nst][:real_batch_len],
                                 gpu_pred_leaves[nst][:real_batch_len])
                gpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))

                # copy prediction from pinned memory to pageable memory on H (from previous iteration)
                if k >= 2:
                    cpu_pred_full[i - 2 * batch_size: i - batch_size] = cpu_pred[nst][:batch_size]

                # copy predictions from device to pinned memory on H
                self.postprocess_fn(gpu_pred[nst][:real_batch_len]).get(out=cpu_pred[nst][:real_batch_len])
                cpu_out_ready_event[nst] = stream.record(cp.cuda.Event(block=True))

                last_batch_size = real_batch_len
                last_n_stream = nst

        # waiting for sync of last two batches
        with map_streams[1 - last_n_stream]:
            cpu_out_ready_event[1 - last_n_stream].synchronize()
            cpu_pred_full[X.shape[0] - (batch_size + last_batch_size): X.shape[0] - last_batch_size] = \
                cpu_pred[1 - last_n_stream][:batch_size]
        with map_streams[last_n_stream]:
            cpu_out_ready_event[last_n_stream].synchronize()
            cpu_pred_full[X.shape[0] - last_batch_size:] = cpu_pred[last_n_stream][:last_batch_size]

        return cpu_pred_full

    def predict_staged(self, X, iterations=None, batch_size=100_000):
        """Make prediction from different stages for the feature matrix X

        Args:
            X: 2d np.ndarray of float32, array of features
            iterations: list of int or None. If list of ints is passed, prediction will be made only
                for given iterations, otherwise - for all iterations
            batch_size: int, inner batch splitting size to avoid OOM

        Returns:
            prediction, np.ndarray 2d of float32, shape (n_iterations, n_data, n_out)
        """
        assert batch_size > 0, 'Batch size must be a positive integer'
        
        if iterations is None:
            iterations = np.arange(len(self.models))
        else:
            iterations = np.array(iterations, dtype=np.int64)
            
        assert len(iterations) > 0, 'Iterations are empty sequence'
        assert (max(iterations) < len(self.models)) and (min(iterations) >= 0), 'Invalid stage numbers'
        assert len(np.unique(iterations)) == len(iterations), 'Duplicate values in stages are not allowed'
        
        ngroups = max((x.ngroups for x in self.models))
        n_out = self.base_score.shape[0]

        self.to_device()
        # special case handle, if X is already on device or size of X <= batch_size
        if type(X) is cp.ndarray or X.shape[0] <= batch_size:
            is_on_gpu = True
            if type(X) is not cp.ndarray:
                is_on_gpu = False
                X = cp.array(X, order='C', dtype=cp.float32)
                pred_full = np.empty((len(iterations), X.shape[0], n_out), dtype=cp.float32)
            else:
                pred_full = cp.empty((len(iterations), X.shape[0], n_out), dtype=cp.float32)
            if not (X.flags['C_CONTIGUOUS'] or X.flags['F_CONTIGUOUS']):
                warnings.warn("X is not 'C_CONTIGUOUS' or 'F_CONTIGUOUS', contiguous copy of array will be created."
                              "To reduce inference time, make sure X is contiguous beforehand")
                X = cp.ascontiguousarray(X)

            gpu_pred = cp.empty((X.shape[0], n_out), dtype=cp.float32)
            gpu_pred_leaves = cp.empty((X.shape[0], ngroups), dtype=cp.int32)

            gpu_pred[:] = self.base_score
            next_out = 0
            for n, tree in enumerate(self.models):
                tree.predict(X, gpu_pred, gpu_pred_leaves)
                if n == iterations[next_out]:
                    if is_on_gpu:
                        pred_full[next_out] = self.postprocess_fn(gpu_pred)
                    else:
                        self.postprocess_fn(gpu_pred).get(out=pred_full[next_out])
                    next_out += 1
                    if next_out >= len(iterations):
                        cp.cuda.get_current_stream().synchronize()
                        return pred_full

        n_streams = 2  # don't change
        map_streams = [cp.cuda.Stream() for _ in range(n_streams)]

        # result allocation
        cpu_pred_full = np.empty((len(iterations), X.shape[0], n_out), dtype=np.float32)
        cpu_pred = [pinned_array(np.empty((len(iterations), batch_size, n_out), dtype=np.float32)) for _ in range(n_streams)]
        gpu_pred = [cp.empty((batch_size, n_out), dtype=cp.float32) for _ in range(n_streams)]

        # temp buffer for leaves
        gpu_pred_leaves = [cp.empty((batch_size, ngroups), dtype=cp.int32) for _ in range(n_streams)]

        # batch allocation
        cpu_batch = [pinned_array(np.empty(X[0:batch_size].shape, dtype=np.float32)) for _ in range(n_streams)]
        gpu_batch = [cp.empty(X[0:batch_size].shape, dtype=np.float32) for _ in range(n_streams)]

        cpu_batch_free_event = [None, None]
        cpu_out_ready_event = [None, None]
        last_batch_size = 0
        last_n_stream = 0
        for k, i in enumerate(range(0, X.shape[0], batch_size)):
            nst = k % n_streams
            with map_streams[nst] as stream:
                real_batch_len = batch_size if i + batch_size <= X.shape[0] else X.shape[0] - i

                # if cpu_batch is available, copy X batch to pinned memory on H
                if k >= 2:
                    cpu_batch_free_event[nst].synchronize()
                cpu_batch[nst][:real_batch_len] = X[i:i + real_batch_len].astype(np.float32)

                # copy X batch from pinned memory to D
                if k >= 2:
                    cpu_out_ready_event[nst].synchronize()
                gpu_batch[nst][:real_batch_len].set(cpu_batch[nst][:real_batch_len])
                cpu_batch_free_event[nst] = stream.record(cp.cuda.Event(block=True))
                gpu_pred[nst][:] = self.base_score

                # copy prediction from pinned memory to pageable memory on H (from previous iteration)
                if k >= 2:
                    cpu_pred_full[:, i - 2 * batch_size: i - batch_size] = cpu_pred[nst][:, :batch_size]

                i_next = 0
                for i_tree, tree in enumerate(self.models):
                    tree.predict(gpu_batch[nst][:real_batch_len], gpu_pred[nst][:real_batch_len],
                                 gpu_pred_leaves[nst][:real_batch_len])
                    if iterations[i_next] == i_tree:
                        self.postprocess_fn(gpu_pred[nst][:real_batch_len]).get(out=cpu_pred[nst][i_next][:real_batch_len])
                        i_next += 1
                        if i_next >= len(iterations):
                            break
                cpu_out_ready_event[nst] = stream.record(cp.cuda.Event(block=True))

                last_batch_size = real_batch_len
                last_n_stream = nst

        # waiting for sync of last two batches
        with map_streams[1 - last_n_stream]:
            cpu_out_ready_event[1 - last_n_stream].synchronize()
            cpu_pred_full[:, X.shape[0] - (batch_size + last_batch_size): X.shape[0] - last_batch_size] =\
                cpu_pred[1 - last_n_stream][:, :batch_size]
        with map_streams[last_n_stream]:
            cpu_out_ready_event[last_n_stream].synchronize()
            cpu_pred_full[:, X.shape[0] - last_batch_size:] = cpu_pred[last_n_stream][:, :last_batch_size]

        return cpu_pred_full
