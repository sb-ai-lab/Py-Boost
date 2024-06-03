import warnings

import numpy as np
import onnx
import onnxruntime
import tqdm
from onnx import helper, TensorProto
from onnx.checker import check_model


def pb_to_onnx(model, output, fltr=None, post_transform='NONE'):
    """Parse the model to ONNX format

    Args:
        model: Py-Boost Ensemble
        output: str, file path
        fltr: str or None, if subset of outputs need to be stored
        post_transform: str, one of 'NONE', 'SOFTMAX', 'LOGISTIC', 'SOFTMAX_ZERO', or 'PROBIT'

    Returns:

    """
    model.to_cpu()

    nout = model.base_score.shape[0]
    if fltr is None:
        fltr = np.arange(nout)
    else:
        fltr_ = np.asarray(fltr)
        fltr = np.sort(fltr_)
        if (fltr != fltr_).any():
            warnings.warn(
                'Selected outputs order changed. Predictions will keep the original model order (fltr array is sorted)'
            )

    nout = len(fltr)

    parsed_ensemble = {

        # const for ensemble
        "base_values": model.base_score[fltr].tolist(),
        "n_targets": nout,
        "aggregate_function": "SUM",
        "post_transform": post_transform

    }

    nodes_attr = [
        "nodes_treeids", "nodes_nodeids", "nodes_modes", "nodes_falsenodeids",
        "nodes_truenodeids", "nodes_featureids", "nodes_values", "nodes_missing_value_tracks_true"
    ]

    leaves_attr = [
        "target_ids", "target_nodeids", "target_treeids", "target_weights"
    ]

    for key in nodes_attr + leaves_attr:
        parsed_ensemble[key] = []

    k = 0
    for tree in tqdm.tqdm(model.models):

        g = 0
        for offset in tree.test_format_offsets:

            offset = offset * 4
            outputs = np.setdiff1d(fltr, np.nonzero(tree.group_index != g)[0])
            # old id and new id
            nodes, n = [(0, 0)], 0

            while len(nodes) > 0:

                # placeholder for new nodes
                new_nodes = []

                # first - adding the node
                for old, new in nodes:

                    parsed_ensemble["nodes_treeids"].append(k)
                    parsed_ensemble["nodes_nodeids"].append(new)

                    if old >= 0:
                        # case - split node
                        i = old * 4
                        f, s, l, r = tree.test_format[offset + i: offset + i + 4]
                        f, l, r = int(f), int(l), int(r)

                        parsed_ensemble["nodes_modes"].append("BRANCH_LEQ")
                        # check NaN condition
                        nan_left = f < 0
                        f = abs(f) - 1

                        parsed_ensemble["nodes_truenodeids"].append(n + 1)
                        parsed_ensemble["nodes_falsenodeids"].append(n + 2)
                        parsed_ensemble["nodes_missing_value_tracks_true"].append(nan_left)
                        parsed_ensemble["nodes_featureids"].append(f)
                        parsed_ensemble["nodes_values"].append(float(s))
                        new_nodes.extend([(l, n + 1), (r, n + 2)])
                        n = n + 2

                    else:
                        # case leaf node
                        leaf = abs(old) - 1
                        parsed_ensemble["nodes_modes"].append("LEAF")
                        # add dummy children info
                        parsed_ensemble["nodes_truenodeids"].append(-1)
                        parsed_ensemble["nodes_falsenodeids"].append(-1)
                        parsed_ensemble["nodes_missing_value_tracks_true"].append(False)
                        parsed_ensemble["nodes_featureids"].append(-1)
                        parsed_ensemble["nodes_values"].append(0.0)
                        # add leaf info
                        for j, o in zip(outputs, np.searchsorted(fltr, outputs)):
                            parsed_ensemble["target_ids"].append(o)
                            parsed_ensemble["target_nodeids"].append(new)
                            parsed_ensemble["target_treeids"].append(k)
                            parsed_ensemble["target_weights"].append(float(tree.values[leaf, j]))

                nodes = new_nodes

            k += 1
            g += 1

    # create a model
    node_proto = helper.make_node(
        op_type="TreeEnsembleRegressor",
        inputs=["X"], outputs=["Y"],
        domain='ai.onnx.ml',
    )
    node_proto.attribute.extend([helper.make_attribute(x, parsed_ensemble[x]) for x in parsed_ensemble])

    X_ft = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y_out = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, nout])

    graph_def = helper.make_graph(
        [node_proto],  # nodes
        'py-boost-ensemble',  # name
        [X_ft],  # inputs
        [Y_out]  # outputs
    )

    model_def = helper.make_model(
        graph_def, producer_name="Py-Boost",
        opset_imports=[
            onnx.helper.make_opsetid('ai.onnx.ml', 3),
            onnx.helper.make_opsetid('', 16),
        ]
    )

    check_model(model_def)

    with open(output, "wb") as f:
        f.write(model_def.SerializeToString())

    return


class ONNXPredictor:
    """
    ONNX parser and CPU predictor. Could be used for inference of Py-Boost model on CPU
    """

    def __init__(self, model, filepath, postprocess_fn=None, fltr=None, post_transform='NONE'):
        """

        Args:
            model: Py-Boost model
            filepath: str, filepath to save
            postprocess_fn: Callable or None, python postprocess_fn. If passed, model postprocessing will be ignored
                and replaced
            fltr: Sequence, indices to use for inference if needed to filter
            post_transform: str, one of 'NONE', ‘SOFTMAX,’ ‘LOGISTIC,’ ‘SOFTMAX_ZERO,’ or ‘PROBIT’.
                Built-in ONNX post_transform function. If passed, both model postprocessing and python postprocess_fn
                will be ignored
        """
        if model is not None:
            pb_to_onnx(model, output=filepath, fltr=fltr, post_transform=post_transform)

        self.filepath = filepath

        # store post transform fn
        if post_transform != 'NONE':
            self.postprocess_fn = None
        else:
            self.postprocess_fn = postprocess_fn
            if postprocess_fn is None and model is not None:
                self.postprocess_fn = model.postprocess_fn

        self.sess = None
        self._start_session()

    @classmethod
    def from_onnx(cls, filepath, postprocess_fn=None):
        """Create ONNX predictor from parsed model

        Args:
            filepath: str, file path
            postprocess_fn: Callable or None

        Returns:

        """
        return cls(None, filepath, postprocess_fn=postprocess_fn, fltr=None, post_transform='NONE')

    def _start_session(self):
        """Start inference session

        Returns:

        """
        self.sess = onnxruntime.InferenceSession(
            self.filepath,
            providers=["CPUExecutionProvider"]
        )

        return

    def predict(self, X):
        """Predict with ONNX runtime

        Args:
            X: np.ndarray, feature matrix

        Returns:
            np.ndarray
        """

        X = X.astype(np.float32, copy=False)
        preds = self.sess.run(['Y'], {'X': X})[0]

        if self.postprocess_fn is not None:
            preds = self.postprocess_fn(preds)

        return preds
