import numpy as np
import tqdm
import ujson as json

from .tree import Tree


def nested_float_cast(arr):
    if type(arr[0]) is list:
        for i in range(len(arr)):
            arr[i] = nested_float_cast(arr[i])

    else:
        for i in range(len(arr)):
            arr[i] = float(arr[i])

    return arr


def handle_float(arr):
    arr = arr.astype(str)
    arr = arr.tolist()
    arr = nested_float_cast(arr)

    return arr


def parse_tree(tree):
    """Parse single tree

    Args:
        tree: Py-Boost Tree

    Returns:
        dict
    """
    D = {}

    for key in tree.__dict__:

        value = tree.__dict__[key]
        if value is None:
            continue

        if type(value) is np.ndarray:

            if np.issubdtype(value.dtype, np.floating):
                value = handle_float(value)
            else:
                value = value.tolist()

        D[key] = value

    return D


def parse_model(model):
    """Parse model

    Args:
        model: Py-Boost Ensemble

    Returns:
        dict
    """
    model.to_cpu()

    D = {'base_score': handle_float(model.base_score)}

    for n, tree in enumerate(tqdm.tqdm(model.models)):
        D[n] = parse_tree(tree)

    return D


def dump(model, file):
    """Parse model and save the results

    Args:
        model: Py-Boost Ensemble
        file: str, path to file

    Returns:

    """
    with open(file, 'w') as f:
        json.dump(parse_model(model), f)

    return


attr_types = {

    'values': np.float32,
    'group_index': np.uint64,
    'feature_importance_gain': np.float32,
    'feature_importance_split': np.float32,
    'test_format': np.float32,
    'test_format_offsets': np.int32

}


def load_tree(D):
    """Create single tree from dict

    Args:
        D: dict

    Returns:
        Py-Boost Tree
    """
    tree = Tree(1, 1, 1)
    # delete unused attrs
    for key in ['gains', 'feats', 'bin_splits', 'nans', 'split', 'val_splits', 'leaves']:
        setattr(tree, key, None)

    # set new attrs
    for key in D:
        value = D[key]

        if type(value) is list:
            value = np.asarray(value)

        if key in attr_types:
            value = value.astype(attr_types[key])

        setattr(tree, key, value)

    return tree


def load_model(D, model):
    """Update model data with dict values

    Args:
        D: dict
        model: Py-Boost Ensemble

    Returns:
        Py-Boost Ensemble
    """
    model.base_score = np.asarray(D.pop('base_score')).astype(np.float32)

    trees = [None] * len(D)

    for key in D:
        trees[int(key)] = load_tree(D[key])

    model.models = trees

    return model


def load(model, file):
    """Read data from json and update Py-Boost model data

    Args:
        model: Py-Boost Ensemble
        file: str, file path

    Returns:
        Py-Boost Ensemble
    """
    with open(file, 'r') as f:
        load_model(json.load(f), model)

    return model
