"""Data and result contracts shared by every tested library version."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def implementation_hash():
    digest = hashlib.sha256()
    for name in ['suite-v1.json', 'protocol.py', 'gpu.py', 'run.py']:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def load_suite():
    return json.loads((ROOT / 'suite-v1.json').read_text())


def make_data(case, seed):
    """PCG64, float32 inputs, eight correlated output groups, no external data."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = case['train'] + case['valid'] + case['test']
    X = rng.standard_normal((n, case['features']), dtype=np.float32)
    outputs = case['outputs']
    if case['task'] == 'multiclass':
        bits = (outputs - 1).bit_length()
        y = np.zeros(n, dtype=np.int32)
        for bit in range(bits):
            y |= (X[:, bit] > 0).astype(np.int32) << bit
        y %= outputs
        # Ensure every class is represented in training, with consistent features.
        for bit in range(bits):
            X[:outputs, bit] = np.where((np.arange(outputs) >> bit) & 1, .5, -.5)
        y[:outputs] = np.arange(outputs)
    else:
        latent = np.stack([X[:, 0], X[:, 1], X[:, 2] * X[:, 3],
                           np.maximum(X[:, 4], 0), np.sin(X[:, 5]),
                           X[:, 6] + X[:, 7], X[:, 8] - X[:, 9],
                           (X[:, 10] > 0).astype(np.float32) - .5], axis=1)
        y = np.empty((n, outputs), dtype=np.float32)
        for start in range(0, outputs, 128):
            stop = min(start + 128, outputs)
            columns = np.arange(start, stop)
            signal = latent[:, columns % 8] + .3 * latent[:, (columns + 3) % 8]
            noise = rng.standard_normal(signal.shape, dtype=np.float32)
            if case['task'] == 'regression':
                y[:, start:stop] = signal + .1 * noise
            else:
                probability = 1 / (1 + np.exp(-signal))
                y[:, start:stop] = rng.random(signal.shape, dtype=np.float32) < probability
    digest = hashlib.sha256()
    for arr in [X, y]:
        digest.update(str((arr.shape, arr.dtype.str)).encode())
        digest.update(memoryview(arr).cast('B'))
    t, v = case['train'], case['train'] + case['valid']
    validation = []
    for indices in np.array_split(np.arange(t, v), case['validation_sets']):
        validation.append({'X': X[indices[0]:indices[-1] + 1],
                           'y': y[indices[0]:indices[-1] + 1]})
    return X[:t], y[:t], validation, X[v:], y[v:], digest.hexdigest()


def quality(task, target, prediction):
    """Holdout metric outside timed regions, with bounded float64 temporaries."""
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite predictions')
    if prediction.shape[0] != target.shape[0]:
        raise ValueError('Prediction row count differs from target')
    total, count = 0., 0
    for start in range(0, len(target), 256):
        y = target[start:start + 256]
        p = prediction[start:start + 256].astype(np.float64)
        if task == 'multiclass':
            if p.ndim != 2 or (y.max() >= p.shape[1]):
                raise ValueError('Missing class probabilities')
            values = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
        elif task == 'regression':
            if p.shape != y.shape:
                raise ValueError('Prediction output count differs from target')
            values = (p - y.astype(np.float64)) ** 2
        else:
            if p.shape != y.shape:
                raise ValueError('Prediction output count differs from target')
            p = np.clip(p, 1e-15, 1 - 1e-15)
            values = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        total += float(values.sum())
        count += values.size
    value = total / count
    return {'name': 'rmse' if task == 'regression' else 'logloss',
            'value': float(np.sqrt(value) if task == 'regression' else value),
            'lower_is_better': True}


def summarize(values):
    return {'median': float(np.median(values)), 'p10': float(np.quantile(values, .1)),
            'p90': float(np.quantile(values, .9)), 'samples': values}


def comparison_errors(base, head):
    errors = []
    for key in ['schema_version', 'benchmark_id', 'implementation_hash', 'suite_hash',
                'case_ids', 'repeats', 'warmups']:
        if base.get(key) != head.get(key):
            errors.append(f'Different {key}')
    if base.get('environment') != head.get('environment'):
        errors.append('Different hardware or software environment')
    for label, report in [('base', base), ('candidate', head)]:
        if not report.get('case_ids') or set(report.get('case_ids', [])) != set(report.get('cases', {})):
            errors.append(f'{label}: missing case results')
    for case_id in sorted(set(base.get('cases', {})) & set(head.get('cases', {}))):
        a, b = base['cases'][case_id], head['cases'][case_id]
        if a.get('status') != 'ok' or b.get('status') != 'ok':
            errors.append(f'{case_id}: failed or incomplete measurement')
        elif a.get('data_hash') != b.get('data_hash'):
            errors.append(f'{case_id}: different dataset checksum')
        elif not a.get('data_hash'):
            errors.append(f'{case_id}: missing dataset checksum')
        if a.get('spec') != b.get('spec'):
            errors.append(f'{case_id}: different case parameters')
    if any(c.get('status') != 'ok' for c in list(base.get('cases', {}).values()) + list(head.get('cases', {}).values())):
        errors.append('One or more cases did not complete')
    if not base.get('complete') or not head.get('complete'):
        errors.append('A report is incomplete')
    return errors
