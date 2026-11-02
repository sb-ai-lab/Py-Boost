"""Pinned Delicious data and the five-fold evaluation protocol from SketchBoost.

No pickled data or models are downloaded or deserialized by this harness.
"""

import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

DATA_URL = 'https://drive.google.com/uc?export=download&id=0B3lPMIHmG6vGdG1jZ19VS2NWRVU'
ARCHIVE_SHA256 = '9a4e79e9bd1835225e59eef09ba0ee29c5430f785045a9245943511836542806'
FILES = {
    'Delicious/Delicious_data.txt': '7c6732b0ba3e2acb8b6f8cd147b83107f798e50fedf47c3429f549d53d971b9b',
    'Delicious/delicious_trSplit.txt': '8dbe2f693e62ae6e45ee617db8372fd5f547f4bb2f59e17c47f55c97b7977d1e',
    'Delicious/delicious_tstSplit.txt': 'aac14f9f84e6fe3f77d81db8ceb04be935e72adf22d9284d78d58d7763f242fa',
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / '.benchmark-data'


def array_hash(*arrays):
    digest = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        digest.update(str((arr.shape, arr.dtype.str)).encode())
        digest.update(memoryview(arr).cast('B'))
    return digest.hexdigest()


def read_archive(archive):
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Delicious archive SHA-256 mismatch')
    with ZipFile(archive) as zipped:
        contents = {}
        for name, expected in FILES.items():
            value = zipped.read(name)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError(f'Delicious file SHA-256 mismatch: {name}')
            contents[name] = value.decode('ascii').splitlines()
    lines = contents['Delicious/Delicious_data.txt']
    if lines[0].split() != ['16105', '500', '983'] or len(lines) != 16106:
        raise ValueError('Unexpected Delicious dimensions')
    X = np.zeros((16105, 500), dtype='<f4')
    y = np.zeros((16105, 983), dtype='<f4')
    for row, line in enumerate(lines[1:]):
        values = line.split()
        if ':' not in values[0]:
            y[row, list(map(int, values.pop(0).split(',')))] = 1
        for item in values:
            column, value = item.split(':')
            X[row, int(column)] = float(value)
    # Match preprocess.py: first column, convert the one-based official split.
    train, test = [np.array([int(line.split()[0]) - 1 for line in contents[name]], dtype='<i8')
                   for name in list(FILES)[1:]]
    if not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(16105)):
        raise ValueError('Official Delicious split must partition all rows')
    return X, y, train, test


def prepare_data(data_dir, archive=None):
    """Download explicitly, check immutable bytes, save arrays with no pickle."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / 'delicious.npz'
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        if archive is None:
            archive = Path(temporary) / 'Delicious.zip'
            with urlopen(DATA_URL, timeout=60) as response, archive.open('wb') as output:
                size = 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 16 * 1024 * 1024:
                        raise ValueError('Unexpected Delicious archive size')
                    output.write(block)
        arrays = read_archive(archive)
        prepared = Path(temporary) / 'delicious.npz'
        np.savez_compressed(prepared, **dict(zip(['X', 'y', 'train', 'test'], arrays)))
        prepared.replace(destination)
    return destination, array_hash(*arrays)


def load_fold(case, seed, data_dir):
    from sklearn.model_selection import KFold

    path = Path(data_dir or DEFAULT_DATA_DIR) / 'delicious.npz'
    if not path.is_file():
        raise ValueError(f'Missing {path}; run benchmarks/run.py prepare-data first')
    with np.load(path, allow_pickle=False) as cache:
        X, y, train, test = [cache[name] for name in ['X', 'y', 'train', 'test']]
    # Validate the actual normalized arrays every time, including split ordering.
    if array_hash(X, y, train, test) != NORMALIZED_SHA256:
        raise ValueError('Prepared Delicious dataset checksum mismatch; prepare it again')
    folds = list(KFold(5, shuffle=True, random_state=seed).split(train))
    fit, valid = [train[indices] for indices in folds[case['fold']]]
    if [len(fit), len(valid), len(test)] != [case['train'], case['valid'], case['test']]:
        raise ValueError('Delicious fold sizes differ from manifest')
    digest = array_hash(X, y, fit, valid, test)
    return X[fit], y[fit], [{'X': X[valid], 'y': y[valid]}], X[test], y[test], digest


# Computed from the checked archive, float32 arrays and ordered official splits.
NORMALIZED_SHA256 = 'a53ca49cdce00c38f90225994ba01e6b352e9445d121df35fcd177a1cbd5c676'
