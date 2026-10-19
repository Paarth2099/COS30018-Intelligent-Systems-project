"""Official EMNIST ByClass files, verified mapping, and upright image batches."""
import gzip
import hashlib
from pathlib import Path
import shutil
import string
import zipfile

import numpy as np
import torch
from torchvision.datasets import EMNIST
from torchvision.datasets.utils import download_url

CLASSES = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
ARCHIVE_MD5 = '58c8d27c78d21e728a6bc7b3cc06412e'
ORIENTATION = 'transpose raw EMNIST rows and columns once to upright'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_byclass(root):
    """Verify the official archive and extract only the five ByClass members."""
    raw = Path(root) / 'EMNIST' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    download_url(URL, str(raw), filename='gzip.zip', md5=ARCHIVE_MD5)
    with zipfile.ZipFile(raw / 'gzip.zip') as archive:
        wanted = ['emnist-byclass-mapping.txt'] + [
            f'emnist-byclass-{partition}-{kind}-idx{idx}-ubyte.gz'
            for partition in ('train', 'test')
            for kind, idx in [('images', 3), ('labels', 1)]
        ]
        for name in wanted:
            members = [m for m in archive.namelist() if Path(m).name == name]
            if len(members) != 1:
                raise ValueError(f'Expected one archive member for {name}')
            destination = raw / name.removesuffix('.gz')
            temporary = destination.with_suffix('.partial')
            with archive.open(members[0]) as source, temporary.open('wb') as target:
                if name.endswith('.gz'):
                    with gzip.GzipFile(fileobj=source) as decoded:
                        shutil.copyfileobj(decoded, target)
                else:
                    shutil.copyfileobj(source, target)
            temporary.replace(destination)
            print(f'Extracted {destination.name}', flush=True)
    return raw


def verify_mapping(root):
    path = Path(root) / 'EMNIST/raw/emnist-byclass-mapping.txt'
    pairs = [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]
    expected = [(i, ord(char)) for i, char in enumerate(CLASSES)]
    if pairs != expected or EMNIST.classes_split_dict['byclass'] != CLASSES:
        raise ValueError('EMNIST mapping differs from the expected 62-class order')
    return path


def load_partition(root, train=True):
    verify_mapping(root)
    dataset = EMNIST(root=root, split='byclass', train=train, download=False)
    expected = 697932 if train else 116323
    if tuple(dataset.data.shape) != (expected, 28, 28):
        raise ValueError(f'Unexpected ByClass shape {tuple(dataset.data.shape)}')
    if dataset.targets.min() != 0 or dataset.targets.max() != 61:
        raise ValueError('Unexpected target labels')
    # The binary dataset uses transposed storage. Ordinary upright photo crops
    # MUST NOT receive this dataset-specific transform at prediction time.
    return dataset.data.transpose(1, 2), dataset.targets


def stratified_split(targets, seed=42, fraction=0.1):
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must be between 0 and 1')
    generator = torch.Generator().manual_seed(seed)
    training, validation = [], []
    for label in range(len(CLASSES)):
        indices = torch.where(targets == label)[0]
        if len(indices) < 2:
            raise ValueError(f'Class {label} has fewer than two examples')
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.append(indices[:count]); training.append(indices[count:])
    return torch.cat(training), torch.cat(validation)


def batch_images(images, indices):
    # Keep the full dataset uint8; allocate floats only for the current batch.
    return images[indices].unsqueeze(1).float().div_(255)


def indices_hash(indices):
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()
