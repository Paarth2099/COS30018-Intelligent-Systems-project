"""A fixed development split, separated from the official final test set."""
import hashlib

import torch
from torchvision.datasets import MNIST


def load_partition(root, train=True, download=False):
    dataset = MNIST(root=root, train=train, download=download)
    images = dataset.data.unsqueeze(1).float().div_(255)
    return images, dataset.targets.clone()


def development_split(count=60000, seed=42, validation_size=6000):
    if not 0 < validation_size < count:
        raise ValueError("Validation size must be between zero and dataset size")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    return order[validation_size:], order[:validation_size]


def indices_hash(indices):
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()
