"""Two small classifiers sharing the same input and output contract."""
from torch import nn


def build_model(name: str) -> nn.Module:
    """Return logits for 10 classes from float32 batches [N, 1, 28, 28]."""
    if name == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    if name == "cnn":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    raise ValueError(f"Unknown model {name!r}; choose 'mlp' or 'cnn'")
