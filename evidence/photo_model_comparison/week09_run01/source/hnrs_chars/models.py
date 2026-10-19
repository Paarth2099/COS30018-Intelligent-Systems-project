"""62-class counterparts of the existing small digit baselines."""
from torch import nn


def build_model(name):
    if name == 'mlp':
        return nn.Sequential(nn.Flatten(), nn.Linear(784,128), nn.ReLU(),
                             nn.Linear(128,64), nn.ReLU(), nn.Linear(64,62))
    if name == 'cnn':
        return nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                             nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                             nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,62))
    raise ValueError("Choose 'mlp' or 'cnn'")
