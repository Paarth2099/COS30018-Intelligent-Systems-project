"""Torchvision classifiers adapted only at input/output for 28x28 grayscale.

No pretrained weights or ImageNet transforms. Original strides, pooling,
residual blocks and MobileNet width are retained; adaptive pooling accepts 28px.
"""
from torch import nn
from torchvision.models import resnet18, mobilenet_v3_small


def build_extended(name, classes):
    if name == 'resnet18':
        model = resnet18(weights=None, num_classes=classes)
        model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        return model
    if name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=None, num_classes=classes)
        model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(model.features[0][0].weight, mode='fan_out', nonlinearity='relu')
        return model
    raise ValueError(f'Unknown architecture: {name}')
