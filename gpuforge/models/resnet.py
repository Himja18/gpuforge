"""Compact ResNet for 32x32 images (CIFAR-style). Conv-heavy on purpose, so the
cuDNN convolution path, channels_last memory format and TensorRT layer fusion
(Conv+BN+ReLU) all have something to work on."""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.short(x))


class ResNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32, blocks=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width),
                                  nn.ReLU(inplace=True))
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * (2 ** i)
            for j in range(n):
                layers.append(BasicBlock(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # zero-init last BN of each residual branch: each block starts as identity
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


def build_model(name: str = "resnet_small", **kw) -> nn.Module:
    if name == "resnet_small":
        return ResNetSmall(**kw)
    if name == "resnet_tiny":  # for CPU smoke tests
        return ResNetSmall(width=8, blocks=(1, 1, 1), **kw)
    raise ValueError(f"unknown model {name}")
