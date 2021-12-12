from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


class DicriminatorBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape: Tuple[int, int]):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DicriminatorBlock(64, 128, 2),
            DicriminatorBlock(128, 128, 1),
            DicriminatorBlock(128, 256, 2),
            DicriminatorBlock(256, 256, 1),
            DicriminatorBlock(256, 512, 1),
            DicriminatorBlock(512, 512, 1),
            DicriminatorBlock(512, 512, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(np.prod(input_shape) * 512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        out = self.features(x)
        return self.classifier(torch.flatten(out))


if __name__ == '__main__':
    print(Discriminator((224, 224))(torch.randn(1,3,224,224)).shape)
