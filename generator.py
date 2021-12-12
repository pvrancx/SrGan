import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return x + self.layers(x)


class ShuffleBlock(nn.Module):
    def __init__(self, in_planes: int = 64, out_planes: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return self.prelu(out)


class Generator(nn.Module):
    def __init__(self, n_blocks: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(n_blocks)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(64)
        self.shuffle = nn.Sequential(ShuffleBlock(64, 256), ShuffleBlock(64, 256))
        self.out = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        features = self.prelu(self.conv1(x))
        out = self.blocks(features)
        out = out + features
        out = self.shuffle(out)
        return self.out(out)


if __name__ == '__main__':
    print(Generator()(torch.randn(1, 3, 224, 224)).shape)


