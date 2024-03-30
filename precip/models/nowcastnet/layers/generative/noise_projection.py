import torch
from torch import nn

from ..shared import SpectralNorm


class ProjectionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SpectralNorm(
            nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=1, padding=0)
        )

        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
        )

    def forward(self, x):
        x1 = torch.cat([x, self.conv1(x)], dim=1)
        x2 = self.conv2(x)
        return x1 + x2


class NoiseProjector(nn.Module):
    FACTOR_EXPANSION = [2, 4, 8, 16]

    def __init__(self, input_seq_len: int) -> None:
        super().__init__()

        self.input_seq_len = input_seq_len
        self.projector = nn.Sequential(
            SpectralNorm(
                nn.Conv2d(self.input_seq_len, self.input_seq_len * 2, kernel_size=3, padding=1)
            ),
            *[
                ProjectionBlock(self.input_seq_len * factor, self.input_seq_len * factor * 2)
                for factor in NoiseProjector.FACTOR_EXPANSION
            ]
        )

    def forward(self, x):
        return self.projector(x)
