import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


def padded_reshape(x: torch.Tensor, y: torch.Tensor):
    """Aligns last two dimensions of y to x via padding."""
    diffY = x.size(-2) - y.size(-2)
    diffX = x.size(-1) - y.size(-1)

    y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return y


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
        )

        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.double_conv(x) + self.single_conv(x)
        return x


class Down(nn.Module):
    """Downscales with maxpool and a double convolution"""

    def __init__(self, in_channels: int, out_channels: int, pool_factor: int = 2) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_factor), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


def unet_up_collate(x: torch.Tensor, y: torch.Tensor, dim: int = 1):
    y = padded_reshape(x, y)
    return torch.cat([x, y], dim=dim)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(unet_up_collate(x1, x2))
