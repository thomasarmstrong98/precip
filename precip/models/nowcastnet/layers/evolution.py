import torch
from torch import nn


from .shared import DoubleConv, Down, Up


class EvolutionNetwork(nn.Module):
    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        base_channels: int = 64,
        bilinear_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = input_seq_len
        self.bilinear = bilinear_upsample
        self.base_channels = base_channels
        self.output_channels = output_seq_len
        factor = 2 if self.bilinear else 1

        # expands dimensionality in our observation domain
        self.inc = DoubleConv(self.n_channels, self.base_channels)

        # iteratively downsample in spatial dimension
        self.down1 = Down(self.base_channels * 1, self.base_channels * 2)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16 // factor)

        # iteratively upsample - motion field
        self.up4 = Up(self.base_channels * 16, self.base_channels * 8 // factor, self.bilinear)
        self.up3 = Up(self.base_channels * 8, self.base_channels * 4 // factor, self.bilinear)
        self.up2 = Up(self.base_channels * 4, self.base_channels * 2 // factor, self.bilinear)
        self.up1 = Up(self.base_channels * 2, self.base_channels, self.bilinear)
        self.gamma = nn.Parameter(torch.zeros(1, self.output_channels, 1, 1), requires_grad=True)

        # iteratively upsample - velocity field
        self.v_up4 = Up(self.base_channels * 16, self.base_channels * 8 // factor, self.bilinear)
        self.v_up3 = Up(self.base_channels * 8, self.base_channels * 4 // factor, self.bilinear)
        self.v_up2 = Up(self.base_channels * 4, self.base_channels * 2 // factor, self.bilinear)
        self.v_up1 = Up(self.base_channels * 2, self.base_channels, self.bilinear)

        # collapse channels
        self.out_motion = nn.Conv2d(self.base_channels, self.output_channels, kernel_size=1)
        self.out_velocity = nn.Conv2d(self.base_channels, self.output_channels * 2, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        v = self.v_up4(x5, x4)
        v = self.v_up3(v, x3)
        v = self.v_up2(v, x2)
        v = self.v_up1(v, x1)

        x = self.out_motion(x) * self.gamma
        v = self.out_velocity(v)
        return x, v
