from einops import rearrange
from torch import nn

from .layers import DownConvLSTM, UpBiLinearConvLSTM


class UNetConvLSTM(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        input_sequence_channels: int,
        output_sequence_length: int,
        output_sequence_channels: int,
        base_channels: int,
    ) -> None:
        super().__init__()
        self.input_seq_length = input_sequence_length
        self.input_seq_channels = input_sequence_channels
        self.output_seq_length = output_sequence_length
        self.output_seq_channels = output_sequence_channels
        self.base_channels = base_channels
        self.factor = 2

        self.conv = nn.Conv2d(
            input_sequence_channels, self.base_channels, kernel_size=3, padding=1
        )

        self.down1 = DownConvLSTM(
            3,
            self.base_channels * 1,
            self.base_channels * 2,
            self.base_channels * 2,
            kernel_size=3,
        )
        self.down2 = DownConvLSTM(
            3,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 4,
            kernel_size=3,
        )
        self.down3 = DownConvLSTM(
            3,
            self.base_channels * 4,
            self.base_channels * 8 // self.factor,
            self.base_channels * 8 // self.factor,
            kernel_size=3,
        )

        self.up3 = UpBiLinearConvLSTM(
            3,
            self.base_channels * 8,
            self.base_channels * 4 // self.factor,
            kernel_size=3,
            upsample_scale=2.0,
        )
        self.up2 = UpBiLinearConvLSTM(
            3,
            self.base_channels * 4,
            self.base_channels * 2 // self.factor,
            kernel_size=3,
            upsample_scale=2.0,
        )
        self.up1 = UpBiLinearConvLSTM(
            3,
            self.base_channels * 2,
            self.base_channels,
            kernel_size=3,
            upsample_scale=2.0,
        )

        self.time_collapse = nn.Conv3d(
            self.input_seq_length, self.output_seq_length, kernel_size=3, padding=1
        )
        self.channel_collapse = nn.Conv2d(
            self.base_channels, self.output_seq_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        if self.input_seq_channels == 1:
            x = x.unsqueeze(dim=2)

        b, t, c, h, w = x.size()

        x = self.conv(rearrange(x, "b t c h w -> (b t) c h w"))
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x = self.up1(x_up, x)

        x = self.time_collapse(x)
        x = self.channel_collapse(rearrange(x, "b t c h w -> (b t) c h w"))

        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=self.output_seq_length)

        if self.output_seq_channels == 1:
            x = x.squeeze(dim=2)
        return x
