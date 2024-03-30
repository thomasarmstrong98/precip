from torch import nn

from ..shared import DoubleConv, Down


class GenerativeEncoder(nn.Module):
    FACTOR_EXPANSION = [1, 2, 4]

    def __init__(self, input_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.output_channel_size = GenerativeEncoder.FACTOR_EXPANSION[-1] * self.base_channels * 2

        self.encoder = nn.Sequential(
            DoubleConv(self.input_channels, self.base_channels, kernel_size=3),
            *[
                Down(self.base_channels * factor, self.base_channels * factor * 2)
                for factor in GenerativeEncoder.FACTOR_EXPANSION
            ]
        )

    def forward(self, x):
        return self.encoder(x)
