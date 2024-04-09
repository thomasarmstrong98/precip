import torch.nn.functional as F
from torch import nn

from ..shared import SequentialMultiInput, SpectralNorm


class SPADE(nn.Module):
    """Spatially Adaptive DeNormalization"""

    def __init__(
        self,
        normalization_channels: int,
        label_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(normalization_channels, affine=False)
        pw = kernel_size // 2
        self.shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(label_channels, hidden_channels, kernel_size=kernel_size, padding=0),
            nn.ReLU(),
        )
        self.pad = nn.ReflectionPad2d(pw)
        self.gamma = nn.Conv2d(
            hidden_channels, normalization_channels, kernel_size=kernel_size, padding=0
        )
        self.beta = nn.Conv2d(
            hidden_channels, normalization_channels, kernel_size=kernel_size, padding=0
        )

    def forward(self, x, y):
        normalized = self.param_free_norm(x)
        y = F.adaptive_max_pool2d(y, output_size=x.size()[2:])
        activation = self.shared(y)
        gamma = self.gamma(self.pad(activation))
        beta = self.beta(self.pad(activation))
        return normalized * (1 + gamma) + beta


class GenerativeBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prediction_length: int,
        dilation: int = 1,
        apply_second_block: bool = False,
    ) -> None:
        super().__init__()
        self.learned_shortcut = in_channels != out_channels
        self.apply_second_block = apply_second_block

        hidden_channels = min(in_channels, out_channels)
        self.pad = nn.ReflectionPad2d(dilation)

        self.first_block = SequentialMultiInput(
            SPADE(in_channels, prediction_length),
            nn.LeakyReLU(negative_slope=2e-1),
            self.pad,
            SpectralNorm(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=0, dilation=dilation)
            ),
        )

        if apply_second_block:
            self.second_block = SequentialMultiInput(
                SPADE(hidden_channels, prediction_length),
                nn.LeakyReLU(negative_slope=2e-1),
                self.pad,
                SpectralNorm(
                    nn.Conv2d(
                        hidden_channels, out_channels, kernel_size=3, padding=0, dilation=dilation
                    )
                ),
            )

        if self.learned_shortcut:
            self.shortcut_module = SequentialMultiInput(
                SPADE(in_channels, prediction_length),
                SpectralNorm(
                    nn.Conv2d(
                        in_channels, hidden_channels, kernel_size=1, padding=0, dilation=dilation
                    )
                ),
            )
        else:
            self.shortcut_module = self._shortcut_identity

    def _shortcut_identity(self, x, y):
        return x

    def forward(self, x, y):
        x_shortcut = self.shortcut_module(x, y)
        dx = self.first_block(x, y)

        if self.apply_second_block:
            dx = self.second_block(dx, y)

        return dx + x_shortcut


class GenerativeDecoder(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv_in = nn.Conv2d(input_channels, 8 * base_channels, kernel_size=3, padding=1)

        self.generative_blocks = nn.ModuleList(
            [
                SequentialMultiInput(
                    GenerativeBlock(
                        8 * base_channels, 8 * base_channels, prediction_length=output_channels
                    ),
                    self.up,
                ),
                SequentialMultiInput(
                    GenerativeBlock(
                        8 * base_channels, 4 * base_channels, prediction_length=output_channels
                    ),
                    self.up,
                ),
                SequentialMultiInput(
                    GenerativeBlock(
                        4 * base_channels, 2 * base_channels, prediction_length=output_channels
                    ),
                    self.up,
                ),
                GenerativeBlock(
                    2 * base_channels, base_channels, prediction_length=output_channels
                ),
                SequentialMultiInput(
                    GenerativeBlock(
                        base_channels, base_channels, prediction_length=output_channels
                    ),
                    nn.LeakyReLU(negative_slope=2e-1),
                ),
            ]
        )

        self.conv_out = nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        # TODO - tidy this up
        x = self.conv_in(x)

        for gen_block in self.generative_blocks:
            x = gen_block(x, y)

        return self.conv_out(x)
