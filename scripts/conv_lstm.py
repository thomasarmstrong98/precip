from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from precip.data.dataset import SwedishPrecipitationDataset
from precip.trainer import Trainer, TrainerArgs

# from torch_lr_finder import LRFinder
# from torch.utils.data import DataLoader
# import torch.optim as optim


def flexible_pool2d(x, kernel_size=3, stride=3):
    if x.dim() == 2:
        return F.avg_pool2d(x.unsqueeze(0), kernel_size=kernel_size, stride=stride).squeeze(0)
    else:
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)


config = TrainerArgs(
    "test_convnet_one_step",
    wandb_track=True,
    number_of_steps=350,
    training_batch_size=2,
    validation_batch_size=2,
    training_size_per_step=100,
    validation_size_per_step=300,
    iterative_multistep=False,
    lr=5e-3,
    subsample=1.0,
    load_from_checkpoint=False,
    forecast_multistep=False,
    lookback_intervals_5_mins_multiple=6,
    lookback_start_5_mins_multiple=36,
    forecast_horizon_start_5_mins_multiple=6,
    forecast_horizon_end_5_mins_multiple=6,
    forecast_gap_5_mins_multiple=6,
)

training_dataset = SwedishPrecipitationDataset(
    split="train",
    forecast_multistep=config.forecast_multistep,
    lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
    lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
    forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
    forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
    forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
    forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
    subsample=config.subsample,
    transform=flexible_pool2d,
)

validation_dataset = SwedishPrecipitationDataset(
    split="val",
    forecast_multistep=config.forecast_multistep,
    lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
    lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
    forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
    forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
    forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
    forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
    subsample=config.subsample,
    transform=flexible_pool2d,
)


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        activation=torch.tanh,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = nn.Conv2d(
            in_channels=self.input_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            padding_mode="replicate",
        )

        self.norm = nn.InstanceNorm2d(num_features=self.input_channels + self.hidden_channels)

    def forward(self, x, previous_state):
        h_prev, c_prev = previous_state

        output = self.conv(self.norm(torch.cat((x, h_prev), dim=1)))
        i, f, o, g = torch.split(output, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = self.activation(g)

        c_new = f * c_prev + i * g
        h_new = o * self.activation(c_new)

        return h_new, c_new

    def init_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state
        Args:
            x: Input tensor to initialize for

        Returns:
            Tuple containing the hidden states
        """
        b, c, h, w = x.size()  # c = 1 even if using grayscale inputs
        state = (
            torch.zeros(b, self.hidden_channels, h, w),
            torch.zeros(b, self.hidden_channels, h, w),
        )
        state = (state[0].type_as(x), state[1].type_as(x))
        return state


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x, hidden=None):
        h, c = hidden if hidden is not None else (None, None)

        hidden_state = list()

        for _x in torch.unbind(x, dim=1):
            if h is None:
                h, c = self.cell.init_hidden(_x)
            h, c = self.cell(_x, (h, c))
            hidden_state.append(h.unsqueeze(1))

        out = torch.cat(hidden_state, dim=1)
        return out


class DownConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_factor: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(pool_factor),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.InstanceNorm2d(num_features=hidden_channels),
        )
        self.conv_lstm = ConvLSTM(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.size()
        x = rearrange(x, "b t c h w -> (b t) c h w ")
        x = rearrange(self.conv(x), "(b t) c h w -> b t c h w", b=b, t=t)
        x = self.conv_lstm(x)
        return x


def padded_reshape(x: torch.Tensor, y: torch.Tensor):
    """Aligns last two dimensions of y to x via padding."""
    diffY = x.size(-2) - y.size(-2)
    diffX = x.size(-1) - y.size(-1)

    y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return y


def unet_up_collate(x: torch.Tensor, y: torch.Tensor, dim: int = 1):
    y = padded_reshape(x, y)
    return torch.cat([x, y], dim=dim)


class UpBiLinearConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_size: Optional[Tuple[int, int]] = None,
        upsample_scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.conv_lstm = ConvLSTM(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
        )

        self.norm = nn.InstanceNorm2d(num_features=in_channels)

        if upsample_size is not None:
            self.up_scale = nn.UpsamplingBilinear2d(size=upsample_size)
        elif upsample_scale is not None:
            self.up_scale = nn.UpsamplingBilinear2d(scale_factor=upsample_scale)
        else:
            raise RuntimeError("Must specify either upsample_size or upsample_scale")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x1.size()
        x1 = rearrange(
            self.up_scale(rearrange(x1, "b t c h w -> (b t) c h w")),
            "(b t) c h w -> b t c h w",
            b=b,
            t=t,
            c=c,
        )

        x1 = unet_up_collate(x2, x1, dim=2)
        x1 = rearrange(
            self.norm(rearrange(x1, "b t c h w -> (b t) c h w")),
            "(b t) c h w -> b t c h w",
            b=b,
            t=t,
            c=c * 2,
        )

        x1 = self.conv_lstm(x1)
        return x1


class UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        input_sequence_length,
        output_channels,
        output_sequence_length,
        base_channels,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_sequence_length = input_sequence_length
        self.output_channels = output_channels
        self.output_sequence_length = output_sequence_length
        self.base_channels = base_channels
        self.factor = 2

        self.conv = nn.Conv2d(self.input_channels, self.base_channels, kernel_size=3, padding=1)

        self.down1 = DownConvLSTM(
            self.base_channels, self.base_channels * 2, self.base_channels * 2, 3
        )
        self.down2 = DownConvLSTM(
            self.base_channels * 2, self.base_channels * 4, self.base_channels * 4, 3
        )
        self.down3 = DownConvLSTM(
            self.base_channels * 4, self.base_channels * 8, self.base_channels * 8, 3
        )

        self.down4 = DownConvLSTM(
            self.base_channels * 8,
            self.base_channels * 16,
            self.base_channels * 16 // self.factor,
            3,
        )

        # self.bottle_neck = nn.Conv3d(
        #     self.input_sequence_length,
        #     self.input_sequence_length,
        #     kernel_size=3,
        #     padding=1,
        # )

        self.bottle_neck = nn.Identity()

        self.up4 = UpBiLinearConvLSTM(
            self.base_channels * 16,
            self.base_channels * 8 // self.factor,
            kernel_size=3,
            upsample_scale=2.0,
        )

        self.up3 = UpBiLinearConvLSTM(
            self.base_channels * 8,
            self.base_channels * 4 // self.factor,
            kernel_size=3,
            upsample_scale=2.0,
        )
        self.up2 = UpBiLinearConvLSTM(
            self.base_channels * 4,
            self.base_channels * 2 // self.factor,
            kernel_size=3,
            upsample_scale=2.0,
        )
        self.up1 = UpBiLinearConvLSTM(
            self.base_channels * 2,
            self.base_channels,
            kernel_size=3,
            upsample_scale=2.0,
        )

        self.time_collapse = nn.Conv3d(
            self.input_sequence_length, self.output_sequence_length, kernel_size=3, padding=1
        )
        self.channel_collapse = nn.Conv2d(
            self.base_channels, self.output_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        b, t, c, h, w = x.size()
        x = self.conv(rearrange(x, "b t c h w -> (b t) c h w"))
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x4 = self.bottle_neck(x4)

        x_up = self.up4(x4, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)

        x = self.time_collapse(x_up)
        x = self.channel_collapse(rearrange(x, "b t c h w -> (b t) c h w"))

        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=self.output_sequence_length)
        return x.squeeze(1, 2)


def main():
    model = UNet(
        input_channels=1,
        input_sequence_length=2,
        output_channels=1,
        output_sequence_length=1,
        base_channels=8,
    )
    trainer = Trainer(
        model, config, training_dataset=training_dataset, val_dataset=validation_dataset
    )
    trainer.train()


# def find_lr():
#     model = UNet(
#         input_channels=1,
#         input_sequence_length=6,
#         output_channels=1,
#         output_sequence_length=1,
#         base_channels=16,
#     )

#     dataloader = DataLoader(
#         training_dataset, sampler=ObservationWeightedOnlineSampler(training_dataset)
#     )
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)
#     lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
#     lr_finder.range_test(dataloader, end_lr=1, num_iter=100)
#     lr_finder.plot()  # to inspect the loss-learning rate graph


if __name__ == "__main__":
    main()
