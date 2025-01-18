import torch
import torch.nn.functional as F
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


def from_pixel_to_rainrate(x, dbz_min=0, dbz_max=75, a=0.0365, b=0.625):
    dbz_values = (x / 255) * (dbz_max - dbz_min) + dbz_min

    # Step 2: Convert dBZ to Z (reflectivity)
    Z_values = 10 ** (dbz_values / 10)

    # Step 3: Convert Z to rain rate (mm/hr) using the empirical formula
    rain_rate = a * (Z_values**b)

    return rain_rate


def digitize(x):
    return (
        torch.bucketize(
            x, boundaries=torch.tensor([-torch.inf, 0.4, 1.0, 2.5, torch.inf]), right=True
        )
        - 1
    )


def transform(x):
    return digitize(from_pixel_to_rainrate(flexible_pool2d(x))).float()


def main(base_channels):
    from precip.models.unet import UNet

    config = TrainerArgs(
        f"unet_classification_proto_{base_channels}",
        wandb_track=True,
        number_of_steps=350,
        training_batch_size=2,
        validation_batch_size=32,
        training_size_per_step=20,
        validation_size_per_step=20,
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
        # device='gpu'
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
        transform=transform,
        scale=False,
    )

    val_dataset = SwedishPrecipitationDataset(
        split="val",
        forecast_multistep=config.forecast_multistep,
        lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
        lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
        forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
        forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
        forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
        forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
        subsample=config.subsample,
        transform=transform,
        scale=False,
    )

    model = UNet(6, 4, base_channels)
    trainer = Trainer(
        model,
        config,
        training_dataset=training_dataset,
        val_dataset=val_dataset,
        loss=nn.CrossEntropyLoss(),
    )
    trainer.train()


if __name__ == "__main__":
    main(64)
