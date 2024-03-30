import math
import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm

import precip
import wandb
from precip.data.dataset import InfiniteSampler, SwedishPrecipitationDataset
from precip.models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Config:
    model_name: str = "unet_64_multistep"

    # dataset
    forecast_multistep: bool = True
    lookback_start_5_mins_multiple: int = 12 * 12
    lookback_intervals_5_mins_multiple: int = 12
    forecast_horizon_start_5_mins_multiple: int = 1
    forecast_horizon_end_5_mins_multiple: int = 12 * 6
    forecast_intervals_5_mins_multiple: int = 12
    forecast_gap_5_mins_multiple: int = 0

    # training params
    wandb_track: bool = True
    training_batch_size: int = 2
    validation_batch_size: int = 2
    number_of_steps: int = 100
    training_size_per_step: int = 500
    validation_size_per_step: int = 1_000

    # optimizer
    lr: float = 5e-5
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9

    # checkpoint
    intermediate_checkpointing: bool = True
    load_from_checkpoint: bool = True
    checkpoint_frequency: int = 10
    checkpoint_path: Path = Path("checkpoints/colorful-disco-1323ORW3/step_num_30.pth")


def get_training_config() -> Config:
    return Config()


def train(config: Config):
    training_dataset = SwedishPrecipitationDataset(
        split="train",
        forecast_multistep=config.forecast_multistep,
        lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
        lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
        forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
        forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
        forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
        forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
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
    )
    output_transform = CenterCrop((256, 256))
    training_dataloader = DataLoader(
        dataset=training_dataset,
        sampler=InfiniteSampler(training_dataset),
        batch_size=config.training_batch_size,
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        sampler=InfiniteSampler(validation_dataset),
        batch_size=config.validation_batch_size,
    )
    train_dataiter = iter(training_dataloader)
    val_dataiter = iter(validation_dataloader)

    if config.wandb_track:
        wandb.init(
            # set the wandb project where this run will be logged
            project="precip",
            config=config.__dict__,
        )

    model = UNet(
        training_dataset.lookback_start_5_mins_multiple
        // training_dataset.lookback_intervals_5_mins_multiple,
        1,
        base_channels=64,
    ).to(device)

    if config.load_from_checkpoint:
        model.load_state_dict(torch.load(config.checkpoint_path)["model_state_dict"])

    loss = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=8,
        threshold=0.01,
        threshold_mode="abs",
        cooldown=3,
    )

    def train(number_of_batches: int = 1_000) -> float:
        model.train()
        loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            forecasts = list()
            (batch_X, batch_y) = next(train_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            for forecast_step in range(batch_y.size(1)):
                out = model(batch_X).unsqueeze(1)
                forecasts.append(out)

                # update the input frame to the model
                # drop the earliest observation (0th index)
                batch_X = torch.cat([batch_X[:, 1:, ...], out], dim=1)

            forecasts = torch.cat(forecasts, dim=1)

            optimizer.zero_grad()
            _loss = loss(output_transform(forecasts), output_transform(batch_y))
            _loss.backward()
            optimizer.step()
            loss_history.append(math.sqrt(_loss.item()))

        return np.mean(loss_history)

    @torch.no_grad()
    def test(number_of_batches: int = 300) -> float:
        # model.eval()
        validation_loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            forecasts = list()
            (batch_X, batch_y) = next(val_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            for forecast_step in range(batch_y.size(1)):
                out = model(batch_X).unsqueeze(1)
                forecasts.append(out)

                # update the input frame to the model
                # drop the earliest observation (0th index)
                batch_X = torch.cat([batch_X[:, 1:, ...], out], dim=1)

            forecasts = torch.cat(forecasts, dim=1)
            _loss = loss(output_transform(forecasts), output_transform(batch_y))
            validation_loss_history.append(math.sqrt(_loss.item()))
        return np.mean(validation_loss_history)

    run_name = wandb.run.name if config.wandb_track else "local"

    folder_name = (
        Path(precip.__file__).parents[1]
        / "checkpoints"
        / (run_name + "".join(random.choices(string.ascii_uppercase + string.digits, k=5)))
    )
    folder_name.mkdir(parents=True, exist_ok=True)
    for step_num in range(0, config.number_of_steps):
        train_loss = train(config.training_size_per_step)
        val_loss = test(config.validation_size_per_step)
        scheduler.step(val_loss)
        # scheduler.step()

        number_of_obs = (
            config.training_batch_size
            * config.training_size_per_step
            * config.number_of_steps
            * (step_num + 1)
        )

        if config.intermediate_checkpointing and not step_num % config.checkpoint_frequency:
            torch.save(
                {
                    "total_number_observations": number_of_obs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                folder_name / f"step_num_{step_num}.pth",
            )

        if config.wandb_track:
            wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})
            # wandb.log({"scheduler": {"lr": scheduler.get_last_lr()}})


def main():
    config = get_training_config()
    train(config)


if __name__ == "__main__":
    main()
