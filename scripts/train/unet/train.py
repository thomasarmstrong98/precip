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
    model_name: str = "unet_64"
    training_batch_size: int = 4
    validation_batch_size: int = 4
    number_of_steps: int = 80
    training_size_per_step: int = 500
    validation_size_per_step: int = 1_000
    lr: float = 3e-4
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9
    intermediate_checkpointing: bool = True
    load_from_checkpoint: bool = True
    checkpoint_frequency: int = 10
    checkpoint_path: Path = Path(
        "/home/tom/dev/precip/checkpoints/pleasant-water-70Q2Y2F/step_num_49.pth"
    )


def get_training_config() -> Config:
    return Config()


def train(config: Config):
    training_dataset = SwedishPrecipitationDataset(split="train")
    validation_dataset = SwedishPrecipitationDataset(split="val")
    output_transform = CenterCrop((128, 128))
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

    wandb.init(
        # set the wandb project where this run will be logged
        project="precip"
    )

    model = UNet(
        training_dataset.lookback_start_5_mins
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
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.number_of_steps,
        eta_min=6e-4,
    )

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=10,
    #     threshold=0.1,
    #     threshold_mode="rel",
    #     cooldown=3,
    # )

    def train(number_of_batches: int = 1_000) -> float:
        model.train()
        loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            (batch_X, batch_y) = next(train_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.squeeze(dim=1).to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            _loss = loss(output_transform(out), output_transform(batch_y))
            _loss.backward()
            optimizer.step()
            loss_history.append(math.sqrt(_loss.item()))

        return np.mean(loss_history)

    @torch.no_grad()
    def test(number_of_batches: int = 300) -> float:
        model.eval()
        validation_loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            batch_X, batch_y = next(val_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X)
            _loss = loss(output_transform(out), output_transform(batch_y))
            validation_loss_history.append(math.sqrt(_loss.item()))
        return np.mean(validation_loss_history)

    folder_name = (
        Path(precip.__file__).parents[1]
        / "checkpoints"
        / (wandb.run.name + "".join(random.choices(string.ascii_uppercase + string.digits, k=5)))
    )
    folder_name.mkdir(parents=True, exist_ok=True)
    for step_num in range(0, config.number_of_steps):
        train_loss = train(config.training_size_per_step)
        val_loss = test(config.validation_size_per_step)
        # scheduler.step(val_loss)
        scheduler.step()

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

        wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})
        wandb.log({"scheudler": {"lr": scheduler.get_lr()}})


def main():
    config = get_training_config()
    train(config)


if __name__ == "__main__":
    main()
