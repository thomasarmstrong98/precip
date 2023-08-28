import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import random
import string
from dataclasses import dataclass
import precip

import wandb
from precip.config import LOCAL_PRECIP_BOUNDARY_MASK
from precip.data.dataset import SwedishPrecipitationDataset, InfiniteSampler, npy_loader
from precip.models.conv_lstm.model import ConvLSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelConfigConvLSTM:
    batch_size: int = 2
    number_of_steps: int = 20
    training_size_per_step: int = 500
    validation_size_per_step: int = 150
    lr: float = 5.34e-03
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.6
    weight_decay: float = 1e-4


class Model(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        num_layers,
        mask: torch.Tensor,
        upscale: float = 255.0,
    ) -> None:
        super().__init__()

        self.lstm_model = ConvLSTM(
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False,
        )
        self.out_conv = nn.ConvTranspose2d(hidden_channels[-1], 1, kernel_size=(3, 3), padding=1)
        # self.mask = mask
        # self.upscale = upscale

    def forward(self, x):
        _, x = self.lstm_model(x)
        x = self.out_conv(x[0][0]).squeeze(1)
        return x


def parse_args() -> ModelConfigConvLSTM:
    return ModelConfigConvLSTM()


def main():
    config = parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="precip",
        # track hyperparameters and run metadata
        config=config.__dict__,
    )

    training_dataset = SwedishPrecipitationDataset(split="train", scale=True)
    validation_dataset = SwedishPrecipitationDataset(split="val", scale=True)

    training_sampler = InfiniteSampler(training_dataset, shuffle=True)
    validation_sampler = InfiniteSampler(validation_dataset, shuffle=True)

    dataloader = DataLoader(
        training_dataset, sampler=training_sampler, batch_size=2, num_workers=12
    )
    val_dataloader = DataLoader(
        validation_dataset, sampler=validation_sampler, batch_size=2, num_workers=12
    )
    train_dataiter, val_dataiter = iter(dataloader), iter(val_dataloader)

    mask = npy_loader(LOCAL_PRECIP_BOUNDARY_MASK)
    model = Model(
        input_channels=1, hidden_channels=[64], kernel_size=(3, 3), num_layers=1, mask=mask
    ).to(device)

    loss = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_scheduler_gamma)

    def train(number_of_batches: int = 100) -> float:
        model.train()
        loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            (batch_X, batch_y) = next(train_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            _loss = loss(out, batch_y)
            _loss.backward()
            optimizer.step()
            loss_history.append(_loss.item())

        return np.mean(loss_history)

    @torch.no_grad()
    def test(number_of_batches: int = 50) -> float:
        model.eval()
        validation_loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            batch_X, batch_y = next(val_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X)
            validation_loss_history.append(loss(out, batch_y).item())
        return np.mean(validation_loss_history)

    folder_name = (
        Path(precip.__file__).parent
        / "checkpoints"
        / "2023_08_28_".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    )
    folder_name.mkdir(parents=True, exist_ok=True)

    for step_num in range(0, config.number_of_steps):
        train_loss = train()
        val_loss = test()
        scheduler.step()

        torch.save(
            {
                "total_number_observations": 500,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            folder_name / f"step_num_{step_num}",
        )

        wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})


if __name__ == "__main__":
    main()
