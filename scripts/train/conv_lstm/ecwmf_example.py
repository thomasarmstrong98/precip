import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm

import precip
import wandb
from precip.data.dataset import InfiniteSampler, SwedishPrecipitationDataset
from precip.models.conv_lstm import ConvLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.depth2space = PixelUnshuffle(downscale_factor=2)
        self.convlstm = ConvLSTM(1, 32, 3, 1)
        self.c1 = nn.Conv2d(32, 1, kernel_size=1, padding="same", padding_mode="replicate")

    def forward(self, x):
        # x = self.depth2space(x)
        _, x = self.convlstm(x)
        x = x[-1][-1]
        return self.c1(x.squeeze(1)).squeeze(1)


@dataclass(frozen=True)
class Config:
    model_name: str = "convlstm_basic"
    batch_size: int = 4
    number_of_steps: int = 10
    training_size_per_step: int = 500
    validation_size_per_step: int = 100
    lr: float = 3e-3
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9
    intermediate_checkpointing: bool = False


def parse_args() -> Config:
    return Config()


def main():
    config = parse_args()

    wandb.init(
        # set the wandb project where this run will be logged
        project="precip",
        # track hyperparameters and run metadata
        config=config.__dict__,
    )

    training_dataset = SwedishPrecipitationDataset(
        split="train",
        scale=True,
        apply_mask_to_zero=True,
        insert_channel_dimension=True,
        transform=CenterCrop((128, 128)),
    )
    validation_dataset = SwedishPrecipitationDataset(
        split="val",
        scale=True,
        apply_mask_to_zero=True,
        insert_channel_dimension=True,
        transform=CenterCrop((128, 128)),
    )

    training_sampler = InfiniteSampler(training_dataset, shuffle=True)
    validation_sampler = InfiniteSampler(validation_dataset, shuffle=True)

    dataloader = DataLoader(
        training_dataset, sampler=training_sampler, batch_size=config.batch_size, num_workers=12
    )
    val_dataloader = DataLoader(
        validation_dataset, sampler=validation_sampler, batch_size=config.batch_size, num_workers=12
    )
    train_dataiter, val_dataiter = iter(dataloader), iter(val_dataloader)

    model = ToyModel().to(device)
    output_transform = CenterCrop((64, 64))

    loss = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_scheduler_gamma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.2,
        patience=5,
        threshold=0.1,
        threshold_mode="rel",
        cooldown=3,
    )

    def train(number_of_batches: int = 1_000) -> float:
        model.train()
        loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            (batch_X, batch_y) = next(train_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.squeeze(dim=1).to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            # out.register_hook(lambda grad: grad * mask)
            _loss = loss(output_transform(out), output_transform(batch_y))
            _loss.backward()
            optimizer.step()
            loss_history.append(np.sqrt(_loss.item()))

        return np.mean(loss_history)

    @torch.no_grad()
    def test(number_of_batches: int = 300) -> float:
        model.eval()
        validation_loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            batch_X, batch_y = next(val_dataiter)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X)
            validation_loss_history.append(np.sqrt(loss(out, batch_y).item()))
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
        scheduler.step(val_loss)

        number_of_obs = (
            config.batch_size
            * config.training_size_per_step
            * config.number_of_steps
            * (step_num + 1)
        )

        torch.save(
            {
                "total_number_observations": number_of_obs,
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
