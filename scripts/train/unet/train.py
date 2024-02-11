import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm

import precip
import wandb
from precip.data.dataset import InfiniteSampler, SwedishPrecipitationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from typing import Optional


def padded_reshape(x: torch.Tensor, y: torch.Tensor):
    """Aligns last two dimensions of y to x via padding."""
    diffY = x.size(-2) - y.size(-2)
    diffX = x.size(-1) - y.size(-1)

    y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return y


def unet_up_collate(x: torch.Tensor, y: torch.Tensor, dim: int = 1):
    y = padded_reshape(x, y)
    return torch.cat([x, y], dim=dim)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
        )

        self.single_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        x = self.double_conv(x) + self.single_conv(x)
        return x


class Down(nn.Module):
    """Downscales with maxpool and a double convolution"""

    def __init__(self, in_channels: int, out_channels: int, pool_factor: int = 2) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pool_factor), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(unet_up_collate(x1, x2))


class UNet(nn.Module):
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

        # expands in our observation domain
        self.inc = DoubleConv(self.n_channels, self.base_channels)

        # iteratively downsample in spatial dimension
        self.down1 = Down(self.base_channels * 1, self.base_channels * 2)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 16 // factor)

        # iteratively upsample
        self.up4 = Up(self.base_channels * 16, self.base_channels * 8 // factor, self.bilinear)
        self.up3 = Up(self.base_channels * 8, self.base_channels * 4 // factor, self.bilinear)
        self.up2 = Up(self.base_channels * 4, self.base_channels * 2 // factor, self.bilinear)
        self.up1 = Up(self.base_channels * 2, self.base_channels, self.bilinear)

        # collapse channels
        self.out = nn.Conv2d(self.base_channels, self.output_channels, kernel_size=1)

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

        return self.out(x).squeeze(1)


@dataclass(frozen=True)
class Config:
    model_name: str = "convlstm_basic"
    batch_size: int = 4
    number_of_steps: int = 25
    training_size_per_step: int = 500
    validation_size_per_step: int = 100
    lr: float = 3e-4
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9
    intermediate_checkpointing: bool = False


def main(config: Config):
    wandb.init(
        # set the wandb project where this run will be logged
        project="precip",
        # track hyperparameters and run metadata
        config=config.__dict__,
    )

    training_dataset = SwedishPrecipitationDataset(split="train")
    validation_dataset = SwedishPrecipitationDataset(split="val")

    sampler = InfiniteSampler(training_dataset)
    training_dataloader = DataLoader(dataset=training_dataset, sampler=sampler, batch_size=8)
    validation_dataloader = DataLoader(dataset=validation_dataset, sampler=sampler, batch_size=8)

    train_dataiter = iter(training_dataloader)
    val_dataiter = iter(validation_dataloader)
    output_transform = CenterCrop((128, 128))

    model = UNet(8, 1, base_channels=64).to(device)

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
        # val_loss = test(config.validation_size_per_step)
        # scheduler.step(val_loss)

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
                # "val_loss": val_loss,
            },
            folder_name / f"step_num_{step_num}",
        )

        # wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})
        wandb.log({"loss": {"train": np.mean(train_loss)}})


if __name__ == "__main__":
    config = Config()
    main(config)
