from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

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
    lr: float = 3e-6
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9
    intermediate_checkpointing: bool = True
    load_from_checkpoint: bool = True
    checkpoint_path: Path = Path(
        "/home/tom/dev/precip/checkpoints/pleasant-water-70Q2Y2F/step_num_49.pth"
    )


def get_training_config() -> Config:
    return Config()


def main(config: Config):
    training_dataset = SwedishPrecipitationDataset(split="train")
    dataloader = DataLoader(
        dataset=training_dataset,
        sampler=InfiniteSampler(training_dataset),
        batch_size=config.training_batch_size,
    )

    model = UNet(
        training_dataset.lookback_start_5_mins
        // training_dataset.lookback_intervals_5_mins_multiple,
        1,
        base_channels=64,
    ).to(device)

    loss = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    config = get_training_config()
    main(config)
