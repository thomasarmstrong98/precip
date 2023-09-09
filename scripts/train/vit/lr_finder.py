import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_lr_finder import LRFinder

from precip.data.dataset import SwedishPrecipitationDataset
from precip.models.conv_lstm.model import ConvLSTM, ConvLSTMCell
import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from precip.data.dataset import InfiniteSampler, SwedishPrecipitationDataset, npy_loader
from torch import nn
from einops.layers.torch import Rearrange
from torchvision.transforms import CenterCrop
from precip.models.vit.model import SimpleViTRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    tf = CenterCrop(64)  # only model center 64 x 64 image!
    training_dataset = SwedishPrecipitationDataset(
        split="train", scale=False, transform=tf, insert_channel_dimension=False
    )
    training_sampler = InfiniteSampler(training_dataset, shuffle=True)
    dataloader = DataLoader(
        training_dataset, sampler=training_sampler, batch_size=2, num_workers=12
    )

    model = SimpleViTRegressor(
        image_size=64,
        patch_size=64,
        dim=1024,
        depth=5,
        heads=5,
        mlp_dim=100,
        channels=4,
    ).to(device)

    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)
    lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    main()
