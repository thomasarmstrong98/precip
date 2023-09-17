import torch
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder

from precip.data.dataset import InfiniteSampler, SwedishPrecipitationDataset
from precip.models.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    training_dataset = SwedishPrecipitationDataset(
        split="train", scale=True, apply_mask_to_zero=True, insert_channel_dimension=False
    )
    training_sampler = InfiniteSampler(training_dataset, shuffle=True)
    dataloader = DataLoader(
        training_dataset, sampler=training_sampler, batch_size=2, num_workers=12
    )

    model = UNet(4, bilinear=True).to(device)

    loss = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-7,
        weight_decay=1e-9,
    )
    lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    main()
