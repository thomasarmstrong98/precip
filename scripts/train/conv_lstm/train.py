import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from precip.config import CONVLSTM_MODEL_CONFIG
from precip.data.dataset import SwedishPrecipitationDataset
from precip.models.conv_lstm.model import ConvLSTM, ConvLSTMCell

wandb.init(
    # set the wandb project where this run will be logged
    project="precip",
    # track hyperparameters and run metadata
    config=CONVLSTM_MODEL_CONFIG.__dict__,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers) -> None:
        super().__init__()

        self.lstm_model = ConvLSTM(
            input_channel=input_channels,
            hidden_channel=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False,
        )
        self.out_conv = nn.ConvTranspose2d(hidden_channels, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        _, x = self.lstm_model(x)
        return self.out_conv(x[0][0]).squeeze(1)


def main():
    training_dataset = SwedishPrecipitationDataset(split="train")
    validation_dataset = SwedishPrecipitationDataset(split="val")

    dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True, num_workers=12)

    model = Model(input_channels=1, hidden_channels=[64], kernel_size=(3, 3), num_layers=1).to(
        device
    )

    loss = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONVLSTM_MODEL_CONFIG.lr,
        weight_decay=CONVLSTM_MODEL_CONFIG.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=CONVLSTM_MODEL_CONFIG.lr_scheduler_gamma
    )

    def train():
        model.train()
        loss_history = list()

        for batch_index, (batch_X, batch_y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X)
            _loss = loss(out, batch_y)
            _loss.backward()
            optimizer.step()

            if batch_index % 10 == 0:
                loss_history.append(_loss.item())

        return loss_history

    @torch.no_grad()
    def test():
        model.eval()
        validation_loss_history = list()

        for _, (batch_X, batch_y) in enumerate(val_dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            out = model(batch_X)
            validation_loss_history.append(loss(out, batch_y).item())
        return validation_loss_history

    for epoch in range(0, CONVLSTM_MODEL_CONFIG.epochs):
        train_loss = train()
        val_loss = test()
        scheduler.step()

        wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})


if __name__ == "__main__":
    main()
