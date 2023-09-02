import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_lr_finder import LRFinder

from precip.data.dataset import SwedishPrecipitationDataset
from precip.models.conv_lstm.model import ConvLSTM, ConvLSTMCell

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
        self.out_conv = nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        _, x = self.lstm_model(x)
        return self.out_conv(x[0][0]).squeeze(1)


def main():
    training_dataset = SwedishPrecipitationDataset(split="train")
    # validation_dataset = SwedishPrecipitationDataset(split='val')

    dataloader = DataLoader(training_dataset, batch_size=2, shuffle=True, num_workers=20)
    # val_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True, num_workers=20)

    model = Model(input_channels=1, hidden_channels=[64], kernel_size=(3, 3), num_layers=1).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state


if __name__ == "__main__":
    main()
