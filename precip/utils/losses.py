import torch
from torch import nn


class TimeWeightedMSELoss(nn.Module):
    def __init__(
        self, exp_weighting: float, time_steps: int, cuda: bool = True
    ) -> None:
        super().__init__()
        self.weights = torch.exp(exp_weighting * torch.arange(time_steps))
        self.weights /= self.weights.mean()
        if cuda:
            self.weights = self.weights.cuda()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x, y):
        mse = self.mse(x, y)
        return torch.mean(mse * self.weights[:, None, None])


class AccumulationLoss(nn.Module):
    def __init__(self, base_measure: nn.Module = nn.MSELoss()) -> None:
        super().__init__()
        self.base_measure = base_measure

    def forward(self, x, y):
        pass
