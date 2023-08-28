from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from precip.config import (
    BOUNDARY_CLASSIFICATION_LABEL,
    LOCAL_PRECIP_DATA_AVERAGES,
    LOCAL_PRECIP_DATA_PATH,
)

TRAINING_KEYS_LAST_INDEX = 25000
VALIDATION_KEYS_LAST_INDEX = 40000


def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample


class SwedishPrecipitationDataset(Dataset):
    def __init__(
        self,
        root: Path = LOCAL_PRECIP_DATA_PATH,
        observation_frequency_5_min: int = 12 * 2,
        lookback_start_5_mins: int = 12 * 4,
        lookback_intervals_5_mins_multiple: int = 12,
        forecast_horizon_5_mins: int = 12,
        split: str = "train",
        scale: bool = True,
    ):
        self.root = root
        self.split = split
        self.observation_frequency_5_min = observation_frequency_5_min
        self.lookback_start_5_mins = lookback_start_5_mins
        self.lookback_intervals_5_mins_multiple = lookback_intervals_5_mins_multiple
        self.forecast_horizon_5_mins = forecast_horizon_5_mins
        self.scale = scale

        self.data, self.keys = self.load(root)

    def load(self, root: Path):
        data = h5py.File(root)
        keys = list(data.keys())
        # keys = list(data.keys())[self.lookback_start_5_mins +1: ]

        if self.split == "train":
            keys = keys[:TRAINING_KEYS_LAST_INDEX]

        elif self.split == "val":
            keys = keys[TRAINING_KEYS_LAST_INDEX:VALIDATION_KEYS_LAST_INDEX]

        else:
            keys = keys[VALIDATION_KEYS_LAST_INDEX:]

        # keys = keys[list(range(self.lookback_start_5_mins + 1, len(keys), self.observation_frequency_5_min))]
        keys = keys[:500]

        return data, keys

    def __len__(self) -> int:
        return len(self.keys) - self.lookback_start_5_mins - self.forecast_horizon_5_mins

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index += self.lookback_start_5_mins
        observation_indicies = [
            (index - lookback)
            for lookback in range(
                0, self.lookback_start_5_mins, self.lookback_intervals_5_mins_multiple
            )
        ]
        forecast_index = index + self.forecast_horizon_5_mins

        X = np.concatenate(
            [
                (np.asarray(self.data[self.keys[_index]]))[np.newaxis, :, :]
                for _index in observation_indicies
            ]
        )
        y = np.asarray(self.data[self.keys[forecast_index]])

        X, y = torch.tensor(X, dtype=torch.float32).unsqueeze(1), torch.tensor(
            y, dtype=torch.float32
        )

        if self.scale:
            X /= BOUNDARY_CLASSIFICATION_LABEL

        return X, y


# for training large dataset, where we don't want to track by epoch
# we can just track by numb. of observations, we need inifite
# sampler from our dataset, to skip StopIteration errors.
class InfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset, shuffle: bool = True, reshuffle: bool = False):
        self.n = len(dataset)
        assert self.n > 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.reshuffle = reshuffle

    def __iter__(self):
        if self.shuffle:
            order = np.random.choice(self.n, self.n)
        else:
            order = np.arange(self.n)

        idx = 0
        while True:
            yield order[idx]
            idx += 1
            if idx == len(order):
                if self.shuffle:
                    # reshuffle
                    order = np.random.choice(self.n, self.n)
                idx = 0  # reset back to beginning without reinit dataset object.
