from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from precip.config import BOUNDARY_CLASSIFICATION_LABEL, LOCAL_PRECIP_DATA_PATH

# TODO - use train/validation splitting method of DGMR paper
# aka no strict train/val split between data
TRAINING_KEYS_LAST_INDEX = 25_000
VALIDATION_KEYS_LAST_INDEX = 40_000


def npy_loader(path):
    sample = torch.from_numpy(np.load(path)).float()
    return sample


def crop_to_region_of_interest(
    radar: torch.Tensor, top: int = 555, left: int = 55, height: int = 256, width: int = 256
) -> torch.Tensor:
    """Crops the radar image to a central, large region which we focus our forecast to."""
    return radar[..., top : top + height, left : left + width]


class SwedishPrecipitationDataset(Dataset):
    def __init__(
        self,
        root: Path = LOCAL_PRECIP_DATA_PATH,
        lookback_start_5_mins: int = 12 * 4,
        lookback_intervals_5_mins_multiple: int = 6,
        forecast_horizon_5_mins_multiple: int = 12,
        forecast_gap_5_mins_multiple: int = 0,
        split: str = "train",
        insert_channel_dimension: bool = False,
        scale: bool = True,
        transform=crop_to_region_of_interest,  # by default we are only using 256x256 patch
        mask_boundary: bool = True,
    ):
        self.root = root
        self.split = split
        self.lookback_start_5_mins = lookback_start_5_mins
        self.lookback_intervals_5_mins_multiple = lookback_intervals_5_mins_multiple
        self.forecast_horizon_5_mins_multiple = forecast_horizon_5_mins_multiple
        self.forecast_gap = forecast_gap_5_mins_multiple
        self.scale = scale
        self.insert_channel_dimension = insert_channel_dimension
        self.transform = transform
        self.mask_boundary = mask_boundary

        self.data, self.keys = self.load(root)

    def load(self, root: Path):
        data = h5py.File(root)
        keys = list(data.keys())

        if self.split == "train":
            keys = keys[:TRAINING_KEYS_LAST_INDEX]

        elif self.split == "val":
            keys = keys[TRAINING_KEYS_LAST_INDEX:VALIDATION_KEYS_LAST_INDEX]

        elif self.split == "test":
            keys = keys[VALIDATION_KEYS_LAST_INDEX:]

        return data, keys

    def __len__(self) -> int:
        return (
            len(self.keys)
            - self.lookback_start_5_mins
            - (self.forecast_horizon_5_mins_multiple + self.forecast_gap)
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index += self.lookback_start_5_mins
        observation_indicies = [
            (index - lookback)
            for lookback in range(
                0, self.lookback_start_5_mins, self.lookback_intervals_5_mins_multiple
            )
        ]
        forecast_index = index + self.forecast_horizon_5_mins_multiple + self.forecast_gap

        X = np.concatenate(
            [
                (np.asarray(self.data[self.keys[_index]]))[np.newaxis, :, :]
                for _index in observation_indicies
            ]
        )
        y = np.asarray(self.data[self.keys[forecast_index]])

        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        if self.mask_boundary:
            X = torch.where(~(X == 255), X, 0.0)
            y = torch.where(~(y == 255), y, 0.0)

        if self.scale:
            X /= BOUNDARY_CLASSIFICATION_LABEL

        if self.transform is not None:
            X, y = self.transform(X), self.transform(y)

        if self.insert_channel_dimension:
            X = X.unsqueeze(1)
        else:
            y = y.squeeze(0)

        return X, y


class InfiniteSampler(Sampler):
    # L2 of a 256 x 256 single frame - frames below this are not considered in our dataset due to low IC
    MEDIAN_SCALED_CROPPED_IMAGE = 950.00

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        reshuffle: bool = False,
        is_scaled: bool = True,
    ):
        self.n = len(dataset)  # type: ignore
        assert self.n > 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.reshuffle = reshuffle
        self.is_scaled = is_scaled

        if self.shuffle:
            self.order = np.random.choice(self.n, self.n)
        else:
            self.order = np.arange(self.n)

    def _increase_index_maybe_reset(self, index: int) -> int:
        index += 1
        if index == self.n:
            if self.reshuffle:
                # reshuffle
                self.order = np.random.choice(self.n, self.n)
            index = 0  # reset back to beginning without reinit dataset object.

        return index

    def sample(self, image: torch.Tensor):
        _sum = torch.sum(image**2)
        if self.is_scaled:
            _sample = _sum < self.MEDIAN_SCALED_CROPPED_IMAGE
        else:
            _sample = _sum < 255**2 * self.MEDIAN_SCALED_CROPPED_IMAGE
        return _sample

    def __iter__(self):
        if self.shuffle:
            order = np.random.choice(self.n, self.n)
        else:
            order = np.arange(self.n)

        idx = 0
        while True:
            X_sample, _ = self.dataset[order[idx]]

            # get single image
            X_sample = X_sample[0][0]
            if not self.sample(X_sample):
                idx = self._increase_index_maybe_reset(idx)
                continue
            else:
                yield order[idx]
            idx = self._increase_index_maybe_reset(idx)
