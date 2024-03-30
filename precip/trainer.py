import math
import random
import string
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
from tqdm import tqdm

import precip
import wandb
from precip.data.dataset import InfiniteSampler


@dataclass(frozen=True)
class TrainerArgs:
    model_name: str
    device: str = "cuda"

    # training params
    wandb_track: bool = True
    iterative_multistep: bool = True
    training_batch_size: int = 1
    validation_batch_size: int = 1
    number_of_steps: int = 100
    training_size_per_step: int = 500
    validation_size_per_step: int = 1_000

    # optimizer
    lr: float = 5e-5
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-9

    # checkpointing
    intermediate_checkpointing: bool = True
    load_from_checkpoint: bool = False
    checkpoint_frequency: int = 10
    checkpoint_path: Optional[Path] = None

    # dataset config, TODO - move this elsewhere?
    shuffle: bool = True
    subsample: float = 1.00
    forecast_multistep: bool = True
    lookback_start_5_mins_multiple: int = 12 * 6
    lookback_intervals_5_mins_multiple: int = 3
    forecast_horizon_start_5_mins_multiple: int = 1
    forecast_horizon_end_5_mins_multiple: int = 12 * 4
    forecast_intervals_5_mins_multiple: int = 3
    forecast_gap_5_mins_multiple: int = 0


class Trainer:
    def __init__(
        self,
        model,
        args: TrainerArgs,
        training_dataset,
        val_dataset=None,
        optimizer=None,
        scheduler=None,
        loss=None,
    ) -> None:
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

        if self.args.wandb_track:
            wandb.init(
                # set the wandb project where this run will be logged
                project="precip",
                config=self.args.__dict__,
            )

        run_name = wandb.run.name if self.args.wandb_track else "local"
        self.folder_name = (
            Path(precip.__file__).parents[1]
            / "checkpoints"
            / (run_name + "".join(random.choices(string.ascii_uppercase + string.digits, k=5)))
        )
        self.folder_name.mkdir(parents=True, exist_ok=True)

        self.forecast_steps = (
            training_dataset.forecast_horizon_end_5_mins_multiple
            + 1
            - training_dataset.forecast_horizon_start_5_mins_multiple
        ) // training_dataset.forecast_intervals_5_mins_multiple

        self.train_dataiter = iter(
            DataLoader(
                dataset=training_dataset,
                sampler=InfiniteSampler(training_dataset, shuffle=self.args.shuffle),
                batch_size=self.args.training_batch_size,
            )
        )
        if val_dataset is not None:
            self.val_dataiter = iter(
                DataLoader(
                    dataset=val_dataset,
                    sampler=InfiniteSampler(val_dataset, shuffle=self.args.shuffle),
                    batch_size=self.args.validation_batch_size,
                )
            )
        else:
            self.val_dataiter = None

        self.output_transform = CenterCrop((256, 256))  # hardcoded

        self._setup()

    def _setup(self):
        self.model = self._get_model()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.loss = self._get_loss()

    def _get_model(self):
        model = deepcopy(self.model)
        if self.args.load_from_checkpoint:
            assert self.args.checkpoint_path is not None
            model.load_state_dict(torch.load(self.args.checkpoint_path)["model_state_dict"])
        return model.to(self.args.device)

    def _get_loss(self):
        if self.loss is not None:
            return self.loss
        else:
            return nn.MSELoss()

    def _get_scheduler(self):
        if self.scheduler is not None:
            return self.scheduler
        else:
            assert self.optimizer is not None
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode="min",
                factor=0.1,
                patience=15,
                threshold=0.01,
                threshold_mode="abs",
                cooldown=3,
            )

    def _get_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        else:
            return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def _scheduler_step(self, validation_loss):
        self.scheduler.step(validation_loss)

    def _multistep_prediction(self, batch_X):
        forecasts = list()

        for forecast_step in range(self.forecast_steps):
            predictions = self.model(batch_X).unsqueeze(1)
            forecasts.append(predictions)

            # update the input frame to the model
            # drop the earliest observation (0th index)
            batch_X = torch.cat([batch_X[:, 1:, ...], predictions], dim=1)

        forecasts = torch.cat(forecasts, dim=1)
        return forecasts

    def _singlestep_prediction(self, batch_X):
        forecasts = self.model(batch_X)
        return forecasts

    def _prediction(self, batch_X):
        if self.args.iterative_multistep:
            return self._multistep_prediction(batch_X)
        else:
            return self._singlestep_prediction(batch_X)

    def _train_step(self, batch_X, batch_y):
        forecasts = self._prediction(batch_X)
        _loss = self.calculate_loss(batch_y, forecasts)
        _loss.backward()
        return _loss.item()

    def calculate_loss(self, target, predictions):
        return self.loss(self.output_transform(target), self.output_transform(predictions))

    def train_step(self, number_of_batches: int) -> float:
        self.model.train()
        loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            (batch_X, batch_y) = next(self.train_dataiter)
            batch_X, batch_y = batch_X.to(self.args.device), batch_y.to(self.args.device)
            self.optimizer.zero_grad()
            _loss = self._train_step(batch_X, batch_y)
            self.optimizer.step()

            loss_history.append(math.sqrt(_loss))

        return np.mean(loss_history).item()

    @torch.no_grad()
    def evaluate_step(self, number_of_batches: int = 300) -> float:
        # model.eval()
        validation_loss_history = list()

        for _ in tqdm(range(number_of_batches)):
            batch_X, batch_y = next(self.val_dataiter)
            batch_X, batch_y = batch_X.to(self.args.device), batch_y.to(self.args.device)
            forecasts = self._prediction(batch_X)
            _loss = self.calculate_loss(batch_y, forecasts)
            validation_loss_history.append(math.sqrt(_loss.item()))
        return np.mean(validation_loss_history).item()

    def train(self):
        for step_num in range(0, self.args.number_of_steps):
            train_loss = self.train_step(self.args.training_size_per_step)
            print(train_loss)

            if self.val_dataiter is not None:
                val_loss = self.evaluate_step(self.args.validation_size_per_step)
                self._scheduler_step(val_loss)
            else:
                val_loss = np.nan

            number_of_obs = (
                self.args.training_batch_size
                * self.args.training_size_per_step
                * self.args.number_of_steps
                * (step_num + 1)
            )

            if (
                self.args.intermediate_checkpointing
                and not step_num % self.args.checkpoint_frequency
            ):
                torch.save(
                    {
                        "total_number_observations": number_of_obs,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    self.folder_name / f"step_num_{step_num}.pth",
                )

            if self.args.wandb_track:
                wandb.log({"loss": {"train": np.mean(train_loss), "val": np.mean(val_loss)}})
                # wandb.log({"scheduler": {"lr": scheduler.get_last_lr()}})
