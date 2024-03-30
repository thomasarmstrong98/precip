from precip.data.dataset import SwedishPrecipitationDataset
from precip.models.nowcastnet import NowcastNet
from precip.trainer import Trainer, TrainerArgs


def main():
    config = TrainerArgs(
        "nowcastnet_test",
        wandb_track=True,
        number_of_steps=100,
        training_batch_size=1,
        validation_batch_size=1,
        training_size_per_step=300,
        validation_size_per_step=600,
        iterative_multistep=False,
        lr=3e-4,
        subsample=0.3,
        load_from_checkpoint=False,
    )

    training_dataset = SwedishPrecipitationDataset(
        split="train",
        forecast_multistep=config.forecast_multistep,
        lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
        lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
        forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
        forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
        forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
        forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
        subsample=config.subsample,
    )

    validation_dataset = SwedishPrecipitationDataset(
        split="val",
        forecast_multistep=config.forecast_multistep,
        lookback_start_5_mins_multiple=config.lookback_start_5_mins_multiple,
        lookback_intervals_5_mins_multiple=config.lookback_intervals_5_mins_multiple,
        forecast_horizon_start_5_mins_multiple=config.forecast_horizon_start_5_mins_multiple,
        forecast_horizon_end_5_mins_multiple=config.forecast_horizon_end_5_mins_multiple,
        forecast_intervals_5_mins_multiple=config.forecast_intervals_5_mins_multiple,
        forecast_gap_5_mins_multiple=config.forecast_gap_5_mins_multiple,
    )

    model = NowcastNet(
        training_dataset.lookback_start_5_mins_multiple
        // training_dataset.lookback_intervals_5_mins_multiple,
        (
            training_dataset.forecast_horizon_end_5_mins_multiple
            + 1
            - training_dataset.forecast_horizon_start_5_mins_multiple
        )
        // training_dataset.forecast_intervals_5_mins_multiple,
    )

    trainer = Trainer(model, config, training_dataset, validation_dataset)
    trainer.train()


if __name__ == "__main__":
    main()
