from dataclasses import dataclass
from pathlib import Path

import precip

PRECIP_ROOT_DIR = Path(precip.__file__).parents[1]

LOCAL_PRECIP_DATA_PATH = Path(PRECIP_ROOT_DIR / "data/sweden_precip.h5")
LOCAL_PRECIP_BOUNDARY_MASK = Path(PRECIP_ROOT_DIR / "data/sweden_precip_observation_boundaries.npy")

CLASSIFICATION_LABELS = list(range(0, 256))
BOUNDARY_CLASSIFICATION_LABEL = 255

# @dataclass
# class ModelConfig:
#     batch_size: int
#     epochs: int
#     lr: float
#     lr_scheduler_step: int
#     lr_scheduler_gamma: float


@dataclass(frozen=True)
class ModelConfigConvLSTM:
    batch_size: int = 2
    epochs: int = 20
    lr: float = 5.34e-03
    lr_scheduler_step: int = 3
    lr_scheduler_gamma: float = 0.85
    weight_decay: float = 1e-4


CONVLSTM_MODEL_CONFIG = ModelConfigConvLSTM()
