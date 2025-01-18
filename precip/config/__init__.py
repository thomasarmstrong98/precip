from dataclasses import dataclass
from pathlib import Path

import precip

PRECIP_ROOT_DIR = Path(precip.__file__).parents[1]

LOCAL_PRECIP_DATA_PATH = Path(PRECIP_ROOT_DIR / "data/sweden_precip.h5")
LOCAL_PRECIP_BOUNDARY_MASK = Path(PRECIP_ROOT_DIR / "data/sweden_precip_observation_boundaries.npy")

CLASSIFICATION_LABELS = list(range(0, 256))
BOUNDARY_CLASSIFICATION_LABEL = 255
