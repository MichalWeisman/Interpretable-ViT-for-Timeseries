"""Data containers used between preprocessing, training, and explanation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


@dataclass
class BinnedTimeSeries:
    """In-memory result returned by `TimeSeriesBinner.transform`."""

    patient_ids: list[str]
    x: np.ndarray
    y: np.ndarray | None
    label_names: list[str] | None


class BinnedTimeSeriesDataset(Dataset):
    """Torch dataset wrapper around binned tensors and optional labels."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        patient_ids: Sequence[str] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("BinnedTimeSeriesDataset requires PyTorch. Install torch to train models.")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)
        self.patient_ids = [str(i) for i in patient_ids] if patient_ids is not None else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        if self.y is None:
            return self.x[index]
        return self.x[index], self.y[index]
