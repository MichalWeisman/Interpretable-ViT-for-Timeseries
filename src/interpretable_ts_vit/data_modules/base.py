"""Reusable data-module abstractions for tensor preparation and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..binning import TimeSeriesBinner
from ..config import Config, DataConfig
from ..data import BinnedTimeSeriesDataset
from ..io import load_split
from ..pipeline import _prepare_tensor_splits


@dataclass
class BaseDataModule:
    """Base class for datasets that can produce train/val/test tensors."""

    processed_dir: str | Path
    data_config: DataConfig = field(default_factory=DataConfig)

    def prepare(self) -> None:
        """Prepare tensor splits under `processed_dir`."""
        records_path, labels_path = self.input_paths()
        _prepare_tensor_splits(records_path, labels_path, Config(data=self.data_config), self.processed_dir)

    def load(self) -> "BaseDataModule":
        """Load preprocessing metadata and return this module for chaining."""
        self._binner = TimeSeriesBinner.load(self.binner_path)
        return self

    def split(self, name: str) -> BinnedTimeSeriesDataset:
        """Load one prepared split by name, such as `train`, `val`, or `test`."""
        return load_split(self.split_path(name))

    def input_paths(self) -> tuple[Path, Path]:
        """Return records and labels paths used by `prepare`."""
        raise NotImplementedError

    def split_path(self, name: str) -> Path:
        return Path(self.processed_dir) / f"{name}.npz"

    @property
    def binner_path(self) -> Path:
        return Path(self.processed_dir) / "binner.json"

    @property
    def binner(self) -> TimeSeriesBinner:
        if not hasattr(self, "_binner"):
            self.load()
        return self._binner

    @property
    def variable_vocab(self) -> list[str]:
        return self.binner.variable_vocab_

    @property
    def label_names(self) -> list[str]:
        return self.binner.index_to_label_


@dataclass
class GenericCSVDataModule(BaseDataModule):
    """Data module for already prepared generic `records.csv`/`labels.csv` files."""

    records_path: str | Path | None = None
    labels_path: str | Path | None = None

    def input_paths(self) -> tuple[Path, Path]:
        if self.records_path is None or self.labels_path is None:
            raise ValueError("records_path and labels_path are required.")
        return Path(self.records_path), Path(self.labels_path)
