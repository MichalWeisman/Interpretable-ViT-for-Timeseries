"""Reusable data-module abstractions for tensor preparation and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

from ..binning import TimeSeriesBinner
from ..config import Config, DataConfig
from ..data import BinnedTimeSeriesDataset
from ..io import load_split
from ..pipeline import _apply_mimic_variable_filter, _prepare_tensor_splits
from ..training import print_class_balance


logger = logging.getLogger(__name__)


@dataclass
class BaseDataModule:
    """Base class for datasets that can produce train/val/test tensors."""

    processed_dir: str | Path
    data_config: DataConfig = field(default_factory=DataConfig)

    def prepare(self) -> None:
        """Prepare tensor splits under `processed_dir`."""
        records_path, labels_path = self.input_paths()
        config = Config(data=self.data_config)
        _apply_mimic_variable_filter(config, records_path)
        required = [self.binner_path, *(self.split_path(split) for split in ("train", "val", "test"))]
        if all(path.exists() for path in required):
            allowed = set(config.data.allowed_variables or [])
            if allowed:
                binner = TimeSeriesBinner.load(self.binner_path)
                extra_variables = sorted(set(binner.variable_vocab_) - allowed)
                if extra_variables:
                    logger.warning(
                        "Prepared tensor files under %s contain %d variable(s) outside the YAML allow-list; rebuilding",
                        self.processed_dir,
                        len(extra_variables),
                    )
                else:
                    logger.info("Prepared tensor files already exist under %s; skipping preparation", self.processed_dir)
                    return
            else:
                logger.info("Prepared tensor files already exist under %s; skipping preparation", self.processed_dir)
                return
        logger.info("Preparing tensor splits under %s from records=%s labels=%s", self.processed_dir, records_path, labels_path)
        _prepare_tensor_splits(records_path, labels_path, config, self.processed_dir)
        logger.info("Prepared tensor splits under %s", self.processed_dir)

    def load(self) -> "BaseDataModule":
        """Load preprocessing metadata and return this module for chaining."""
        logger.info("Loading data module metadata from %s", self.binner_path)
        self._binner = TimeSeriesBinner.load(self.binner_path)
        logger.info("Loaded data module metadata from %s", self.binner_path)
        if not getattr(self, "_class_balance_printed", False):
            for split in ("train", "val", "test"):
                split_path = self.split_path(split)
                if split_path.exists():
                    print_class_balance(split, load_split(split_path), self.label_names)
            self._class_balance_printed = True
        return self

    def split(self, name: str) -> BinnedTimeSeriesDataset:
        """Load one prepared split by name, such as `train`, `val`, or `test`."""
        logger.info("Loading %s split from data module at %s", name, self.split_path(name))
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
