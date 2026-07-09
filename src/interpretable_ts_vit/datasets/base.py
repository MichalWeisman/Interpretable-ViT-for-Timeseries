"""Small adapter interface for plugging new datasets into the ViT pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    """Generic records/labels payload consumed by `TimeSeriesBinner`.

    `records` must contain patient id, variable name, numeric value, and
    timestamp columns. `labels` must contain patient id and class label columns.
    Adapters may include richer cohort details in `metadata`.
    """

    records: pd.DataFrame
    labels: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> None:
        """Write records, labels, and metadata files to `output_dir`."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving prepared dataset to %s: records=%d, labels=%d",
            output,
            len(self.records),
            len(self.labels),
        )
        self.records.to_csv(output / "records.csv", index=False)
        self.labels.to_csv(output / "labels.csv", index=False)
        with (output / "dataset_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)
        logger.info("Saved prepared dataset to %s", output)


class DatasetAdapter(ABC):
    """Base class for dataset-specific preprocessing modules."""

    name: str

    @abstractmethod
    def prepare(self) -> PreparedDataset:
        """Return generic records and labels for downstream binning/training."""


_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_dataset_adapter(name: str, adapter_cls: type[DatasetAdapter]) -> None:
    """Register an adapter class under a stable dataset name."""
    _REGISTRY[name] = adapter_cls


def get_dataset_adapter(name: str) -> type[DatasetAdapter]:
    """Look up a registered dataset adapter by name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown dataset adapter '{name}'. Available adapters: {available}") from exc
