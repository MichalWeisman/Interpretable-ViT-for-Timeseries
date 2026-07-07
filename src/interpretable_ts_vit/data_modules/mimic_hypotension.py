"""Data module for the MIMIC-IV hypotension task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from .base import GenericCSVDataModule


@dataclass
class MIMICHypotensionDataModule(GenericCSVDataModule):
    """Prepare/load MIMIC-IV hypotension records, labels, and tensor splits.

    Pass `records_path` and `labels_path` to reuse already prepared CSV files.
    Pass `mimic_config` or `mimic_path` to build those CSV files from MIMIC-IV
    before tensor preparation.
    """

    mimic_path: str | Path | None = None
    dataset_dir: str | Path | None = None
    mimic_config: MIMICHypotensionConfig | None = None

    def prepare_source(self) -> tuple[Path, Path]:
        """Create generic records/labels CSV files from MIMIC-IV if needed."""
        if self.records_path is not None and self.labels_path is not None:
            return Path(self.records_path), Path(self.labels_path)
        if self.mimic_config is None and self.mimic_path is None:
            raise ValueError("Provide records/labels paths, mimic_config, or mimic_path.")
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required when preparing MIMIC-IV source files.")
        config = self.mimic_config or MIMICHypotensionConfig(mimic_path=self.mimic_path)
        prepared = MIMICIVHypotensionAdapter(config).prepare()
        prepared.save(self.dataset_dir)
        self.records_path = Path(self.dataset_dir) / "records.csv"
        self.labels_path = Path(self.dataset_dir) / "labels.csv"
        return Path(self.records_path), Path(self.labels_path)

    def input_paths(self) -> tuple[Path, Path]:
        return self.prepare_source()
