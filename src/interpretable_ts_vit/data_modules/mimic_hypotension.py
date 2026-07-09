"""Data module for the MIMIC-IV hypotension task."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

from ..datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from ..datasets.mimic_iv import standardize_temperature_to_celsius
from .base import GenericCSVDataModule


logger = logging.getLogger(__name__)


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
            logger.info("Using existing MIMIC hypotension CSV files: records=%s labels=%s", self.records_path, self.labels_path)
            return self._prepared_csv_paths()
        if self.mimic_config is None and self.mimic_path is None:
            raise ValueError("Provide records/labels paths, mimic_config, or mimic_path.")
        if self.dataset_dir is None:
            raise ValueError("dataset_dir is required when preparing MIMIC-IV source files.")
        config = self.mimic_config or MIMICHypotensionConfig(mimic_path=self.mimic_path)
        logger.info("Preparing MIMIC hypotension source files into %s", self.dataset_dir)
        prepared = MIMICIVHypotensionAdapter(config).prepare()
        prepared.save(self.dataset_dir)
        self.records_path = Path(self.dataset_dir) / "records.csv"
        self.labels_path = Path(self.dataset_dir) / "labels.csv"
        logger.info("Prepared MIMIC hypotension source files: records=%s labels=%s", self.records_path, self.labels_path)
        return Path(self.records_path), Path(self.labels_path)

    def input_paths(self) -> tuple[Path, Path]:
        return self.prepare_source()

    def _prepared_csv_paths(self) -> tuple[Path, Path]:
        records_path = Path(self.records_path)
        labels_path = Path(self.labels_path)
        logger.info("Checking prepared MIMIC records for temperature standardization: %s", records_path)
        records = pd.read_csv(records_path)
        if "variable" not in records.columns or "value" not in records.columns:
            return records_path, labels_path
        is_temperature = records["variable"].astype(str) == "temperature"
        if not is_temperature.any():
            return records_path, labels_path
        cleaned = records.copy()
        cleaned.loc[is_temperature, "value"] = standardize_temperature_to_celsius(cleaned.loc[is_temperature, "value"])
        cleaned = cleaned.dropna(subset=["value"])
        output = Path(self.processed_dir) / "mimic_records_celsius.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output, index=False)
        logger.info("Wrote temperature-standardized MIMIC records to %s: rows=%d -> %d", output, len(records), len(cleaned))
        return output, labels_path
