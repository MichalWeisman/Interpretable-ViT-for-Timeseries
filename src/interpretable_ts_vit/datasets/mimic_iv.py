"""MIMIC-IV adapter for hypotension prediction from ICU chart events.

The adapter produces the generic `records.csv`/`labels.csv` pair expected by
the existing binner. Each patient id is an ICU `stay_id`; timestamps are
relative to ICU admission by default because MIMIC-IV dates are shifted
independently per patient and absolute dates are not comparable across stays.
"""

from __future__ import annotations

import gzip
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .base import DatasetAdapter, PreparedDataset, register_dataset_adapter


DEFAULT_VARIABLE_ITEMIDS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "systolic_bp": [220050, 220179, 224167, 227243],
    "diastolic_bp": [220051, 220180, 224643, 227242],
    "mean_bp": [220052, 220181],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "temperature": [223761, 223762],
    "glucose": [220621, 225664, 226537],
    "lactate": [225668],
    "creatinine": [220615],
}


@dataclass
class MIMICHypotensionConfig:
    """Options for deriving a MIMIC-IV hypotension classification dataset."""

    mimic_path: str | Path
    observation_hours: float = 24.0
    prediction_hours: float = 6.0
    hypotension_threshold: float = 65.0
    variable_itemids: dict[str, list[int]] = field(default_factory=lambda: dict(DEFAULT_VARIABLE_ITEMIDS))
    outcome_itemids: list[int] = field(default_factory=lambda: [220052, 220181])
    chunk_size: int = 1_000_000
    max_stays: int | None = None
    min_observations: int = 1
    require_full_prediction_window: bool = True
    require_outcome_measurement: bool = True
    relative_time_anchor: str = "2000-01-01 00:00:00"


class MIMICIVHypotensionAdapter(DatasetAdapter):
    """Prepare MIMIC-IV ICU chart data for future hypotension prediction.

    A label is `true` when any configured outcome measurement, by default mean
    arterial pressure from item IDs 220052 or 220181, is less than or equal to
    `hypotension_threshold` during the prediction window immediately following
    the observation window.
    """

    name = "mimic_iv_hypotension"

    def __init__(self, config: MIMICHypotensionConfig) -> None:
        self.config = config
        self.source = _MIMICSource(config.mimic_path)

    def prepare(self) -> PreparedDataset:
        """Build generic time-series records and binary hypotension labels."""
        cohort = self._load_cohort()
        item_to_variable = {
            itemid: variable
            for variable, itemids in self.config.variable_itemids.items()
            for itemid in itemids
        }
        desired_itemids = sorted(set(item_to_variable) | set(self.config.outcome_itemids))
        records_parts: list[pd.DataFrame] = []
        outcome_parts: list[pd.DataFrame] = []

        for chunk in self.source.iter_table(
            "icu/chartevents.csv.gz",
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            chunksize=self.config.chunk_size,
        ):
            chunk = chunk.dropna(subset=["stay_id", "itemid", "valuenum"])
            chunk["stay_id"] = chunk["stay_id"].astype("int64")
            chunk["itemid"] = chunk["itemid"].astype("int64")
            chunk = chunk[chunk["stay_id"].isin(cohort.index) & chunk["itemid"].isin(desired_itemids)]
            if chunk.empty:
                continue
            chunk["charttime"] = pd.to_datetime(chunk["charttime"])
            chunk = chunk.merge(cohort, left_on="stay_id", right_index=True, how="inner")

            observed = chunk[
                (chunk["charttime"] >= chunk["intime"])
                & (chunk["charttime"] < chunk["observation_end"])
                & chunk["itemid"].isin(item_to_variable)
            ].copy()
            if not observed.empty:
                observed["patient_id"] = observed["stay_id"].astype(str)
                observed["variable"] = observed["itemid"].map(item_to_variable)
                observed["value"] = observed["valuenum"].astype(float)
                observed["timestamp"] = self._relative_timestamp(observed["charttime"], observed["intime"])
                records_parts.append(observed[["patient_id", "variable", "value", "timestamp"]])

            outcome = chunk[
                (chunk["charttime"] >= chunk["observation_end"])
                & (chunk["charttime"] < chunk["prediction_end"])
                & chunk["itemid"].isin(self.config.outcome_itemids)
            ].copy()
            if not outcome.empty:
                outcome_parts.append(outcome[["stay_id", "valuenum"]])

        records = self._finalize_records(records_parts)
        labels = self._build_labels(cohort, records, outcome_parts)
        records = records[records["patient_id"].isin(set(labels["patient_id"]))].reset_index(drop=True)
        metadata = self._metadata(cohort, records, labels)
        return PreparedDataset(records=records, labels=labels, metadata=metadata)

    def _load_cohort(self) -> pd.DataFrame:
        stays = self.source.read_table(
            "icu/icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        )
        stays["intime"] = pd.to_datetime(stays["intime"])
        stays["outtime"] = pd.to_datetime(stays["outtime"])
        stays["observation_end"] = stays["intime"] + pd.to_timedelta(self.config.observation_hours, unit="h")
        stays["prediction_end"] = stays["observation_end"] + pd.to_timedelta(self.config.prediction_hours, unit="h")
        if self.config.require_full_prediction_window:
            stays = stays[stays["outtime"] >= stays["prediction_end"]]
        else:
            stays = stays[stays["outtime"] > stays["observation_end"]].copy()
            stays["prediction_end"] = stays[["prediction_end", "outtime"]].min(axis=1)
        stays = stays.sort_values("stay_id")
        if self.config.max_stays is not None:
            stays = stays.head(self.config.max_stays)
        return stays.set_index("stay_id")

    def _relative_timestamp(self, charttime: pd.Series, intime: pd.Series) -> pd.Series:
        elapsed = charttime - intime
        anchor = pd.Timestamp(self.config.relative_time_anchor)
        return (anchor + elapsed).dt.strftime("%Y-%m-%d %H:%M:%S")

    def _finalize_records(self, records_parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not records_parts:
            return pd.DataFrame(columns=["patient_id", "variable", "value", "timestamp"])
        records = pd.concat(records_parts, ignore_index=True)
        return records.sort_values(["patient_id", "timestamp", "variable"]).reset_index(drop=True)

    def _build_labels(
        self,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        outcome_parts: list[pd.DataFrame],
    ) -> pd.DataFrame:
        observed_counts = records.groupby("patient_id").size() if not records.empty else pd.Series(dtype=int)
        eligible_ids = set(observed_counts[observed_counts >= self.config.min_observations].index)
        if outcome_parts:
            outcomes = pd.concat(outcome_parts, ignore_index=True)
            outcome_summary = outcomes.groupby("stay_id")["valuenum"].agg(
                any_hypotension=lambda s: bool((s.astype(float) <= self.config.hypotension_threshold).any()),
                n_outcome_measurements="size",
            )
        else:
            outcome_summary = pd.DataFrame(columns=["any_hypotension", "n_outcome_measurements"])
        rows = []
        for stay_id in cohort.index:
            patient_id = str(stay_id)
            if patient_id not in eligible_ids:
                continue
            if stay_id not in outcome_summary.index:
                if self.config.require_outcome_measurement:
                    continue
                label = "false"
            else:
                label = "true" if bool(outcome_summary.loc[stay_id, "any_hypotension"]) else "false"
            rows.append({"patient_id": patient_id, "label": label})
        return pd.DataFrame(rows, columns=["patient_id", "label"])

    def _metadata(self, cohort: pd.DataFrame, records: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
        return _jsonable({
            "dataset": self.name,
            "source": str(self.config.mimic_path),
            "config": _jsonable(asdict(self.config)),
            "n_candidate_stays": int(len(cohort)),
            "n_labeled_stays": int(len(labels)),
            "n_records": int(len(records)),
            "label_counts": labels["label"].value_counts().to_dict() if not labels.empty else {},
            "physionet_reference": "MIMIC-IV v3.1, https://physionet.org/content/mimiciv/3.1/",
            "time_axis": "Relative ICU time anchored at " + self.config.relative_time_anchor,
        })


class _MIMICSource:
    """Read MIMIC-IV CSV.GZ tables from either a zip archive or directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.is_zip = self.path.is_file() and self.path.suffix.lower() == ".zip"

    def read_table(self, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
        if self.is_zip:
            with ZipFile(self.path) as zf:
                entry = self._zip_entry(zf, table)
                with gzip.GzipFile(fileobj=zf.open(entry)) as fh:
                    return pd.read_csv(fh, usecols=usecols)
        return pd.read_csv(self._file_path(table), usecols=usecols)

    def iter_table(
        self,
        table: str,
        usecols: list[str] | None,
        chunksize: int,
    ) -> Iterator[pd.DataFrame]:
        if self.is_zip:
            with ZipFile(self.path) as zf:
                entry = self._zip_entry(zf, table)
                with gzip.GzipFile(fileobj=zf.open(entry)) as fh:
                    yield from pd.read_csv(fh, usecols=usecols, chunksize=chunksize)
            return
        yield from pd.read_csv(self._file_path(table), usecols=usecols, chunksize=chunksize)

    def _zip_entry(self, zf: ZipFile, table: str) -> str:
        matches = [name for name in zf.namelist() if name.endswith(table)]
        if not matches:
            raise FileNotFoundError(f"Could not find {table} inside {self.path}")
        return matches[0]

    def _file_path(self, table: str) -> Path:
        filename = Path(table).name
        matches = [path for path in self.path.rglob(filename) if path.as_posix().endswith(table)]
        if not matches:
            raise FileNotFoundError(f"Could not find {table} under {self.path}")
        return matches[0]


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


register_dataset_adapter(MIMICIVHypotensionAdapter.name, MIMICIVHypotensionAdapter)
