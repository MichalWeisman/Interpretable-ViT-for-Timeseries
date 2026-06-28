"""Convert irregular event records into fixed value/mask time-series tensors."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .data import BinnedTimeSeries


AGGREGATIONS = {"mean", "median", "min", "max", "first", "last"}


class TimeSeriesBinner:
    """Fit and apply the preprocessing contract used by the ViT pipeline.

    The binner learns training-only metadata: variable order, label encoding,
    global time bins, and per-variable normalization statistics. `transform`
    then applies that frozen contract to any split without changing the tensor
    shape, which avoids train/test leakage.

    Output tensors have shape `[patients, 2, variables, timesteps]`; channel 0
    contains normalized values and channel 1 contains the observation mask.
    """

    def __init__(self, config: DataConfig | None = None, **kwargs: Any) -> None:
        """Create a binner from a `DataConfig` plus optional field overrides."""
        self.config = config or DataConfig()
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        if self.config.aggregation not in AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: {self.config.aggregation}")
        self.variable_vocab_: list[str] = []
        self.time_start_: pd.Timestamp | None = None
        self.time_end_: pd.Timestamp | None = None
        self.time_bins_: list[str] = []
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self.label_to_index_: dict[str, int] = {}
        self.index_to_label_: list[str] = []

    def fit(self, records: pd.DataFrame | str | Path, labels: pd.DataFrame | dict | str | Path | None = None):
        """Learn vocabulary, global bins, normalization stats, and labels.

        Parameters may be in-memory pandas objects or paths to CSV/Parquet
        files. If time bounds are not configured, they are inferred from the
        records passed to `fit`, so callers should pass training records only.
        """
        records_df = self._read_table(records)
        records_df = self._prepare_records(records_df)
        self.variable_vocab_ = sorted(records_df[self.config.variable_col].astype(str).unique().tolist())
        self.time_start_ = pd.Timestamp(self.config.time_start) if self.config.time_start else records_df[self.config.timestamp_col].min().floor(self.config.granularity)
        self.time_end_ = pd.Timestamp(self.config.time_end) if self.config.time_end else records_df[self.config.timestamp_col].max().ceil(self.config.granularity)
        self._refresh_time_bins()
        for variable, group in records_df.groupby(self.config.variable_col):
            values = pd.to_numeric(group[self.config.value_col], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else 0.0
            std = float(values.std(ddof=0)) if len(values) else 1.0
            self.means_[str(variable)] = mean
            self.stds_[str(variable)] = std if std > 0 else 1.0
        if labels is not None:
            labels_df = self._prepare_labels(labels)
            names = sorted(labels_df[self.config.label_col].astype(str).unique().tolist())
            self.index_to_label_ = names
            self.label_to_index_ = {label: idx for idx, label in enumerate(names)}
        return self

    def transform(self, records: pd.DataFrame | str | Path, labels: pd.DataFrame | dict | str | Path | None = None) -> BinnedTimeSeries:
        """Convert records into value/mask tensors using fitted metadata.

        Unknown variables are ignored so validation/test/inference tensors keep
        the same feature shape as training. Labels are optional for inference.
        """
        if self.time_start_ is None or self.time_end_ is None:
            raise RuntimeError("TimeSeriesBinner must be fitted before transform.")
        records_df = self._prepare_records(self._read_table(records))
        labels_df = self._prepare_labels(labels) if labels is not None else None
        if labels_df is not None:
            patient_ids = labels_df[self.config.patient_id_col].astype(str).tolist()
        else:
            patient_ids = sorted(records_df[self.config.patient_id_col].astype(str).unique().tolist())
        var_index = {var: idx for idx, var in enumerate(self.variable_vocab_)}
        n_patients = len(patient_ids)
        n_vars = len(self.variable_vocab_)
        n_steps = len(self.time_bins_)
        x = np.zeros((n_patients, 2, n_vars, n_steps), dtype=np.float32)
        pid_index = {pid: idx for idx, pid in enumerate(patient_ids)}
        work = records_df.copy()
        work[self.config.patient_id_col] = work[self.config.patient_id_col].astype(str)
        work[self.config.variable_col] = work[self.config.variable_col].astype(str)
        work = work[work[self.config.patient_id_col].isin(pid_index)]
        work = work[work[self.config.variable_col].isin(var_index)]
        work = work[(work[self.config.timestamp_col] >= self.time_start_) & (work[self.config.timestamp_col] < self.time_end_)]
        if len(work):
            delta = pd.to_timedelta(self.config.granularity)
            bin_idx = ((work[self.config.timestamp_col] - self.time_start_) / delta).astype(int)
            work = work.assign(_bin=bin_idx, _value=pd.to_numeric(work[self.config.value_col], errors="coerce"))
            work = work.dropna(subset=["_value"])
            grouped = work.groupby([self.config.patient_id_col, self.config.variable_col, "_bin"], sort=False)["_value"]
            aggregated = self._aggregate(grouped).reset_index()
            for row in aggregated.to_dict("records"):
                pid = str(row[self.config.patient_id_col])
                variable = str(row[self.config.variable_col])
                step = int(row["_bin"])
                value = float(row["_value"])
                if 0 <= step < n_steps:
                    p = pid_index[pid]
                    v = var_index[variable]
                    x[p, 0, v, step] = (value - self.means_.get(variable, 0.0)) / self.stds_.get(variable, 1.0)
                    x[p, 1, v, step] = 1.0
        y = None
        label_names = None
        if labels_df is not None:
            label_names = self.index_to_label_
            y = labels_df[self.config.label_col].astype(str).map(self.label_to_index_).to_numpy()
            if np.any(pd.isna(y)):
                missing = labels_df.loc[pd.isna(y), self.config.label_col].unique().tolist()
                raise ValueError(f"Labels not seen during fit: {missing}")
            y = y.astype(np.int64)
        return BinnedTimeSeries(patient_ids=patient_ids, x=x, y=y, label_names=label_names)

    def fit_transform(self, records: pd.DataFrame | str | Path, labels: pd.DataFrame | dict | str | Path | None = None) -> BinnedTimeSeries:
        """Fit on `records` and immediately transform them."""
        return self.fit(records, labels).transform(records, labels)

    def save(self, path: str | Path) -> None:
        """Persist fitted preprocessing metadata as JSON."""
        payload = {
            "config": asdict(self.config),
            "variable_vocab": self.variable_vocab_,
            "time_start": self.time_start_.isoformat() if self.time_start_ is not None else None,
            "time_end": self.time_end_.isoformat() if self.time_end_ is not None else None,
            "time_bins": self.time_bins_,
            "means": self.means_,
            "stds": self.stds_,
            "label_to_index": self.label_to_index_,
            "index_to_label": self.index_to_label_,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TimeSeriesBinner":
        """Load a previously saved binner."""
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        binner = cls(DataConfig(**payload["config"]))
        binner.variable_vocab_ = payload["variable_vocab"]
        binner.time_start_ = pd.Timestamp(payload["time_start"])
        binner.time_end_ = pd.Timestamp(payload["time_end"])
        binner.time_bins_ = payload["time_bins"]
        binner.means_ = {str(k): float(v) for k, v in payload["means"].items()}
        binner.stds_ = {str(k): float(v) for k, v in payload["stds"].items()}
        binner.label_to_index_ = {str(k): int(v) for k, v in payload["label_to_index"].items()}
        binner.index_to_label_ = payload["index_to_label"]
        return binner

    def _refresh_time_bins(self) -> None:
        if self.time_start_ is None or self.time_end_ is None:
            return
        bins = pd.date_range(self.time_start_, self.time_end_, freq=self.config.granularity, inclusive="left")
        self.time_bins_ = [ts.isoformat() for ts in bins]

    def _aggregate(self, grouped):
        if self.config.aggregation == "mean":
            return grouped.mean()
        if self.config.aggregation == "median":
            return grouped.median()
        if self.config.aggregation == "min":
            return grouped.min()
        if self.config.aggregation == "max":
            return grouped.max()
        if self.config.aggregation == "first":
            return grouped.first()
        return grouped.last()

    def _prepare_records(self, records: pd.DataFrame) -> pd.DataFrame:
        required = [self.config.patient_id_col, self.config.variable_col, self.config.value_col, self.config.timestamp_col]
        missing = [col for col in required if col not in records.columns]
        if missing:
            raise ValueError(f"Records table is missing columns: {missing}")
        out = records.copy()
        out[self.config.timestamp_col] = pd.to_datetime(out[self.config.timestamp_col])
        if getattr(out[self.config.timestamp_col].dt, "tz", None) is not None:
            out[self.config.timestamp_col] = out[self.config.timestamp_col].dt.tz_convert(None)
        return out

    def _prepare_labels(self, labels: pd.DataFrame | dict | str | Path) -> pd.DataFrame:
        if isinstance(labels, dict):
            labels = pd.DataFrame(
                {
                    self.config.patient_id_col: list(labels.keys()),
                    self.config.label_col: list(labels.values()),
                }
            )
        else:
            labels = self._read_table(labels)
        required = [self.config.patient_id_col, self.config.label_col]
        missing = [col for col in required if col not in labels.columns]
        if missing:
            raise ValueError(f"Labels table is missing columns: {missing}")
        out = labels.copy()
        out[self.config.patient_id_col] = out[self.config.patient_id_col].astype(str)
        return out

    @staticmethod
    def _read_table(table: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(table, pd.DataFrame):
            return table
        path = Path(table)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)
