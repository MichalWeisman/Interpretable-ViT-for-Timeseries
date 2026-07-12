"""Configurable MIMIC-IV multi-target dataset creation.

This module creates portable pre-tensor datasets from MIMIC-IV hospital
admissions or ICU stays. It intentionally stops at the generic records/labels
schema used by the binner: one `records.csv`, one `labels.csv`, and one
metadata JSON per target/window combination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..base import PreparedDataset, TargetWindowConfig, register_dataset_adapter
from .mimic_iv import _MIMICSource, _jsonable, standardize_temperature_to_celsius


logger = logging.getLogger(__name__)


TARGET_NAMES = [
    "cardiovascular_event",
    "nosocomial_infection",
    "hypoglycemia",
    "hypokalemia",
    "prolonged_hyperglycemia",
    "in_hospital_mortality",
    "hypotension",
]

COHORT_LEVELS = {"admission", "icu"}
DEFAULT_TARGETS = [
    "cardiovascular_event",
    "nosocomial_infection",
    "hypoglycemia",
    "hypokalemia",
    "prolonged_hyperglycemia",
    "in_hospital_mortality",
]

REQUIRED_CONFIG_FIELDS = {"windows"}


MIMICTargetWindowConfig = TargetWindowConfig


@dataclass
class MIMICTargetVariableConfig:
    """Per-target source variable mappings used to build records and labels."""

    lab_regexes: dict[str, list[str]] = field(default_factory=dict)
    lab_itemids: dict[str, list[int]] = field(default_factory=dict)
    chart_regexes: dict[str, list[str]] = field(default_factory=dict)
    chart_itemids: dict[str, list[int]] = field(default_factory=dict)
    drug_regexes: dict[str, list[str]] = field(default_factory=dict)
    inputevent_regexes: dict[str, list[str]] = field(default_factory=dict)
    inputevent_itemids: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class MIMICTargetsConfig:
    """Options for creating multiple MIMIC-IV target datasets."""

    mimic_path: str | Path
    output_dir: str | Path = "data/mimic_targets"
    cache_dir: str | Path | None = "data/mimic_targets/cache"
    cohort_level: str = "admission"
    windows: list[TargetWindowConfig] = field(default_factory=list)
    targets: list[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    lab_regexes: dict[str, list[str]] = field(default_factory=dict)
    lab_itemids: dict[str, list[int]] = field(default_factory=dict)
    chart_regexes: dict[str, list[str]] = field(default_factory=dict)
    chart_itemids: dict[str, list[int]] = field(default_factory=dict)
    drug_regexes: dict[str, list[str]] = field(default_factory=dict)
    inputevent_regexes: dict[str, list[str]] = field(default_factory=dict)
    inputevent_itemids: dict[str, list[int]] = field(default_factory=dict)
    target_variables: dict[str, MIMICTargetVariableConfig] = field(default_factory=dict)
    chunk_size: int = 1_000_000
    use_extracted_files: bool = True
    use_filtered_cache: bool = True
    require_full_window: bool = False
    min_observations: int = 1
    relative_time_anchor: str = "2000-01-01 00:00:00"
    fever_threshold_celsius: float = 37.8
    fever_prior_clean_threshold_celsius: float = 37.7
    leukocytosis_threshold: float = 11_000.0
    leukopenia_threshold: float = 4_500.0
    neutropenia_threshold: float = 1_500.0
    hypoglycemia_threshold: float = 70.0
    hypokalemia_threshold: float = 3.6
    hyperglycemia_threshold: float = 180.0
    hyperglycemia_min_day: int = 3
    hyperglycemia_max_day: int = 14
    hyperglycemia_duration_hours: float = 48.0
    troponin_i_threshold: float = 0.12
    hypotension_systolic_threshold: float = 90.0
    hypotension_diastolic_threshold: float = 60.0
    temperature_celsius_min: float = 25.0
    temperature_celsius_max: float = 45.0
    progress_interval_chunks: int = 1
    max_admissions: int | None = None
    max_stays: int | None = None
    require_outcome_measurement: bool = False


class MIMICIVMultiTargetAdapter:
    """Prepare configurable MIMIC-IV hospital-admission target datasets."""

    name = "mimic_iv_multi_target"

    def __init__(self, config: MIMICTargetsConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir) if config.cache_dir is not None else None
        extraction_dir = self.cache_dir / "extracted" if self.cache_dir and config.use_extracted_files else None
        self.source = _MIMICSource(config.mimic_path, extraction_dir=extraction_dir)

    def prepare_all(self) -> dict[tuple[str, str], PreparedDataset]:
        """Return all configured datasets keyed by `(window_name, target_name)`."""
        self._validate_targets()
        target_mappings = self._resolve_target_mappings()
        all_lab_itemids = _merge_item_mappings(mapping.lab_itemids for mapping in target_mappings.values())
        all_chart_itemids = _merge_item_mappings(mapping.chart_itemids for mapping in target_mappings.values())
        all_input_itemids = _merge_item_mappings(mapping.inputevent_itemids for mapping in target_mappings.values())
        all_drug_regexes = _merge_regex_mappings(mapping.drug_regexes for mapping in target_mappings.values())
        cohort_base = self._load_cohort()
        desired_chart = sorted({itemid for itemids in all_chart_itemids.values() for itemid in itemids})
        desired_labs = sorted({itemid for itemids in all_lab_itemids.values() for itemid in itemids})
        chart = self._load_events("icu/chartevents.csv.gz", "charttime", "itemid", desired_chart, ["hadm_id", "stay_id", "charttime", "itemid", "valuenum"])
        labs = self._load_events("hosp/labevents.csv.gz", "charttime", "itemid", desired_labs, ["hadm_id", "charttime", "itemid", "valuenum", "valueuom"])
        micro = self._load_microbiology()
        prescriptions = self._load_prescriptions(all_drug_regexes)
        inputevents = self._load_inputevents(sorted({itemid for itemids in all_input_itemids.values() for itemid in itemids}))
        datasets: dict[tuple[str, str], PreparedDataset] = {}
        for window in self.config.windows:
            cohort = self._windowed_cohort(cohort_base, window)
            logger.info("Preparing MIMIC targets for window %s: cohort_level=%s rows=%d", window.name, self.config.cohort_level, len(cohort))
            for target in self.config.targets:
                mapping = target_mappings[target]
                records = self._build_records(cohort, window, mapping.lab_itemids, mapping.chart_itemids, chart, labs, micro, prescriptions, inputevents, mapping.inputevent_itemids, mapping.drug_regexes)
                target_inputs = _TargetInputs(cohort, window, mapping.lab_itemids, mapping.chart_itemids, chart, labs, micro, prescriptions, inputevents)
                labels, target_metadata = self._build_target_labels(target, target_inputs, records)
                target_records = records[records["patient_id"].isin(set(labels["patient_id"]))].reset_index(drop=True)
                metadata = self._metadata(
                    window,
                    target,
                    cohort,
                    target_records,
                    labels,
                    mapping.lab_itemids,
                    mapping.chart_itemids,
                    mapping.inputevent_itemids,
                    mapping.drug_regexes,
                    target_metadata,
                )
                datasets[(window.name, target)] = PreparedDataset(target_records, labels, metadata)
        return datasets

    def save_all(self, output_dir: str | Path | None = None) -> dict[tuple[str, str], Path]:
        """Create and save all configured datasets."""
        root = Path(output_dir) if output_dir is not None else Path(self.config.output_dir)
        outputs: dict[tuple[str, str], Path] = {}
        for (window_name, target), prepared in self.prepare_all().items():
            out = root / window_name / target
            prepared.save(out)
            outputs[(window_name, target)] = out
        return outputs

    def _validate_targets(self) -> None:
        if self.config.cohort_level not in COHORT_LEVELS:
            raise ValueError(f"cohort_level must be one of {sorted(COHORT_LEVELS)}; got {self.config.cohort_level!r}")
        unknown = sorted(set(self.config.targets) - set(TARGET_NAMES))
        if unknown:
            raise ValueError(f"Unknown target(s): {unknown}. Available targets: {TARGET_NAMES}")
        if self.config.cohort_level == "admission" and "hypotension" in self.config.targets:
            raise ValueError("Target 'hypotension' requires cohort_level='icu' because it is defined per ICU stay.")

    def _load_cohort(self) -> pd.DataFrame:
        if self.config.cohort_level == "icu":
            return self._load_icu_stays()
        return self._load_admissions()

    def _load_admissions(self) -> pd.DataFrame:
        admissions = self.source.read_table(
            "hosp/admissions.csv.gz",
            usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"],
        )
        admissions["hadm_id"] = admissions["hadm_id"].astype("int64")
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
        admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")
        admissions = admissions.sort_values("hadm_id").set_index("hadm_id")
        if self.config.max_admissions is not None:
            admissions = admissions.head(int(self.config.max_admissions))
        return admissions

    def _load_icu_stays(self) -> pd.DataFrame:
        stays = self.source.read_table(
            "icu/icustays.csv.gz",
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        )
        stays["hadm_id"] = stays["hadm_id"].astype("int64")
        stays["stay_id"] = stays["stay_id"].astype("int64")
        stays["intime"] = pd.to_datetime(stays["intime"])
        stays["outtime"] = pd.to_datetime(stays["outtime"])
        try:
            admissions = self.source.read_table("hosp/admissions.csv.gz", usecols=["hadm_id", "deathtime"])
            admissions["hadm_id"] = admissions["hadm_id"].astype("int64")
            admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")
            stays = stays.merge(admissions, on="hadm_id", how="left")
        except FileNotFoundError:
            stays["deathtime"] = pd.NaT
        stays = stays.sort_values("stay_id").set_index("stay_id")
        if self.config.max_stays is not None:
            stays = stays.head(int(self.config.max_stays))
        return stays

    def _windowed_cohort(self, admissions: pd.DataFrame, window: TargetWindowConfig) -> pd.DataFrame:
        cohort = admissions.copy()
        start_col = self._cohort_start_col()
        end_col = self._cohort_end_col()
        cohort["observation_end"] = cohort[start_col] + pd.to_timedelta(window.observation_hours, unit="h")
        cohort["prediction_start"] = cohort["observation_end"] + pd.to_timedelta(window.gap_hours, unit="h")
        cohort["prediction_end"] = cohort["prediction_start"] + pd.to_timedelta(window.prediction_hours, unit="h")
        if self.config.require_full_window:
            cohort = cohort[cohort[end_col] >= cohort["prediction_end"]]
        else:
            cohort = cohort[cohort[end_col] >= cohort["observation_end"]].copy()
            cohort["prediction_end"] = cohort[["prediction_end", end_col]].min(axis=1)
        return cohort

    def _resolve_target_mappings(self) -> dict[str, MIMICTargetVariableConfig]:
        mappings: dict[str, MIMICTargetVariableConfig] = {}
        for target in self.config.targets:
            raw = self.config.target_variables.get(target, MIMICTargetVariableConfig())
            if not isinstance(raw, MIMICTargetVariableConfig):
                raw = MIMICTargetVariableConfig(**(raw or {}))
            lab_itemids = self._resolve_lab_itemids(raw.lab_itemids or self.config.lab_itemids, raw.lab_regexes or self.config.lab_regexes)
            chart_itemids = self._resolve_chart_itemids(raw.chart_itemids or self.config.chart_itemids, raw.chart_regexes or self.config.chart_regexes)
            inputevent_itemids = self._resolve_inputevent_itemids(raw.inputevent_itemids or self.config.inputevent_itemids, raw.inputevent_regexes or self.config.inputevent_regexes)
            mappings[target] = MIMICTargetVariableConfig(
                lab_regexes=dict(raw.lab_regexes or self.config.lab_regexes),
                lab_itemids=lab_itemids,
                chart_regexes=dict(raw.chart_regexes or self.config.chart_regexes),
                chart_itemids=chart_itemids,
                drug_regexes=dict(raw.drug_regexes or self.config.drug_regexes),
                inputevent_regexes=dict(raw.inputevent_regexes or self.config.inputevent_regexes),
                inputevent_itemids=inputevent_itemids,
            )
        return mappings

    def _resolve_lab_itemids(self, itemids_by_variable: dict[str, list[int]], regexes_by_variable: dict[str, list[str]]) -> dict[str, list[int]]:
        configured = {name: sorted({int(itemid) for itemid in itemids}) for name, itemids in itemids_by_variable.items()}
        try:
            labs = self.source.read_table("hosp/d_labitems.csv.gz", usecols=["itemid", "label", "fluid", "category"])
        except FileNotFoundError:
            return configured
        text = (
            labs[["label", "fluid", "category"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        for variable, patterns in regexes_by_variable.items():
            matches = _match_patterns(text, patterns)
            itemids = set(configured.get(variable, []))
            itemids.update(labs.loc[matches, "itemid"].astype(int).tolist())
            if itemids:
                configured[variable] = sorted(itemids)
        return configured

    def _resolve_inputevent_itemids(self, itemids_by_variable: dict[str, list[int]], regexes_by_variable: dict[str, list[str]]) -> dict[str, list[int]]:
        configured = {name: sorted({int(itemid) for itemid in itemids}) for name, itemids in itemids_by_variable.items()}
        try:
            items = self.source.read_table("icu/d_items.csv.gz", usecols=["itemid", "label", "abbreviation", "category"])
        except FileNotFoundError:
            return configured
        text = (
            items[["label", "abbreviation", "category"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        for variable, patterns in regexes_by_variable.items():
            matches = _match_patterns(text, patterns)
            itemids = set(configured.get(variable, []))
            itemids.update(items.loc[matches, "itemid"].astype(int).tolist())
            if itemids:
                configured[variable] = sorted(itemids)
        return configured

    def _resolve_chart_itemids(self, itemids_by_variable: dict[str, list[int]], regexes_by_variable: dict[str, list[str]]) -> dict[str, list[int]]:
        configured = _canonical_chart_itemids(itemids_by_variable)
        try:
            items = self.source.read_table("icu/d_items.csv.gz", usecols=["itemid", "label", "abbreviation", "category"])
        except FileNotFoundError:
            return configured
        text = (
            items[["label", "abbreviation", "category"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        for variable, patterns in regexes_by_variable.items():
            matches = _match_patterns(text, patterns)
            variable = _canonical_chart_variable(variable)
            itemids = set(configured.get(variable, []))
            itemids.update(items.loc[matches, "itemid"].astype(int).tolist())
            if itemids:
                configured[variable] = sorted(itemids)
        return configured

    def _load_events(
        self,
        table: str,
        time_col: str,
        item_col: str,
        desired_itemids: list[int],
        usecols: list[str],
    ) -> pd.DataFrame:
        if not desired_itemids:
            return pd.DataFrame(columns=usecols)
        cache_path = self._cache_path(table, {"itemids": desired_itemids, "usecols": usecols})
        if cache_path is not None and self.config.use_filtered_cache and cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if time_col in frame.columns:
                frame[time_col] = pd.to_datetime(frame[time_col])
            return frame
        parts: list[pd.DataFrame] = []
        desired = set(desired_itemids)
        for chunk in self._iter_optional_table(table, usecols=usecols, chunksize=self.config.chunk_size):
            if chunk.empty:
                continue
            required = ["hadm_id", item_col, time_col]
            if "stay_id" in usecols and "stay_id" in chunk.columns:
                required.append("stay_id")
            chunk = chunk.dropna(subset=required)
            if chunk.empty:
                continue
            chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
            if "stay_id" in chunk.columns:
                chunk["stay_id"] = chunk["stay_id"].astype("int64")
            chunk[item_col] = chunk[item_col].astype("int64")
            filtered = chunk[chunk[item_col].isin(desired)].copy()
            if not filtered.empty:
                filtered[time_col] = pd.to_datetime(filtered[time_col])
                if "valuenum" in filtered.columns:
                    filtered["valuenum"] = pd.to_numeric(filtered["valuenum"], errors="coerce")
                parts.append(filtered)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        if cache_path is not None and self.config.use_filtered_cache:
            self._write_cache(frame, cache_path)
        return frame

    def _load_microbiology(self) -> pd.DataFrame:
        usecols = ["subject_id", "hadm_id", "charttime", "chartdate", "spec_type_desc", "test_name"]
        cache_path = self._cache_path("hosp/microbiologyevents.csv.gz", {"cultures": ["blood", "urine"], "usecols": usecols})
        if cache_path is not None and self.config.use_filtered_cache and cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if not frame.empty:
                frame["charttime"] = pd.to_datetime(frame["charttime"])
            return frame
        parts: list[pd.DataFrame] = []
        for chunk in self._iter_optional_table("hosp/microbiologyevents.csv.gz", usecols=usecols, chunksize=self.config.chunk_size):
            if chunk.empty or "hadm_id" not in chunk:
                continue
            chunk = chunk.dropna(subset=["hadm_id"])
            text = (
                chunk[["spec_type_desc", "test_name"]]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            filtered = chunk[text.str.contains("blood|urine", regex=True, na=False)].copy()
            if not filtered.empty:
                filtered["hadm_id"] = filtered["hadm_id"].astype("int64")
                filtered["charttime"] = _coalesce_datetime(filtered.get("charttime"), filtered.get("chartdate"))
                parts.append(filtered.dropna(subset=["charttime"]))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        if cache_path is not None and self.config.use_filtered_cache:
            self._write_cache(frame, cache_path)
        return frame

    def _load_prescriptions(self, drug_regexes: dict[str, list[str]]) -> pd.DataFrame:
        usecols = ["subject_id", "hadm_id", "starttime", "stoptime", "drug"]
        patterns = [pattern for values in drug_regexes.values() for pattern in values]
        if not patterns:
            return pd.DataFrame(columns=usecols)
        cache_path = self._cache_path("hosp/prescriptions.csv.gz", {"drug_regexes": drug_regexes, "usecols": usecols})
        if cache_path is not None and self.config.use_filtered_cache and cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if not frame.empty:
                frame["starttime"] = pd.to_datetime(frame["starttime"])
            return frame
        parts: list[pd.DataFrame] = []
        for chunk in self._iter_optional_table("hosp/prescriptions.csv.gz", usecols=usecols, chunksize=self.config.chunk_size):
            if chunk.empty:
                continue
            chunk = chunk.dropna(subset=["hadm_id", "drug", "starttime"])
            text = chunk["drug"].astype(str).str.lower()
            filtered = chunk[_match_patterns(text, patterns)].copy()
            if not filtered.empty:
                filtered["hadm_id"] = filtered["hadm_id"].astype("int64")
                filtered["starttime"] = pd.to_datetime(filtered["starttime"])
                parts.append(filtered)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        if cache_path is not None and self.config.use_filtered_cache:
            self._write_cache(frame, cache_path)
        return frame

    def _load_inputevents(self, desired_itemids: list[int]) -> pd.DataFrame:
        return self._load_events(
            "icu/inputevents.csv.gz",
            "starttime",
            "itemid",
            desired_itemids,
            ["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"],
        )

    def _build_records(
        self,
        cohort: pd.DataFrame,
        window: TargetWindowConfig,
        lab_itemids: dict[str, list[int]],
        chart_itemids: dict[str, list[int]],
        chart: pd.DataFrame,
        labs: pd.DataFrame,
        micro: pd.DataFrame,
        prescriptions: pd.DataFrame,
        inputevents: pd.DataFrame,
        input_itemids: dict[str, list[int]],
        drug_regexes: dict[str, list[str]],
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        item_to_lab = {itemid: variable for variable, itemids in lab_itemids.items() for itemid in itemids}
        item_to_chart = {itemid: variable for variable, itemids in chart_itemids.items() for itemid in itemids}
        parts.append(self._numeric_records(labs, cohort, "charttime", item_to_lab, source="labs"))
        parts.append(self._numeric_records(chart, cohort, "charttime", item_to_chart, source="chart"))
        parts.append(self._culture_records(micro, cohort))
        parts.append(self._prescription_records(prescriptions, cohort, drug_regexes))
        parts.append(self._inputevent_records(inputevents, cohort, input_itemids))
        records = pd.concat([part for part in parts if not part.empty], ignore_index=True) if parts else pd.DataFrame()
        if records.empty:
            return pd.DataFrame(columns=["patient_id", "variable", "value", "timestamp"])
        records = records.dropna(subset=["value"])
        observed_counts = records.groupby("patient_id").size()
        eligible = set(observed_counts[observed_counts >= self.config.min_observations].index)
        records = records[records["patient_id"].isin(eligible)]
        return records.sort_values(["patient_id", "timestamp", "variable"]).reset_index(drop=True)

    def _numeric_records(
        self,
        events: pd.DataFrame,
        cohort: pd.DataFrame,
        time_col: str,
        item_to_variable: dict[int, str],
        *,
        source: str,
    ) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if events.empty:
            return pd.DataFrame(columns=columns)
        frame = self._attach_cohort_for_records(events, cohort, time_col)
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["variable"] = frame["itemid"].astype(int).map(item_to_variable)
        frame = frame.dropna(subset=["variable"])
        frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")
        is_temperature = frame["variable"] == "temperature"
        if is_temperature.any():
            frame.loc[is_temperature, "value"] = standardize_temperature_to_celsius(
                frame.loc[is_temperature, "value"],
                itemids=frame.loc[is_temperature, "itemid"] if source == "chart" else None,
                celsius_min=self.config.temperature_celsius_min,
                celsius_max=self.config.temperature_celsius_max,
            )
        frame["patient_id"] = frame["_cohort_id"].astype(str)
        frame["timestamp"] = self._relative_timestamp(frame[time_col], frame["_cohort_start"])
        return frame[columns]

    def _culture_records(self, micro: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if micro.empty:
            return pd.DataFrame(columns=columns)
        frame = self._attach_cohort_for_records(micro, cohort, "charttime")
        if frame.empty:
            return pd.DataFrame(columns=columns)
        text = (
            frame[["spec_type_desc", "test_name"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        frame["variable"] = pd.Series(pd.NA, index=frame.index, dtype="object")
        frame.loc[text.str.contains("blood", regex=False), "variable"] = "blood_culture"
        frame.loc[text.str.contains("urine", regex=False), "variable"] = "urine_culture"
        frame = frame.dropna(subset=["variable"])
        frame["patient_id"] = frame["_cohort_id"].astype(str)
        frame["value"] = 1.0
        frame["timestamp"] = self._relative_timestamp(frame["charttime"], frame["_cohort_start"])
        return frame[columns]

    def _prescription_records(self, prescriptions: pd.DataFrame, cohort: pd.DataFrame, drug_regexes: dict[str, list[str]]) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if prescriptions.empty:
            return pd.DataFrame(columns=columns)
        frame = self._attach_cohort_for_records(prescriptions, cohort, "starttime")
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["drug_text"] = frame["drug"].astype(str).str.lower()
        rows = []
        for variable, patterns in drug_regexes.items():
            matched = frame[_match_patterns(frame["drug_text"], patterns)]
            if not matched.empty:
                out = matched[["_cohort_id", "starttime", "_cohort_start"]].copy()
                out["variable"] = variable
                rows.append(out)
        if not rows:
            return pd.DataFrame(columns=columns)
        records = pd.concat(rows, ignore_index=True)
        records["patient_id"] = records["_cohort_id"].astype(str)
        records["value"] = 1.0
        records["timestamp"] = self._relative_timestamp(records["starttime"], records["_cohort_start"])
        return records[columns]

    def _inputevent_records(self, inputevents: pd.DataFrame, cohort: pd.DataFrame, input_itemids: dict[str, list[int]]) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if inputevents.empty:
            return pd.DataFrame(columns=columns)
        frame = self._attach_cohort_for_records(inputevents, cohort, "starttime")
        if frame.empty:
            return pd.DataFrame(columns=columns)
        item_to_variables: dict[int, list[str]] = {}
        for variable, itemids in input_itemids.items():
            for itemid in itemids:
                item_to_variables.setdefault(itemid, []).append(variable)
        frame["variable"] = frame["itemid"].astype(int).map(item_to_variables)
        frame = frame.dropna(subset=["variable"]).explode("variable")
        frame["patient_id"] = frame["_cohort_id"].astype(str)
        frame["value"] = 1.0
        frame["timestamp"] = self._relative_timestamp(frame["starttime"], frame["_cohort_start"])
        return frame[columns]

    def _build_target_labels(
        self,
        target: str,
        inputs: _TargetInputs,
        records: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if target == "cardiovascular_event":
            return self._threshold_target(
                inputs,
                "troponin_i",
                self.config.troponin_i_threshold,
                op=">=",
                definition=f"Troponin I >= {self.config.troponin_i_threshold} ng/mL in prediction window",
            )
        if target == "hypoglycemia":
            return self._threshold_target(
                inputs,
                "blood_glucose",
                self.config.hypoglycemia_threshold,
                op="<",
                exclude_prior=False,
                definition=f"Glucose < {self.config.hypoglycemia_threshold} mg/dL in prediction window",
            )
        if target == "hypokalemia":
            return self._threshold_target(
                inputs,
                "potassium",
                self.config.hypokalemia_threshold,
                op="<",
                exclude_prior=False,
                definition=f"Potassium < {self.config.hypokalemia_threshold} mmol/L in prediction window",
            )
        if target == "prolonged_hyperglycemia":
            return self._prolonged_hyperglycemia_target(inputs)
        if target == "in_hospital_mortality":
            return self._mortality_target(inputs, records)
        if target == "nosocomial_infection":
            return self._infection_target(inputs)
        if target == "hypotension":
            return self._hypotension_target(inputs)
        raise ValueError(target)

    def _threshold_target(
        self,
        inputs: _TargetInputs,
        variable: str,
        threshold: float,
        *,
        op: str,
        exclude_prior: bool = True,
        definition: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        itemids = set(inputs.lab_itemids.get(variable, [])) | set(inputs.chart_itemids.get(variable, []))
        events = _events_for_itemids(inputs.labs, inputs.chart, itemids)
        if events.empty:
            return self._labels_from_sets(inputs.cohort, set(), set(), definition)
        events = self._attach_window(events, inputs.cohort)
        abnormal = events[pd.to_numeric(events["valuenum"], errors="coerce").map(lambda value: _compare(value, threshold, op))]
        id_col = self._cohort_id_col()
        prior = set(abnormal.loc[abnormal["charttime"] < abnormal["prediction_start"], id_col].astype(int)) if exclude_prior else set()
        positive = set(
            abnormal.loc[
                (abnormal["charttime"] >= abnormal["prediction_start"])
                & (abnormal["charttime"] < abnormal["prediction_end"]),
                id_col,
            ].astype(int)
        )
        return self._labels_from_sets(inputs.cohort, positive, prior, definition)

    def _prolonged_hyperglycemia_target(self, inputs: _TargetInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
        itemids = set(inputs.lab_itemids.get("blood_glucose", [])) | set(inputs.chart_itemids.get("blood_glucose", []))
        events = _events_for_itemids(inputs.labs, inputs.chart, itemids)
        definition = (
            f"All glucose values > {self.config.hyperglycemia_threshold} mg/dL over a "
            f"{self.config.hyperglycemia_duration_hours:g}h period between hospital days "
            f"{self.config.hyperglycemia_min_day}-{self.config.hyperglycemia_max_day}"
        )
        if events.empty:
            return self._labels_from_sets(inputs.cohort, set(), set(), definition)
        events = self._attach_window(events, inputs.cohort)
        events["value"] = pd.to_numeric(events["valuenum"], errors="coerce")
        id_col = self._cohort_id_col()
        prior = set(events.loc[(events["charttime"] < events["prediction_start"]) & (events["value"] > self.config.hyperglycemia_threshold), id_col].astype(int))
        positive: set[int] = set()
        min_offset = pd.to_timedelta(self.config.hyperglycemia_min_day, unit="D")
        max_offset = pd.to_timedelta(self.config.hyperglycemia_max_day, unit="D")
        duration = pd.to_timedelta(self.config.hyperglycemia_duration_hours, unit="h")
        for cohort_id, group in events.groupby(id_col):
            start = inputs.cohort.loc[int(cohort_id), self._cohort_start_col()]
            window_start = max(inputs.cohort.loc[int(cohort_id), "prediction_start"], start + min_offset)
            window_end = min(inputs.cohort.loc[int(cohort_id), "prediction_end"], start + max_offset)
            candidate = group[(group["charttime"] >= window_start) & (group["charttime"] < window_end)].dropna(subset=["value"]).sort_values("charttime")
            if self._has_persistent_high_glucose(candidate, duration):
                positive.add(int(cohort_id))
        return self._labels_from_sets(inputs.cohort, positive, prior, definition)

    def _has_persistent_high_glucose(self, glucose_events: pd.DataFrame, duration: pd.Timedelta) -> bool:
        if glucose_events.empty:
            return False
        times = glucose_events["charttime"].tolist()
        values = glucose_events["value"].astype(float).tolist()
        for idx, start in enumerate(times):
            end = start + duration
            mask = [(time >= start and time <= end) for time in times]
            if not any(mask):
                continue
            span_values = [value for value, keep in zip(values, mask) if keep]
            span_times = [time for time, keep in zip(times, mask) if keep]
            if span_times[-1] - span_times[0] >= duration and all(value > self.config.hyperglycemia_threshold for value in span_values):
                return True
        return False

    def _mortality_target(self, inputs: _TargetInputs, records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        cohort = inputs.cohort
        death = cohort["deathtime"]
        start_col = self._cohort_start_col()
        positive_mask = (
            death.notna()
            & (death >= cohort["prediction_start"])
            & (death < cohort["prediction_end"])
            & (death >= cohort[start_col] + pd.to_timedelta(48, unit="h"))
        )
        positive = set(cohort.loc[positive_mask].index.astype(int))
        prior = set(cohort.loc[death.notna() & (death < cohort["prediction_start"])].index.astype(int))
        return self._labels_from_sets(cohort, positive, prior, f"Death in prediction window and at least 48h after {self.config.cohort_level} start")

    def _infection_target(self, inputs: _TargetInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
        signs = self._infection_sign_events(inputs)
        cultures = self._culture_events_any_time(inputs.micro, inputs.cohort)
        positive: set[int] = set()
        id_col = self._cohort_id_col()
        prior: set[int] = set(signs.loc[signs["charttime"] < signs["prediction_start"], id_col].astype(int)) if not signs.empty else set()
        if not signs.empty and not cultures.empty:
            pred_signs = signs[(signs["charttime"] >= signs["prediction_start"]) & (signs["charttime"] < signs["prediction_end"])]
            culture_groups = {int(cohort_id): group.sort_values("charttime") for cohort_id, group in cultures.groupby(id_col)}
            for _, sign in pred_signs.sort_values("charttime").iterrows():
                group = culture_groups.get(int(sign[id_col]))
                if group is None:
                    continue
                has_culture = (
                    (group["charttime"] >= sign["charttime"])
                    & (group["charttime"] <= sign["charttime"] + pd.to_timedelta(24, unit="h"))
                ).any()
                if has_culture:
                    positive.add(int(sign[id_col]))
        return self._labels_from_sets(
            inputs.cohort,
            positive,
            prior,
            "Nosocomial infection sign at least 48h after admission with blood/urine culture within 24h",
        )

    def _infection_sign_events(self, inputs: _TargetInputs) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        temperature_itemids = set(inputs.chart_itemids.get("temperature", []))
        temp = inputs.chart[inputs.chart["itemid"].isin(temperature_itemids)].copy() if not inputs.chart.empty else pd.DataFrame()
        if not temp.empty:
            temp = self._attach_window(temp, inputs.cohort)
            temp["value"] = standardize_temperature_to_celsius(
                temp["valuenum"],
                itemids=temp["itemid"],
                celsius_min=self.config.temperature_celsius_min,
                celsius_max=self.config.temperature_celsius_max,
            )
            fever_rows = []
            id_col = self._cohort_id_col()
            for _, group in temp.dropna(subset=["value"]).groupby(id_col):
                group = group.sort_values("charttime")
                for _, row in group.iterrows():
                    if row["value"] < self.config.fever_threshold_celsius:
                        continue
                    same_day = group[(group["charttime"] >= row["charttime"]) & (group["charttime"] <= row["charttime"] + pd.to_timedelta(24, unit="h"))]
                    prior = group[(group["charttime"] >= row["charttime"] - pd.to_timedelta(48, unit="h")) & (group["charttime"] < row["charttime"])]
                    if (same_day["value"] >= self.config.fever_threshold_celsius).sum() >= 2 and (prior.empty or (prior["value"] < self.config.fever_prior_clean_threshold_celsius).all()):
                        fever_rows.append(row)
                        break
            if fever_rows:
                fever = pd.DataFrame(fever_rows)
                fever["sign"] = "fever"
                pieces.append(fever[self._event_identity_columns("charttime", "prediction_start", "prediction_end", "sign")])
        wbc = self._lab_variable_events(inputs, "wbc")
        if not wbc.empty:
            wbc["value"] = pd.to_numeric(wbc["valuenum"], errors="coerce")
            leukocytosis_rows = []
            leukopenia = wbc[wbc["value"] < self.config.leukopenia_threshold].copy()
            if not leukopenia.empty:
                leukopenia["sign"] = "leukopenia"
                pieces.append(leukopenia[self._event_identity_columns("charttime", "prediction_start", "prediction_end", "sign")])
            for _, row in wbc[wbc["value"] > self.config.leukocytosis_threshold].sort_values("charttime").iterrows():
                group = wbc[wbc[self._cohort_id_col()] == row[self._cohort_id_col()]]
                prior = group[(group["charttime"] >= row["charttime"] - pd.to_timedelta(48, unit="h")) & (group["charttime"] < row["charttime"])]
                if prior.empty or (prior["value"] <= self.config.leukocytosis_threshold).all():
                    leukocytosis_rows.append(row)
            if leukocytosis_rows:
                leukocytosis = pd.DataFrame(leukocytosis_rows)
                leukocytosis["sign"] = "leukocytosis"
                pieces.append(leukocytosis[self._event_identity_columns("charttime", "prediction_start", "prediction_end", "sign")])
        neut = self._lab_variable_events(inputs, "neutrophils")
        if not neut.empty:
            neut["value"] = pd.to_numeric(neut["valuenum"], errors="coerce")
            neutropenia = neut[neut["value"] < self.config.neutropenia_threshold].copy()
            if not neutropenia.empty:
                neutropenia["sign"] = "neutropenia"
                pieces.append(neutropenia[self._event_identity_columns("charttime", "prediction_start", "prediction_end", "sign")])
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=self._event_identity_columns("charttime", "prediction_start", "prediction_end", "sign"))

    def _lab_variable_events(self, inputs: _TargetInputs, variable: str) -> pd.DataFrame:
        itemids = set(inputs.lab_itemids.get(variable, []))
        frame = inputs.labs[inputs.labs["itemid"].isin(itemids)].copy() if not inputs.labs.empty else pd.DataFrame()
        return self._attach_window(frame, inputs.cohort) if not frame.empty else frame

    def _culture_events_any_time(self, micro: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        if micro.empty:
            return pd.DataFrame(columns=self._event_identity_columns("charttime"))
        frame = self._attach_cohort_for_records(micro, cohort, "charttime", through_prediction=True)
        if frame.empty:
            return pd.DataFrame(columns=self._event_identity_columns("charttime"))
        text = (
            frame[["spec_type_desc", "test_name"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        out = frame[text.str.contains("blood|urine", regex=True, na=False)].copy()
        out[self._cohort_id_col()] = out["_cohort_id"].astype(int)
        return out[self._event_identity_columns("charttime")]

    def _hypotension_target(self, inputs: _TargetInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
        if self.config.cohort_level != "icu":
            raise ValueError("Target 'hypotension' requires cohort_level='icu'.")
        systolic_itemids = set(inputs.chart_itemids.get("systolic_bp", []))
        diastolic_itemids = set(inputs.chart_itemids.get("diastolic_bp", []))
        itemids = systolic_itemids | diastolic_itemids
        events = inputs.chart[inputs.chart["itemid"].isin(itemids)].copy() if not inputs.chart.empty else pd.DataFrame()
        definition = (
            f"Systolic blood pressure < {self.config.hypotension_systolic_threshold:g} mmHg "
            f"or diastolic blood pressure < {self.config.hypotension_diastolic_threshold:g} mmHg "
            "in ICU prediction window"
        )
        if events.empty:
            return self._labels_from_outcomes(inputs.cohort, set(), set(), definition)
        events = self._attach_window(events, inputs.cohort)
        events["value"] = pd.to_numeric(events["valuenum"], errors="coerce")
        outcome = events[(events["charttime"] >= events["prediction_start"]) & (events["charttime"] < events["prediction_end"])]
        id_col = self._cohort_id_col()
        measured = set(outcome[id_col].astype(int))
        systolic_low = outcome["itemid"].isin(systolic_itemids) & (outcome["value"] < self.config.hypotension_systolic_threshold)
        diastolic_low = outcome["itemid"].isin(diastolic_itemids) & (outcome["value"] < self.config.hypotension_diastolic_threshold)
        positive = set(outcome.loc[systolic_low | diastolic_low, id_col].astype(int))
        return self._labels_from_outcomes(inputs.cohort, positive, measured, definition)

    def _labels_from_sets(
        self,
        cohort: pd.DataFrame,
        positive: set[int],
        excluded_prior: set[int],
        definition: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows = []
        for hadm_id in cohort.index:
            if int(hadm_id) in excluded_prior:
                continue
            rows.append({"patient_id": str(int(hadm_id)), "label": "true" if int(hadm_id) in positive else "false"})
        labels = pd.DataFrame(rows, columns=["patient_id", "label"])
        metadata = {
            "target_definition": definition,
            "excluded_prior_positive": len(excluded_prior),
            f"positive_{self._cohort_id_col()}s": len(positive),
        }
        return labels, metadata

    def _labels_from_outcomes(
        self,
        cohort: pd.DataFrame,
        positive: set[int],
        measured: set[int],
        definition: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows = []
        for cohort_id in cohort.index:
            cohort_id = int(cohort_id)
            if self.config.require_outcome_measurement and cohort_id not in measured:
                continue
            rows.append({"patient_id": str(cohort_id), "label": "true" if cohort_id in positive else "false"})
        labels = pd.DataFrame(rows, columns=["patient_id", "label"])
        metadata = {
            "target_definition": definition,
            "required_outcome_measurement": self.config.require_outcome_measurement,
            "n_outcome_measured": len(measured),
            f"positive_{self._cohort_id_col()}s": len(positive),
        }
        return labels, metadata

    def _attach_window(self, events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        frame = self._attach_cohort_for_records(events, cohort, "charttime", through_prediction=True)
        if frame.empty:
            return frame
        id_col = self._cohort_id_col()
        frame[id_col] = frame["_cohort_id"].astype(int)
        return frame

    def _attach_cohort_for_records(
        self,
        events: pd.DataFrame,
        cohort: pd.DataFrame,
        time_col: str,
        *,
        through_prediction: bool = False,
    ) -> pd.DataFrame:
        if events.empty:
            return events.copy()
        start_col = self._cohort_start_col()
        if self.config.cohort_level == "icu" and "stay_id" in events.columns:
            direct = events[events["stay_id"].notna() & events["stay_id"].isin(cohort.index)].copy()
            frames = []
            if not direct.empty:
                direct["_cohort_id"] = direct["stay_id"].astype(int)
                direct["_cohort_start"] = direct["_cohort_id"].map(cohort[start_col])
                direct["observation_end"] = direct["_cohort_id"].map(cohort["observation_end"])
                direct["prediction_start"] = direct["_cohort_id"].map(cohort["prediction_start"])
                direct["prediction_end"] = direct["_cohort_id"].map(cohort["prediction_end"])
                frames.append(direct)
            indirect = events[events["stay_id"].isna()].copy()
            if not indirect.empty:
                indirect = indirect.drop(columns=["stay_id"])
                join_cols = [self._cohort_id_col(), "hadm_id", start_col, "observation_end", "prediction_start", "prediction_end"]
                cohort_join = cohort.reset_index()[list(dict.fromkeys(join_cols))]
                indirect = indirect.merge(cohort_join, on="hadm_id", how="inner")
                if not indirect.empty:
                    indirect["_cohort_id"] = indirect[self._cohort_id_col()].astype(int)
                    indirect["_cohort_start"] = indirect[start_col]
                    frames.append(indirect)
            frame = pd.concat(frames, ignore_index=True) if frames else events.iloc[0:0].copy()
        else:
            join_cols = [self._cohort_id_col(), "hadm_id", start_col, "observation_end", "prediction_start", "prediction_end"]
            cohort_join = cohort.reset_index()[list(dict.fromkeys(join_cols))]
            frame = events.merge(cohort_join, on="hadm_id", how="inner")
            if frame.empty:
                return frame
            frame["_cohort_id"] = frame[self._cohort_id_col()].astype(int)
            frame["_cohort_start"] = frame[start_col]
        if frame.empty:
            return frame
        end = frame["prediction_end"] if through_prediction else frame["observation_end"]
        return frame[(frame[time_col] >= frame["_cohort_start"]) & (frame[time_col] < end)].copy()

    def _cohort_id_col(self) -> str:
        return "stay_id" if self.config.cohort_level == "icu" else "hadm_id"

    def _cohort_start_col(self) -> str:
        return "intime" if self.config.cohort_level == "icu" else "admittime"

    def _cohort_end_col(self) -> str:
        return "outtime" if self.config.cohort_level == "icu" else "dischtime"

    def _event_identity_columns(self, *columns: str) -> list[str]:
        return list(dict.fromkeys([self._cohort_id_col(), "hadm_id", *columns]))

    def _relative_timestamp(self, event_time: pd.Series, start_time: pd.Series) -> pd.Series:
        elapsed = event_time - start_time
        anchor = pd.Timestamp(self.config.relative_time_anchor)
        return (anchor + elapsed).dt.strftime("%Y-%m-%d %H:%M:%S")

    def _iter_optional_table(self, table: str, usecols: list[str], chunksize: int):
        try:
            yield from self.source.iter_table(table, usecols=usecols, chunksize=chunksize)
        except FileNotFoundError:
            logger.warning("MIMIC table %s not found; continuing with empty data", table)
            return

    def _cache_path(self, table: str, payload: Mapping[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True).encode("utf-8")).hexdigest()[:16]
        safe_table = re.sub(r"[^A-Za-z0-9_.-]+", "_", table.replace(".csv.gz", ""))
        return self.cache_dir / "filtered" / f"{safe_table}_{digest}.parquet"

    def _write_cache(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            logger.warning("Skipping Parquet cache because a Parquet engine is not installed: %s", exc)

    def _metadata(
        self,
        window: TargetWindowConfig,
        target: str,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        labels: pd.DataFrame,
        lab_itemids: dict[str, list[int]],
        chart_itemids: dict[str, list[int]],
        input_itemids: dict[str, list[int]],
        drug_regexes: dict[str, list[str]],
        target_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return _jsonable(
            {
                "dataset": self.name,
                "target": target,
                "source": str(self.config.mimic_path),
                "cohort_level": self.config.cohort_level,
                "patient_id": self._cohort_id_col(),
                "patient_id_source": self._cohort_id_col(),
                "time_axis": self._time_axis_description(),
                "window": asdict(window),
                "config": asdict(self.config),
                "source_tables": [
                    "hosp/admissions.csv.gz",
                    "hosp/labevents.csv.gz",
                    "hosp/microbiologyevents.csv.gz",
                    "hosp/prescriptions.csv.gz",
                    "icu/chartevents.csv.gz",
                    "icu/inputevents.csv.gz",
                ],
                "variable_mappings": {
                    "lab_itemids": lab_itemids,
                    "chart_itemids": chart_itemids,
                    "inputevent_itemids": input_itemids,
                    "drug_regexes": drug_regexes,
                },
                "unit_conversions": {
                    "temperature": "Fahrenheit values are converted to Celsius and implausible Celsius values are dropped."
                },
                f"n_candidate_{self.config.cohort_level}s": int(len(cohort)),
                f"n_labeled_{self.config.cohort_level}s": int(len(labels)),
                "n_records": int(len(records)),
                "record_counts_by_variable": records["variable"].value_counts().to_dict() if not records.empty else {},
                "label_counts": labels["label"].value_counts().to_dict() if not labels.empty else {},
                "target_metadata": target_metadata,
            }
        )

    def _time_axis_description(self) -> str:
        if self.config.cohort_level == "icu":
            return "Relative ICU admission time anchored at " + self.config.relative_time_anchor
        return "Relative hospital admission time anchored at " + self.config.relative_time_anchor


@dataclass
class _TargetInputs:
    cohort: pd.DataFrame
    window: TargetWindowConfig
    lab_itemids: dict[str, list[int]]
    chart_itemids: dict[str, list[int]]
    chart: pd.DataFrame
    labs: pd.DataFrame
    micro: pd.DataFrame
    prescriptions: pd.DataFrame
    inputevents: pd.DataFrame


def _merge_item_mappings(mappings: Any) -> dict[str, list[int]]:
    merged: dict[str, set[int]] = {}
    for mapping in mappings:
        for variable, itemids in mapping.items():
            merged.setdefault(str(variable), set()).update(int(itemid) for itemid in itemids)
    return {variable: sorted(itemids) for variable, itemids in merged.items()}


def _merge_regex_mappings(mappings: Any) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for mapping in mappings:
        for variable, patterns in mapping.items():
            bucket = merged.setdefault(str(variable), [])
            for pattern in patterns:
                if pattern not in bucket:
                    bucket.append(str(pattern))
    return merged


def _canonical_chart_variable(variable: str) -> str:
    variable = str(variable)
    if variable in {"temperature_c", "temperature_f", "temperature_celsius", "temperature_fahrenheit"}:
        return "temperature"
    return variable


def _canonical_chart_itemids(mapping: dict[str, list[int]]) -> dict[str, list[int]]:
    merged: dict[str, set[int]] = {}
    for variable, itemids in mapping.items():
        merged.setdefault(_canonical_chart_variable(variable), set()).update(int(itemid) for itemid in itemids)
    return {variable: sorted(itemids) for variable, itemids in merged.items()}


def load_mimic_targets_config(path: str | Path) -> MIMICTargetsConfig:
    """Load a JSON/YAML config for the multi-target MIMIC adapter."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json":
            raw = json.load(fh) or {}
        else:
            import yaml

            raw = yaml.safe_load(fh) or {}
    raw = dict(raw)
    missing = sorted(field for field in REQUIRED_CONFIG_FIELDS if field not in raw)
    if missing:
        raise ValueError(f"MIMIC target config is missing required field(s): {', '.join(missing)}")
    if "windows" in raw:
        raw["windows"] = [window if isinstance(window, TargetWindowConfig) else TargetWindowConfig(**window) for window in raw["windows"]]
    if "target_variables" in raw:
        raw["target_variables"] = {
            str(target): value if isinstance(value, MIMICTargetVariableConfig) else MIMICTargetVariableConfig(**(value or {}))
            for target, value in (raw["target_variables"] or {}).items()
        }
    return MIMICTargetsConfig(**raw)


def _match_patterns(text: pd.Series, patterns: list[str]) -> pd.Series:
    if not patterns:
        return pd.Series(False, index=text.index)
    combined = "|".join(f"(?:{pattern})" for pattern in patterns)
    return text.str.contains(combined, case=False, regex=True, na=False)


def _coalesce_datetime(primary: pd.Series | None, fallback: pd.Series | None) -> pd.Series:
    if primary is None:
        return pd.to_datetime(fallback, errors="coerce")
    out = pd.to_datetime(primary, errors="coerce")
    if fallback is not None:
        out = out.fillna(pd.to_datetime(fallback, errors="coerce"))
    return out


def _events_for_itemids(labs: pd.DataFrame, chart: pd.DataFrame, itemids: set[int]) -> pd.DataFrame:
    parts = []
    if not labs.empty:
        parts.append(labs[labs["itemid"].isin(itemids)][["hadm_id", "charttime", "itemid", "valuenum"]])
    if not chart.empty:
        columns = ["hadm_id", "charttime", "itemid", "valuenum"]
        if "stay_id" in chart.columns:
            columns.insert(1, "stay_id")
        parts.append(chart[chart["itemid"].isin(itemids)][columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["hadm_id", "charttime", "itemid", "valuenum"])


def _compare(value: float, threshold: float, op: str) -> bool:
    if pd.isna(value):
        return False
    if op == ">=":
        return float(value) >= threshold
    if op == "<":
        return float(value) < threshold
    raise ValueError(op)


register_dataset_adapter(MIMICIVMultiTargetAdapter.name, MIMICIVMultiTargetAdapter)
