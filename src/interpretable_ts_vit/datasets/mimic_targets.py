"""Configurable MIMIC-IV multi-target dataset creation.

This module creates portable pre-tensor datasets from MIMIC-IV hospital
admissions. It intentionally stops at the generic records/labels schema used by
the binner: one `records.csv`, one `labels.csv`, and one metadata JSON per
target/window combination.
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

from .base import PreparedDataset, register_dataset_adapter
from .mimic_iv import _MIMICSource, _jsonable, standardize_temperature_to_celsius


logger = logging.getLogger(__name__)


TARGET_NAMES = [
    "cardiovascular_event",
    "nosocomial_infection",
    "hypoglycemia",
    "prolonged_hyperglycemia",
    "in_hospital_mortality",
]


DEFAULT_WINDOWS = [
    {"name": "obs48_target24_gap0", "observation_hours": 48.0, "prediction_hours": 24.0, "gap_hours": 0.0},
    {"name": "obs48_target24_gap8", "observation_hours": 48.0, "prediction_hours": 24.0, "gap_hours": 8.0},
]


DEFAULT_LAB_REGEXES: dict[str, list[str]] = {
    "blood_glucose": [r"\bglucose\b"],
    "creatinine": [r"\bcreatinine\b"],
    "albumin": [r"\balbumin\b"],
    "wbc": [r"white blood cells?", r"\bwbc\b"],
    "lactate": [r"\blactate\b"],
    "ldh": [r"lactate dehydrogenase", r"\bldh\b"],
    "neutrophils": [r"\bneutrophils?\b"],
    "troponin_i": [r"troponin\s*i\b"],
    "blood_ph": [r"\bph\b"],
    "bicarbonate": [r"\bbicarbonate\b"],
}


DEFAULT_CHART_ITEMIDS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "temperature": [223761, 223762],
    "blood_glucose": [220621, 225664, 226537],
}


DEFAULT_DRUG_REGEXES: dict[str, list[str]] = {
    "antibiotics": [
        r"cef",
        r"cillin",
        r"cycline",
        r"floxacin",
        r"meropenem",
        r"vancomycin",
        r"azithromycin",
        r"metronidazole",
        r"trimethoprim",
        r"sulfamethoxazole",
    ],
    "dextrose_5": [r"dextrose\s*5", r"\bd5w\b"],
    "dextrose_10": [r"dextrose\s*10", r"\bd10w\b"],
    "insulin_rapid_acting": [r"lispro", r"aspart", r"glulisine"],
    "insulin_short_acting": [r"regular insulin", r"\binsulin regular\b"],
    "insulin_intermediate_acting": [r"\bnph\b"],
    "insulin_long_acting": [r"glargine", r"detemir"],
    "insulin_ultra_long_acting": [r"degludec"],
    "insulin_premixed": [r"70/30", r"75/25", r"premix", r"premixed"],
    "metformin": [r"metformin"],
    "meglitinides": [r"repaglinide", r"nateglinide"],
    "glp1_agonists": [r"exenatide", r"liraglutide", r"dulaglutide", r"semaglutide", r"albiglutide"],
    "sulfonylurea": [r"glyburide", r"glipizide", r"glimepiride"],
    "dpp4_inhibitors_with_metformin": [r"sitagliptin.*metformin", r"metformin.*sitagliptin", r"janumet"],
    "dpp4_inhibitors": [r"sitagliptin", r"saxagliptin", r"linagliptin", r"alogliptin"],
    "alpha_glucosidase_inhibitors": [r"acarbose", r"miglitol"],
    "furosemide": [r"furosemide", r"\blasix\b"],
}


DEFAULT_INPUTEVENT_REGEXES: dict[str, list[str]] = {
    "dextrose_5": [r"dextrose\s*5", r"\bd5w\b"],
    "dextrose_10": [r"dextrose\s*10", r"\bd10w\b"],
    "insulin_rapid_acting": [r"lispro", r"aspart", r"glulisine"],
    "insulin_short_acting": [r"regular insulin", r"\binsulin regular\b"],
    "insulin_intermediate_acting": [r"\bnph\b"],
    "insulin_long_acting": [r"glargine", r"detemir"],
    "insulin_ultra_long_acting": [r"degludec"],
    "insulin_premixed": [r"70/30", r"75/25", r"premix", r"premixed"],
}


@dataclass
class MIMICTargetWindowConfig:
    """Observation/gap/prediction window definition in hours."""

    name: str
    observation_hours: float = 48.0
    prediction_hours: float = 24.0
    gap_hours: float = 0.0


@dataclass
class MIMICTargetsConfig:
    """Options for creating multiple MIMIC-IV target datasets."""

    mimic_path: str | Path
    output_dir: str | Path = "data/mimic_targets"
    cache_dir: str | Path | None = "data/mimic_targets/cache"
    windows: list[MIMICTargetWindowConfig] = field(
        default_factory=lambda: [MIMICTargetWindowConfig(**window) for window in DEFAULT_WINDOWS]
    )
    targets: list[str] = field(default_factory=lambda: list(TARGET_NAMES))
    lab_regexes: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_LAB_REGEXES))
    lab_itemids: dict[str, list[int]] = field(default_factory=dict)
    chart_itemids: dict[str, list[int]] = field(default_factory=lambda: dict(DEFAULT_CHART_ITEMIDS))
    drug_regexes: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_DRUG_REGEXES))
    inputevent_regexes: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_INPUTEVENT_REGEXES))
    chunk_size: int = 1_000_000
    use_extracted_files: bool = True
    use_filtered_cache: bool = True
    require_full_window: bool = True
    min_observations: int = 1
    relative_time_anchor: str = "2000-01-01 00:00:00"
    fever_threshold_celsius: float = 37.8
    fever_prior_clean_threshold_celsius: float = 37.7
    leukocytosis_threshold: float = 11_000.0
    leukopenia_threshold: float = 4_500.0
    neutropenia_threshold: float = 1_500.0
    hypoglycemia_threshold: float = 70.0
    hyperglycemia_threshold: float = 180.0
    hyperglycemia_min_day: int = 3
    hyperglycemia_max_day: int = 14
    hyperglycemia_duration_hours: float = 48.0
    troponin_i_threshold: float = 0.12
    temperature_celsius_min: float = 25.0
    temperature_celsius_max: float = 45.0
    progress_interval_chunks: int = 1
    max_admissions: int | None = None


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
        lab_itemids = self._resolve_lab_itemids()
        input_itemids = self._resolve_inputevent_itemids()
        cohort_base = self._load_admissions()
        desired_chart = sorted({itemid for itemids in self.config.chart_itemids.values() for itemid in itemids})
        desired_labs = sorted({itemid for itemids in lab_itemids.values() for itemid in itemids})
        chart = self._load_events("icu/chartevents.csv.gz", "charttime", "itemid", desired_chart, ["hadm_id", "charttime", "itemid", "valuenum"])
        labs = self._load_events("hosp/labevents.csv.gz", "charttime", "itemid", desired_labs, ["hadm_id", "charttime", "itemid", "valuenum", "valueuom"])
        micro = self._load_microbiology()
        prescriptions = self._load_prescriptions()
        inputevents = self._load_inputevents(sorted({itemid for itemids in input_itemids.values() for itemid in itemids}))
        datasets: dict[tuple[str, str], PreparedDataset] = {}
        for window in self.config.windows:
            cohort = self._windowed_cohort(cohort_base, window)
            logger.info("Preparing MIMIC targets for window %s: admissions=%d", window.name, len(cohort))
            records = self._build_records(cohort, window, lab_itemids, chart, labs, micro, prescriptions, inputevents, input_itemids)
            target_inputs = _TargetInputs(cohort, window, lab_itemids, chart, labs, micro, prescriptions, inputevents)
            for target in self.config.targets:
                labels, target_metadata = self._build_target_labels(target, target_inputs, records)
                target_records = records[records["patient_id"].isin(set(labels["patient_id"]))].reset_index(drop=True)
                metadata = self._metadata(
                    window,
                    target,
                    cohort,
                    target_records,
                    labels,
                    lab_itemids,
                    input_itemids,
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
        unknown = sorted(set(self.config.targets) - set(TARGET_NAMES))
        if unknown:
            raise ValueError(f"Unknown target(s): {unknown}. Available targets: {TARGET_NAMES}")

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

    def _windowed_cohort(self, admissions: pd.DataFrame, window: MIMICTargetWindowConfig) -> pd.DataFrame:
        cohort = admissions.copy()
        cohort["observation_end"] = cohort["admittime"] + pd.to_timedelta(window.observation_hours, unit="h")
        cohort["prediction_start"] = cohort["observation_end"] + pd.to_timedelta(window.gap_hours, unit="h")
        cohort["prediction_end"] = cohort["prediction_start"] + pd.to_timedelta(window.prediction_hours, unit="h")
        if self.config.require_full_window:
            cohort = cohort[cohort["dischtime"] >= cohort["prediction_end"]]
        else:
            cohort = cohort[cohort["dischtime"] > cohort["observation_end"]].copy()
            cohort["prediction_end"] = cohort[["prediction_end", "dischtime"]].min(axis=1)
        return cohort

    def _resolve_lab_itemids(self) -> dict[str, list[int]]:
        configured = {name: sorted({int(itemid) for itemid in itemids}) for name, itemids in self.config.lab_itemids.items()}
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
        for variable, patterns in self.config.lab_regexes.items():
            matches = _match_patterns(text, patterns)
            itemids = set(configured.get(variable, []))
            itemids.update(labs.loc[matches, "itemid"].astype(int).tolist())
            if itemids:
                configured[variable] = sorted(itemids)
        return configured

    def _resolve_inputevent_itemids(self) -> dict[str, list[int]]:
        try:
            items = self.source.read_table("icu/d_items.csv.gz", usecols=["itemid", "label", "abbreviation", "category"])
        except FileNotFoundError:
            return {}
        text = (
            items[["label", "abbreviation", "category"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        resolved: dict[str, list[int]] = {}
        for variable, patterns in self.config.inputevent_regexes.items():
            matches = _match_patterns(text, patterns)
            itemids = sorted(set(items.loc[matches, "itemid"].astype(int).tolist()))
            if itemids:
                resolved[variable] = itemids
        return resolved

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
            chunk = chunk.dropna(subset=["hadm_id", item_col, time_col])
            if chunk.empty:
                continue
            chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
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

    def _load_prescriptions(self) -> pd.DataFrame:
        usecols = ["subject_id", "hadm_id", "starttime", "stoptime", "drug"]
        cache_path = self._cache_path("hosp/prescriptions.csv.gz", {"drug_regexes": self.config.drug_regexes, "usecols": usecols})
        if cache_path is not None and self.config.use_filtered_cache and cache_path.exists():
            frame = pd.read_parquet(cache_path)
            if not frame.empty:
                frame["starttime"] = pd.to_datetime(frame["starttime"])
            return frame
        patterns = [pattern for values in self.config.drug_regexes.values() for pattern in values]
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
        window: MIMICTargetWindowConfig,
        lab_itemids: dict[str, list[int]],
        chart: pd.DataFrame,
        labs: pd.DataFrame,
        micro: pd.DataFrame,
        prescriptions: pd.DataFrame,
        inputevents: pd.DataFrame,
        input_itemids: dict[str, list[int]],
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        item_to_lab = {itemid: variable for variable, itemids in lab_itemids.items() for itemid in itemids}
        item_to_chart = {itemid: variable for variable, itemids in self.config.chart_itemids.items() for itemid in itemids}
        parts.append(self._numeric_records(labs, cohort, "charttime", item_to_lab, source="labs"))
        parts.append(self._numeric_records(chart, cohort, "charttime", item_to_chart, source="chart"))
        parts.append(self._culture_records(micro, cohort))
        parts.append(self._prescription_records(prescriptions, cohort))
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
        frame = events[events["hadm_id"].isin(cohort.index)].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["observation_end"] = frame["hadm_id"].map(cohort["observation_end"])
        frame = frame[(frame[time_col] >= frame["admittime"]) & (frame[time_col] < frame["observation_end"])]
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
        frame["patient_id"] = frame["hadm_id"].astype(str)
        frame["timestamp"] = self._relative_timestamp(frame[time_col], frame["admittime"])
        return frame[columns]

    def _culture_records(self, micro: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if micro.empty:
            return pd.DataFrame(columns=columns)
        frame = micro[micro["hadm_id"].isin(cohort.index)].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["observation_end"] = frame["hadm_id"].map(cohort["observation_end"])
        frame = frame[(frame["charttime"] >= frame["admittime"]) & (frame["charttime"] < frame["observation_end"])]
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
        frame["patient_id"] = frame["hadm_id"].astype(str)
        frame["value"] = 1.0
        frame["timestamp"] = self._relative_timestamp(frame["charttime"], frame["admittime"])
        return frame[columns]

    def _prescription_records(self, prescriptions: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if prescriptions.empty:
            return pd.DataFrame(columns=columns)
        frame = prescriptions[prescriptions["hadm_id"].isin(cohort.index)].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["observation_end"] = frame["hadm_id"].map(cohort["observation_end"])
        frame = frame[(frame["starttime"] >= frame["admittime"]) & (frame["starttime"] < frame["observation_end"])]
        frame["drug_text"] = frame["drug"].astype(str).str.lower()
        rows = []
        for variable, patterns in self.config.drug_regexes.items():
            matched = frame[_match_patterns(frame["drug_text"], patterns)]
            if not matched.empty:
                out = matched[["hadm_id", "starttime", "admittime"]].copy()
                out["variable"] = variable
                rows.append(out)
        if not rows:
            return pd.DataFrame(columns=columns)
        records = pd.concat(rows, ignore_index=True)
        records["patient_id"] = records["hadm_id"].astype(str)
        records["value"] = 1.0
        records["timestamp"] = self._relative_timestamp(records["starttime"], records["admittime"])
        return records[columns]

    def _inputevent_records(self, inputevents: pd.DataFrame, cohort: pd.DataFrame, input_itemids: dict[str, list[int]]) -> pd.DataFrame:
        columns = ["patient_id", "variable", "value", "timestamp"]
        if inputevents.empty:
            return pd.DataFrame(columns=columns)
        frame = inputevents[inputevents["hadm_id"].isin(cohort.index)].copy()
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["observation_end"] = frame["hadm_id"].map(cohort["observation_end"])
        frame = frame[(frame["starttime"] >= frame["admittime"]) & (frame["starttime"] < frame["observation_end"])]
        item_to_variable = {itemid: variable for variable, itemids in input_itemids.items() for itemid in itemids}
        frame["variable"] = frame["itemid"].astype(int).map(item_to_variable)
        frame = frame.dropna(subset=["variable"])
        frame["patient_id"] = frame["hadm_id"].astype(str)
        frame["value"] = 1.0
        frame["timestamp"] = self._relative_timestamp(frame["starttime"], frame["admittime"])
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
                definition=f"Glucose < {self.config.hypoglycemia_threshold} mg/dL in prediction window",
            )
        if target == "prolonged_hyperglycemia":
            return self._prolonged_hyperglycemia_target(inputs)
        if target == "in_hospital_mortality":
            return self._mortality_target(inputs, records)
        if target == "nosocomial_infection":
            return self._infection_target(inputs)
        raise ValueError(target)

    def _threshold_target(
        self,
        inputs: _TargetInputs,
        variable: str,
        threshold: float,
        *,
        op: str,
        definition: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        itemids = set(inputs.lab_itemids.get(variable, [])) | set(self.config.chart_itemids.get(variable, []))
        events = _events_for_itemids(inputs.labs, inputs.chart, itemids)
        if events.empty:
            return self._labels_from_sets(inputs.cohort, set(), set(), definition)
        events = self._attach_window(events, inputs.cohort)
        abnormal = events[pd.to_numeric(events["valuenum"], errors="coerce").map(lambda value: _compare(value, threshold, op))]
        prior = set(abnormal.loc[abnormal["charttime"] < abnormal["prediction_start"], "hadm_id"].astype(int))
        positive = set(
            abnormal.loc[
                (abnormal["charttime"] >= abnormal["prediction_start"])
                & (abnormal["charttime"] < abnormal["prediction_end"]),
                "hadm_id",
            ].astype(int)
        )
        return self._labels_from_sets(inputs.cohort, positive, prior, definition)

    def _prolonged_hyperglycemia_target(self, inputs: _TargetInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
        itemids = set(inputs.lab_itemids.get("blood_glucose", [])) | set(self.config.chart_itemids.get("blood_glucose", []))
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
        prior = set(events.loc[(events["charttime"] < events["prediction_start"]) & (events["value"] > self.config.hyperglycemia_threshold), "hadm_id"].astype(int))
        positive: set[int] = set()
        min_offset = pd.to_timedelta(self.config.hyperglycemia_min_day, unit="D")
        max_offset = pd.to_timedelta(self.config.hyperglycemia_max_day, unit="D")
        duration = pd.to_timedelta(self.config.hyperglycemia_duration_hours, unit="h")
        for hadm_id, group in events.groupby("hadm_id"):
            admit = inputs.cohort.loc[int(hadm_id), "admittime"]
            window_start = max(inputs.cohort.loc[int(hadm_id), "prediction_start"], admit + min_offset)
            window_end = min(inputs.cohort.loc[int(hadm_id), "prediction_end"], admit + max_offset)
            candidate = group[(group["charttime"] >= window_start) & (group["charttime"] < window_end)].dropna(subset=["value"]).sort_values("charttime")
            if self._has_persistent_high_glucose(candidate, duration):
                positive.add(int(hadm_id))
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
        positive_mask = (
            death.notna()
            & (death >= cohort["prediction_start"])
            & (death < cohort["prediction_end"])
            & (death >= cohort["admittime"] + pd.to_timedelta(48, unit="h"))
        )
        positive = set(cohort.loc[positive_mask].index.astype(int))
        prior = set(cohort.loc[death.notna() & (death < cohort["prediction_start"])].index.astype(int))
        return self._labels_from_sets(cohort, positive, prior, "Death in prediction window and at least 48h after admission")

    def _infection_target(self, inputs: _TargetInputs) -> tuple[pd.DataFrame, dict[str, Any]]:
        signs = self._infection_sign_events(inputs)
        cultures = self._culture_events_any_time(inputs.micro, inputs.cohort)
        positive: set[int] = set()
        prior: set[int] = set(signs.loc[signs["charttime"] < signs["prediction_start"], "hadm_id"].astype(int)) if not signs.empty else set()
        if not signs.empty and not cultures.empty:
            pred_signs = signs[(signs["charttime"] >= signs["prediction_start"]) & (signs["charttime"] < signs["prediction_end"])]
            culture_groups = {int(hadm_id): group.sort_values("charttime") for hadm_id, group in cultures.groupby("hadm_id")}
            for _, sign in pred_signs.sort_values("charttime").iterrows():
                group = culture_groups.get(int(sign["hadm_id"]))
                if group is None:
                    continue
                has_culture = (
                    (group["charttime"] >= sign["charttime"])
                    & (group["charttime"] <= sign["charttime"] + pd.to_timedelta(24, unit="h"))
                ).any()
                if has_culture:
                    positive.add(int(sign["hadm_id"]))
        return self._labels_from_sets(
            inputs.cohort,
            positive,
            prior,
            "Nosocomial infection sign at least 48h after admission with blood/urine culture within 24h",
        )

    def _infection_sign_events(self, inputs: _TargetInputs) -> pd.DataFrame:
        pieces: list[pd.DataFrame] = []
        temperature_itemids = set(self.config.chart_itemids.get("temperature", []))
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
            for hadm_id, group in temp.dropna(subset=["value"]).groupby("hadm_id"):
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
                pieces.append(fever[["hadm_id", "charttime", "prediction_start", "prediction_end", "sign"]])
        wbc = self._lab_variable_events(inputs, "wbc")
        if not wbc.empty:
            wbc["value"] = pd.to_numeric(wbc["valuenum"], errors="coerce")
            leukocytosis_rows = []
            leukopenia = wbc[wbc["value"] < self.config.leukopenia_threshold].copy()
            if not leukopenia.empty:
                leukopenia["sign"] = "leukopenia"
                pieces.append(leukopenia[["hadm_id", "charttime", "prediction_start", "prediction_end", "sign"]])
            for _, row in wbc[wbc["value"] > self.config.leukocytosis_threshold].sort_values("charttime").iterrows():
                group = wbc[wbc["hadm_id"] == row["hadm_id"]]
                prior = group[(group["charttime"] >= row["charttime"] - pd.to_timedelta(48, unit="h")) & (group["charttime"] < row["charttime"])]
                if prior.empty or (prior["value"] <= self.config.leukocytosis_threshold).all():
                    leukocytosis_rows.append(row)
            if leukocytosis_rows:
                leukocytosis = pd.DataFrame(leukocytosis_rows)
                leukocytosis["sign"] = "leukocytosis"
                pieces.append(leukocytosis[["hadm_id", "charttime", "prediction_start", "prediction_end", "sign"]])
        neut = self._lab_variable_events(inputs, "neutrophils")
        if not neut.empty:
            neut["value"] = pd.to_numeric(neut["valuenum"], errors="coerce")
            neutropenia = neut[neut["value"] < self.config.neutropenia_threshold].copy()
            if not neutropenia.empty:
                neutropenia["sign"] = "neutropenia"
                pieces.append(neutropenia[["hadm_id", "charttime", "prediction_start", "prediction_end", "sign"]])
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["hadm_id", "charttime", "prediction_start", "prediction_end", "sign"])

    def _lab_variable_events(self, inputs: _TargetInputs, variable: str) -> pd.DataFrame:
        itemids = set(inputs.lab_itemids.get(variable, []))
        frame = inputs.labs[inputs.labs["itemid"].isin(itemids)].copy() if not inputs.labs.empty else pd.DataFrame()
        return self._attach_window(frame, inputs.cohort) if not frame.empty else frame

    def _culture_events_any_time(self, micro: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        if micro.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime"])
        frame = micro[micro["hadm_id"].isin(cohort.index)].copy()
        if frame.empty:
            return pd.DataFrame(columns=["hadm_id", "charttime"])
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["prediction_end"] = frame["hadm_id"].map(cohort["prediction_end"])
        frame = frame[(frame["charttime"] >= frame["admittime"]) & (frame["charttime"] <= frame["prediction_end"])]
        text = (
            frame[["spec_type_desc", "test_name"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        return frame[text.str.contains("blood|urine", regex=True, na=False)][["hadm_id", "charttime"]]

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
            "positive_hadm_ids": len(positive),
        }
        return labels, metadata

    def _attach_window(self, events: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        frame = events[events["hadm_id"].isin(cohort.index)].copy()
        frame["admittime"] = frame["hadm_id"].map(cohort["admittime"])
        frame["prediction_start"] = frame["hadm_id"].map(cohort["prediction_start"])
        frame["prediction_end"] = frame["hadm_id"].map(cohort["prediction_end"])
        return frame

    def _relative_timestamp(self, event_time: pd.Series, admittime: pd.Series) -> pd.Series:
        elapsed = event_time - admittime
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
        window: MIMICTargetWindowConfig,
        target: str,
        cohort: pd.DataFrame,
        records: pd.DataFrame,
        labels: pd.DataFrame,
        lab_itemids: dict[str, list[int]],
        input_itemids: dict[str, list[int]],
        target_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return _jsonable(
            {
                "dataset": self.name,
                "target": target,
                "source": str(self.config.mimic_path),
                "patient_id": "hadm_id",
                "time_axis": "Relative hospital admission time anchored at " + self.config.relative_time_anchor,
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
                    "chart_itemids": self.config.chart_itemids,
                    "drug_regexes": self.config.drug_regexes,
                    "inputevent_itemids": input_itemids,
                },
                "unit_conversions": {
                    "temperature": "Fahrenheit values are converted to Celsius and implausible Celsius values are dropped."
                },
                "n_candidate_admissions": int(len(cohort)),
                "n_labeled_admissions": int(len(labels)),
                "n_records": int(len(records)),
                "record_counts_by_variable": records["variable"].value_counts().to_dict() if not records.empty else {},
                "label_counts": labels["label"].value_counts().to_dict() if not labels.empty else {},
                "target_metadata": target_metadata,
            }
        )


@dataclass
class _TargetInputs:
    cohort: pd.DataFrame
    window: MIMICTargetWindowConfig
    lab_itemids: dict[str, list[int]]
    chart: pd.DataFrame
    labs: pd.DataFrame
    micro: pd.DataFrame
    prescriptions: pd.DataFrame
    inputevents: pd.DataFrame


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
    if "windows" in raw:
        raw["windows"] = [window if isinstance(window, MIMICTargetWindowConfig) else MIMICTargetWindowConfig(**window) for window in raw["windows"]]
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
        parts.append(chart[chart["itemid"].isin(itemids)][["hadm_id", "charttime", "itemid", "valuenum"]])
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
