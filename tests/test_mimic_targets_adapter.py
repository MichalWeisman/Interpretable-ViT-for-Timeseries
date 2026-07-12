import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from interpretable_ts_vit.datasets import TargetWindowConfig
from interpretable_ts_vit.datasets.mimic import (
    MIMICIVMultiTargetAdapter,
    MIMICTargetVariableConfig,
    MIMICTargetsConfig,
    configured_variables_for_target,
    load_mimic_targets_config,
)
from interpretable_ts_vit.cli import main as cli_main
from interpretable_ts_vit.visualization import load_normal_ranges


def _write_gz_csv(zf, name, frame):
    zf.writestr(name, gzip.compress(frame.to_csv(index=False).encode("utf-8")))


def _mini_mimic_zip(tmp_path):
    zip_path = tmp_path / "mimic-targets-mini.zip"
    admissions = pd.DataFrame(
        [
            [1, 10, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [1, 11, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [1, 12, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [2, 20, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [2, 21, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [2, 22, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [3, 30, "2100-01-01 00:00:00", "2100-01-07 00:00:00", "2100-01-03 12:00:00"],
            [3, 31, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [3, 32, "2100-01-01 00:00:00", "2100-01-07 00:00:00", "2100-01-02 16:00:00"],
            [4, 40, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [4, 41, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [5, 50, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
            [5, 51, "2100-01-01 00:00:00", "2100-01-07 00:00:00", ""],
        ],
        columns=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"],
    )
    icustays = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    icustays["stay_id"] = icustays["hadm_id"] * 10
    icustays["intime"] = icustays["admittime"]
    icustays["outtime"] = icustays["dischtime"]
    icustays["los"] = 6.0
    icustays = icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]]
    d_labitems = pd.DataFrame(
        [
            [1001, "Glucose", "Blood", "Chemistry"],
            [1002, "Creatinine", "Blood", "Chemistry"],
            [1003, "Albumin", "Blood", "Chemistry"],
            [1004, "White Blood Cells", "Blood", "Hematology"],
            [1005, "Lactate", "Blood", "Chemistry"],
            [1006, "LDH", "Blood", "Chemistry"],
            [1007, "Neutrophils", "Blood", "Hematology"],
            [1008, "Troponin I", "Blood", "Chemistry"],
            [1009, "pH", "Blood", "Blood Gas"],
            [1010, "Bicarbonate", "Blood", "Chemistry"],
            [1011, "Potassium", "Blood", "Chemistry"],
        ],
        columns=["itemid", "label", "fluid", "category"],
    )
    lab_rows = []

    def lab(hadm_id, hours, itemid, value, unit=""):
        lab_rows.append([1, hadm_id, f"2100-01-{1 + hours // 24:02d} {hours % 24:02d}:00:00", itemid, value, unit])

    for hadm_id in admissions["hadm_id"]:
        lab(hadm_id, 2, 1002, 1.0, "mg/dL")
    lab(10, 50, 1008, 0.2, "ng/mL")
    lab(11, 50, 1008, 0.01, "ng/mL")
    lab(12, 20, 1008, 0.2, "ng/mL")
    lab(12, 50, 1008, 0.3, "ng/mL")
    lab(20, 52, 1001, 60, "mg/dL")
    lab(21, 52, 1001, 90, "mg/dL")
    lab(22, 20, 1001, 60, "mg/dL")
    lab(22, 52, 1001, 65, "mg/dL")
    lab(20, 52, 1011, 3.2, "mmol/L")
    lab(21, 52, 1011, 4.0, "mmol/L")
    lab(22, 20, 1011, 3.2, "mmol/L")
    lab(22, 52, 1011, 3.4, "mmol/L")
    lab(40, 50, 1004, 12000, "cells/uL")
    lab(41, 50, 1004, 12000, "cells/uL")
    lab(50, 72, 1001, 220, "mg/dL")
    lab(50, 120, 1001, 230, "mg/dL")
    lab(51, 72, 1001, 220, "mg/dL")
    lab(51, 120, 1001, 100, "mg/dL")
    labevents = pd.DataFrame(lab_rows, columns=["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"])

    d_items = pd.DataFrame(
        [
            [220045, "Heart Rate", "HR", "Routine Vital Signs"],
            [223761, "Temperature Fahrenheit", "Temp F", "Routine Vital Signs"],
            [223762, "Temperature Celsius", "Temp C", "Routine Vital Signs"],
            [220050, "Arterial Blood Pressure systolic", "ABPs", "Routine Vital Signs"],
            [220051, "Arterial Blood Pressure diastolic", "ABPd", "Routine Vital Signs"],
            [300001, "Dextrose 10%", "D10", "Fluids/Intake"],
        ],
        columns=["itemid", "label", "abbreviation", "category"],
    )
    chart = pd.DataFrame(
        [
            [1, 10, 100, "2100-01-01 01:00:00", 220045, 80.0],
            [1, 10, 100, "2100-01-01 02:00:00", 223761, 98.6],
            [1, 10, 100, "2100-01-03 02:00:00", 220050, 85.0],
            [1, 11, 110, "2100-01-01 01:00:00", 220045, 82.0],
            [1, 11, 110, "2100-01-03 02:00:00", 220050, 100.0],
            [1, 11, 110, "2100-01-03 02:00:00", 220051, 70.0],
            [1, 12, 120, "2100-01-01 20:00:00", 220051, 55.0],
            [1, 12, 120, "2100-01-03 02:00:00", 220050, 100.0],
            [1, 12, 120, "2100-01-03 02:00:00", 220051, 70.0],
            [4, 40, 400, "2100-01-01 10:00:00", 223762, 37.0],
            [4, 40, 400, "2100-01-03 02:00:00", 223762, 38.1],
            [4, 40, 400, "2100-01-03 03:00:00", 223762, 38.0],
            [4, 41, 410, "2100-01-03 02:00:00", 223762, 38.2],
            [4, 41, 410, "2100-01-03 03:00:00", 223762, 38.3],
        ],
        columns=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
    )
    micro = pd.DataFrame(
        [
            [4, 40, "2100-01-03 04:00:00", "", "Blood Culture", "Blood Culture"],
            [4, 40, "2100-01-01 04:00:00", "", "Blood Culture", "Blood Culture"],
            [1, 10, "2100-01-01 03:00:00", "", "Urine", "Urine Culture"],
        ],
        columns=["subject_id", "hadm_id", "charttime", "chartdate", "spec_type_desc", "test_name"],
    )
    prescriptions = pd.DataFrame(
        [
            [1, 10, "2100-01-01 04:00:00", "2100-01-01 06:00:00", "Vancomycin"],
            [1, 11, "2100-01-01 04:00:00", "2100-01-01 06:00:00", "Metformin"],
        ],
        columns=["subject_id", "hadm_id", "starttime", "stoptime", "drug"],
    )
    inputevents = pd.DataFrame(
        [[1, 10, 100, "2100-01-01 05:00:00", "2100-01-01 06:00:00", 300001, 100.0]],
        columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"],
    )
    with ZipFile(zip_path, "w") as zf:
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/admissions.csv.gz", admissions)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/d_labitems.csv.gz", d_labitems)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/labevents.csv.gz", labevents)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/microbiologyevents.csv.gz", micro)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/prescriptions.csv.gz", prescriptions)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/icustays.csv.gz", icustays)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/d_items.csv.gz", d_items)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/chartevents.csv.gz", chart)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/inputevents.csv.gz", inputevents)
    return zip_path


def _config(zip_path, tmp_path, *, windows=None, targets=None, cohort_level="admission"):
    config = load_mimic_targets_config("configs/datasets/mimic/targets.yaml")
    config.mimic_path = zip_path
    config.output_dir = tmp_path / "out"
    config.cache_dir = tmp_path / "cache"
    config.cohort_level = cohort_level
    config.windows = windows or [TargetWindowConfig(name="wide", observation_hours=48, prediction_hours=96, gap_hours=0)]
    config.targets = targets or [
        "cardiovascular_event",
        "nosocomial_infection",
        "hypoglycemia",
        "hypokalemia",
        "prolonged_hyperglycemia",
        "in_hospital_mortality",
    ]
    config.target_variables = {}
    config.lab_itemids = {
        "blood_glucose": [1001],
        "creatinine": [1002],
        "albumin": [1003],
        "wbc": [1004],
        "lactate": [1005],
        "ldh": [1006],
        "neutrophils": [1007],
        "troponin_i": [1008],
        "blood_ph": [1009],
        "bicarbonate": [1010],
        "potassium": [1011],
    }
    config.inputevent_itemids = {"dextrose_10": [300001]}
    config.chart_itemids = {
        "heart_rate": [220045],
        "systolic_bp": [220050],
        "diastolic_bp": [220051],
        "temperature": [223761, 223762],
    }
    config.drug_regexes = {
        "antibiotics": [r"vancomycin"],
        "metformin": [r"metformin"],
    }
    config.chunk_size = 4
    return config


def _labels(prepared):
    return prepared.labels.set_index("patient_id")["label"].to_dict()


def test_multi_target_adapter_builds_each_target_and_keeps_prior_hypoglycemia_and_hypokalemia(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    datasets = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path)).prepare_all()

    assert _labels(datasets[("wide", "cardiovascular_event")])["10"] == "true"
    assert _labels(datasets[("wide", "cardiovascular_event")])["11"] == "false"
    assert "12" not in _labels(datasets[("wide", "cardiovascular_event")])
    assert _labels(datasets[("wide", "hypoglycemia")])["20"] == "true"
    assert _labels(datasets[("wide", "hypoglycemia")])["21"] == "false"
    assert _labels(datasets[("wide", "hypoglycemia")])["22"] == "true"
    assert _labels(datasets[("wide", "hypokalemia")])["20"] == "true"
    assert _labels(datasets[("wide", "hypokalemia")])["21"] == "false"
    assert _labels(datasets[("wide", "hypokalemia")])["22"] == "true"
    assert _labels(datasets[("wide", "in_hospital_mortality")])["30"] == "true"
    assert _labels(datasets[("wide", "in_hospital_mortality")])["31"] == "false"
    assert "32" not in _labels(datasets[("wide", "in_hospital_mortality")])
    assert _labels(datasets[("wide", "nosocomial_infection")])["40"] == "true"
    assert _labels(datasets[("wide", "nosocomial_infection")])["41"] == "false"
    assert _labels(datasets[("wide", "prolonged_hyperglycemia")])["50"] == "true"
    assert _labels(datasets[("wide", "prolonged_hyperglycemia")])["51"] == "false"


def test_mimic_target_windows_support_optional_gap(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    windows = [
        TargetWindowConfig(name="gap0", observation_hours=48, prediction_hours=24, gap_hours=0),
        TargetWindowConfig(name="gap8", observation_hours=48, prediction_hours=24, gap_hours=8),
    ]
    datasets = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, windows=windows, targets=["hypoglycemia"])).prepare_all()

    assert _labels(datasets[("gap0", "hypoglycemia")])["20"] == "true"
    assert _labels(datasets[("gap8", "hypoglycemia")])["20"] == "false"


def test_mimic_target_records_standardize_units_and_event_variables(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    dataset = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypoglycemia"])).prepare_all()[("wide", "hypoglycemia")]
    records = dataset.records

    assert round(records.loc[records["variable"] == "temperature", "value"].iloc[0], 1) == 37.0
    assert {"blood_culture", "urine_culture", "antibiotics", "metformin", "dextrose_10"}.issubset(set(records["variable"]))
    assert set(records.loc[records["variable"].isin(["blood_culture", "urine_culture", "antibiotics"]), "value"]) == {1.0}
    assert dataset.metadata["patient_id"] == "hadm_id"
    assert dataset.metadata["window"]["name"] == "wide"


def test_temperature_c_and_f_chart_variables_collapse_to_temperature(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config = _config(zip_path, tmp_path, targets=["hypoglycemia"])
    config.chart_itemids = {
        "temperature_f": [223761],
        "temperature_c": [223762],
    }

    dataset = MIMICIVMultiTargetAdapter(config).prepare_all()[("wide", "hypoglycemia")]
    temperature_records = dataset.records[dataset.records["variable"] == "temperature"]

    assert not temperature_records.empty
    assert "temperature_f" not in set(dataset.records["variable"])
    assert "temperature_c" not in set(dataset.records["variable"])
    assert round(temperature_records["value"].min(), 1) == 37.0
    assert dataset.metadata["variable_mappings"]["chart_itemids"]["temperature"] == [223761, 223762]


def test_target_variables_scope_records_per_target(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config = _config(zip_path, tmp_path, targets=["hypoglycemia", "hypokalemia"])
    config.lab_itemids = {}
    config.chart_itemids = {}
    config.inputevent_itemids = {}
    config.drug_regexes = {}
    config.target_variables = {
        "hypoglycemia": MIMICTargetVariableConfig(lab_itemids={"blood_glucose": [1001]}),
        "hypokalemia": MIMICTargetVariableConfig(lab_itemids={"potassium": [1011]}),
    }

    datasets = MIMICIVMultiTargetAdapter(config).prepare_all()

    assert "potassium" not in set(datasets[("wide", "hypoglycemia")].records["variable"])
    assert "blood_glucose" not in set(datasets[("wide", "hypokalemia")].records["variable"])
    assert datasets[("wide", "hypoglycemia")].metadata["variable_mappings"]["lab_itemids"] == {"blood_glucose": [1001]}
    assert datasets[("wide", "hypokalemia")].metadata["variable_mappings"]["lab_itemids"] == {"potassium": [1011]}


def test_configured_variables_for_target_reflects_yaml_mappings(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config = _config(zip_path, tmp_path, targets=["hypoglycemia"])
    config.lab_itemids = {}
    config.chart_itemids = {}
    config.inputevent_itemids = {}
    config.drug_regexes = {}
    config.target_variables = {
        "hypoglycemia": MIMICTargetVariableConfig(
            lab_itemids={"blood_glucose": [1001]},
            chart_itemids={"temperature_f": [223761]},
            inputevent_itemids={"dextrose_10": [300001]},
        )
    }

    assert configured_variables_for_target(config, "hypoglycemia") == ["blood_glucose", "dextrose_10", "temperature"]


def test_prepare_mimic_targets_endpoint_outputs_only_pre_tensor_files(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config_path = tmp_path / "mimic_targets.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"mimic_path: {zip_path.as_posix()}",
                f"output_dir: {(tmp_path / 'prepared').as_posix()}",
                f"cache_dir: {(tmp_path / 'cache').as_posix()}",
                "cohort_level: admission",
                "chunk_size: 4",
                "windows:",
                "  - name: obs48_target24_gap0",
                "    observation_hours: 48",
                "    prediction_hours: 24",
                "    gap_hours: 0",
            ]
        ),
        encoding="utf-8",
    )
    cli_main(["prepare-mimic-targets", "--config", str(config_path), "--targets", "hypoglycemia"])

    out = tmp_path / "prepared" / "obs48_target24_gap0" / "hypoglycemia"
    assert sorted(path.name for path in out.iterdir()) == ["dataset_metadata.json", "labels.csv", "records.csv"]
    metadata = json.loads((out / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["target"] == "hypoglycemia"


def test_prepare_mimic_targets_endpoint_accepts_cohort_level_override(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config_path = tmp_path / "mimic_targets.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"mimic_path: {zip_path.as_posix()}",
                f"output_dir: {(tmp_path / 'prepared').as_posix()}",
                f"cache_dir: {(tmp_path / 'cache').as_posix()}",
                "cohort_level: admission",
                "chunk_size: 4",
                "windows:",
                "  - name: obs48_target24_gap0",
                "    observation_hours: 48",
                "    prediction_hours: 24",
                "    gap_hours: 0",
            ]
        ),
        encoding="utf-8",
    )

    cli_main(["prepare-mimic-targets", "--config", str(config_path), "--cohort-level", "icu", "--targets", "hypotension"])

    metadata = json.loads((tmp_path / "prepared" / "obs48_target24_gap0" / "hypotension" / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["cohort_level"] == "icu"
    assert metadata["patient_id_source"] == "stay_id"


def test_mimic_legacy_imports_remain_available():
    from interpretable_ts_vit.datasets.mimic_targets import (
        MIMICIVMultiTargetAdapter as LegacyAdapter,
        MIMICTargetWindowConfig as LegacyWindowConfig,
        MIMICTargetsConfig as LegacyConfig,
    )

    assert LegacyAdapter is MIMICIVMultiTargetAdapter
    assert LegacyConfig is MIMICTargetsConfig
    assert LegacyWindowConfig is TargetWindowConfig


def test_mimic_config_uses_interface_target_window(tmp_path):
    config_path = tmp_path / "targets.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mimic_path: mimic-iv-3.1.zip",
                f"output_dir: {(tmp_path / 'prepared').as_posix()}",
                "windows:",
                "  - name: obs12_target6_gap2",
                "    observation_hours: 12",
                "    prediction_hours: 6",
                "    gap_hours: 2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_mimic_targets_config(config_path)

    assert isinstance(config.windows[0], TargetWindowConfig)
    assert config.windows[0].name == "obs12_target6_gap2"


def test_legacy_and_canonical_mimic_config_paths_are_present():
    assert Path("configs/mimic_targets.yaml").exists()
    assert Path("configs/datasets/mimic/targets.yaml").exists()


def test_icu_mode_uses_stay_ids_and_icu_relative_timestamps(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    dataset = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypotension"], cohort_level="icu")).prepare_all()[("wide", "hypotension")]

    assert dataset.metadata["cohort_level"] == "icu"
    assert dataset.metadata["patient_id"] == "stay_id"
    assert set(dataset.records["patient_id"]).issuperset({"100", "110", "120"})
    assert "2000-01-03 02:00:00" not in dataset.records["timestamp"].tolist()
    assert dataset.records.loc[dataset.records["patient_id"] == "100", "timestamp"].iloc[0] == "2000-01-01 01:00:00"


def test_hypotension_target_labels_icu_prediction_window(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    dataset = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypotension"], cohort_level="icu")).prepare_all()[("wide", "hypotension")]
    labels = _labels(dataset)

    assert labels["100"] == "true"
    assert labels["110"] == "false"
    assert labels["120"] == "false"


def test_icu_lab_targets_attach_admission_level_events_to_stays(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    dataset = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypoglycemia"], cohort_level="icu")).prepare_all()[("wide", "hypoglycemia")]
    labels = _labels(dataset)

    assert labels["200"] == "true"
    assert labels["210"] == "false"
    assert "stay_id_x" not in dataset.records.columns
    assert "stay_id_y" not in dataset.records.columns


def test_hypotension_requires_icu_cohort_level(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    with pytest.raises(ValueError, match="hypotension.*cohort_level='icu'"):
        MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypotension"], cohort_level="admission")).prepare_all()


def test_normal_ranges_include_mimic_target_variables():
    ranges = load_normal_ranges()
    expected = {
        "blood_glucose",
        "potassium",
        "creatinine",
        "wbc",
        "bun",
        "magnesium",
        "hemoglobin",
        "pco2",
        "platelet_count",
        "po2",
        "sodium",
        "heart_rate",
        "temperature",
        "systolic_bp",
        "diastolic_bp",
        "o2_saturation_pulseox",
        "previous_dextrose_treatment",
        "previous_insulin_treatment",
        "previous_potassium_chloride_treatment",
        "previous_dopamine_treatment",
        "previous_norepinephrine_treatment",
        "previous_fluids_given",
    }
    assert expected.issubset(set(ranges))
