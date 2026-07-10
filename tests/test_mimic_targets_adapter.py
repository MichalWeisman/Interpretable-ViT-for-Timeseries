import gzip
import json
from zipfile import ZipFile

import pandas as pd

from interpretable_ts_vit.datasets import (
    MIMICIVMultiTargetAdapter,
    MIMICTargetsConfig,
    MIMICTargetWindowConfig,
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
            [300001, "Dextrose 10%", "D10", "Fluids/Intake"],
        ],
        columns=["itemid", "label", "abbreviation", "category"],
    )
    chart = pd.DataFrame(
        [
            [1, 10, 1, "2100-01-01 01:00:00", 220045, 80.0],
            [1, 10, 1, "2100-01-01 02:00:00", 223761, 98.6],
            [4, 40, 1, "2100-01-01 10:00:00", 223762, 37.0],
            [4, 40, 1, "2100-01-03 02:00:00", 223762, 38.1],
            [4, 40, 1, "2100-01-03 03:00:00", 223762, 38.0],
            [4, 41, 1, "2100-01-03 02:00:00", 223762, 38.2],
            [4, 41, 1, "2100-01-03 03:00:00", 223762, 38.3],
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
        [[1, 10, 1, "2100-01-01 05:00:00", "2100-01-01 06:00:00", 300001, 100.0]],
        columns=["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount"],
    )
    with ZipFile(zip_path, "w") as zf:
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/admissions.csv.gz", admissions)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/d_labitems.csv.gz", d_labitems)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/labevents.csv.gz", labevents)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/microbiologyevents.csv.gz", micro)
        _write_gz_csv(zf, "mimic-iv-3.1/hosp/prescriptions.csv.gz", prescriptions)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/d_items.csv.gz", d_items)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/chartevents.csv.gz", chart)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/inputevents.csv.gz", inputevents)
    return zip_path


def _config(zip_path, tmp_path, *, windows=None, targets=None):
    return MIMICTargetsConfig(
        mimic_path=zip_path,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        windows=windows or [MIMICTargetWindowConfig(name="wide", observation_hours=48, prediction_hours=96, gap_hours=0)],
        targets=targets or [
            "cardiovascular_event",
            "nosocomial_infection",
            "hypoglycemia",
            "prolonged_hyperglycemia",
            "in_hospital_mortality",
        ],
        chunk_size=4,
    )


def _labels(prepared):
    return prepared.labels.set_index("patient_id")["label"].to_dict()


def test_multi_target_adapter_builds_each_target_and_excludes_prior_positives(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    datasets = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path)).prepare_all()

    assert _labels(datasets[("wide", "cardiovascular_event")])["10"] == "true"
    assert _labels(datasets[("wide", "cardiovascular_event")])["11"] == "false"
    assert "12" not in _labels(datasets[("wide", "cardiovascular_event")])
    assert _labels(datasets[("wide", "hypoglycemia")])["20"] == "true"
    assert _labels(datasets[("wide", "hypoglycemia")])["21"] == "false"
    assert "22" not in _labels(datasets[("wide", "hypoglycemia")])
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
        MIMICTargetWindowConfig(name="gap0", observation_hours=48, prediction_hours=24, gap_hours=0),
        MIMICTargetWindowConfig(name="gap8", observation_hours=48, prediction_hours=24, gap_hours=8),
    ]
    datasets = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, windows=windows, targets=["hypoglycemia"])).prepare_all()

    assert _labels(datasets[("gap0", "hypoglycemia")])["20"] == "true"
    assert "20" not in _labels(datasets[("gap8", "hypoglycemia")])


def test_mimic_target_records_standardize_units_and_event_variables(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    dataset = MIMICIVMultiTargetAdapter(_config(zip_path, tmp_path, targets=["hypoglycemia"])).prepare_all()[("wide", "hypoglycemia")]
    records = dataset.records

    assert round(records.loc[records["variable"] == "temperature", "value"].iloc[0], 1) == 37.0
    assert {"blood_culture", "urine_culture", "antibiotics", "metformin", "dextrose_10"}.issubset(set(records["variable"]))
    assert set(records.loc[records["variable"].isin(["blood_culture", "urine_culture", "antibiotics"]), "value"]) == {1.0}
    assert dataset.metadata["patient_id"] == "hadm_id"
    assert dataset.metadata["window"]["name"] == "wide"


def test_prepare_mimic_targets_endpoint_outputs_only_pre_tensor_files(tmp_path):
    zip_path = _mini_mimic_zip(tmp_path)
    config_path = tmp_path / "mimic_targets.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"mimic_path: {zip_path.as_posix()}",
                f"output_dir: {(tmp_path / 'prepared').as_posix()}",
                f"cache_dir: {(tmp_path / 'cache').as_posix()}",
                "chunk_size: 4",
                "windows:",
                "  - name: obs48_target24_gap0",
                "    observation_hours: 48",
                "    prediction_hours: 24",
                "    gap_hours: 0",
                "targets:",
                "  - hypoglycemia",
            ]
        ),
        encoding="utf-8",
    )
    cli_main(["prepare-mimic-targets", "--config", str(config_path)])

    out = tmp_path / "prepared" / "obs48_target24_gap0" / "hypoglycemia"
    assert sorted(path.name for path in out.iterdir()) == ["dataset_metadata.json", "labels.csv", "records.csv"]
    metadata = json.loads((out / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["target"] == "hypoglycemia"


def test_normal_ranges_include_mimic_target_variables():
    ranges = load_normal_ranges()
    expected = {
        "blood_glucose",
        "creatinine",
        "albumin",
        "wbc",
        "lactate",
        "ldh",
        "neutrophils",
        "troponin_i",
        "blood_ph",
        "bicarbonate",
        "blood_culture",
        "urine_culture",
        "heart_rate",
        "temperature",
        "antibiotics",
        "dextrose_5",
        "dextrose_10",
        "insulin_rapid_acting",
        "insulin_short_acting",
        "insulin_intermediate_acting",
        "insulin_long_acting",
        "insulin_ultra_long_acting",
        "insulin_premixed",
        "metformin",
        "meglitinides",
        "glp1_agonists",
        "sulfonylurea",
        "dpp4_inhibitors_with_metformin",
        "dpp4_inhibitors",
        "alpha_glucosidase_inhibitors",
        "furosemide",
    }
    assert expected.issubset(set(ranges))
