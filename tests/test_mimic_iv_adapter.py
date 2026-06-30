import gzip
from zipfile import ZipFile

import pandas as pd

from interpretable_ts_vit.datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig


def _write_gz_csv(zf, name, frame):
    zf.writestr(name, gzip.compress(frame.to_csv(index=False).encode("utf-8")))


def test_mimic_hypotension_adapter_builds_generic_records_and_labels(tmp_path):
    zip_path = tmp_path / "mimic-mini.zip"
    stays = pd.DataFrame(
        [
            [1, 10, 100, "2100-01-01 00:00:00", "2100-01-02 12:00:00", 1.5],
            [2, 20, 200, "2100-02-01 00:00:00", "2100-02-02 12:00:00", 1.5],
        ],
        columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    items = pd.DataFrame(
        [
            [220045, "Heart Rate", "HR", "chartevents", "Routine Vital Signs", "bpm", "Numeric", None, None],
            [220052, "Arterial Blood Pressure mean", "ABPm", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", None, None],
        ],
        columns=["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"],
    )
    chart = pd.DataFrame(
        [
            [1, 10, 100, 1, "2100-01-01 01:00:00", "2100-01-01 01:05:00", 220045, "80", 80.0, "bpm", 0],
            [1, 10, 100, 1, "2100-01-02 01:00:00", "2100-01-02 01:05:00", 220052, "60", 60.0, "mmHg", 0],
            [2, 20, 200, 1, "2100-02-01 01:00:00", "2100-02-01 01:05:00", 220045, "70", 70.0, "bpm", 0],
            [2, 20, 200, 1, "2100-02-02 01:00:00", "2100-02-02 01:05:00", 220052, "72", 72.0, "mmHg", 0],
        ],
        columns=["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"],
    )
    with ZipFile(zip_path, "w") as zf:
        _write_gz_csv(zf, "mimic-iv-3.1/icu/icustays.csv.gz", stays)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/d_items.csv.gz", items)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/chartevents.csv.gz", chart)

    config = MIMICHypotensionConfig(
        mimic_path=zip_path,
        observation_hours=24,
        prediction_hours=6,
        variable_itemids={"heart_rate": [220045]},
        outcome_itemids=[220052],
        chunk_size=2,
    )
    prepared = MIMICIVHypotensionAdapter(config).prepare()

    assert list(prepared.records.columns) == ["patient_id", "variable", "value", "timestamp"]
    assert set(prepared.records["patient_id"]) == {"100", "200"}
    assert prepared.records["timestamp"].tolist() == ["2000-01-01 01:00:00", "2000-01-01 01:00:00"]
    assert prepared.labels.sort_values("patient_id").to_dict("records") == [
        {"patient_id": "100", "label": "true"},
        {"patient_id": "200", "label": "false"},
    ]
    assert prepared.metadata["n_labeled_stays"] == 2


def test_prepared_dataset_save_writes_expected_files(tmp_path):
    zip_path = tmp_path / "mimic-mini.zip"
    stays = pd.DataFrame(
        [[1, 10, 100, "2100-01-01 00:00:00", "2100-01-02 12:00:00", 1.5]],
        columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    chart = pd.DataFrame(
        [
            [1, 10, 100, 1, "2100-01-01 01:00:00", "2100-01-01 01:05:00", 220045, "80", 80.0, "bpm", 0],
            [1, 10, 100, 1, "2100-01-02 01:00:00", "2100-01-02 01:05:00", 220052, "60", 60.0, "mmHg", 0],
        ],
        columns=["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"],
    )
    with ZipFile(zip_path, "w") as zf:
        _write_gz_csv(zf, "mimic-iv-3.1/icu/icustays.csv.gz", stays)
        _write_gz_csv(zf, "mimic-iv-3.1/icu/chartevents.csv.gz", chart)
    config = MIMICHypotensionConfig(
        mimic_path=zip_path,
        variable_itemids={"heart_rate": [220045]},
        outcome_itemids=[220052],
    )
    prepared = MIMICIVHypotensionAdapter(config).prepare()
    prepared.save(tmp_path / "prepared")

    assert (tmp_path / "prepared" / "records.csv").exists()
    assert (tmp_path / "prepared" / "labels.csv").exists()
    assert (tmp_path / "prepared" / "dataset_metadata.json").exists()
