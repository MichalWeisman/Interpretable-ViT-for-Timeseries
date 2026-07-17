import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from interpretable_ts_vit import TimeSeriesBinner
from interpretable_ts_vit.cli import cmd_report
from interpretable_ts_vit.io import save_metadata, save_split
from interpretable_ts_vit.reporting import (
    ExperimentReportSpec,
    build_experiment_report,
    class_pattern_payloads,
    discover_experiment_specs,
    infer_variable_display_metadata,
    order_variables,
    top_importance_sparse_values,
    write_mimic_variable_display_metadata,
)


def _make_synthetic_run(tmp_path, use_base_as_run_dir=False):
    records = pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "heart_rate", "value": 60.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p1", "variable": "blood_glucose", "value": 80.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p1", "variable": "dextrose_hypertonic", "value": 1.0, "timestamp": "2026-01-01 01:00:00"},
            {"patient_id": "p2", "variable": "heart_rate", "value": 100.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "blood_glucose", "value": 55.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "dextrose_hypertonic", "value": 1.0, "timestamp": "2026-01-01 01:00:00"},
            {"patient_id": "p3", "variable": "heart_rate", "value": 70.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p3", "variable": "blood_glucose", "value": 95.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p4", "variable": "heart_rate", "value": 105.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p4", "variable": "blood_glucose", "value": 50.0, "timestamp": "2026-01-01 00:00:00"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"patient_id": "p1", "label": "false"},
            {"patient_id": "p2", "label": "true"},
            {"patient_id": "p3", "label": "false"},
            {"patient_id": "p4", "label": "true"},
        ]
    )
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01 00:00:00", time_end="2026-01-01 02:00:00")
    binned = binner.fit_transform(records, labels)
    run_dir = tmp_path if use_base_as_run_dir else tmp_path / "run"
    dataset_dir = tmp_path / "dataset"
    run_dir.mkdir(parents=True)
    dataset_dir.mkdir()
    save_split(run_dir / "test.npz", binned.patient_ids, np.asarray(binned.x), np.asarray(binned.y))
    save_metadata(run_dir, binner)
    (run_dir / "explanations" / "test").mkdir(parents=True)
    explanations = {
        "p1": np.array([[0.1, 0.2], [0.2, 0.9], [0.4, 0.1]]),
        "p2": np.array([[0.7, 0.1], [0.8, 0.2], [0.2, 0.1]]),
        "p3": np.array([[0.2, 0.3], [0.1, 0.7], [0.1, 0.1]]),
        "p4": np.array([[0.9, 0.2], [0.6, 0.1], [0.3, 0.1]]),
    }
    for patient_id, matrix in explanations.items():
        np.save(run_dir / "explanations" / "test" / f"{patient_id}.npy", matrix)
    clusters = run_dir / "clusters" / "test"
    clusters.mkdir(parents=True)
    pd.DataFrame(
        [
            {"patient_id": "p1", "predicted_label": "false", "cluster": 0},
            {"patient_id": "p2", "predicted_label": "true", "cluster": 0},
            {"patient_id": "p3", "predicted_label": "false", "cluster": 0},
            {"patient_id": "p4", "predicted_label": "true", "cluster": 0},
        ]
    ).to_csv(clusters / "cluster_assignments.csv", index=False)
    (clusters / "cluster_metadata.json").write_text(
        json.dumps({"feature_mode": "autoencoder", "clustering_method": "kmeans", "n_clusters_used": {"false": 1, "true": 1}}),
        encoding="utf-8",
    )
    (run_dir / "test_evaluation_metrics.json").write_text(
        json.dumps({"accuracy": 0.75, "macro_f1": 0.7, "auroc": 0.8, "tpr": 0.6, "fpr": 0.2, "tnr": 0.8, "ppv": 0.75, "confusion_matrix": [[1, 1], [0, 2]]}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"best_epoch": 2, "epochs_ran": 3}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "pattern": "true",
                "more_similar_class": "true",
                "n_same_class": 2,
                "n_other_class": 2,
                "mean_same_class": 0.9,
                "mean_other_class": 0.2,
                "mean_difference": 0.7,
                "p_value": 0.01,
                "is_significant": True,
            }
        ]
    ).to_csv(run_dir / "test_class_similarity_tests.csv", index=False)
    pd.DataFrame(
        [
            {"patient_id": "p1", "similarity_to_true": 0.2, "similarity_to_false": 0.8},
            {"patient_id": "p2", "similarity_to_true": 0.9, "similarity_to_false": 0.1},
        ]
    ).to_csv(run_dir / "test_pattern_similarity.csv", index=False)
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "target": "synthetic_target",
                "window": {"name": "obs2_target1_gap0", "observation_hours": 2, "prediction_hours": 1, "gap_hours": 0},
                "target_metadata": {"target_definition": "Synthetic event in prediction window"},
                "variable_mappings": {
                    "chart_itemids": {"heart_rate": [220045]},
                    "lab_itemids": {"blood_glucose": [50809]},
                    "inputevent_itemids": {"dextrose_hypertonic": [221014]},
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir, dataset_dir, binner, binned


def test_top_importance_sparse_values_keeps_only_top_cells():
    values = np.array([[1.0, 2.0], [3.0, np.nan]])
    importance = np.array([[0.1, 0.9], [0.8, 0.7]])

    sparse, mask = top_importance_sparse_values(values, importance, 0.5)

    assert np.isnan(sparse[0, 0])
    assert np.isclose(sparse[0, 1], 2.0)
    assert np.isclose(sparse[1, 0], 3.0)
    assert np.isnan(sparse[1, 1])
    assert mask.tolist() == [[False, True], [True, False]]


def test_variable_grouping_orders_metadata_sources():
    metadata = infer_variable_display_metadata(
        {
            "variable_mappings": {
                "chart_itemids": {"heart_rate": [{"id": 1, "name": "Heart Rate"}]},
                "lab_itemids": {"blood_glucose": [2]},
                "inputevent_itemids": {"dextrose_hypertonic": [3]},
            }
        },
        ["dextrose_hypertonic", "blood_glucose", "heart_rate", "unknown_signal"],
    )

    assert order_variables(["dextrose_hypertonic", "blood_glucose", "heart_rate", "unknown_signal"], metadata) == [
        "heart_rate",
        "blood_glucose",
        "dextrose_hypertonic",
        "unknown_signal",
    ]
    assert metadata["heart_rate"]["items"][0] == {"id": "1", "name": "Heart Rate"}
    assert metadata["dextrose_hypertonic"]["items"][0]["name"] == "name unavailable"
    assert metadata["dextrose_hypertonic"]["items"][0]["missing_name"] is True


def test_class_pattern_payloads_use_predicted_classes_and_grouped_variables(tmp_path):
    run_dir, dataset_dir, binner, binned = _make_synthetic_run(tmp_path)
    from interpretable_ts_vit.io import load_split
    from interpretable_ts_vit.reporting import load_variable_display_metadata

    dataset = load_split(run_dir / "test.npz")
    assignments = pd.read_csv(run_dir / "clusters" / "test" / "cluster_assignments.csv")
    dataset_metadata = json.loads((dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    display_metadata = load_variable_display_metadata(dataset_dir, dataset_metadata, binner.variable_vocab_)
    ordered = order_variables(binner.variable_vocab_, display_metadata)

    patterns = class_pattern_payloads(
        dataset,
        binner,
        assignments,
        run_dir / "explanations" / "test",
        ordered,
        display_metadata,
        {},
        top_importance_fraction=0.5,
    )

    assert [pattern["class_label"] for pattern in patterns] == ["False", "True"]
    assert patterns[0]["variables"][0]["group"] == "measurements"
    assert patterns[0]["variables"][1]["group"] == "lab_tests"
    assert patterns[0]["variables"][2]["group"] == "inputs"
    assert any(cell is not None for row in patterns[0]["values"] for cell in row)
    assert "range_status" in patterns[0]


def test_build_experiment_report_writes_embedded_html(tmp_path):
    run_dir, dataset_dir, _, _ = _make_synthetic_run(tmp_path)
    out = tmp_path / "report.html"

    payload = build_experiment_report([ExperimentReportSpec(run_dir, dataset_dir, "Synthetic")], out)

    text = out.read_text(encoding="utf-8")
    assert payload["experiments"][0]["name"] == "Synthetic"
    assert "Experiment Results" in text
    assert "Compare" in text
    assert "experimentSelect" in text
    assert "Configuration" in text
    assert "Test Results" in text
    assert "Pattern similarity statistical test" in text
    assert "Target definition" in text
    assert "Most similar class" in text
    assert "Clinical range status" in text
    assert "variable-description-grid" in text
    assert "Class Pattern Visualization" in text
    assert "report-data" in text
    assert "MIMIC" not in text
    assert payload["experiments"][0]["statistics"]["class_similarity_tests"][0]["is_significant"] is True
    assert payload["experiments"][0]["class_patterns"][0]["range_status"]


def test_mimic_dictionary_metadata_writes_item_names_and_missing_markers(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "variable_mappings": {
                    "chart_itemids": {"heart_rate": [220045, 999999]},
                    "lab_itemids": {"blood_glucose": [50809]},
                    "inputevent_itemids": {"dextrose_hypertonic": [221014]},
                }
            }
        ),
        encoding="utf-8",
    )
    mimic_root = tmp_path / "mimic"
    (mimic_root / "hosp").mkdir(parents=True)
    (mimic_root / "icu").mkdir(parents=True)
    pd.DataFrame([{"itemid": 50809, "label": "Glucose"}]).to_csv(mimic_root / "hosp" / "d_labitems.csv.gz", index=False)
    pd.DataFrame(
        [
            {"itemid": 220045, "label": "Heart Rate"},
            {"itemid": 221014, "label": "Dextrose 50%"},
        ]
    ).to_csv(mimic_root / "icu" / "d_items.csv.gz", index=False)

    output = write_mimic_variable_display_metadata(dataset_dir, mimic_root)

    written = json.loads(Path(output).read_text(encoding="utf-8"))["variables"]
    assert written["heart_rate"]["items"][0]["name"] == "Heart Rate"
    assert written["blood_glucose"]["items"][0]["name"] == "Glucose"
    assert written["dextrose_hypertonic"]["items"][0]["name"] == "Dextrose 50%"
    assert written["heart_rate"]["items"][1]["name"] == "name unavailable"
    assert written["heart_rate"]["items"][1]["missing_name"] is True


def test_cmd_report_validates_dataset_dir_count(tmp_path):
    run_dir, _, _, _ = _make_synthetic_run(tmp_path)

    class Args:
        run = [str(run_dir), str(run_dir)]
        runs_root = None
        dataset_root = None
        dataset_dir = [str(tmp_path / "dataset")]
        name = None
        out = str(tmp_path / "report.html")
        split = "test"
        top_importance_fraction = 0.1
        mimic_path = None

    with pytest.raises(SystemExit, match="--dataset-dir"):
        cmd_report(Args())


def test_discover_experiment_specs_matches_dataset_root_by_relative_path(tmp_path):
    runs_root = tmp_path / "runs_root"
    dataset_root = tmp_path / "dataset_root"
    run_dir, dataset_dir, _, _ = _make_synthetic_run(runs_root / "obs24_target8_gap0" / "hypotension", use_base_as_run_dir=True)
    matched_dataset_dir = dataset_root / "obs24_target8_gap0" / "hypotension"
    matched_dataset_dir.mkdir(parents=True)
    matched_dataset_dir.joinpath("dataset_metadata.json").write_text(
        dataset_dir.joinpath("dataset_metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    specs = discover_experiment_specs(runs_root, dataset_root=dataset_root)

    assert len(specs) == 1
    assert specs[0].run_dir == run_dir
    assert specs[0].dataset_dir == matched_dataset_dir
    assert specs[0].name == "obs24_target8_gap0/hypotension"


def test_cmd_report_supports_runs_root(tmp_path):
    runs_root = tmp_path / "runs_root"
    dataset_root = tmp_path / "dataset_root"
    _run_dir, dataset_dir, _, _ = _make_synthetic_run(runs_root / "obs24_target8_gap0" / "hypotension", use_base_as_run_dir=True)
    matched_dataset_dir = dataset_root / "obs24_target8_gap0" / "hypotension"
    matched_dataset_dir.mkdir(parents=True)
    matched_dataset_dir.joinpath("dataset_metadata.json").write_text(
        dataset_dir.joinpath("dataset_metadata.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    Args = type(
        "Args",
        (),
        {
            "run": None,
            "runs_root": str(runs_root),
            "dataset_root": str(dataset_root),
            "dataset_dir": None,
            "name": None,
            "out": str(tmp_path / "root_report.html"),
            "split": "test",
            "top_importance_fraction": 0.1,
            "mimic_path": None,
        },
    )

    cmd_report(Args())

    text = (tmp_path / "root_report.html").read_text(encoding="utf-8")
    assert "obs24_target8_gap0/hypotension" in text
