import numpy as np
import pandas as pd
import pytest

from interpretable_ts_vit import TimeSeriesBinner
from interpretable_ts_vit.clustering import cluster_explanations
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.visualization import aggregate_cluster_value_matrices, plot_value_heatmap


def test_cluster_value_aggregation_uses_observed_raw_values(tmp_path):
    records = pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "heart_rate", "value": 60.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p1", "variable": "heart_rate", "value": 80.0, "timestamp": "2026-01-01 01:00:00"},
            {"patient_id": "p2", "variable": "heart_rate", "value": 100.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "mean_bp", "value": 70.0, "timestamp": "2026-01-01 01:00:00"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"patient_id": "p1", "label": "false"},
            {"patient_id": "p2", "label": "true"},
        ]
    )
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01 00:00:00", time_end="2026-01-01 02:00:00")
    binned = binner.fit_transform(records, labels)
    dataset = BinnedTimeSeriesDataset(binned.x, binned.y, binned.patient_ids)
    assignments = pd.DataFrame({"patient_id": ["p1", "p2"], "cluster": [0, 0]})

    aggregates = aggregate_cluster_value_matrices(dataset, assignments, binner, output_dir=tmp_path / "values")
    matrix = aggregates[0]
    hr_index = binner.variable_vocab_.index("heart_rate")
    bp_index = binner.variable_vocab_.index("mean_bp")

    assert np.isclose(matrix[hr_index, 0], 80.0)
    assert np.isclose(matrix[hr_index, 1], 80.0)
    assert np.isnan(matrix[bp_index, 0])
    assert np.isclose(matrix[bp_index, 1], 70.0)
    assert (tmp_path / "values" / "cluster_0.npy").exists()

    plot_value_heatmap(matrix, binner.variable_vocab_, binner.time_bins_, tmp_path / "cluster_0.png")
    assert (tmp_path / "cluster_0.png").exists()


def test_importance_clustering_and_value_importance_opacity(tmp_path):
    records = pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "heart_rate", "value": 60.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p1", "variable": "mean_bp", "value": 80.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "heart_rate", "value": 100.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "mean_bp", "value": 50.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p3", "variable": "heart_rate", "value": 62.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p3", "variable": "mean_bp", "value": 79.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p4", "variable": "heart_rate", "value": 98.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p4", "variable": "mean_bp", "value": 51.0, "timestamp": "2026-01-01 00:00:00"},
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
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01 00:00:00", time_end="2026-01-01 01:00:00")
    binned = binner.fit_transform(records, labels)
    dataset = BinnedTimeSeriesDataset(binned.x, binned.y, binned.patient_ids)
    explanations = {
        "p1": np.array([[0.1], [0.9]]),
        "p2": np.array([[0.9], [0.1]]),
        "p3": np.array([[0.2], [0.8]]),
        "p4": np.array([[0.8], [0.2]]),
    }

    clustered = cluster_explanations(
        explanations,
        predictions={"p1": "false", "p2": "true", "p3": "false", "p4": "true"},
        n_clusters=2,
        output_dir=tmp_path / "clusters",
    )
    assert set(clustered["assignments"].columns) == {"patient_id", "predicted_label", "cluster"}
    assert (tmp_path / "clusters" / "cluster_metadata.json").exists()
    assert (tmp_path / "clusters" / "false" / "cluster_0.npy").exists()
    assert (tmp_path / "clusters" / "true" / "cluster_0.npy").exists()

    value_matrices = aggregate_cluster_value_matrices(dataset, clustered["assignments"], binner, output_dir=tmp_path / "values")
    assert all(isinstance(key, tuple) for key in value_matrices)
    assert (tmp_path / "values" / "false" / "cluster_0.npy").exists()
    assert (tmp_path / "values" / "true" / "cluster_0.npy").exists()
    for (predicted_label, cluster), value_matrix in value_matrices.items():
        class_dir = tmp_path / predicted_label
        class_dir.mkdir(exist_ok=True)
        plot_value_heatmap(
            value_matrix,
            binner.variable_vocab_,
            binner.time_bins_,
            class_dir / f"overlay_{cluster}.png",
            importance_matrix=clustered["aggregates"][predicted_label][cluster],
            importance_threshold=0.8,
        )
        plot_value_heatmap(
            value_matrix,
            binner.variable_vocab_,
            binner.time_bins_,
            class_dir / f"border_{cluster}.png",
            importance_matrix=clustered["aggregates"][predicted_label][cluster],
            importance_style="border",
            importance_threshold=0.8,
        )
    assert any(tmp_path.rglob("overlay_*.png"))
    assert any(tmp_path.rglob("border_*.png"))


def test_importance_threshold_must_be_quantile(tmp_path):
    matrix = np.array([[1.0, 2.0]])
    importance = np.array([[0.2, 0.8]])

    with pytest.raises(ValueError, match="importance_threshold"):
        plot_value_heatmap(
            matrix,
            ["heart_rate"],
            ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
            tmp_path / "bad_threshold.png",
            importance_matrix=importance,
            importance_threshold=1.5,
        )
