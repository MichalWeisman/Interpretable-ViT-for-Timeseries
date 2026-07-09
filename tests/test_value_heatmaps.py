import numpy as np
import pandas as pd
import pytest

from interpretable_ts_vit import TimeSeriesBinner
from interpretable_ts_vit.autoencoder import cluster_explanation_value_autoencoder
from interpretable_ts_vit.config import ClusterConfig
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.pipeline import _cluster_title, _denormalized_patient_value_maps
from interpretable_ts_vit.visualization import (
    aggregate_cluster_value_matrices,
    cluster_assignment_counts,
    filter_value_matrix_by_explanation,
    load_normal_ranges,
    normal_range_status_matrix,
    patient_class_frame,
    plot_value_heatmap,
    patient_value_matrix,
    plot_patient_matrix_comparison,
    plot_patient_matrices,
    select_patient_ids,
    value_ranges_by_variable,
)
from interpretable_ts_vit.visualization import _relative_time_ticks


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

    plot_value_heatmap(matrix, binner.variable_vocab_, binner.time_bins_, tmp_path / "cluster_0.png", show_values=True)
    assert (tmp_path / "cluster_0.png").exists()


def test_cluster_heatmap_defaults_show_values_and_predicted_class_title():
    assert ClusterConfig().show_values is True
    assert _cluster_title(("true", 0), 7) == "Predicted class true: cluster_0 (n=7)"


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

    values = _denormalized_patient_value_maps(dataset, binner)
    clustered = cluster_explanation_value_autoencoder(
        explanations,
        values,
        validation_explanations=explanations,
        validation_values=values,
        cluster_explanations=explanations,
        cluster_values=values,
        predictions={"p1": "false", "p2": "true", "p3": "false", "p4": "true"},
        n_clusters=2,
        output_dir=tmp_path / "clusters",
        latent_dim=2,
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    assert {"patient_id", "predicted_label", "cluster", "distance_to_centroid", "is_centroid"}.issubset(clustered["assignments"].columns)
    assert (tmp_path / "clusters" / "cluster_metadata.json").exists()
    assert (tmp_path / "clusters" / "false" / "cluster_0.npy").exists()
    assert (tmp_path / "clusters" / "true" / "cluster_0.npy").exists()

    value_matrices = aggregate_cluster_value_matrices(dataset, clustered["assignments"], binner, output_dir=tmp_path / "values")
    counts = cluster_assignment_counts(clustered["assignments"])
    assert all(isinstance(key, tuple) for key in value_matrices)
    assert set(counts) == set(value_matrices)
    assert all(count >= 1 for count in counts.values())
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
            importance_matrix=np.load(tmp_path / "clusters" / predicted_label / f"cluster_{cluster}.npy"),
            importance_threshold=0.8,
        )
        plot_value_heatmap(
            value_matrix,
            binner.variable_vocab_,
            binner.time_bins_,
            class_dir / f"border_{cluster}.png",
            importance_matrix=np.load(tmp_path / "clusters" / predicted_label / f"cluster_{cluster}.npy"),
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


def test_value_heatmap_supports_per_variable_color_ranges(tmp_path):
    matrices = [
        np.array([[60.0, 80.0], [16.0, 20.0]]),
        np.array([[100.0, 120.0], [28.0, 32.0]]),
    ]
    vmin, vmax = value_ranges_by_variable(matrices)

    assert np.allclose(vmin, [60.0, 16.0])
    assert np.allclose(vmax, [120.0, 32.0])

    plot_value_heatmap(
        matrices[0],
        ["mean_bp", "respiratory_rate"],
        ["2026-01-01 00:00:00", "2026-01-01 01:00:00"],
        tmp_path / "per_variable.png",
        vmin=vmin,
        vmax=vmax,
        show_values=True,
    )

    assert (tmp_path / "per_variable.png").exists()


def test_value_heatmap_supports_normal_range_status_coloring(tmp_path):
    matrix = np.array([[55.0, 80.0, 120.0], [10.0, 16.0, 24.0]])
    variables = ["heart_rate", "respiratory_rate"]
    ranges = load_normal_ranges()
    status = normal_range_status_matrix(matrix, variables, ranges)

    assert np.allclose(status[0], [-1.0, 0.0, 1.0])
    assert np.allclose(status[1], [-1.0, 0.0, 1.0])

    plot_value_heatmap(
        matrix,
        variables,
        ["2026-01-01 00:00:00", "2026-01-01 01:00:00", "2026-01-01 02:00:00"],
        tmp_path / "normal_ranges.png",
        normal_ranges=ranges,
    )
    assert (tmp_path / "normal_ranges.png").exists()


def test_normal_range_status_coloring_uses_shades_within_each_range():
    matrix = np.array([[55.0, 58.0, 60.0, 70.0, 90.0, 100.0, 110.0, 120.0]])
    ranges = {"heart_rate": {"low": 60.0, "high": 100.0}}

    status = normal_range_status_matrix(matrix, ["heart_rate"], ranges)[0]

    assert np.isclose(status[0], -1.0)
    assert -1.0 < status[1] < 0.0
    assert status[2] < status[3] < status[4] < status[5]
    assert 0.0 < status[6] < 1.0
    assert np.isclose(status[7], 1.0)
    assert len(set(np.round(status, 3))) == len(status)


def test_relative_time_ticks_use_one_unit():
    positions, labels, granularity = _relative_time_ticks(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 00:30:00",
            "2026-01-01 01:00:00",
            "2026-01-01 01:30:00",
        ]
    )

    assert positions.tolist() == [0, 1, 2, 3]
    assert labels == ["0", "30min", "60min", "90min"]
    assert granularity == "30min"


def test_patient_matrix_plots_denormalize_values_and_write_both_heatmaps(tmp_path):
    records = pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "heart_rate", "value": 60.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p1", "variable": "heart_rate", "value": 80.0, "timestamp": "2026-01-01 01:00:00"},
            {"patient_id": "p1", "variable": "mean_bp", "value": 70.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "heart_rate", "value": 100.0, "timestamp": "2026-01-01 00:00:00"},
        ]
    )
    labels = pd.DataFrame(
        [
            {"patient_id": "p1", "label": "true"},
            {"patient_id": "p2", "label": "false"},
        ]
    )
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01 00:00:00", time_end="2026-01-01 02:00:00")
    binned = binner.fit_transform(records, labels)
    dataset = BinnedTimeSeriesDataset(binned.x, binned.y, binned.patient_ids)
    explanations = {"p1": np.array([[0.1, 0.2], [0.3, 0.4]])}

    matrix = patient_value_matrix(dataset, binner, "p1")
    hr_index = binner.variable_vocab_.index("heart_rate")
    bp_index = binner.variable_vocab_.index("mean_bp")
    assert np.isclose(matrix[hr_index, 0], 60.0)
    assert np.isclose(matrix[hr_index, 1], 80.0)
    assert np.isclose(matrix[bp_index, 0], 70.0)
    assert np.isnan(matrix[bp_index, 1])

    paths = plot_patient_matrices("p1", dataset, binner, explanations, tmp_path, show_values=True)

    assert paths["explanation"].exists()
    assert paths["values"].exists()
    filtered_paths = plot_patient_matrices(
        "p1",
        dataset,
        binner,
        explanations,
        tmp_path / "filtered",
        explanation_threshold=0.25,
        plot_explanation=False,
    )
    assert set(filtered_paths) == {"values"}
    assert filtered_paths["values"].exists()

    filtered = filter_value_matrix_by_explanation(matrix, explanations["p1"], 0.25)
    assert np.isnan(filtered[hr_index, 0])
    assert np.isnan(filtered[hr_index, 1])
    assert np.isclose(filtered[bp_index, 0], 70.0)
    assert np.isnan(filtered[bp_index, 1])


def test_patient_selection_and_comparison_plots_support_class_comparisons(tmp_path):
    records = pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "heart_rate", "value": 60.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "heart_rate", "value": 100.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p3", "variable": "heart_rate", "value": 65.0, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p4", "variable": "heart_rate", "value": 105.0, "timestamp": "2026-01-01 00:00:00"},
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
    predictions = pd.DataFrame(
        [
            {"patient_id": "p1", "predicted_label": "false"},
            {"patient_id": "p2", "predicted_label": "true"},
            {"patient_id": "p3", "predicted_label": "true"},
            {"patient_id": "p4", "predicted_label": "true"},
        ]
    )
    explanations = {
        "p1": np.array([[0.1]]),
        "p2": np.array([[0.8]]),
        "p3": np.array([[0.6]]),
        "p4": np.array([[0.9]]),
    }

    frame = patient_class_frame(dataset, binner, predictions)
    assert set(frame.columns).issuperset({"patient_id", "true_label", "predicted_label"})
    assert select_patient_ids(dataset, binner, predictions=predictions, predicted_label="true", n=2) == ["p2", "p3"]
    assert select_patient_ids(dataset, binner, true_label="false") == ["p1", "p3"]

    selected = ["p1", "p2", "p3"]
    paths = plot_patient_matrix_comparison(selected, dataset, binner, explanations, tmp_path / "compare", explanation_threshold=0.5)

    assert set(paths) == set(selected)
    for patient_paths in paths.values():
        assert set(patient_paths) == {"values"}
        assert patient_paths["values"].exists()
