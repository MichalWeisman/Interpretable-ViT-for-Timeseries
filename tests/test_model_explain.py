import numpy as np
import pytest
import torch

from interpretable_ts_vit import ViTConfig, ViTTimeSeriesClassifier, cluster_embeddings, cluster_explanations, explain_model
from interpretable_ts_vit.config import TrainConfig
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import extract_model_embeddings, train_model


def test_model_forward_shape_with_padding():
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=3, num_timesteps=5, num_classes=2, patch_size=(2, 4), embed_dim=16, depth=1, num_heads=2)
    )
    logits = model(torch.randn(4, 2, 3, 5))
    assert logits.shape == (4, 2)


def test_training_and_explanation_shape():
    x = np.random.default_rng(0).normal(size=(8, 2, 3, 5)).astype("float32")
    x[:, 1] = (x[:, 1] > 0).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ds = BinnedTimeSeriesDataset(x, y, [f"p{i}" for i in range(8)])
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=3, num_timesteps=5, num_classes=2, patch_size=(1, 2), embed_dim=16, depth=1, num_heads=2)
    )
    metrics = train_model(model, ds, ds, config=type("Cfg", (), {"device": "cpu", "batch_size": 4, "epochs": 1, "learning_rate": 1e-3, "weight_decay": 0.0})())
    assert "accuracy" in metrics
    explanations = explain_model(model, ds, target_class=1, device="cpu")
    assert explanations["p0"].shape == (3, 5)


def test_explanation_supports_progress_toggle():
    x = np.random.default_rng(1).normal(size=(4, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1]), [f"p{i}" for i in range(4)])
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    explanations = explain_model(model, ds, target_class=1, device="cpu", show_progress=False)
    assert set(explanations) == {"p0", "p1", "p2", "p3"}


def test_transformer_attribution_method():
    x = np.random.default_rng(2).normal(size=(4, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1]), [f"p{i}" for i in range(4)])
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    explanations = explain_model(model, ds, method="transformer_attribution", target_class=1, device="cpu", show_progress=False)
    assert explanations["p0"].shape == (2, 3)


def test_integrated_gradients_missing_dependency_message():
    x = np.zeros((1, 2, 2, 2), dtype="float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0]), ["p0"])
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=2, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    try:
        import captum  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="captum"):
            explain_model(model, ds, method="integrated_gradients", device="cpu")


def test_training_early_stopping_restores_best_epoch():
    x = np.random.default_rng(1).normal(size=(8, 2, 3, 5)).astype("float32")
    x[:, 1] = (x[:, 1] > 0).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ds = BinnedTimeSeriesDataset(x, y, [f"p{i}" for i in range(8)])
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=3, num_timesteps=5, num_classes=2, patch_size=(1, 2), embed_dim=16, depth=1, num_heads=2)
    )

    metrics = train_model(
        model,
        ds,
        ds,
        config=TrainConfig(
            device="cpu",
            batch_size=4,
            epochs=5,
            learning_rate=0.0,
            weight_decay=0.0,
            early_stopping_patience=1,
            early_stopping_monitor="val_loss",
        ),
    )

    assert metrics["stopped_early"] is True
    assert metrics["epochs_ran"] == 2
    assert metrics["best_epoch"] == 1
    assert "val_loss" in metrics["history"][0]


def test_cluster_explanations():
    explanations = {f"p{i}": np.ones((2, 3)) * i for i in range(4)}
    clustered = cluster_explanations(explanations, n_clusters=2)
    assert set(clustered.keys()) == {"assignments", "aggregates"}
    assert len(clustered["assignments"]) == 4


def test_extract_and_cluster_vit_embeddings(tmp_path):
    x = np.random.default_rng(3).normal(size=(6, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1, 0, 1]), [f"p{i}" for i in range(6)])
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, patch_size=(1, 1), embed_dim=8, depth=1, num_heads=2)
    )

    patient_ids, embeddings = extract_model_embeddings(model, ds, config=type("Cfg", (), {"device": "cpu", "batch_size": 3})())
    assert patient_ids == ds.patient_ids
    assert embeddings.shape == (6, 8)

    clustered = cluster_embeddings(
        embeddings,
        patient_ids=patient_ids,
        predictions={patient_id: "true" if idx % 2 else "false" for idx, patient_id in enumerate(patient_ids)},
        n_clusters=2,
        output_dir=tmp_path / "embedding_clusters",
    )
    assignments = clustered["assignments"]
    assert {"patient_id", "predicted_label", "cluster", "distance_to_centroid", "is_centroid"}.issubset(assignments.columns)
    assert any(assignments["is_centroid"])
    assert (tmp_path / "embedding_clusters" / "cluster_centroids.csv").exists()


def test_cluster_explanations_uses_predicted_classes():
    explanations = {"p0": np.ones((2, 3)), "p1": np.ones((2, 3)) * 2}
    clustered = cluster_explanations(explanations, predictions={"p0": "0", "p1": "1"}, n_clusters=1)
    assert set(clustered["assignments"]["predicted_label"].tolist()) == {"0", "1"}


def test_cluster_explanations_can_combine_explanations_and_values():
    explanations = {f"p{i}": np.ones((2, 3)) for i in range(4)}
    values = {
        "p0": np.array([[0.0, 0.0], [0.0, np.nan]]),
        "p1": np.array([[0.1, 0.0], [0.0, np.nan]]),
        "p2": np.array([[10.0, 10.0], [10.0, np.nan]]),
        "p3": np.array([[10.1, 10.0], [10.0, np.nan]]),
    }

    clustered = cluster_explanations(
        explanations,
        values=values,
        feature_mode="combined",
        explanation_weight=1.0,
        value_weight=1.0,
        n_clusters=2,
    )

    assert len(clustered["assignments"]) == 4
    assert set(clustered["assignments"]["cluster"]) == {0, 1}


def test_value_feature_mode_requires_values():
    explanations = {f"p{i}": np.ones((2, 3)) for i in range(2)}
    with pytest.raises(ValueError, match="values are required"):
        cluster_explanations(explanations, feature_mode="combined")


def test_cluster_explanations_supports_hdbscan(monkeypatch):
    import interpretable_ts_vit.clustering as clustering

    calls = {}

    class FakeHDBSCAN:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def fit_predict(self, _x):
            return np.array([0, 0, -1, 1, 1])

    monkeypatch.setattr(clustering, "HDBSCAN", FakeHDBSCAN)
    explanations = {f"p{i}": np.array([float(i), float(i + 1)]) for i in range(5)}

    clustered = cluster_explanations(
        explanations,
        method="hdbscan",
        hdbscan_min_cluster_size=2,
        hdbscan_min_samples=1,
    )

    assert calls == {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"}
    assert clustered["assignments"]["cluster"].tolist() == [0, 0, -1, 1, 1]
    assert set(clustered["aggregates"]) == {-1, 0, 1}


def test_hdbscan_marks_too_small_groups_as_noise(monkeypatch):
    import interpretable_ts_vit.clustering as clustering

    monkeypatch.setattr(clustering, "HDBSCAN", object)
    explanations = {f"p{i}": np.array([float(i), float(i + 1)]) for i in range(3)}

    clustered = cluster_explanations(explanations, method="hdbscan", hdbscan_min_cluster_size=5)

    assert clustered["assignments"]["cluster"].tolist() == [-1, -1, -1]
    assert set(clustered["aggregates"]) == {-1}
