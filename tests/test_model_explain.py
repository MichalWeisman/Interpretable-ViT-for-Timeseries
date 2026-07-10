import numpy as np
import torch

from interpretable_ts_vit import ViTConfig, ViTTimeSeriesClassifier, explain_model
from interpretable_ts_vit.autoencoder import cluster_explanation_value_autoencoder
from interpretable_ts_vit.config import TrainConfig
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import evaluate_model, train_model


class ScoreModel(torch.nn.Module):
    def forward(self, x):
        scores = x[:, 0, 0, 0]
        return torch.stack([-scores, scores], dim=1)


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


def test_evaluation_reports_binary_confusion_rates_and_auc():
    x = np.zeros((4, 2, 1, 1), dtype="float32")
    x[:, 0, 0, 0] = np.array([-0.1, 0.9, 0.8, -0.2], dtype="float32")
    y = np.array([0, 0, 1, 1])
    ds = BinnedTimeSeriesDataset(x, y, [f"p{i}" for i in range(4)])

    metrics = evaluate_model(ScoreModel(), ds, TrainConfig(device="cpu", batch_size=4))

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["auc"] == metrics["auroc"] == 0.25
    assert metrics["tpr"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["tnr"] == 0.5
    assert metrics["fnr"] == 0.5
    assert metrics["ppv"] == 0.5


def test_explanation_supports_progress_toggle():
    x = np.random.default_rng(1).normal(size=(4, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1]), [f"p{i}" for i in range(4)])
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    explanations = explain_model(model, ds, target_class=1, device="cpu", show_progress=False)
    assert set(explanations) == {"p0", "p1", "p2", "p3"}


def test_batched_explanations_match_single_patient_rollout():
    x = np.random.default_rng(3).normal(size=(5, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1, 0]), [f"p{i}" for i in range(5)])
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2, dropout=0.0)
    )

    single = explain_model(model, ds, target_class=1, device="cpu", show_progress=False, batch_size=1)
    batched = explain_model(model, ds, target_class=1, device="cpu", show_progress=False, batch_size=3)

    assert set(single) == set(batched)
    for patient_id in single:
        np.testing.assert_allclose(single[patient_id], batched[patient_id], rtol=1e-5, atol=1e-6)


def test_explanations_skip_existing_patient_files(tmp_path):
    x = np.random.default_rng(4).normal(size=(3, 2, 2, 3)).astype("float32")
    patient_ids = [f"p{i}" for i in range(3)]
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0]), patient_ids)
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    existing = np.full((2, 3), 7.0, dtype=np.float32)
    np.save(tmp_path / "p0.npy", existing)

    explanations = explain_model(
        model,
        ds,
        target_class=1,
        output_dir=tmp_path,
        device="cpu",
        show_progress=False,
        batch_size=2,
    )

    np.testing.assert_array_equal(explanations["p0"], existing)
    np.testing.assert_array_equal(np.load(tmp_path / "p0.npy"), existing)
    assert (tmp_path / "p1.npy").exists()
    assert (tmp_path / "p2.npy").exists()


def test_only_grad_attention_rollout_is_supported():
    x = np.random.default_rng(2).normal(size=(4, 2, 2, 3)).astype("float32")
    ds = BinnedTimeSeriesDataset(x, np.array([0, 1, 0, 1]), [f"p{i}" for i in range(4)])
    model = ViTTimeSeriesClassifier(ViTConfig(num_variables=2, num_timesteps=3, num_classes=2, embed_dim=8, depth=1, num_heads=2))
    try:
        explain_model(model, ds, method="other_method", target_class=1, device="cpu", show_progress=False)
    except ValueError as exc:
        assert "grad_attention_rollout" in str(exc)
    else:
        raise AssertionError("Expected non-rollout explanation methods to be rejected.")


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


def test_cluster_explanation_value_autoencoder_uses_predicted_classes(tmp_path):
    explanations = {
        f"p{idx}": np.array([[idx, idx + 1.0], [idx + 2.0, idx + 3.0]], dtype=float)
        for idx in range(6)
    }
    values = {
        f"p{idx}": np.array([[70 + idx, 80 + idx], [30 + idx, 35 + idx]], dtype=float)
        for idx in range(6)
    }
    predictions = {f"p{idx}": "true" if idx % 2 else "false" for idx in range(6)}

    clustered = cluster_explanation_value_autoencoder(
        {patient_id: explanations[patient_id] for patient_id in ["p0", "p1", "p2", "p3"]},
        {patient_id: values[patient_id] for patient_id in ["p0", "p1", "p2", "p3"]},
        validation_explanations={patient_id: explanations[patient_id] for patient_id in ["p4", "p5"]},
        validation_values={patient_id: values[patient_id] for patient_id in ["p4", "p5"]},
        cluster_explanations=explanations,
        cluster_values=values,
        predictions=predictions,
        n_clusters=2,
        output_dir=tmp_path / "autoencoder_clusters",
        latent_dim=4,
        epochs=2,
        batch_size=3,
        device="cpu",
    )

    assignments = clustered["assignments"]
    assert {"patient_id", "predicted_label", "cluster", "distance_to_centroid", "is_centroid"}.issubset(assignments.columns)
    assert set(assignments["predicted_label"]) == {"false", "true"}
    assert (tmp_path / "autoencoder_clusters" / "autoencoder_embeddings.csv").exists()
    assert (tmp_path / "autoencoder_clusters" / "autoencoder.pt").exists()
    assert (tmp_path / "autoencoder_clusters" / "autoencoder_metrics.json").exists()
    assert (tmp_path / "autoencoder_clusters" / "cluster_centroids.csv").exists()
    assert (tmp_path / "autoencoder_clusters" / "false" / "cluster_0.npy").exists()
    assert clustered["metrics"]["val_loss"] is not None
    assert clustered["metrics"]["cluster_loss"] is not None


def test_autoencoder_reuses_saved_embeddings_and_metrics(tmp_path):
    explanations = {f"p{idx}": np.ones((2, 2), dtype=float) * idx for idx in range(4)}
    values = {f"p{idx}": np.ones((2, 2), dtype=float) * (70 + idx) for idx in range(4)}
    out = tmp_path / "autoencoder_clusters"

    first = cluster_explanation_value_autoencoder(
        explanations,
        values,
        validation_explanations=explanations,
        validation_values=values,
        cluster_explanations=explanations,
        cluster_values=values,
        n_clusters=2,
        output_dir=out,
        latent_dim=2,
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    second = cluster_explanation_value_autoencoder(
        explanations,
        values,
        validation_explanations=explanations,
        validation_values=values,
        cluster_explanations=explanations,
        cluster_values=values,
        n_clusters=2,
        output_dir=out,
        latent_dim=2,
        epochs=10,
        batch_size=2,
        device="cpu",
    )

    assert first["embeddings"].shape == second["embeddings"].shape
    assert second["loaded_from_cache"] is True


def test_autoencoder_uses_requested_patch_size(tmp_path):
    explanations = {f"p{idx}": np.ones((2, 4), dtype=float) * idx for idx in range(4)}
    values = {f"p{idx}": np.ones((2, 4), dtype=float) * (70 + idx) for idx in range(4)}
    out = tmp_path / "autoencoder_clusters"

    clustered = cluster_explanation_value_autoencoder(
        explanations,
        values,
        validation_explanations=explanations,
        validation_values=values,
        cluster_explanations=explanations,
        cluster_values=values,
        n_clusters=2,
        output_dir=out,
        latent_dim=2,
        epochs=1,
        batch_size=2,
        device="cpu",
        patch_size=(2, 1),
    )

    checkpoint = torch.load(out / "autoencoder.pt", map_location="cpu")
    assert checkpoint["patch_size"] == [2, 1]
    assert clustered["metrics"]["cluster_loss"] is not None
