import numpy as np
import pytest
import torch

from interpretable_ts_vit import ViTConfig, ViTTimeSeriesClassifier, cluster_explanations, explain_model
from interpretable_ts_vit.config import TrainConfig
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import train_model


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
