import pandas as pd

from interpretable_ts_vit.config import ClusterConfig, DataConfig, ExplainConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.data_modules import GenericCSVDataModule, MIMICHypotensionDataModule
from interpretable_ts_vit.model_modules import ViTTimeSeriesModule


def test_data_and_model_modules_run_split_workflow(tmp_path):
    records = []
    labels = []
    for i in range(8):
        pid = f"p{i}"
        labels.append({"patient_id": pid, "label": "true" if i % 2 else "false"})
        for t in range(4):
            records.append({"patient_id": pid, "variable": "hr", "value": 70 + i + t, "timestamp": f"2026-01-01 0{t}:00:00"})
            records.append({"patient_id": pid, "variable": "map", "value": 65 + i - t, "timestamp": f"2026-01-01 0{t}:00:00"})

    records_path = tmp_path / "records.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    data = GenericCSVDataModule(
        records_path=records_path,
        labels_path=labels_path,
        processed_dir=tmp_path / "processed",
        data_config=DataConfig(
            granularity="1h",
            time_start="2026-01-01 00:00:00",
            time_end="2026-01-01 04:00:00",
            val_fraction=0.25,
            test_fraction=0.25,
        ),
    )
    model = ViTTimeSeriesModule(
        run_dir=tmp_path / "run",
        model_config=ModelConfig(patch_size=(1, 2), embed_dim=16, depth=1, num_heads=2),
        train_config=TrainConfig(batch_size=4, epochs=1, learning_rate=1e-3, weight_decay=0.0, device="cpu", verbose=False),
        explain_config=ExplainConfig(method="grad_attention_rollout", target_class=1),
        cluster_config=ClusterConfig(n_clusters=2, show_values=True),
    )

    data.prepare()
    assert data.split("train").x.shape[2:] == (2, 4)
    assert data.variable_vocab == ["hr", "map"]

    train_metrics = model.fit(data)
    eval_metrics = model.evaluate(data, split="test")
    explanations_dir = model.explain(data, split="test")
    clusters_dir = model.cluster_explanations(data, split="test")
    heatmaps_dir = model.plot_cluster_values(data, split="test")

    assert "history" in train_metrics
    assert "accuracy" in eval_metrics
    assert model.metrics_path("test").exists()
    assert model.predictions_path("test").exists()
    assert not model.predictions("test").empty
    assert any(explanations_dir.glob("*.npy"))
    assert (clusters_dir / "cluster_assignments.csv").exists()
    assert any(heatmaps_dir.rglob("*.png"))
    assert model.display_cluster_heatmaps("test")


def test_mimic_data_module_can_reuse_prepared_csv_files(tmp_path):
    records_path = tmp_path / "records.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"patient_id": "p1", "variable": "hr", "value": 70, "timestamp": "2026-01-01 00:00:00"},
            {"patient_id": "p2", "variable": "hr", "value": 80, "timestamp": "2026-01-01 00:00:00"},
        ]
    ).to_csv(records_path, index=False)
    pd.DataFrame([{"patient_id": "p1", "label": "false"}, {"patient_id": "p2", "label": "true"}]).to_csv(labels_path, index=False)

    data = MIMICHypotensionDataModule(
        records_path=records_path,
        labels_path=labels_path,
        processed_dir=tmp_path / "processed",
    )

    assert data.input_paths() == (records_path, labels_path)
