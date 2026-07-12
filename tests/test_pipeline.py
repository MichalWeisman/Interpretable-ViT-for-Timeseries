import pandas as pd

from interpretable_ts_vit.config import ClusterConfig, Config, DataConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.binning import TimeSeriesBinner
from interpretable_ts_vit.pipeline import _prepare_tensor_splits
from interpretable_ts_vit.pipeline import PipelinePaths, PipelineRunConfig, run_pipeline


def test_run_pipeline_without_cli_on_generic_records(tmp_path):
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

    config = Config(
        data=DataConfig(
            granularity="1h",
            time_start="2026-01-01 00:00:00",
            time_end="2026-01-01 04:00:00",
            val_fraction=0.25,
            test_fraction=0.25,
        ),
        model=ModelConfig(patch_size=(1, 2), embed_dim=16, depth=1, num_heads=2),
        train=TrainConfig(batch_size=4, epochs=1, learning_rate=1e-3, weight_decay=0.0, device="cpu"),
        cluster=ClusterConfig(n_clusters=2),
    )
    result = run_pipeline(
        PipelineRunConfig(
            paths=PipelinePaths(
                records_path=records_path,
                labels_path=labels_path,
                processed_dir=tmp_path / "processed",
                run_dir=tmp_path / "run",
            ),
            config=config,
            prepare_mimic=False,
            split="test",
        )
    )

    assert (tmp_path / "processed" / "binner.json").exists()
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "test_evaluation_metrics.json").exists()
    assert (tmp_path / "run" / "explanations" / "test").exists()
    assert (tmp_path / "run" / "clusters" / "test" / "cluster_assignments.csv").exists()
    assignments = pd.read_csv(tmp_path / "run" / "clusters" / "test" / "cluster_assignments.csv")
    assert "predicted_label" in assignments.columns
    assert any((tmp_path / "run" / "cluster_heatmaps" / "test").rglob("*.png"))
    assert "evaluation_metrics" in result.artifacts


def test_mimic_target_tensor_preparation_infers_yaml_variable_filter(tmp_path):
    source_dir = tmp_path / "data" / "mimic_targets" / "obs24_target8_gap0" / "hypoglycemia"
    source_dir.mkdir(parents=True)
    records = []
    labels = []
    for i in range(8):
        patient_id = f"p{i}"
        labels.append({"patient_id": patient_id, "label": "true" if i % 2 else "false"})
        records.append({"patient_id": patient_id, "variable": "blood_glucose", "value": 80 + i, "timestamp": "2026-01-01 00:00:00"})
        records.append({"patient_id": patient_id, "variable": "not_in_yaml", "value": 1.0, "timestamp": "2026-01-01 00:00:00"})
    records_path = source_dir / "records.csv"
    labels_path = source_dir / "labels.csv"
    pd.DataFrame(records).to_csv(records_path, index=False)
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    _prepare_tensor_splits(
        records_path,
        labels_path,
        Config(data=DataConfig(granularity="1h", time_start="2026-01-01", time_end="2026-01-02")),
        tmp_path / "processed",
    )

    binner = TimeSeriesBinner.load(tmp_path / "processed" / "binner.json")
    assert binner.variable_vocab_ == ["blood_glucose"]
