import numpy as np
import pandas as pd

from interpretable_ts_vit import TimeSeriesBinner, ViTConfig, ViTTimeSeriesClassifier, cluster_explanations, explain_model
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import train_model
from interpretable_ts_vit.visualization import plot_explanation_heatmap


def test_end_to_end_smoke(tmp_path):
    records = []
    labels = {}
    for i in range(10):
        pid = f"p{i}"
        labels[pid] = "case" if i % 2 else "control"
        for t in range(4):
            records.append([pid, "hr", 70 + i + t, f"2026-01-01 0{t}:00:00"])
            if t % 2 == 0:
                records.append([pid, "bp", 110 + i, f"2026-01-01 0{t}:30:00"])
    records = pd.DataFrame(records, columns=["patient_id", "variable", "value", "timestamp"])
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01 00:00:00", time_end="2026-01-01 04:00:00")
    binned = binner.fit_transform(records, labels)
    ds = BinnedTimeSeriesDataset(binned.x, binned.y, binned.patient_ids)
    model = ViTTimeSeriesClassifier(
        ViTConfig(num_variables=2, num_timesteps=4, num_classes=2, patch_size=(1, 2), embed_dim=16, depth=1, num_heads=2)
    )
    train_model(model, ds, ds, config=type("Cfg", (), {"device": "cpu", "batch_size": 5, "epochs": 1, "learning_rate": 1e-3, "weight_decay": 0.0})())
    explanations = explain_model(model, ds, output_dir=tmp_path / "explanations", device="cpu")
    clustered = cluster_explanations(explanations, n_clusters=2, output_dir=tmp_path / "clusters")
    for cluster, matrix in clustered["aggregates"].items():
        plot_explanation_heatmap(matrix, binner.variable_vocab_, binner.time_bins_, tmp_path / f"cluster_{cluster}.png")
    assert (tmp_path / "explanations" / "p0.npy").exists()
    assert (tmp_path / "clusters" / "cluster_assignments.csv").exists()
    assert any(tmp_path.glob("cluster_*.png"))
