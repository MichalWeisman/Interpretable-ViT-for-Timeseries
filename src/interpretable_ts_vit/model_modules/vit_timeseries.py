"""Notebook-friendly module wrapper for the ViT time-series classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ClusterConfig, Config, ExplainConfig, ModelConfig, TrainConfig
from ..data_modules import BaseDataModule
from ..pipeline import _cluster_and_save, _explain_and_save, _load_evaluate_and_save, _plot_and_save, _train_and_save


@dataclass
class ViTTimeSeriesModule:
    """High-level model workflow for training, evaluation, explanation, and plots."""

    run_dir: str | Path
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    explain_config: ExplainConfig = field(default_factory=ExplainConfig)
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)

    def fit(self, data: BaseDataModule) -> dict[str, Any]:
        """Train the model on prepared tensors from `data`."""
        data.load()
        return _train_and_save(self._config(data), data.processed_dir, self.run_dir)

    def evaluate(self, data: BaseDataModule, split: str = "test") -> dict[str, Any]:
        """Evaluate a saved model and write metrics/predictions for `split`."""
        data.load()
        return _load_evaluate_and_save(self._config(data), self.run_dir, split)

    def explain(self, data: BaseDataModule, split: str = "test", show_progress: bool = True, batch_size: int | None = None) -> Path:
        """Generate explanation maps for a saved model and split."""
        data.load()
        config = self._config(data)
        if batch_size is not None:
            config.explain.batch_size = batch_size
        _explain_and_save(config, self.run_dir, split, show_progress=show_progress)
        return self.explanations_dir(split)

    def cluster_explanations(self, data: BaseDataModule | None = None, split: str = "test") -> Path:
        """Cluster patients with autoencoder embeddings from explanation/value maps."""
        _cluster_and_save(self._config(data), self.run_dir, split)
        return self.clusters_dir(split)

    def cluster_autoencoder(self, data: BaseDataModule | None = None, split: str = "test") -> Path:
        """Cluster patients by autoencoded explanation/value maps."""
        config = self._config(data)
        config.cluster.feature_mode = "autoencoder"
        _cluster_and_save(config, self.run_dir, split)
        return self.clusters_dir(split)

    def plot_cluster_values(
        self,
        data: BaseDataModule,
        split: str = "test",
        render_instance_heatmaps: bool = False,
    ) -> Path:
        """Plot cluster-level value heatmaps using saved clusters and tensors."""
        data.load()
        _plot_and_save(
            self.run_dir,
            split,
            render_instance_heatmaps,
            self.cluster_config.plot_mode,
            self.cluster_config.importance_threshold,
            self.cluster_config.show_values,
            self.cluster_config.normal_ranges_path
            if self.cluster_config.normal_ranges_path
            else (Path(__file__).parents[1] / "normal_ranges.json" if self.cluster_config.use_normal_ranges else None),
        )
        return self.cluster_heatmaps_dir(split)

    def visualize(
        self,
        data: BaseDataModule,
        split: str = "test",
        render_instance_heatmaps: bool = False,
    ) -> Path:
        """Cluster explanations and plot value heatmaps."""
        self.cluster_explanations(data, split=split)
        return self.plot_cluster_values(data, split=split, render_instance_heatmaps=render_instance_heatmaps)

    def display_cluster_heatmaps(self, split: str = "test") -> list[Path]:
        """Display cluster heatmaps in notebooks when IPython is available."""
        paths = sorted(self.cluster_heatmaps_dir(split).rglob("*.png"))
        try:
            from IPython.display import Image, display
        except ImportError:
            return paths
        for path in paths:
            print(path.relative_to(self.cluster_heatmaps_dir(split)))
            display(Image(filename=str(path)))
        return paths

    def metrics_path(self, split: str = "test") -> Path:
        return Path(self.run_dir) / f"{split}_evaluation_metrics.json"

    def predictions_path(self, split: str = "test") -> Path:
        return Path(self.run_dir) / f"{split}_predictions.csv"

    def predictions(self, split: str = "test") -> pd.DataFrame:
        return pd.read_csv(self.predictions_path(split))

    def explanations_dir(self, split: str = "test") -> Path:
        return Path(self.run_dir) / "explanations" / split

    def clusters_dir(self, split: str = "test") -> Path:
        return Path(self.run_dir) / "clusters" / split

    def cluster_heatmaps_dir(self, split: str = "test") -> Path:
        return Path(self.run_dir) / "cluster_heatmaps" / split

    def _config(self, data: BaseDataModule | None = None) -> Config:
        return Config(
            data=data.data_config if data is not None else Config().data,
            model=self.model_config,
            train=self.train_config,
            explain=self.explain_config,
            cluster=self.cluster_config,
        )
