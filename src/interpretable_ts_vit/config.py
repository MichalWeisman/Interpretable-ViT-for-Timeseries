"""Configuration dataclasses and file loading helpers for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Input-table, binning, split, and leakage-control options."""

    patient_id_col: str = "patient_id"
    variable_col: str = "variable"
    value_col: str = "value"
    timestamp_col: str = "timestamp"
    label_col: str = "label"
    granularity: str = "30min"
    time_start: str | None = None
    time_end: str | None = None
    aggregation: str = "mean"
    allowed_variables: list[str] | None = None
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    random_state: int = 13


@dataclass
class ModelConfig:
    """Architecture hyperparameters shared by the ViT model."""

    patch_size: tuple[int, int] = (1, 4)
    embed_dim: int = 64
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training-loop options."""

    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    early_stopping_patience: int | None = None
    early_stopping_monitor: str = "val_loss"
    early_stopping_min_delta: float = 0.0
    early_stopping_mode: str = "auto"
    restore_best_model: bool = True
    verbose: bool = True
    progress_interval_batches: int | None = 50


@dataclass
class ExplainConfig:
    """Explanation method options."""

    method: str = "grad_attention_rollout"
    target_class: int | None = None
    batch_size: int = 16


@dataclass
class ClusterConfig:
    """Patient clustering and heatmap visualization options."""

    n_clusters: int = 8
    method: str = "kmeans"
    feature_mode: str = "autoencoder"
    autoencoder_latent_dim: int = 16
    autoencoder_epochs: int = 50
    autoencoder_learning_rate: float = 1e-3
    autoencoder_batch_size: int = 32
    autoencoder_early_stopping_patience: int | None = 10
    hdbscan_min_cluster_size: int | None = 5
    hdbscan_min_samples: int | None = None
    aggregate: str = "mean"
    plot_mode: str = "value_with_importance_opacity"
    importance_threshold: float | None = None
    show_values: bool = True
    normal_ranges_path: str | None = None
    use_normal_ranges: bool = False


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif key == "patch_size" and isinstance(value, list):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path | None = None) -> Config:
    """Load a JSON/YAML config file over the defaults.

    JSON works without optional dependencies. YAML files require PyYAML, which
    is declared as a project dependency but imported lazily for lighter use of
    the Python API.
    """
    config = Config()
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as fh:
        if Path(path).suffix.lower() == ".json":
            raw = json.load(fh) or {}
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML config files require PyYAML. Use JSON config or install PyYAML.") from exc
            raw = yaml.safe_load(fh) or {}
    return _merge_dataclass(config, raw)
