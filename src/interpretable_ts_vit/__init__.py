"""Public Python API for the interpretable time-series ViT package.

Heavy optional modules such as PyTorch-based training are imported lazily so
pure preprocessing code can run in lighter environments.
"""

from .config import Config, load_config
from .binning import TimeSeriesBinner


def __getattr__(name):
    if name == "cluster_explanations":
        from .clustering import cluster_explanations

        return cluster_explanations
    if name == "explain_model":
        from .explain import explain_model

        return explain_model
    if name in {"ViTConfig", "ViTTimeSeriesClassifier"}:
        from .model import ViTConfig, ViTTimeSeriesClassifier

        return {"ViTConfig": ViTConfig, "ViTTimeSeriesClassifier": ViTTimeSeriesClassifier}[name]
    if name == "train_model":
        from .training import train_model

        return train_model
    if name == "plot_explanation_heatmap":
        from .visualization import plot_explanation_heatmap

        return plot_explanation_heatmap
    raise AttributeError(name)

__all__ = [
    "Config",
    "TimeSeriesBinner",
    "ViTConfig",
    "ViTTimeSeriesClassifier",
    "cluster_explanations",
    "explain_model",
    "load_config",
    "plot_explanation_heatmap",
    "train_model",
]
