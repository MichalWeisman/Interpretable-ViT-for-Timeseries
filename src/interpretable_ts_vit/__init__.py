"""Public Python API for the interpretable time-series ViT package.

Heavy optional modules such as PyTorch-based training are imported lazily so
pure preprocessing code can run in lighter environments.
"""

from .config import Config, load_config
from .binning import TimeSeriesBinner


def __getattr__(name):
    if name in {
        "cluster_autoencoder_embeddings",
        "cluster_explanation_value_autoencoder",
        "create_explanation_value_embeddings",
        "train_explanation_value_autoencoder",
    }:
        from .autoencoder import (
            cluster_autoencoder_embeddings,
            cluster_explanation_value_autoencoder,
            create_explanation_value_embeddings,
            train_explanation_value_autoencoder,
        )

        return {
            "cluster_autoencoder_embeddings": cluster_autoencoder_embeddings,
            "cluster_explanation_value_autoencoder": cluster_explanation_value_autoencoder,
            "create_explanation_value_embeddings": create_explanation_value_embeddings,
            "train_explanation_value_autoencoder": train_explanation_value_autoencoder,
        }[name]
    if name == "explain_model":
        from .explain import explain_model

        return explain_model
    if name in {"ViTConfig", "ViTTimeSeriesClassifier"}:
        from .model import ViTConfig, ViTTimeSeriesClassifier

        return {"ViTConfig": ViTConfig, "ViTTimeSeriesClassifier": ViTTimeSeriesClassifier}[name]
    if name == "train_model":
        from .training import train_model

        return train_model
    if name in {
        "aggregate_cluster_value_matrices",
        "filter_value_matrix_by_explanation",
        "load_patient_explanation_matrix",
        "load_normal_ranges",
        "normal_range_status_matrix",
        "patient_value_matrix",
        "patient_class_frame",
        "plot_explanation_heatmap",
        "plot_patient_matrix_comparison",
        "plot_patient_matrices",
        "plot_value_heatmap",
        "select_patient_ids",
    }:
        from .visualization import (
            aggregate_cluster_value_matrices,
            filter_value_matrix_by_explanation,
            load_patient_explanation_matrix,
            load_normal_ranges,
            normal_range_status_matrix,
            patient_class_frame,
            patient_value_matrix,
            plot_explanation_heatmap,
            plot_patient_matrix_comparison,
            plot_patient_matrices,
            plot_value_heatmap,
            select_patient_ids,
        )

        return {
            "aggregate_cluster_value_matrices": aggregate_cluster_value_matrices,
            "filter_value_matrix_by_explanation": filter_value_matrix_by_explanation,
            "load_patient_explanation_matrix": load_patient_explanation_matrix,
            "load_normal_ranges": load_normal_ranges,
            "normal_range_status_matrix": normal_range_status_matrix,
            "patient_class_frame": patient_class_frame,
            "patient_value_matrix": patient_value_matrix,
            "plot_explanation_heatmap": plot_explanation_heatmap,
            "plot_patient_matrix_comparison": plot_patient_matrix_comparison,
            "plot_patient_matrices": plot_patient_matrices,
            "plot_value_heatmap": plot_value_heatmap,
            "select_patient_ids": select_patient_ids,
        }[name]
    if name in {"MIMICIVHypotensionAdapter", "MIMICHypotensionConfig"}:
        from .datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig

        return {
            "MIMICIVHypotensionAdapter": MIMICIVHypotensionAdapter,
            "MIMICHypotensionConfig": MIMICHypotensionConfig,
        }[name]
    if name in {"PipelinePaths", "PipelineResult", "PipelineRunConfig", "run_pipeline"}:
        from .pipeline import PipelinePaths, PipelineResult, PipelineRunConfig, run_pipeline

        return {
            "PipelinePaths": PipelinePaths,
            "PipelineResult": PipelineResult,
            "PipelineRunConfig": PipelineRunConfig,
            "run_pipeline": run_pipeline,
        }[name]
    if name in {"BaseDataModule", "GenericCSVDataModule", "MIMICHypotensionDataModule"}:
        from .data_modules import BaseDataModule, GenericCSVDataModule, MIMICHypotensionDataModule

        return {
            "BaseDataModule": BaseDataModule,
            "GenericCSVDataModule": GenericCSVDataModule,
            "MIMICHypotensionDataModule": MIMICHypotensionDataModule,
        }[name]
    if name == "ViTTimeSeriesModule":
        from .model_modules import ViTTimeSeriesModule

        return ViTTimeSeriesModule
    raise AttributeError(name)

__all__ = [
    "Config",
    "TimeSeriesBinner",
    "ViTConfig",
    "ViTTimeSeriesClassifier",
    "aggregate_cluster_value_matrices",
    "cluster_autoencoder_embeddings",
    "cluster_explanation_value_autoencoder",
    "create_explanation_value_embeddings",
    "explain_model",
    "filter_value_matrix_by_explanation",
    "load_config",
    "load_patient_explanation_matrix",
    "load_normal_ranges",
    "MIMICIVHypotensionAdapter",
    "MIMICHypotensionConfig",
    "normal_range_status_matrix",
    "patient_class_frame",
    "patient_value_matrix",
    "PipelinePaths",
    "PipelineResult",
    "PipelineRunConfig",
    "plot_explanation_heatmap",
    "plot_patient_matrix_comparison",
    "plot_patient_matrices",
    "plot_value_heatmap",
    "run_pipeline",
    "select_patient_ids",
    "train_explanation_value_autoencoder",
    "train_model",
    "BaseDataModule",
    "GenericCSVDataModule",
    "MIMICHypotensionDataModule",
    "ViTTimeSeriesModule",
]
