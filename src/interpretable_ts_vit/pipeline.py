"""Programmatic end-to-end pipeline without using the CLI entrypoint."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .binning import TimeSeriesBinner
from .clustering import cluster_explanations
from .config import Config
from .datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from .explain import explain_model
from .io import load_model, load_split, save_metadata, save_predictions, save_split
from .model import ViTConfig, ViTTimeSeriesClassifier
from .training import evaluate_model, predict_model, train_model
from .visualization import aggregate_cluster_value_matrices, plot_value_heatmap


@dataclass
class PipelinePaths:
    """Filesystem locations used by the end-to-end pipeline."""

    mimic_path: str | Path | None = "mimic-iv-3.1.zip"
    records_path: str | Path | None = None
    labels_path: str | Path | None = None
    dataset_dir: str | Path = "data/mimic_hypotension"
    processed_dir: str | Path = "data/processed"
    run_dir: str | Path = "runs/hypotension_v1"


@dataclass
class PipelineRunConfig:
    """Options controlling which pipeline stages run."""

    paths: PipelinePaths = field(default_factory=PipelinePaths)
    config: Config = field(default_factory=Config)
    mimic_config: MIMICHypotensionConfig | None = None
    prepare_mimic: bool = True
    prepare_tensors: bool = True
    train: bool = True
    evaluate: bool = True
    explain: bool = True
    cluster: bool = True
    plot: bool = True
    split: str = "test"
    render_instance_heatmaps: bool = False


@dataclass
class PipelineResult:
    """Summary returned after a pipeline run."""

    artifacts: dict[str, str]
    train_metrics: dict[str, Any] | None = None
    evaluation_metrics: dict[str, Any] | None = None


def run_pipeline(run_config: PipelineRunConfig | None = None) -> PipelineResult:
    """Run data preparation, training, loading, evaluation, and explanations.

    This is the programmatic equivalent of running the CLI commands in order:
    `prepare-mimic-hypotension`, `prepare-data`, `train`, `explain`,
    `cluster`, and `plot`.
    """

    run_config = run_config or PipelineRunConfig()
    paths = run_config.paths
    config = run_config.config
    artifacts: dict[str, str] = {}
    records_path, labels_path = _resolve_input_tables(run_config)
    train_metrics = None
    evaluation_metrics = None

    if run_config.prepare_mimic:
        mimic_config = run_config.mimic_config or MIMICHypotensionConfig(mimic_path=paths.mimic_path)
        prepared = MIMICIVHypotensionAdapter(mimic_config).prepare()
        prepared.save(paths.dataset_dir)
        records_path = Path(paths.dataset_dir) / "records.csv"
        labels_path = Path(paths.dataset_dir) / "labels.csv"
        artifacts["dataset_dir"] = str(Path(paths.dataset_dir))

    if run_config.prepare_tensors:
        _prepare_tensor_splits(records_path, labels_path, config, paths.processed_dir)
        artifacts["processed_dir"] = str(Path(paths.processed_dir))

    if run_config.train:
        train_metrics = _train_and_save(config, paths.processed_dir, paths.run_dir)
        artifacts["run_dir"] = str(Path(paths.run_dir))

    if run_config.evaluate:
        evaluation_metrics = _load_evaluate_and_save(config, paths.run_dir, run_config.split)
        artifacts["evaluation_metrics"] = str(Path(paths.run_dir) / f"{run_config.split}_evaluation_metrics.json")

    if run_config.explain:
        _explain_and_save(config, paths.run_dir, run_config.split)
        artifacts["explanations"] = str(Path(paths.run_dir) / "explanations" / run_config.split)

    if run_config.cluster:
        _cluster_and_save(config, paths.run_dir, run_config.split)
        artifacts["clusters"] = str(Path(paths.run_dir) / "clusters" / run_config.split)

    if run_config.plot:
        _plot_and_save(paths.run_dir, run_config.split, run_config.render_instance_heatmaps, config.cluster.plot_mode)
        artifacts["cluster_heatmaps"] = str(Path(paths.run_dir) / "cluster_heatmaps" / run_config.split)
        artifacts["cluster_values"] = str(Path(paths.run_dir) / "cluster_values" / run_config.split)

    return PipelineResult(artifacts=artifacts, train_metrics=train_metrics, evaluation_metrics=evaluation_metrics)


def _resolve_input_tables(run_config: PipelineRunConfig) -> tuple[Path, Path]:
    paths = run_config.paths
    if run_config.prepare_mimic:
        if paths.mimic_path is None and run_config.mimic_config is None:
            raise ValueError("prepare_mimic=True requires paths.mimic_path or mimic_config.")
        return Path(paths.dataset_dir) / "records.csv", Path(paths.dataset_dir) / "labels.csv"
    if paths.records_path is None or paths.labels_path is None:
        raise ValueError("prepare_mimic=False requires records_path and labels_path.")
    return Path(paths.records_path), Path(paths.labels_path)


def _prepare_tensor_splits(records_path: str | Path, labels_path: str | Path, config: Config, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(records_path)
    labels = pd.read_csv(labels_path)
    patient_ids = labels[config.data.patient_id_col].astype(str).tolist()
    y = labels[config.data.label_col].astype(str).tolist()
    holdout_fraction = config.data.val_fraction + config.data.test_fraction
    if not 0 < holdout_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be between 0 and 1.")
    train_ids, holdout_ids, _, holdout_y = train_test_split(
        patient_ids,
        y,
        test_size=holdout_fraction,
        random_state=config.data.random_state,
        stratify=_stratify_or_none(y),
    )
    relative_test = config.data.test_fraction / holdout_fraction
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=relative_test,
        random_state=config.data.random_state,
        stratify=_stratify_or_none(holdout_y),
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    train_labels = labels[labels[config.data.patient_id_col].astype(str).isin(train_ids)]
    train_records = records[records[config.data.patient_id_col].astype(str).isin(train_ids)]
    binner = TimeSeriesBinner(config.data).fit(train_records, train_labels)
    for split, ids in split_ids.items():
        split_labels = labels[labels[config.data.patient_id_col].astype(str).isin(ids)]
        split_records = records[records[config.data.patient_id_col].astype(str).isin(ids)]
        binned = binner.transform(split_records, split_labels)
        save_split(out / f"{split}.npz", binned.patient_ids, binned.x, binned.y)
    save_metadata(out, binner)
    with (out / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(split_ids, fh, indent=2)


def _train_and_save(config: Config, data_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_ds = load_split(data_dir / "train.npz")
    val_ds = load_split(data_dir / "val.npz")
    binner = TimeSeriesBinner.load(data_dir / "binner.json")
    x_shape = train_ds.x.shape
    model_config = ViTConfig(
        **config.model.__dict__,
        num_variables=int(x_shape[2]),
        num_timesteps=int(x_shape[3]),
        num_classes=len(binner.index_to_label_),
    )
    model = ViTTimeSeriesClassifier(model_config)
    metrics = train_model(model, train_ds, val_ds, config.train, run_dir)
    shutil.copyfile(data_dir / "binner.json", run_dir / "binner.json")
    shutil.copyfile(data_dir / "variable_vocab.json", run_dir / "variable_vocab.json")
    for split in ["train", "val", "test"]:
        split_path = data_dir / f"{split}.npz"
        if split_path.exists():
            shutil.copyfile(split_path, run_dir / split_path.name)
    return metrics


def _load_evaluate_and_save(config: Config, run_dir: str | Path, split: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    model = load_model(run_dir)
    dataset = load_split(run_dir / f"{split}.npz")
    binner = TimeSeriesBinner.load(run_dir / "binner.json")
    metrics = evaluate_model(model, dataset, config.train)
    with (run_dir / f"{split}_evaluation_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(metrics), fh, indent=2)
    logits, _ = predict_model(model, dataset, config.train)
    save_predictions(run_dir / f"{split}_predictions.csv", dataset.patient_ids or [], logits, binner.index_to_label_)
    return metrics


def _explain_and_save(config: Config, run_dir: str | Path, split: str) -> None:
    run_dir = Path(run_dir)
    model = load_model(run_dir)
    dataset = load_split(run_dir / f"{split}.npz")
    explain_model(
        model,
        dataset,
        method=config.explain.method,
        target_class=config.explain.target_class,
        output_dir=run_dir / "explanations" / split,
        device=config.train.device,
    )


def _cluster_and_save(config: Config, run_dir: str | Path, split: str) -> None:
    run_dir = Path(run_dir)
    cluster_explanations(
        run_dir / "explanations" / split,
        n_clusters=config.cluster.n_clusters,
        method=config.cluster.method,
        aggregate=config.cluster.aggregate,
        output_dir=run_dir / "clusters" / split,
    )


def _plot_and_save(run_dir: str | Path, split: str, render_instance_heatmaps: bool, plot_mode: str = "value_with_importance_opacity") -> None:
    run_dir = Path(run_dir)
    binner = TimeSeriesBinner.load(run_dir / "binner.json")
    dataset = load_split(run_dir / f"{split}.npz")
    assignments_path = run_dir / "clusters" / split / "cluster_assignments.csv"
    cluster_dir = run_dir / "clusters" / split
    value_dir = run_dir / "cluster_values" / split
    heatmap_dir = run_dir / "cluster_heatmaps" / split
    matrices_by_cluster = aggregate_cluster_value_matrices(dataset, assignments_path, binner, output_dir=value_dir)
    importance_by_cluster = _cluster_importance_matrices(cluster_dir) if plot_mode == "value_with_importance_opacity" else {}
    matrices = list(matrices_by_cluster.values())
    if matrices:
        vmin = min(float(np.nanmin(matrix)) for matrix in matrices)
        vmax = max(float(np.nanmax(matrix)) for matrix in matrices)
        for cluster, matrix in matrices_by_cluster.items():
            plot_value_heatmap(
                matrix,
                binner.variable_vocab_,
                binner.time_bins_,
                heatmap_dir / f"cluster_{cluster}.png",
                title=f"cluster_{cluster}",
                vmin=vmin,
                vmax=vmax,
                importance_matrix=importance_by_cluster.get(cluster),
            )
    if render_instance_heatmaps:
        instance_dir = run_dir / "instance_heatmaps" / split
        instance_maps = _denormalized_patient_value_maps(dataset, binner)
        if instance_maps:
            vmin = min(float(np.nanmin(matrix)) for matrix in instance_maps.values())
            vmax = max(float(np.nanmax(matrix)) for matrix in instance_maps.values())
            for patient_id, matrix in instance_maps.items():
                plot_value_heatmap(
                    matrix,
                    binner.variable_vocab_,
                    binner.time_bins_,
                    instance_dir / f"{patient_id}.png",
                    title=patient_id,
                    vmin=vmin,
                    vmax=vmax,
                )


def _cluster_importance_matrices(cluster_dir: Path) -> dict[int, np.ndarray]:
    matrices: dict[int, np.ndarray] = {}
    for path in sorted(cluster_dir.glob("cluster_*.npy")):
        try:
            cluster = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        matrices[cluster] = np.load(path)
    return matrices


def _denormalized_patient_value_maps(dataset, binner: TimeSeriesBinner) -> dict[str, np.ndarray]:
    if dataset.patient_ids is None:
        return {}
    x = dataset.x.detach().cpu().numpy()
    values = x[:, 0].astype(np.float64)
    mask = x[:, 1].astype(np.float64)
    means = np.array([binner.means_.get(variable, 0.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    stds = np.array([binner.stds_.get(variable, 1.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    raw_values = values * stds[None, :, :] + means[None, :, :]
    raw_values[mask == 0] = np.nan
    return {patient_id: raw_values[idx] for idx, patient_id in enumerate(dataset.patient_ids)}


def _stratify_or_none(labels: list[str]) -> list[str] | None:
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
