"""Command-line entrypoint for prepare/train/explain/cluster/plot workflows."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .autoencoder import cluster_autoencoder_embeddings, create_explanation_value_embeddings, train_explanation_value_autoencoder
from .binning import TimeSeriesBinner
from .config import load_config
from .data import BinnedTimeSeriesDataset
from .datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from .explain import explain_model
from .io import load_model, load_split, save_metadata, save_predictions, save_split
from .model import ViTConfig, ViTTimeSeriesClassifier
from .training import predict_model, train_model
from .visualization import aggregate_cluster_value_matrices, cluster_assignment_counts, patient_value_matrix, plot_value_heatmap, value_ranges_by_variable


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected workflow command."""
    parser = argparse.ArgumentParser(prog="tsvit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--records", required=True)
    prepare.add_argument("--labels", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--config")

    mimic = sub.add_parser("prepare-mimic-hypotension")
    mimic.add_argument("--mimic-path", required=True, help="Path to MIMIC-IV zip archive or extracted directory.")
    mimic.add_argument("--out", required=True, help="Directory for records.csv, labels.csv, and metadata.")
    mimic.add_argument("--observation-hours", type=float, default=24.0)
    mimic.add_argument("--prediction-hours", type=float, default=6.0)
    mimic.add_argument("--threshold", type=float, default=65.0, help="MAP threshold in mmHg for hypotension.")
    mimic.add_argument("--chunk-size", type=int, default=1_000_000)
    mimic.add_argument("--cache-dir", default="data/hypotension/mimic_cache")
    mimic.add_argument("--read-zip-directly", action="store_true", help="Do not extract selected .csv.gz files before reading.")
    mimic.add_argument("--no-filtered-cache", action="store_true", help="Do not read/write the filtered chartevents Parquet cache.")
    mimic.add_argument("--progress-interval-chunks", type=int, default=1)
    mimic.add_argument("--max-stays", type=int)
    mimic.add_argument("--min-observations", type=int, default=1)
    mimic.add_argument("--allow-short-prediction-window", action="store_true")
    mimic.add_argument("--allow-missing-outcome-measurement", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--config")

    explain = sub.add_parser("explain")
    explain.add_argument("--run", required=True)
    explain.add_argument("--split", default="test")
    explain.add_argument("--target-class", type=int)
    explain.add_argument("--batch-size", type=int)

    cluster = sub.add_parser("cluster")
    cluster.add_argument("--run", required=True)
    cluster.add_argument("--split", default="test")
    cluster.add_argument("--n-clusters", type=int)
    cluster.add_argument("--method", choices=["kmeans", "hdbscan"])
    cluster.add_argument("--hdbscan-min-cluster-size", type=int)
    cluster.add_argument("--hdbscan-min-samples", type=int)
    cluster.add_argument("--config")

    plot = sub.add_parser("plot")
    plot.add_argument("--run", required=True)
    plot.add_argument("--split", default="test")
    plot.add_argument("--instances", action="store_true")
    plot.add_argument("--config")
    plot.add_argument("--plot-mode", choices=["value", "value_with_importance_opacity", "value_with_importance_border"])
    plot.add_argument(
        "--importance-threshold",
        type=float,
        help="Optional importance quantile in [0, 1]; for example, 0.8 shows only the top 20%% most important cells.",
    )
    plot.add_argument("--show-values", action="store_true", help="Annotate heatmap cells with mean observed values.")
    plot.add_argument("--use-normal-ranges", action="store_true", help="Color values as low/normal/high using normal ranges.")
    plot.add_argument("--normal-ranges", help="Path to a JSON file of normal ranges.")

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("Running command %s", args.command)
    if args.command == "prepare-data":
        cmd_prepare_data(args)
    elif args.command == "prepare-mimic-hypotension":
        cmd_prepare_mimic_hypotension(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "cluster":
        cmd_cluster(args)
    elif args.command == "plot":
        cmd_plot(args)


def cmd_prepare_data(args) -> None:
    """Split records, fit preprocessing on train only, and save `.npz` splits."""
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing data tensors from records=%s labels=%s into %s", args.records, args.labels, out)
    records = pd.read_csv(args.records)
    labels = pd.read_csv(args.labels)
    logger.info("Loaded source tables for tensor preparation: records=%d, labels=%d", len(records), len(labels))
    patient_ids = labels[config.data.patient_id_col].astype(str).tolist()
    y = labels[config.data.label_col].astype(str).tolist()
    train_ids, holdout_ids, _, holdout_y = train_test_split(
        patient_ids,
        y,
        test_size=config.data.val_fraction + config.data.test_fraction,
        random_state=config.data.random_state,
        stratify=y if len(set(y)) > 1 else None,
    )
    relative_test = config.data.test_fraction / (config.data.val_fraction + config.data.test_fraction)
    val_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=relative_test,
        random_state=config.data.random_state,
        stratify=holdout_y if len(set(holdout_y)) > 1 else None,
    )
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    logger.info("Created patient splits: train=%d, val=%d, test=%d", len(train_ids), len(val_ids), len(test_ids))
    train_labels = labels[labels[config.data.patient_id_col].astype(str).isin(train_ids)]
    train_records = records[records[config.data.patient_id_col].astype(str).isin(train_ids)]
    logger.info("Fitting binner on train split: records=%d, labels=%d", len(train_records), len(train_labels))
    binner = TimeSeriesBinner(config.data).fit(train_records, train_labels)
    for split, ids in split_ids.items():
        split_labels = labels[labels[config.data.patient_id_col].astype(str).isin(ids)]
        split_records = records[records[config.data.patient_id_col].astype(str).isin(ids)]
        logger.info("Preparing %s tensor split: records=%d, labels=%d", split, len(split_records), len(split_labels))
        binned = binner.transform(split_records, split_labels)
        save_split(out / f"{split}.npz", binned.patient_ids, binned.x, binned.y)
    save_metadata(out, binner)
    with (out / "splits.json").open("w", encoding="utf-8") as fh:
        json.dump(split_ids, fh, indent=2)
    logger.info("Finished preparing data tensors into %s", out)


def cmd_prepare_mimic_hypotension(args) -> None:
    """Create generic records/labels files for MIMIC-IV hypotension prediction."""
    logger.info("Preparing MIMIC-IV hypotension CSV files into %s", args.out)
    config = MIMICHypotensionConfig(
        mimic_path=args.mimic_path,
        observation_hours=args.observation_hours,
        prediction_hours=args.prediction_hours,
        hypotension_threshold=args.threshold,
        chunk_size=args.chunk_size,
        cache_dir=args.cache_dir,
        use_extracted_files=not args.read_zip_directly,
        use_filtered_cache=not args.no_filtered_cache,
        progress_interval_chunks=args.progress_interval_chunks,
        max_stays=args.max_stays,
        min_observations=args.min_observations,
        require_full_prediction_window=not args.allow_short_prediction_window,
        require_outcome_measurement=not args.allow_missing_outcome_measurement,
    )
    prepared = MIMICIVHypotensionAdapter(config).prepare()
    prepared.save(args.out)
    logger.info("Finished preparing MIMIC-IV hypotension CSV files into %s", args.out)


def cmd_train(args) -> None:
    """Train the ViT on prepared data and save model/metrics/predictions."""
    config = load_config(args.config)
    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Loading training data from %s", data_dir)
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
    metrics = train_model(model, train_ds, val_ds, config.train, out)
    logger.info("Copying prepared data artifacts from %s to %s", data_dir, out)
    shutil.copyfile(data_dir / "binner.json", out / "binner.json")
    shutil.copyfile(data_dir / "variable_vocab.json", out / "variable_vocab.json")
    for split in ["train", "val", "test"]:
        if (data_dir / f"{split}.npz").exists():
            shutil.copyfile(data_dir / f"{split}.npz", out / f"{split}.npz")
    test_path = data_dir / "test.npz"
    if test_path.exists():
        test_ds = load_split(test_path)
        logits, _ = predict_model(model, test_ds, config.train)
        save_predictions(out / "predictions.csv", test_ds.patient_ids or [], logits, binner.index_to_label_)
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def cmd_explain(args) -> None:
    """Generate per-patient explanation maps for a prepared split."""
    run = Path(args.run)
    config = load_config(None)
    model = load_model(run)
    dataset = load_split(run / f"{args.split}.npz")
    out = run / "explanations" / args.split
    explain_model(
        model,
        dataset,
        method="grad_attention_rollout",
        target_class=args.target_class,
        output_dir=out,
        batch_size=args.batch_size or config.explain.batch_size,
    )


def cmd_cluster(args) -> None:
    """Cluster patients by model-importance maps."""
    run = Path(args.run)
    config = load_config(args.config).cluster
    n_clusters = args.n_clusters or config.n_clusters
    method = args.method or config.method
    hdbscan_min_cluster_size = (
        args.hdbscan_min_cluster_size
        if args.hdbscan_min_cluster_size is not None
        else config.hdbscan_min_cluster_size
    )
    hdbscan_min_samples = args.hdbscan_min_samples if args.hdbscan_min_samples is not None else config.hdbscan_min_samples
    explanation_dir = run / "explanations" / args.split
    out = run / "clusters" / args.split
    full_config = load_config(args.config)
    model = load_model(run)
    for autoencoder_split in _autoencoder_required_splits(args.split):
        autoencoder_explanation_dir = run / "explanations" / autoencoder_split
        explain_model(
            model,
            load_split(run / f"{autoencoder_split}.npz"),
            method="grad_attention_rollout",
            target_class=full_config.explain.target_class,
            output_dir=autoencoder_explanation_dir,
            device=full_config.train.device,
            show_progress=False,
            batch_size=full_config.explain.batch_size,
        )
    train_dataset = load_split(run / "train.npz")
    val_dataset = load_split(run / "val.npz")
    dataset = load_split(run / f"{args.split}.npz")
    binner = TimeSeriesBinner.load(run / "binner.json")
    trained = train_explanation_value_autoencoder(
        run / "explanations" / "train",
        _denormalized_patient_value_maps(train_dataset, binner),
        validation_explanations=run / "explanations" / "val",
        validation_values=_denormalized_patient_value_maps(val_dataset, binner),
        output_dir=out,
        latent_dim=config.autoencoder_latent_dim,
        epochs=config.autoencoder_epochs,
        learning_rate=config.autoencoder_learning_rate,
        batch_size=config.autoencoder_batch_size,
        device=full_config.train.device,
        early_stopping_patience=config.autoencoder_early_stopping_patience,
        patch_size=full_config.model.patch_size,
    )
    embedded = create_explanation_value_embeddings(
        explanation_dir,
        _denormalized_patient_value_maps(dataset, binner),
        model=trained["model"],
        preprocessor=trained["preprocessor"],
        output_dir=out,
        batch_size=config.autoencoder_batch_size,
        device=full_config.train.device,
    )
    cluster_autoencoder_embeddings(
        embedded["embedding_frame"],
        explanations=embedded["explanations"],
        predictions=_find_predictions_path(run, args.split),
        n_clusters=n_clusters,
        method=method,
        output_dir=out,
        autoencoder_metrics=trained["metrics"] | {"cluster_loss": embedded["loss"]},
        autoencoder_metadata={
            **trained["metadata"],
            "cluster": {"n_patients": len(embedded["patient_ids"]), **embedded["metadata"]},
        },
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )


def cmd_plot(args) -> None:
    """Render cluster-level value heatmaps and optionally per-patient value heatmaps."""
    run = Path(args.run)
    config = load_config(args.config).cluster
    plot_mode = args.plot_mode or config.plot_mode
    importance_threshold = args.importance_threshold if args.importance_threshold is not None else config.importance_threshold
    show_values = args.show_values or config.show_values
    normal_ranges = args.normal_ranges or config.normal_ranges_path
    if normal_ranges is None and (args.use_normal_ranges or config.use_normal_ranges):
        normal_ranges = Path(__file__).with_name("normal_ranges.json")
    binner = TimeSeriesBinner.load(run / "binner.json")
    dataset = load_split(run / f"{args.split}.npz")
    assignments_path = run / "clusters" / args.split / "cluster_assignments.csv"
    cluster_dir = run / "clusters" / args.split
    value_dir = run / "cluster_values" / args.split
    heatmap_dir = run / "cluster_heatmaps" / args.split
    matrices_by_cluster = aggregate_cluster_value_matrices(dataset, assignments_path, binner, output_dir=value_dir)
    counts_by_cluster = cluster_assignment_counts(assignments_path)
    importance_style = _importance_style(plot_mode)
    importance_by_cluster = _cluster_importance_matrices(cluster_dir) if importance_style is not None else {}
    matrices = list(matrices_by_cluster.values())
    if matrices:
        vmin, vmax = value_ranges_by_variable(matrices)
        for cluster_key, matrix in matrices_by_cluster.items():
            plot_value_heatmap(
                matrix,
                binner.variable_vocab_,
                binner.time_bins_,
                _cluster_heatmap_path(heatmap_dir, cluster_key),
                title=_cluster_title(cluster_key, counts_by_cluster.get(cluster_key)),
                vmin=vmin,
                vmax=vmax,
                importance_matrix=_importance_for_key(importance_by_cluster, cluster_key),
                importance_style=importance_style or "opacity",
                importance_threshold=importance_threshold,
                show_values=True,
                normal_ranges=normal_ranges,
            )
    _plot_embedding_centroids(run, args.split, dataset, binner, normal_ranges, show_values)
    if args.instances:
        instance_dir = run / "instance_heatmaps" / args.split
        maps = _denormalized_patient_value_maps(dataset, binner)
        if maps:
            vmin, vmax = value_ranges_by_variable(list(maps.values()))
            for patient_id, matrix in maps.items():
                plot_value_heatmap(
                    matrix,
                    binner.variable_vocab_,
                    binner.time_bins_,
                    instance_dir / f"{patient_id}.png",
                    title=patient_id,
                    vmin=vmin,
                    vmax=vmax,
                    show_values=True,
                    normal_ranges=normal_ranges,
                )


def _plot_embedding_centroids(
    run: Path,
    split: str,
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    normal_ranges: str | Path | None,
    show_values: bool,
) -> None:
    centroids_path = run / "clusters" / split / "cluster_centroids.csv"
    if not centroids_path.exists():
        return
    centroid_frame = pd.read_csv(centroids_path)
    if centroid_frame.empty:
        return
    output_dir = run / "cluster_centroid_heatmaps" / split
    for row in centroid_frame.to_dict("records"):
        patient_id = str(row["patient_id"])
        prefix_parts = []
        if "predicted_label" in row and pd.notna(row["predicted_label"]):
            prefix_parts.append(_safe_path_component(str(row["predicted_label"])))
        prefix_parts.append(f"cluster_{int(row['cluster'])}")
        output = output_dir.joinpath(*prefix_parts)
        matrix = patient_value_matrix(dataset, binner, patient_id)
        plot_value_heatmap(
            matrix,
            binner.variable_vocab_,
            binner.time_bins_,
            output / f"centroid_patient_{_safe_path_component(patient_id)}.png",
            title=_centroid_title(patient_id, row),
            show_values=True,
            normal_ranges=normal_ranges,
        )


def _denormalized_patient_value_maps(dataset: BinnedTimeSeriesDataset, binner: TimeSeriesBinner) -> dict[str, np.ndarray]:
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


def _cluster_importance_matrices(cluster_dir: Path) -> dict[int | tuple[str, int], np.ndarray]:
    matrices: dict[int | tuple[str, int], np.ndarray] = {}
    for path in sorted(cluster_dir.glob("cluster_*.npy")):
        try:
            cluster = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        matrices[cluster] = np.load(path)
    for class_dir in sorted(path for path in cluster_dir.iterdir() if path.is_dir()):
        for path in sorted(class_dir.glob("cluster_*.npy")):
            try:
                cluster = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            matrices[(class_dir.name, cluster)] = np.load(path)
    return matrices


def _find_predictions_path(run_dir: Path, split: str) -> Path | None:
    split_predictions = run_dir / f"{split}_predictions.csv"
    if split_predictions.exists():
        return split_predictions
    legacy_predictions = run_dir / "predictions.csv"
    if legacy_predictions.exists():
        return legacy_predictions
    return None


def _autoencoder_required_splits(split: str) -> list[str]:
    return list(dict.fromkeys(["train", "val", split]))


def _cluster_heatmap_path(output_dir: Path, key: int | tuple[str, int]) -> Path:
    if isinstance(key, tuple):
        predicted_label, cluster = key
        return output_dir / _safe_path_component(predicted_label) / f"cluster_{cluster}.png"
    return output_dir / f"cluster_{key}.png"


def _cluster_title(key: int | tuple[str, int], count: int | None = None) -> str:
    suffix = f" (n={count})" if count is not None else ""
    if isinstance(key, tuple):
        predicted_label, cluster = key
        return f"Predicted class {predicted_label}: {_cluster_label(cluster)}{suffix}"
    return f"{_cluster_label(key)}{suffix}"


def _centroid_title(patient_id: str, row: dict[str, object]) -> str:
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return f"Predicted class {row['predicted_label']}: centroid representative {patient_id}"
    return f"Centroid representative {patient_id}"


def _cluster_label(cluster: int) -> str:
    return "noise (-1)" if cluster == -1 else f"cluster_{cluster}"


def _importance_for_key(
    importance_by_cluster: dict[int | tuple[str, int], np.ndarray],
    key: int | tuple[str, int],
) -> np.ndarray | None:
    matrix = importance_by_cluster.get(key)
    if matrix is not None or not isinstance(key, tuple):
        return matrix
    predicted_label, cluster = key
    return importance_by_cluster.get((_safe_path_component(predicted_label), cluster))


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "class"


def _importance_style(plot_mode: str) -> str | None:
    if plot_mode == "value_with_importance_opacity":
        return "opacity"
    if plot_mode == "value_with_importance_border":
        return "border"
    if plot_mode == "value":
        return None
    raise ValueError("plot_mode must be 'value', 'value_with_importance_opacity', or 'value_with_importance_border'.")


if __name__ == "__main__":
    main()
