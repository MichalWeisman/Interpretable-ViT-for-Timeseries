"""Command-line entrypoint for prepare/train/explain/cluster/plot workflows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .binning import TimeSeriesBinner
from .clustering import cluster_explanations
from .config import load_config
from .data import BinnedTimeSeriesDataset
from .datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from .explain import explain_model
from .io import load_model, load_split, save_metadata, save_predictions, save_split
from .model import ViTConfig, ViTTimeSeriesClassifier
from .training import predict_model, train_model
from .visualization import plot_explanation_heatmap


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and dispatch to the selected workflow command."""
    parser = argparse.ArgumentParser(prog="tsvit")
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
    explain.add_argument("--method")
    explain.add_argument("--target-class", type=int)

    cluster = sub.add_parser("cluster")
    cluster.add_argument("--run", required=True)
    cluster.add_argument("--split", default="test")
    cluster.add_argument("--n-clusters", type=int)

    plot = sub.add_parser("plot")
    plot.add_argument("--run", required=True)
    plot.add_argument("--split", default="test")
    plot.add_argument("--instances", action="store_true")

    args = parser.parse_args(argv)
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
    records = pd.read_csv(args.records)
    labels = pd.read_csv(args.labels)
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


def cmd_prepare_mimic_hypotension(args) -> None:
    """Create generic records/labels files for MIMIC-IV hypotension prediction."""
    config = MIMICHypotensionConfig(
        mimic_path=args.mimic_path,
        observation_hours=args.observation_hours,
        prediction_hours=args.prediction_hours,
        hypotension_threshold=args.threshold,
        chunk_size=args.chunk_size,
        max_stays=args.max_stays,
        min_observations=args.min_observations,
        require_full_prediction_window=not args.allow_short_prediction_window,
        require_outcome_measurement=not args.allow_missing_outcome_measurement,
    )
    prepared = MIMICIVHypotensionAdapter(config).prepare()
    prepared.save(args.out)


def cmd_train(args) -> None:
    """Train the ViT on prepared data and save model/metrics/predictions."""
    config = load_config(args.config)
    data_dir = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    method = args.method or config.explain.method
    out = run / "explanations" / args.split
    explain_model(model, dataset, method=method, target_class=args.target_class, output_dir=out)


def cmd_cluster(args) -> None:
    """Cluster saved explanation maps and save cluster-level averages."""
    run = Path(args.run)
    n_clusters = args.n_clusters or load_config(None).cluster.n_clusters
    explanation_dir = run / "explanations" / args.split
    out = run / "clusters" / args.split
    cluster_explanations(explanation_dir, n_clusters=n_clusters, output_dir=out)


def cmd_plot(args) -> None:
    """Render cluster-level heatmaps and optionally per-instance heatmaps."""
    run = Path(args.run)
    binner = TimeSeriesBinner.load(run / "binner.json")
    cluster_dir = run / "clusters" / args.split
    heatmap_dir = run / "cluster_heatmaps" / args.split
    matrices = [np.load(path) for path in sorted(cluster_dir.glob("cluster_*.npy"))]
    if matrices:
        vmin = min(float(matrix.min()) for matrix in matrices)
        vmax = max(float(matrix.max()) for matrix in matrices)
        for path, matrix in zip(sorted(cluster_dir.glob("cluster_*.npy")), matrices):
            plot_explanation_heatmap(
                matrix,
                binner.variable_vocab_,
                binner.time_bins_,
                heatmap_dir / f"{path.stem}.png",
                title=path.stem,
                vmin=vmin,
                vmax=vmax,
            )
    if args.instances:
        explanation_dir = run / "explanations" / args.split
        instance_dir = run / "instance_heatmaps" / args.split
        paths = sorted(explanation_dir.glob("*.npy"))
        if paths:
            maps = [np.load(path) for path in paths]
            vmin = min(float(matrix.min()) for matrix in maps)
            vmax = max(float(matrix.max()) for matrix in maps)
            for path, matrix in zip(paths, maps):
                plot_explanation_heatmap(
                    matrix,
                    binner.variable_vocab_,
                    binner.time_bins_,
                    instance_dir / f"{path.stem}.png",
                    title=path.stem,
                    vmin=vmin,
                    vmax=vmax,
                )


if __name__ == "__main__":
    main()
