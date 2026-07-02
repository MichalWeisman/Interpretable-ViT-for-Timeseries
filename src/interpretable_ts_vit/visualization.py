"""Heatmap rendering utilities for value and explanation matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .binning import TimeSeriesBinner
from .data import BinnedTimeSeriesDataset


def plot_explanation_heatmap(
    matrix: np.ndarray,
    variables: Sequence[str],
    time_bins: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a variable-by-time explanation heatmap to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(8, min(24, len(time_bins) * 0.28))
    fig_height = max(4, min(18, len(variables) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(variables)
    tick_count = min(10, len(time_bins))
    if tick_count:
        positions = np.linspace(0, len(time_bins) - 1, tick_count, dtype=int)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(time_bins[i]) for i in positions], rotation=45, ha="right")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Variable")
    if title:
        ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Explanation score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def aggregate_cluster_value_matrices(
    dataset: BinnedTimeSeriesDataset,
    assignments: pd.DataFrame | str | Path,
    binner: TimeSeriesBinner,
    output_dir: str | Path | None = None,
) -> dict[int, np.ndarray]:
    """Aggregate observed clinical values for each cluster.

    The prepared tensor stores normalized values in channel 0 and an observed
    mask in channel 1. Missing values are represented as `D=0, M=0`, so a plain
    mean over `D` would incorrectly treat missing observations as real zeros.
    This function denormalizes observed cells back to the training-scale value
    for each variable, then computes:

    `mean observed value = sum(value * mask) / sum(mask)`

    Cells with no observations in a cluster are returned as `np.nan`, which the
    heatmap renderer displays as gray.
    """
    assignment_frame = _read_assignments(assignments)
    if dataset.patient_ids is None:
        raise ValueError("Dataset must include patient_ids to aggregate cluster values.")

    x = dataset.x.detach().cpu().numpy()
    values = x[:, 0].astype(np.float64)
    mask = x[:, 1].astype(np.float64)
    means = np.array([binner.means_.get(variable, 0.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    stds = np.array([binner.stds_.get(variable, 1.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    raw_values = values * stds[None, :, :] + means[None, :, :]

    patient_to_index = {str(patient_id): idx for idx, patient_id in enumerate(dataset.patient_ids)}
    aggregates: dict[int, np.ndarray] = {}
    for cluster, group in assignment_frame.groupby("cluster", sort=True):
        indices = [patient_to_index[str(patient_id)] for patient_id in group["patient_id"] if str(patient_id) in patient_to_index]
        if not indices:
            continue
        cluster_values = raw_values[indices]
        cluster_mask = mask[indices]
        numerator = np.sum(cluster_values * cluster_mask, axis=0)
        denominator = np.sum(cluster_mask, axis=0)
        matrix = np.full_like(numerator, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=matrix, where=denominator > 0)
        aggregates[int(cluster)] = matrix

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for cluster, matrix in aggregates.items():
            np.save(out / f"cluster_{cluster}.npy", matrix)
    return aggregates


def plot_value_heatmap(
    matrix: np.ndarray,
    variables: Sequence[str],
    time_bins: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    importance_matrix: np.ndarray | None = None,
    importance_quantile: float = 0.75,
) -> None:
    """Save a variable-by-time heatmap of mean observed clinical values.

    If `importance_matrix` is provided, the value heatmap is overlaid with
    black contours around cells whose mean importance is in the upper
    `importance_quantile` of finite importance values. Color always represents
    clinical value; the contour represents model importance.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(8, min(24, len(time_bins) * 0.28))
    fig_height = max(4, min(18, len(variables) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("viridis").with_extremes(bad="#d9d9d9")
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if importance_matrix is not None:
        _draw_importance_contours(ax, np.asarray(importance_matrix), importance_quantile)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(variables)
    tick_count = min(10, len(time_bins))
    if tick_count:
        positions = np.linspace(0, len(time_bins) - 1, tick_count, dtype=int)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(time_bins[i]) for i in positions], rotation=45, ha="right")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Variable")
    if title:
        ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean observed value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_importance_contours(ax, importance_matrix: np.ndarray, quantile: float) -> None:
    if importance_matrix.shape[0] < 2 or importance_matrix.shape[1] < 2:
        return
    finite = importance_matrix[np.isfinite(importance_matrix)]
    if finite.size == 0:
        return
    threshold = float(np.quantile(finite, quantile))
    if not np.isfinite(threshold) or float(np.nanmax(finite)) <= threshold:
        return
    y = np.arange(importance_matrix.shape[0])
    x = np.arange(importance_matrix.shape[1])
    ax.contour(x, y, importance_matrix, levels=[threshold], colors="black", linewidths=0.8)
    ax.text(
        0.99,
        1.01,
        f"black contour: top {int((1 - quantile) * 100)}% importance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="black",
    )


def _read_assignments(assignments: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(assignments, pd.DataFrame):
        frame = assignments.copy()
    else:
        frame = pd.read_csv(assignments)
    required = {"patient_id", "cluster"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cluster assignments are missing columns: {sorted(missing)}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["cluster"] = frame["cluster"].astype(int)
    return frame
