"""Heatmap rendering utilities for value and explanation matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    importance_style: str = "opacity",
    min_importance_alpha: float = 0.15,
    importance_threshold: float | None = None,
) -> None:
    """Save a variable-by-time heatmap of mean observed clinical values.

    Color represents clinical value using a blue-low to red-high scale. If
    `importance_matrix` is provided, `importance_style` controls the visual
    encoding: `"opacity"` makes important cells more opaque, while `"border"`
    uses thicker cell borders for more important cells. The colorbar
    intentionally has no numeric ticks because variables can have different
    clinical units. `importance_threshold` is an optional quantile in `[0, 1]`;
    for example, `0.8` emphasizes only cells at or above the 80th percentile
    of finite importance scores.
    """
    if importance_style not in {"opacity", "border"}:
        raise ValueError("importance_style must be either 'opacity' or 'border'.")
    _validate_importance_threshold(importance_threshold)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(8, min(24, len(time_bins) * 0.28))
    fig_height = max(4, min(18, len(variables) * 0.32))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("coolwarm").with_extremes(bad="#d9d9d9")
    importance = np.asarray(importance_matrix) if importance_matrix is not None else None
    alpha = (
        _importance_alpha(importance, matrix, min_importance_alpha, importance_threshold)
        if importance is not None and importance_style == "opacity"
        else None
    )
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    if importance is not None and importance_style == "border":
        _draw_importance_borders(ax, importance, matrix, importance_threshold)
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(variables)
    positions, labels, granularity = _relative_time_ticks(time_bins)
    if len(positions):
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel(f"Relative time ({granularity} bins)" if granularity else "Relative time")
    ax.set_ylabel("Variable")
    if title:
        ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, label="Mean Observed Value")
    colorbar.set_ticks([])
    colorbar.ax.text(0.5, -0.02, "Low", transform=colorbar.ax.transAxes, ha="center", va="top")
    colorbar.ax.text(0.5, 1.02, "High", transform=colorbar.ax.transAxes, ha="center", va="bottom")
    if importance_matrix is not None:
        label = "opacity: model importance" if importance_style == "opacity" else "border width: model importance"
        if importance_threshold is not None:
            label = f"{label} (top {int(round((1.0 - importance_threshold) * 100))}%)"
        ax.text(
            0.99,
            1.01,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _relative_time_ticks(time_bins: Sequence[str]) -> tuple[np.ndarray, list[str], str | None]:
    if len(time_bins) == 0:
        return np.array([], dtype=int), [], None
    tick_count = min(10, len(time_bins))
    positions = np.unique(np.linspace(0, len(time_bins) - 1, tick_count, dtype=int))
    parsed = pd.to_datetime(list(time_bins), errors="coerce")
    if len(parsed) < 2 or pd.isna(parsed[0]) or pd.isna(parsed[1]):
        return positions, [str(int(position)) for position in positions], None
    step = parsed[1] - parsed[0]
    granularity = _format_timedelta(step)
    labels = [_format_timedelta(parsed[int(position)] - parsed[0]) for position in positions]
    return positions, labels, granularity


def _format_timedelta(delta) -> str:
    total_seconds = int(pd.Timedelta(delta).total_seconds())
    if total_seconds == 0:
        return "0"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def _validate_importance_threshold(importance_threshold: float | None) -> None:
    if importance_threshold is None:
        return
    if not 0.0 <= importance_threshold <= 1.0:
        raise ValueError("importance_threshold must be a quantile between 0 and 1.")


def _importance_cutoff(importance_matrix: np.ndarray, importance_threshold: float | None) -> float | None:
    if importance_threshold is None:
        return None
    finite = importance_matrix[np.isfinite(importance_matrix)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, importance_threshold))


def _importance_alpha(
    importance_matrix: np.ndarray,
    value_matrix: np.ndarray,
    min_alpha: float,
    importance_threshold: float | None,
) -> np.ndarray:
    alpha = np.ones_like(value_matrix, dtype=np.float64)
    if importance_matrix.shape != value_matrix.shape:
        return alpha
    cutoff = _importance_cutoff(importance_matrix, importance_threshold)
    finite_mask = np.isfinite(importance_matrix)
    if cutoff is not None:
        finite_mask &= importance_matrix >= cutoff
    finite = importance_matrix[finite_mask]
    if finite.size == 0:
        return alpha
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        alpha[:] = min_alpha
        alpha[finite_mask] = 1.0
        alpha[~np.isfinite(value_matrix)] = 1.0
        return alpha
    scaled = (importance_matrix - low) / (high - low)
    alpha = min_alpha + (1.0 - min_alpha) * np.clip(scaled, 0.0, 1.0)
    if cutoff is not None:
        alpha[~finite_mask] = min_alpha
    alpha[~np.isfinite(value_matrix)] = 1.0
    return alpha


def _draw_importance_borders(
    ax,
    importance_matrix: np.ndarray,
    value_matrix: np.ndarray,
    importance_threshold: float | None,
) -> None:
    if importance_matrix.shape != value_matrix.shape:
        return
    cutoff = _importance_cutoff(importance_matrix, importance_threshold)
    finite_mask = np.isfinite(importance_matrix)
    if cutoff is not None:
        finite_mask &= importance_matrix >= cutoff
    finite = importance_matrix[finite_mask]
    if finite.size == 0:
        return
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        scaled = np.where(finite_mask, 1.0, np.nan)
    else:
        scaled = np.clip((importance_matrix - low) / (high - low), 0.0, 1.0)
        scaled[~finite_mask] = np.nan
    for row in range(value_matrix.shape[0]):
        for col in range(value_matrix.shape[1]):
            if not np.isfinite(value_matrix[row, col]) or not np.isfinite(scaled[row, col]):
                continue
            linewidth = 0.05 + 1.8 * float(scaled[row, col])
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=linewidth,
                )
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
