"""Heatmap rendering utilities for value and explanation matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .binning import TimeSeriesBinner
from .data import BinnedTimeSeriesDataset


VALUE_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "value_blue_purple_red",
    ["#1e88e5", "#7b2cbf", "#ff0051"],
)
NORMAL_RANGE_CMAP = LinearSegmentedColormap.from_list(
    "value_low_normal_high",
    ["#1e88e5", "#f2f2f2", "#ff0051"],
)


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
) -> dict[int | tuple[str, int], np.ndarray]:
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
    group_columns = ["predicted_label", "cluster"] if "predicted_label" in assignment_frame.columns else ["cluster"]
    aggregates: dict[int | tuple[str, int], np.ndarray] = {}
    for group_key, group in assignment_frame.groupby(group_columns, sort=True):
        indices = [patient_to_index[str(patient_id)] for patient_id in group["patient_id"] if str(patient_id) in patient_to_index]
        if not indices:
            continue
        cluster_values = raw_values[indices]
        cluster_mask = mask[indices]
        numerator = np.sum(cluster_values * cluster_mask, axis=0)
        denominator = np.sum(cluster_mask, axis=0)
        matrix = np.full_like(numerator, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=matrix, where=denominator > 0)
        key = _aggregate_key(group_key, group_columns)
        aggregates[key] = matrix

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for key, matrix in aggregates.items():
            path = _cluster_matrix_path(out, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
    return aggregates


def cluster_assignment_counts(assignments: pd.DataFrame | str | Path) -> dict[int | tuple[str, int], int]:
    """Return the number of assigned patients in each cluster key."""
    assignment_frame = _read_assignments(assignments)
    group_columns = ["predicted_label", "cluster"] if "predicted_label" in assignment_frame.columns else ["cluster"]
    counts: dict[int | tuple[str, int], int] = {}
    for group_key, group in assignment_frame.groupby(group_columns, sort=True):
        counts[_aggregate_key(group_key, group_columns)] = int(len(group))
    return counts


def patient_value_matrix(
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    patient_id: str,
) -> np.ndarray:
    """Return one patient's denormalized observed-value matrix.

    Missing cells are returned as `np.nan` so heatmaps render them as gray
    instead of treating the preprocessing zero-fill as a clinical value.
    """
    if dataset.patient_ids is None:
        raise ValueError("Dataset must include patient_ids to select a patient.")
    patient_ids = [str(pid) for pid in dataset.patient_ids]
    patient_id = str(patient_id)
    if patient_id not in patient_ids:
        raise KeyError(f"Patient id {patient_id!r} was not found in the dataset.")

    idx = patient_ids.index(patient_id)
    x = dataset.x.detach().cpu().numpy()
    values = x[idx, 0].astype(np.float64)
    mask = x[idx, 1].astype(bool)
    means = np.array([binner.means_.get(variable, 0.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    stds = np.array([binner.stds_.get(variable, 1.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    raw_values = values * stds + means
    raw_values[~mask] = np.nan
    return raw_values


def load_patient_explanation_matrix(
    explanations: Mapping[str, np.ndarray] | str | Path,
    patient_id: str,
) -> np.ndarray:
    """Load one patient's explanation matrix from a mapping, `.npy` file, or directory."""
    patient_id = str(patient_id)
    if isinstance(explanations, Mapping):
        if patient_id not in explanations:
            raise KeyError(f"Patient id {patient_id!r} was not found in the explanations.")
        return np.asarray(explanations[patient_id], dtype=np.float64)

    path = Path(explanations)
    explanation_path = path if path.is_file() else path / f"{patient_id}.npy"
    if not explanation_path.exists():
        raise FileNotFoundError(explanation_path)
    return np.load(explanation_path).astype(np.float64)


def patient_class_frame(
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    predictions: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Return patient ids with true labels and optional predicted labels."""
    if dataset.patient_ids is None:
        raise ValueError("Dataset must include patient_ids.")
    frame = pd.DataFrame({"patient_id": [str(patient_id) for patient_id in dataset.patient_ids]})
    if dataset.y is not None:
        y = dataset.y.detach().cpu().numpy().astype(int)
        labels = [binner.index_to_label_[idx] if 0 <= idx < len(binner.index_to_label_) else str(idx) for idx in y]
        frame["true_label"] = [str(label) for label in labels]
    if predictions is not None:
        pred_frame = predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.read_csv(predictions)
        if "patient_id" not in pred_frame.columns:
            raise ValueError("Predictions are missing column: patient_id")
        pred_frame = pred_frame.copy()
        pred_frame["patient_id"] = pred_frame["patient_id"].astype(str)
        keep_columns = ["patient_id"] + [col for col in pred_frame.columns if col != "patient_id"]
        frame = frame.merge(pred_frame[keep_columns], on="patient_id", how="left")
        if "predicted_label" in frame.columns:
            frame["predicted_label"] = frame["predicted_label"].astype("string")
    return frame


def select_patient_ids(
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    *,
    predictions: pd.DataFrame | str | Path | None = None,
    true_label: str | None = None,
    predicted_label: str | None = None,
    patient_ids: Sequence[str] | None = None,
    n: int | None = None,
    random_state: int | None = None,
) -> list[str]:
    """Select patient ids by true class, predicted class, explicit ids, or both."""
    frame = patient_class_frame(dataset, binner, predictions)
    if patient_ids is not None:
        wanted = {str(patient_id) for patient_id in patient_ids}
        frame = frame[frame["patient_id"].isin(wanted)]
    if true_label is not None:
        if "true_label" not in frame.columns:
            raise ValueError("Dataset does not include labels for true_label selection.")
        frame = frame[frame["true_label"].astype(str) == str(true_label)]
    if predicted_label is not None:
        if "predicted_label" not in frame.columns:
            raise ValueError("Predictions with a predicted_label column are required for predicted_label selection.")
        frame = frame[frame["predicted_label"].astype(str) == str(predicted_label)]
    if n is not None:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if random_state is None:
            frame = frame.head(n)
        else:
            frame = frame.sample(n=min(n, len(frame)), random_state=random_state)
    return frame["patient_id"].astype(str).tolist()


def plot_patient_matrices(
    patient_id: str,
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    explanations: Mapping[str, np.ndarray] | str | Path,
    output_dir: str | Path,
    *,
    show_values: bool = False,
    explanation_threshold: float | None = None,
    explanation_threshold_mode: str = "absolute",
    plot_explanation: bool = True,
    value_vmin: float | Sequence[float] | np.ndarray | None = None,
    value_vmax: float | Sequence[float] | np.ndarray | None = None,
    explanation_vmin: float | None = None,
    explanation_vmax: float | None = None,
    normal_ranges: Mapping[str, object] | str | Path | None = None,
) -> dict[str, Path]:
    """Plot one patient's observed-value matrix and optionally explanation matrix.

    If `explanation_threshold` is provided, values are shown only where the
    explanation score passes the threshold; other cells are set to `np.nan` and
    render as gray. `explanation_threshold_mode` can be `"absolute"` for raw
    scores or `"quantile"` for per-patient quantiles.
    """
    patient_id = str(patient_id)
    output_dir = Path(output_dir)
    explanation_matrix = load_patient_explanation_matrix(explanations, patient_id)
    value_matrix = patient_value_matrix(dataset, binner, patient_id)
    value_matrix = filter_value_matrix_by_explanation(
        value_matrix,
        explanation_matrix,
        explanation_threshold,
        explanation_threshold_mode,
    )
    if value_vmin is None and value_vmax is None:
        value_vmin, value_vmax = value_ranges_by_variable([value_matrix])

    stem = _safe_path_component(patient_id)
    paths = {"values": output_dir / f"{stem}_values.png"}
    if plot_explanation:
        paths["explanation"] = output_dir / f"{stem}_explanation.png"
        plot_explanation_heatmap(
            explanation_matrix,
            binner.variable_vocab_,
            binner.time_bins_,
            paths["explanation"],
            title=f"Patient {patient_id}: model explanation",
            vmin=explanation_vmin,
            vmax=explanation_vmax,
        )
    title = f"Patient {patient_id}: observed values"
    if explanation_threshold is not None:
        title = f"{title} where explanation >= {_threshold_label(explanation_threshold, explanation_threshold_mode)}"
    plot_value_heatmap(
        value_matrix,
        binner.variable_vocab_,
        binner.time_bins_,
        paths["values"],
        title=title,
        vmin=value_vmin,
        vmax=value_vmax,
        show_values=show_values,
        normal_ranges=normal_ranges,
    )
    return paths


def plot_patient_matrix_comparison(
    patient_ids: Sequence[str],
    dataset: BinnedTimeSeriesDataset,
    binner: TimeSeriesBinner,
    explanations: Mapping[str, np.ndarray] | str | Path,
    output_dir: str | Path,
    *,
    show_values: bool = False,
    explanation_threshold: float | None = None,
    explanation_threshold_mode: str = "absolute",
    plot_explanation: bool = False,
    shared_value_scale: bool = True,
    shared_explanation_scale: bool = True,
    normal_ranges: Mapping[str, object] | str | Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Plot several patients with optional shared scales for comparison."""
    patient_ids = [str(patient_id) for patient_id in patient_ids]
    if not patient_ids:
        return {}

    value_vmin = value_vmax = None
    if shared_value_scale:
        value_matrices = [
            filter_value_matrix_by_explanation(
                patient_value_matrix(dataset, binner, pid),
                load_patient_explanation_matrix(explanations, pid),
                explanation_threshold,
                explanation_threshold_mode,
            )
            for pid in patient_ids
        ]
        value_vmin, value_vmax = value_ranges_by_variable(value_matrices)

    explanation_vmin = explanation_vmax = None
    if shared_explanation_scale:
        explanation_matrices = [load_patient_explanation_matrix(explanations, pid) for pid in patient_ids]
        finite_parts = [matrix[np.isfinite(matrix)].reshape(-1) for matrix in explanation_matrices if np.isfinite(matrix).any()]
        if finite_parts:
            finite = np.concatenate(finite_parts)
            explanation_vmin = float(np.min(finite))
            explanation_vmax = float(np.max(finite))

    output_dir = Path(output_dir)
    return {
        patient_id: plot_patient_matrices(
            patient_id,
            dataset,
            binner,
            explanations,
            output_dir / _safe_path_component(patient_id),
            show_values=show_values,
            explanation_threshold=explanation_threshold,
            explanation_threshold_mode=explanation_threshold_mode,
            plot_explanation=plot_explanation,
            value_vmin=value_vmin,
            value_vmax=value_vmax,
            explanation_vmin=explanation_vmin,
            explanation_vmax=explanation_vmax,
            normal_ranges=normal_ranges,
        )
        for patient_id in patient_ids
    }


def filter_value_matrix_by_explanation(
    value_matrix: np.ndarray,
    explanation_matrix: np.ndarray,
    explanation_threshold: float | None,
    threshold_mode: str = "absolute",
) -> np.ndarray:
    """Return values only where explanation scores pass the threshold."""
    value_matrix = np.asarray(value_matrix, dtype=np.float64)
    explanation_matrix = np.asarray(explanation_matrix, dtype=np.float64)
    if value_matrix.shape != explanation_matrix.shape:
        raise ValueError("value_matrix and explanation_matrix must have the same shape.")
    if explanation_threshold is None:
        return value_matrix.copy()
    if threshold_mode not in {"absolute", "quantile"}:
        raise ValueError("threshold_mode must be 'absolute' or 'quantile'.")
    threshold = float(explanation_threshold)
    if threshold_mode == "quantile":
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("quantile explanation_threshold must be between 0 and 1.")
        finite = explanation_matrix[np.isfinite(explanation_matrix)]
        if finite.size == 0:
            return np.full_like(value_matrix, np.nan, dtype=np.float64)
        threshold = float(np.quantile(finite, threshold))
    mask = np.isfinite(explanation_matrix) & (explanation_matrix >= threshold)
    return np.where(mask, value_matrix, np.nan)


def _threshold_label(threshold: float, threshold_mode: str) -> str:
    if threshold_mode == "quantile":
        return f"q{threshold:.2f}"
    return f"{threshold:g}"


def load_normal_ranges(normal_ranges: Mapping[str, object] | str | Path | None = None) -> dict[str, dict[str, object]]:
    """Load clinical low/normal/high plotting ranges."""
    if normal_ranges is None:
        path = Path(__file__).with_name("normal_ranges.json")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    elif isinstance(normal_ranges, Mapping):
        raw = dict(normal_ranges)
    else:
        with Path(normal_ranges).open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    return {str(variable): dict(spec) for variable, spec in raw.items()}


def normal_range_status_matrix(
    matrix: np.ndarray,
    variables: Sequence[str],
    normal_ranges: Mapping[str, object] | str | Path | None = None,
) -> np.ndarray:
    """Convert values to -1 low, 0 normal, 1 high using per-variable ranges."""
    ranges = load_normal_ranges(normal_ranges)
    matrix = np.asarray(matrix, dtype=np.float64)
    status = np.full_like(matrix, np.nan, dtype=np.float64)
    for row, variable in enumerate(variables):
        spec = ranges.get(str(variable))
        if spec is None:
            continue
        low = spec.get("low")
        high = spec.get("high")
        if low is None or high is None:
            continue
        values = matrix[row]
        status[row] = np.where(values < float(low), -1.0, np.where(values > float(high), 1.0, 0.0))
        status[row, ~np.isfinite(values)] = np.nan
    return status


def _normal_range_lines(variables: Sequence[str], ranges: Mapping[str, Mapping[str, object]]) -> list[str]:
    lines = []
    for variable in variables:
        spec = ranges.get(str(variable))
        if spec is None or spec.get("low") is None or spec.get("high") is None:
            continue
        label = str(variable).replace("_", " ")
        unit = f" {spec['unit']}" if spec.get("unit") else ""
        lines.append(f"{label}: {spec['low']}-{spec['high']}{unit}")
    return lines


def plot_value_heatmap(
    matrix: np.ndarray,
    variables: Sequence[str],
    time_bins: Sequence[str],
    output_path: str | Path,
    title: str | None = None,
    vmin: float | Sequence[float] | np.ndarray | None = None,
    vmax: float | Sequence[float] | np.ndarray | None = None,
    importance_matrix: np.ndarray | None = None,
    importance_style: str = "opacity",
    min_importance_alpha: float = 0.15,
    importance_threshold: float | None = None,
    show_values: bool = False,
    value_text_format: str = ".1f",
    normal_ranges: Mapping[str, object] | str | Path | None = None,
) -> None:
    """Save a variable-by-time heatmap of mean observed clinical values.

    Color represents clinical value using a blue-low to red-high scale. Scalar
    `vmin`/`vmax` values use one shared numeric scale; row-shaped values use
    one low-to-high scale per variable, which is usually the right choice when
    rows have different clinical units. If
    `importance_matrix` is provided, `importance_style` controls the visual
    encoding: `"opacity"` makes important cells more opaque, while `"border"`
    uses thicker cell borders for more important cells. The colorbar
    intentionally has no numeric ticks because variables can have different
    clinical units. `importance_threshold` is an optional quantile in `[0, 1]`;
    for example, `0.8` emphasizes only cells at or above the 80th percentile
    of finite importance scores. Set `show_values=True` to annotate each
    finite cell with its mean observed value.
    """
    if importance_style not in {"opacity", "border"}:
        raise ValueError("importance_style must be either 'opacity' or 'border'.")
    _validate_importance_threshold(importance_threshold)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranges = load_normal_ranges(normal_ranges) if normal_ranges is not None else None
    fig_width = max(8, min(24, len(time_bins) * 0.28))
    range_lines = _normal_range_lines(variables, ranges) if ranges is not None else []
    if ranges is not None:
        fig_width += max(4.0, min(7.0, max((len(line) for line in range_lines), default=18) * 0.07))
    fig_height = max(4, min(18, len(variables) * 0.32))
    if ranges is not None:
        fig, (ax, cax, ranges_ax) = plt.subplots(
            1,
            3,
            figsize=(fig_width, fig_height),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 0.035, 0.28], "wspace": 0.18},
        )
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cax = None
        ranges_ax = None
    cmap = (NORMAL_RANGE_CMAP if ranges is not None else VALUE_HEATMAP_CMAP).with_extremes(bad="#d9d9d9")
    importance = np.asarray(importance_matrix) if importance_matrix is not None else None
    alpha = (
        _importance_alpha(importance, matrix, min_importance_alpha, importance_threshold)
        if importance is not None and importance_style == "opacity"
        else None
    )
    if ranges is not None:
        display_matrix = normal_range_status_matrix(matrix, variables, ranges)
        display_vmin, display_vmax = -1.0, 1.0
        value_scale_label = "Clinical range status"
    else:
        display_matrix, display_vmin, display_vmax, value_scale_label = _display_value_matrix(matrix, vmin, vmax)
    image = ax.imshow(
        np.ma.masked_invalid(display_matrix),
        aspect="auto",
        cmap=cmap,
        vmin=display_vmin,
        vmax=display_vmax,
        alpha=alpha,
    )
    if importance is not None and importance_style == "border":
        _draw_importance_borders(ax, importance, matrix, importance_threshold)
    if show_values:
        _annotate_heatmap_values(ax, image, matrix, display_matrix, value_text_format)
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
    colorbar = fig.colorbar(image, ax=ax, cax=cax, label=value_scale_label)
    if ranges is not None:
        colorbar.set_ticks([-1.0, 0.0, 1.0])
        colorbar.set_ticklabels(["Low", "Normal", "High"])
        colorbar.ax.yaxis.set_ticks_position("left")
        colorbar.ax.yaxis.set_label_position("right")
        colorbar.ax.tick_params(labelsize=9, pad=6)
        if ranges_ax is not None:
            ranges_ax.axis("off")
        if range_lines and ranges_ax is not None:
            ranges_ax.text(
                0.0,
                1.0,
                "Normal ranges",
                transform=ranges_ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
            ranges_ax.text(
                0.0,
                0.92,
                "\n".join(range_lines),
                transform=ranges_ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="black",
                linespacing=1.45,
            )
    else:
        colorbar.set_ticks([])
        colorbar.ax.text(-0.35, -0.02, "Low", transform=colorbar.ax.transAxes, ha="right", va="top")
        colorbar.ax.text(-0.35, 1.02, "High", transform=colorbar.ax.transAxes, ha="right", va="bottom")
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
    if ranges is None:
        fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def value_ranges_by_variable(matrices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-variable finite min/max values across matrices."""
    finite_matrices = [np.asarray(matrix, dtype=np.float64) for matrix in matrices if np.asarray(matrix).size]
    if not finite_matrices:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    stacked = np.stack(finite_matrices)
    n_variables = stacked.shape[1]
    vmin = np.zeros(n_variables, dtype=np.float64)
    vmax = np.zeros(n_variables, dtype=np.float64)
    for variable_idx in range(n_variables):
        values = stacked[:, variable_idx, :]
        finite = values[np.isfinite(values)]
        if finite.size:
            vmin[variable_idx] = float(np.min(finite))
            vmax[variable_idx] = float(np.max(finite))
    return vmin.astype(np.float64), vmax.astype(np.float64)


def _display_value_matrix(
    matrix: np.ndarray,
    vmin: float | Sequence[float] | np.ndarray | None,
    vmax: float | Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, float | None, float | None, str]:
    matrix = np.asarray(matrix, dtype=np.float64)
    row_vmin = _row_scale(vmin, matrix.shape[0])
    row_vmax = _row_scale(vmax, matrix.shape[0])
    if row_vmin is None and row_vmax is None:
        return matrix, None, None, "Mean Observed Value"
    if row_vmin is None or row_vmax is None:
        return matrix, _scalar_or_none(vmin), _scalar_or_none(vmax), "Mean Observed Value"

    span = row_vmax - row_vmin
    display = np.full_like(matrix, np.nan, dtype=np.float64)
    valid_span = np.isfinite(span) & (span > 0)
    if np.any(valid_span):
        display[valid_span] = (matrix[valid_span] - row_vmin[valid_span, None]) / span[valid_span, None]
    flat_span = np.isfinite(row_vmin) & np.isfinite(row_vmax) & ~valid_span
    if np.any(flat_span):
        display[flat_span] = np.where(np.isfinite(matrix[flat_span]), 0.5, np.nan)
    return np.clip(display, 0.0, 1.0), 0.0, 1.0, "Mean Observed Value (per variable)"


def _row_scale(values: float | Sequence[float] | np.ndarray | None, n_rows: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return None
    array = array.reshape(-1)
    if array.size != n_rows:
        return None
    return array


def _scalar_or_none(values: float | Sequence[float] | np.ndarray | None) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return None


def _annotate_heatmap_values(
    ax,
    image,
    matrix: np.ndarray,
    display_matrix: np.ndarray,
    value_text_format: str,
) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(display_matrix[row, col]))
            alpha = image.get_alpha()
            if isinstance(alpha, np.ndarray):
                alpha_value = float(alpha[row, col])
            elif alpha is None:
                alpha_value = 1.0
            else:
                alpha_value = float(alpha)
            red = red * alpha_value + (1.0 - alpha_value)
            green = green * alpha_value + (1.0 - alpha_value)
            blue = blue * alpha_value + (1.0 - alpha_value)
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            text_color = "white" if luminance < 0.45 else "black"
            ax.text(
                col,
                row,
                format(float(value), value_text_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=6,
                clip_on=True,
            )


def _relative_time_ticks(time_bins: Sequence[str]) -> tuple[np.ndarray, list[str], str | None]:
    if len(time_bins) == 0:
        return np.array([], dtype=int), [], None
    tick_count = min(10, len(time_bins))
    positions = np.unique(np.linspace(0, len(time_bins) - 1, tick_count, dtype=int))
    parsed = pd.to_datetime(list(time_bins), errors="coerce")
    if len(parsed) < 2 or pd.isna(parsed[0]) or pd.isna(parsed[1]):
        return positions, [str(int(position)) for position in positions], None
    step = parsed[1] - parsed[0]
    unit, unit_seconds = _time_axis_unit(step)
    granularity = _format_timedelta_in_unit(step, unit, unit_seconds)
    labels = [_format_timedelta_in_unit(parsed[int(position)] - parsed[0], unit, unit_seconds) for position in positions]
    return positions, labels, granularity


def _time_axis_unit(step) -> tuple[str, int]:
    step_seconds = abs(int(pd.Timedelta(step).total_seconds()))
    if step_seconds >= 3600 and step_seconds % 3600 == 0:
        return "h", 3600
    if step_seconds >= 60 and step_seconds % 60 == 0:
        return "min", 60
    return "s", 1


def _format_timedelta_in_unit(delta, unit: str, unit_seconds: int) -> str:
    total_seconds = int(pd.Timedelta(delta).total_seconds())
    if total_seconds == 0:
        return "0"
    value = total_seconds / unit_seconds
    if float(value).is_integer():
        return f"{int(value)}{unit}"
    return f"{value:g}{unit}"


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
    if "predicted_label" in frame.columns:
        frame["predicted_label"] = frame["predicted_label"].astype(str)
    return frame


def _aggregate_key(group_key, group_columns: list[str]) -> int | tuple[str, int]:
    if group_columns == ["cluster"]:
        if isinstance(group_key, tuple):
            group_key = group_key[0]
        return int(group_key)
    predicted_label, cluster = group_key
    return str(predicted_label), int(cluster)


def _cluster_matrix_path(output_dir: Path, key: int | tuple[str, int]) -> Path:
    if isinstance(key, tuple):
        predicted_label, cluster = key
        return output_dir / _safe_path_component(predicted_label) / f"cluster_{cluster}.npy"
    return output_dir / f"cluster_{key}.npy"


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "class"
