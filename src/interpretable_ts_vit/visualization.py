"""Heatmap rendering utilities for explanation matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
