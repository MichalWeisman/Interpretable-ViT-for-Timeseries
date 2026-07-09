"""Persistence helpers for datasets, models, predictions, and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .binning import TimeSeriesBinner
from .data import BinnedTimeSeriesDataset
from .model import ViTConfig, ViTTimeSeriesClassifier


logger = logging.getLogger(__name__)


def save_split(path: str | Path, patient_ids: list[str], x: np.ndarray, y: np.ndarray | None) -> None:
    """Save one prepared split as a compressed NumPy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving split to %s: patients=%d, x_shape=%s, labels=%s", path, len(patient_ids), x.shape, y is not None)
    payload = {"patient_ids": np.array(patient_ids, dtype=str), "x": x}
    if y is not None:
        payload["y"] = y
    np.savez_compressed(path, **payload)
    logger.info("Saved split to %s", path)


def load_split(path: str | Path) -> BinnedTimeSeriesDataset:
    """Load a prepared split as a `BinnedTimeSeriesDataset`."""
    path = Path(path)
    logger.info("Loading split from %s", path)
    raw = np.load(path, allow_pickle=False)
    y = raw["y"] if "y" in raw.files else None
    dataset = BinnedTimeSeriesDataset(raw["x"], y, raw["patient_ids"].astype(str).tolist())
    logger.info("Loaded split from %s: patients=%d, x_shape=%s, labels=%s", path, len(dataset), tuple(dataset.x.shape), y is not None)
    return dataset


def save_predictions(path: str | Path, patient_ids: list[str], logits: np.ndarray, label_names: list[str]) -> None:
    """Save predicted labels and class probabilities as CSV."""
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    frame = pd.DataFrame({"patient_id": patient_ids, "predicted_label": [label_names[i] for i in pred]})
    for idx, label in enumerate(label_names):
        frame[f"prob_{label}"] = probs[:, idx]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_model(run_dir: str | Path) -> ViTTimeSeriesClassifier:
    """Load `model.pt` from a run directory."""
    checkpoint = torch.load(Path(run_dir) / "model.pt", map_location="cpu")
    model = ViTTimeSeriesClassifier(ViTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_metadata(out_dir: str | Path, binner: TimeSeriesBinner) -> None:
    """Save preprocessing metadata expected by downstream CLI commands."""
    out = Path(out_dir)
    logger.info("Saving preprocessing metadata to %s", out)
    binner.save(out / "binner.json")
    with (out / "variable_vocab.json").open("w", encoding="utf-8") as fh:
        json.dump(binner.variable_vocab_, fh, indent=2)
    logger.info("Saved preprocessing metadata to %s: variables=%d, time_bins=%d", out, len(binner.variable_vocab_), len(binner.time_bins_))
