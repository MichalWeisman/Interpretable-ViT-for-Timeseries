"""Training, prediction, and metric helpers for the ViT classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import BinnedTimeSeriesDataset


def resolve_device(device: str) -> torch.device:
    """Resolve `auto` to CUDA when available, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_model(
    model: nn.Module,
    train_dataset: BinnedTimeSeriesDataset,
    val_dataset: BinnedTimeSeriesDataset | None = None,
    config: TrainConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a model and optionally save `model.pt` plus `metrics.json`."""
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses))}
        if val_dataset is not None:
            row.update({f"val_{k}": v for k, v in evaluate_model(model, val_dataset, config).items() if isinstance(v, float)})
        history.append(row)
    metrics = evaluate_model(model, val_dataset or train_dataset, config)
    metrics["history"] = history
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": model.config.__dict__}, out / "model.pt")
        with (out / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(_jsonable(metrics), fh, indent=2)
    return metrics


@torch.no_grad()
def predict_model(model: nn.Module, dataset: BinnedTimeSeriesDataset, config: TrainConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and labels for a dataset without gradient tracking."""
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size)
    logits = []
    labels = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
            labels.append(y.numpy())
        else:
            x = batch
        logits.append(model(x.to(device)).detach().cpu().numpy())
    y_true = np.concatenate(labels) if labels else np.array([])
    return np.concatenate(logits), y_true


def evaluate_model(model: nn.Module, dataset: BinnedTimeSeriesDataset, config: TrainConfig | None = None) -> dict[str, Any]:
    """Compute accuracy, macro F1, AUROC when possible, and confusion matrix."""
    logits, y_true = predict_model(model, dataset, config)
    y_pred = logits.argmax(axis=1)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
        if probs.shape[1] == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except ValueError:
        metrics["auroc"] = None
    return metrics


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
