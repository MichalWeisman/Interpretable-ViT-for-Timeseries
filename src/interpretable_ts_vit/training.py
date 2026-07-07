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
    early_stopping_patience = getattr(config, "early_stopping_patience", None)
    monitor = getattr(config, "early_stopping_monitor", "val_loss")
    min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
    mode = _resolve_monitor_mode(monitor, getattr(config, "early_stopping_mode", "auto"))
    restore_best_model = bool(getattr(config, "restore_best_model", True))
    verbose = bool(getattr(config, "verbose", True))
    progress_interval_batches = getattr(config, "progress_interval_batches", 50)
    best_value: float | None = None
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    if verbose:
        print(
            f"Training on {len(train_dataset)} examples for up to {config.epochs} epoch(s) "
            f"with {len(loader)} batch(es)/epoch on {device}.",
            flush=True,
        )
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if _should_print_batch_progress(verbose, progress_interval_batches, batch_idx, len(loader)):
                running_loss = float(np.mean(losses))
                print(
                    f"Epoch {epoch + 1}/{config.epochs} - batch {batch_idx}/{len(loader)} "
                    f"- train_loss_running={running_loss:.4f}",
                    flush=True,
                )
        row = {"epoch": float(epoch + 1), "train_loss": float(np.mean(losses))}
        improved_this_epoch = False
        if val_dataset is not None:
            row["val_loss"] = evaluate_loss(model, val_dataset, config, criterion)
            row.update({f"val_{k}": v for k, v in evaluate_model(model, val_dataset, config).items() if isinstance(v, float)})
        if val_dataset is not None and monitor in row:
            current = float(row[monitor])
            if best_value is None or _is_improvement(current, best_value, mode, min_delta):
                best_value = current
                best_epoch = epoch + 1
                bad_epochs = 0
                best_state = _copy_state_dict_to_cpu(model)
                improved_this_epoch = True
            else:
                bad_epochs += 1
        elif val_dataset is not None and early_stopping_patience is not None:
            available = ", ".join(sorted(row))
            raise ValueError(f"early_stopping_monitor '{monitor}' is not available. Available metrics: {available}")
        history.append(row)
        if verbose:
            print(_format_epoch_progress(row, epoch + 1, config.epochs), flush=True)
        if early_stopping_patience is not None and val_dataset is not None and not improved_this_epoch and bad_epochs >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after epoch {epoch + 1}: {monitor} did not improve for {bad_epochs} epoch(s).", flush=True)
            break
    if restore_best_model and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    metrics = evaluate_model(model, val_dataset or train_dataset, config)
    metrics["history"] = history
    metrics["epochs_ran"] = len(history)
    metrics["stopped_early"] = len(history) < config.epochs
    if best_epoch is not None:
        metrics["best_epoch"] = best_epoch
        metrics["best_monitor"] = monitor
        metrics["best_monitor_value"] = best_value
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": model.config.__dict__}, out / "model.pt")
        with (out / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(_jsonable(metrics), fh, indent=2)
    return metrics


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataset: BinnedTimeSeriesDataset,
    config: TrainConfig | None = None,
    criterion: nn.Module | None = None,
) -> float:
    """Compute average cross-entropy loss over a labeled dataset."""
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size)
    criterion = criterion or nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_size = int(y.shape[0])
        total_loss += float(criterion(model(x), y).detach().cpu()) * batch_size
        total_examples += batch_size
    if total_examples == 0:
        raise ValueError("Cannot evaluate loss on an empty dataset.")
    return total_loss / total_examples


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


@torch.no_grad()
def extract_model_embeddings(
    model: nn.Module,
    dataset: BinnedTimeSeriesDataset,
    config: TrainConfig | None = None,
    embedding: str = "cls",
) -> tuple[list[str], np.ndarray]:
    """Return patient ids and ViT embeddings from a trained model."""
    if embedding != "cls":
        raise ValueError("Only embedding='cls' is currently supported.")
    if not hasattr(model, "forward_features"):
        raise ValueError("Model does not expose forward_features for embedding extraction.")
    config = config or TrainConfig()
    device = resolve_device(config.device)
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    embeddings = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        features = model.forward_features(x.to(device))
        embeddings.append(features.detach().cpu().numpy())
    patient_ids = [str(patient_id) for patient_id in (dataset.patient_ids or [str(i) for i in range(len(dataset))])]
    return patient_ids, np.concatenate(embeddings, axis=0)


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


def _resolve_monitor_mode(monitor: str, mode: str) -> str:
    if mode not in {"auto", "min", "max"}:
        raise ValueError("early_stopping_mode must be 'auto', 'min', or 'max'.")
    if mode != "auto":
        return mode
    return "min" if monitor.endswith("loss") else "max"


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if not np.isfinite(current):
        return False
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


def _copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _format_epoch_progress(row: dict[str, float], epoch: int, total_epochs: int) -> str:
    metrics = []
    for key, value in row.items():
        if key == "epoch":
            continue
        metrics.append(f"{key}={_format_metric_value(value)}")
    return f"Epoch {epoch}/{total_epochs} - " + " - ".join(metrics)


def _should_print_batch_progress(verbose: bool, interval: int | None, batch_idx: int, total_batches: int) -> bool:
    if not verbose or interval is None or interval <= 0:
        return False
    return batch_idx == 1 or batch_idx == total_batches or batch_idx % interval == 0


def _format_metric_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.4f}"
        return str(float(value))
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
