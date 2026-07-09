"""Autoencoder embeddings for explanation/value patient maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .clustering import cluster_latent_vectors
from .model import TransformerEncoderLayerWithAttention
from .training import resolve_device


class MapAutoencoder(nn.Module):
    """ViT-style autoencoder for two-channel variable-time maps."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int = 16,
        *,
        patch_size: tuple[int, int] = (1, 4),
        embed_dim: int | None = None,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(dim) for dim in input_shape)
        channels, variables, timesteps = self.input_shape
        patch_vars, patch_steps = patch_size
        self.patch_vars = int(patch_vars)
        self.patch_steps = int(patch_steps)
        self.padded_variables = ((variables + self.patch_vars - 1) // self.patch_vars) * self.patch_vars
        self.padded_timesteps = ((timesteps + self.patch_steps - 1) // self.patch_steps) * self.patch_steps
        self.num_patches = (self.padded_variables // self.patch_vars) * (self.padded_timesteps // self.patch_steps)
        self.patch_dim = channels * self.patch_vars * self.patch_steps
        self.embed_dim = int(embed_dim or max(32, min(128, latent_dim * 4)))
        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(
                    self.embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        self.latent_head = nn.Linear(self.embed_dim, latent_dim)
        self.decoder_seed = nn.Linear(latent_dim, self.num_patches * self.embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttention(
                    self.embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.patch_decode = nn.Linear(self.embed_dim, self.patch_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        tokens = self.decoder_seed(latent).view(x.shape[0], self.num_patches, self.embed_dim)
        tokens = tokens + self.decoder_pos_embed
        for block in self.decoder_blocks:
            tokens = block(tokens)
        tokens = self.decoder_norm(tokens)
        patches = self.patch_decode(tokens)
        return self.unpatchify(patches)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.encoder_pos_embed
        for block in self.encoder_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        return self.latent_head(tokens[:, 0])

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, variables, timesteps = x.shape
        expected_channels = self.input_shape[0]
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, got {channels}.")
        pad_vars = self.padded_variables - variables
        pad_steps = self.padded_timesteps - timesteps
        if pad_vars or pad_steps:
            x = F.pad(x, (0, pad_steps, 0, pad_vars))
        x = x.unfold(2, self.patch_vars, self.patch_vars).unfold(3, self.patch_steps, self.patch_steps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(bsz, self.num_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        bsz = patches.shape[0]
        channels, variables, timesteps = self.input_shape
        vars_p = self.padded_variables // self.patch_vars
        steps_p = self.padded_timesteps // self.patch_steps
        x = patches.view(bsz, vars_p, steps_p, channels, self.patch_vars, self.patch_steps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(bsz, channels, self.padded_variables, self.padded_timesteps)
        return x[:, :, :variables, :timesteps]


def cluster_explanation_value_autoencoder(
    explanations: Mapping[str, np.ndarray] | str | Path,
    values: Mapping[str, np.ndarray],
    *,
    validation_explanations: Mapping[str, np.ndarray] | str | Path | None = None,
    validation_values: Mapping[str, np.ndarray] | None = None,
    cluster_explanations: Mapping[str, np.ndarray] | str | Path | None = None,
    cluster_values: Mapping[str, np.ndarray] | None = None,
    predictions: pd.DataFrame | Mapping[str, str] | str | Path | None = None,
    n_clusters: int = 8,
    method: str = "kmeans",
    output_dir: str | Path | None = None,
    latent_dim: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    device: str = "auto",
    early_stopping_patience: int | None = 10,
    show_progress: bool = False,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> dict[str, object]:
    """Train, embed, and cluster explanation/value maps.

    This compatibility wrapper keeps the original one-call workflow while the
    underlying stages are exposed as separate public functions.
    """
    out = Path(output_dir) if output_dir is not None else None
    cached = _load_cached_outputs(out)
    if cached is not None:
        return cached

    trained = train_explanation_value_autoencoder(
        explanations,
        values,
        validation_explanations=validation_explanations,
        validation_values=validation_values,
        output_dir=out,
        latent_dim=latent_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        early_stopping_patience=early_stopping_patience,
        show_progress=show_progress,
    )
    embedded = create_explanation_value_embeddings(
        cluster_explanations if cluster_explanations is not None else explanations,
        cluster_values if cluster_values is not None else values,
        model=trained["model"],
        preprocessor=trained["preprocessor"],
        output_dir=out,
        batch_size=batch_size,
        device=device,
    )
    clustered = cluster_autoencoder_embeddings(
        embedded["embedding_frame"],
        explanations=embedded["explanations"],
        predictions=predictions,
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
    return {
        "assignments": clustered["assignments"],
        "centroids": clustered["centroids"],
        "embeddings": embedded["embeddings"],
        "metrics": trained["metrics"] | {"cluster_loss": embedded["loss"]},
    }


def train_explanation_value_autoencoder(
    explanations: Mapping[str, np.ndarray] | str | Path,
    values: Mapping[str, np.ndarray],
    *,
    validation_explanations: Mapping[str, np.ndarray] | str | Path | None = None,
    validation_values: Mapping[str, np.ndarray] | None = None,
    output_dir: str | Path | None = None,
    latent_dim: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    device: str = "auto",
    early_stopping_patience: int | None = 10,
    show_progress: bool = False,
) -> dict[str, object]:
    """Train the autoencoder on train maps and optionally validate it."""
    out = Path(output_dir) if output_dir is not None else None
    cached = _load_cached_autoencoder(out, device=device)
    if cached is not None:
        return cached

    train_maps = _load_maps(explanations)
    train_ids, train_raw, train_metadata = _raw_autoencoder_tensor(train_maps, values)
    preprocessor = _fit_preprocessor(train_raw)
    train_tensors = _transform_tensor(train_raw, preprocessor)

    validation_tensors = None
    validation_ids: list[str] = []
    if validation_explanations is not None and validation_values is not None:
        validation_maps = _load_maps(validation_explanations)
        validation_ids, validation_raw, validation_metadata = _raw_autoencoder_tensor(validation_maps, validation_values)
        validation_tensors = _transform_tensor(validation_raw, preprocessor)
    else:
        validation_metadata = {"n_patients": 0, "nan_cells": 0}

    model, history = _train_autoencoder(
        train_tensors,
        validation_tensors=validation_tensors,
        latent_dim=latent_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        early_stopping_patience=early_stopping_patience,
        show_progress=show_progress,
    )
    metrics = {
        "train_loss": _evaluate_autoencoder(model, train_tensors, batch_size=batch_size, device=device),
        "val_loss": (
            _evaluate_autoencoder(model, validation_tensors, batch_size=batch_size, device=device)
            if validation_tensors is not None
            else None
        ),
        "cluster_loss": None,
        "best_epoch": _best_epoch(history),
        "history": history,
    }
    metadata = {
        "autoencoder_architecture": "vit",
        "train": {"n_patients": len(train_ids), **train_metadata},
        "validation": {"n_patients": len(validation_ids), **validation_metadata},
        "input_shape": list(train_tensors.shape[1:]),
        "latent_dim": latent_dim,
        "autoencoder_epochs": epochs,
        "autoencoder_learning_rate": learning_rate,
        "autoencoder_batch_size": batch_size,
        "autoencoder_early_stopping_patience": early_stopping_patience,
    }
    if out is not None:
        _write_autoencoder_training_outputs(
            out,
            model,
            preprocessor,
            latent_dim,
            metrics,
            metadata,
        )
    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "metadata": metadata,
    }


def create_explanation_value_embeddings(
    explanations: Mapping[str, np.ndarray] | str | Path,
    values: Mapping[str, np.ndarray],
    *,
    model: MapAutoencoder | None = None,
    preprocessor: dict[str, np.ndarray] | None = None,
    autoencoder_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, object]:
    """Create latent embeddings for a split using a trained autoencoder."""
    out = Path(output_dir) if output_dir is not None else None
    maps = _load_maps(explanations)
    cached = _load_cached_embeddings(out)
    if cached is not None:
        cached["explanations"] = maps
        return cached
    if model is None or preprocessor is None:
        artifact_path = Path(autoencoder_path) if autoencoder_path is not None else None
        if artifact_path is None and out is not None:
            artifact_path = out / "autoencoder.pt"
        if artifact_path is None:
            raise ValueError("model/preprocessor or autoencoder_path is required to create embeddings.")
        model, preprocessor, _ = _load_autoencoder_artifact(artifact_path, device=device)

    patient_ids, raw, metadata = _raw_autoencoder_tensor(maps, values)
    tensors = _transform_tensor(raw, preprocessor)
    embeddings = _encode_autoencoder(model, tensors, batch_size=batch_size, device=device)
    loss = _evaluate_autoencoder(model, tensors, batch_size=batch_size, device=device)
    embedding_frame = _embedding_frame(patient_ids, embeddings)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        embedding_frame.to_csv(out / "autoencoder_embeddings.csv", index=False)
        (out / "autoencoder_embedding_metadata.json").write_text(
            json.dumps(
                {
                    "autoencoder_architecture": "vit",
                    "cluster": {"n_patients": len(patient_ids), **metadata},
                    "cluster_loss": loss,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "patient_ids": patient_ids,
        "embeddings": embeddings,
        "embedding_frame": embedding_frame,
        "explanations": maps,
        "metadata": metadata,
        "tensors": tensors,
        "loss": loss,
    }


def cluster_autoencoder_embeddings(
    embeddings: pd.DataFrame | Mapping[str, np.ndarray] | np.ndarray,
    *,
    explanations: Mapping[str, np.ndarray] | str | Path | None = None,
    patient_ids: list[str] | None = None,
    predictions: pd.DataFrame | Mapping[str, str] | str | Path | None = None,
    n_clusters: int = 8,
    method: str = "kmeans",
    output_dir: str | Path | None = None,
    autoencoder_metrics: dict[str, object] | None = None,
    autoencoder_metadata: dict[str, object] | None = None,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> dict[str, object]:
    """Cluster a saved or in-memory autoencoder embedding table."""
    clustered = cluster_latent_vectors(
        embeddings,
        patient_ids=patient_ids,
        predictions=predictions,
        n_clusters=n_clusters,
        method=method,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )
    assignments = clustered["assignments"]
    if output_dir is not None:
        out = Path(output_dir)
        _write_cluster_outputs(
            out,
            assignments,
            method,
            n_clusters,
            autoencoder_metrics or {},
            autoencoder_metadata or {},
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
        )
        if explanations is not None:
            _write_explanation_aggregates(out, assignments, _load_maps(explanations) if not isinstance(explanations, Mapping) else explanations)
    return clustered


def _load_maps(explanations: Mapping[str, np.ndarray] | str | Path) -> dict[str, np.ndarray]:
    if isinstance(explanations, Mapping):
        return {str(patient_id): np.asarray(matrix, dtype=np.float64) for patient_id, matrix in explanations.items()}
    path = Path(explanations)
    return {p.stem: np.load(p).astype(np.float64) for p in sorted(path.glob("*.npy"))}


def _raw_autoencoder_tensor(
    explanations: Mapping[str, np.ndarray],
    values: Mapping[str, np.ndarray],
    patient_ids: list[str] | None = None,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    if patient_ids is None:
        patient_ids = [patient_id for patient_id in explanations if patient_id in values]
    if not patient_ids:
        raise ValueError("No patients had both explanation maps and value maps.")
    stacked = []
    for patient_id in patient_ids:
        explanation = np.asarray(explanations[patient_id], dtype=np.float64)
        value = np.asarray(values[patient_id], dtype=np.float64)
        if explanation.shape != value.shape:
            raise ValueError(f"Explanation/value shape mismatch for patient {patient_id}: {explanation.shape} != {value.shape}")
        stacked.append(np.stack([explanation, value], axis=0))
    tensor = np.stack(stacked, axis=0)
    return patient_ids, tensor, {"nan_cells": int(np.sum(~np.isfinite(tensor)))}


def _fit_preprocessor(tensor: np.ndarray) -> dict[str, np.ndarray]:
    flat = tensor.reshape(tensor.shape[0], -1).copy()
    column_means = np.nanmean(flat, axis=0)
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    rows, cols = np.where(~np.isfinite(flat))
    if len(rows):
        flat[rows, cols] = column_means[cols]
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    return {"column_means": column_means, "means": means, "stds": stds}


def _transform_tensor(tensor: np.ndarray, preprocessor: dict[str, np.ndarray]) -> np.ndarray:
    flat = tensor.reshape(tensor.shape[0], -1).copy()
    rows, cols = np.where(~np.isfinite(flat))
    if len(rows):
        flat[rows, cols] = preprocessor["column_means"][cols]
    scaled = (flat - preprocessor["means"]) / preprocessor["stds"]
    return scaled.reshape(tensor.shape).astype(np.float32)


def _train_autoencoder(
    tensors: np.ndarray,
    *,
    validation_tensors: np.ndarray | None = None,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: str,
    early_stopping_patience: int | None = 10,
    show_progress: bool = False,
) -> tuple[MapAutoencoder, list[dict[str, float | int | None]]]:
    resolved_device = resolve_device(device)
    x = torch.as_tensor(tensors, dtype=torch.float32)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MapAutoencoder(input_shape=tuple(int(dim) for dim in tensors.shape[1:]), latent_dim=latent_dim).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[dict[str, float | int | None]] = []
    best_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    for epoch in range(max(1, int(epochs))):
        model.train()
        epoch_losses = []
        for (batch,) in loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(epoch_losses))
        val_loss = (
            _evaluate_autoencoder(model, validation_tensors, batch_size=batch_size, device=device)
            if validation_tensors is not None
            else None
        )
        monitor = train_loss if val_loss is None else val_loss
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if show_progress:
            val_text = "none" if val_loss is None else f"{val_loss:.6f}"
            print(f"autoencoder epoch {epoch + 1}/{max(1, int(epochs))} - train_loss={train_loss:.6f} - val_loss={val_text}")
        if best_loss is None or monitor < best_loss:
            best_loss = monitor
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if early_stopping_patience is not None and bad_epochs >= early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(resolved_device)
    return model, history


@torch.no_grad()
def _evaluate_autoencoder(model: MapAutoencoder, tensors: np.ndarray, *, batch_size: int, device: str) -> float:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    loader = DataLoader(TensorDataset(torch.as_tensor(tensors, dtype=torch.float32)), batch_size=batch_size)
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_values = 0
    for (batch,) in loader:
        batch = batch.to(resolved_device)
        total_loss += float(criterion(model(batch), batch).detach().cpu())
        total_values += int(batch.numel())
    if total_values == 0:
        raise ValueError("Cannot evaluate autoencoder on an empty tensor set.")
    return total_loss / total_values


def _best_epoch(history: list[dict[str, float | int | None]]) -> int | None:
    if not history:
        return None
    metric = "val_loss" if history[0].get("val_loss") is not None else "train_loss"
    return int(min(history, key=lambda row: float(row[metric]))["epoch"])


@torch.no_grad()
def _encode_autoencoder(model: MapAutoencoder, tensors: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()
    loader = DataLoader(TensorDataset(torch.as_tensor(tensors, dtype=torch.float32)), batch_size=batch_size)
    embeddings = []
    for (batch,) in loader:
        embeddings.append(model.encode(batch.to(resolved_device)).detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _embedding_frame(patient_ids: list[str], embeddings: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(embeddings, columns=[f"embedding_{idx}" for idx in range(embeddings.shape[1])])
    frame.insert(0, "patient_id", patient_ids)
    return frame


def _write_autoencoder_training_outputs(
    output_dir: Path,
    model: MapAutoencoder,
    preprocessor: dict[str, np.ndarray],
    latent_dim: int,
    metrics: dict[str, object],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_autoencoder_artifact(output_dir / "autoencoder.pt", model, preprocessor, latent_dim)
    (output_dir / "autoencoder_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "autoencoder_losses.json").write_text(json.dumps(metrics["history"], indent=2), encoding="utf-8")
    (output_dir / "autoencoder_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _save_autoencoder_artifact(
    path: Path,
    model: MapAutoencoder,
    preprocessor: dict[str, np.ndarray],
    latent_dim: int,
) -> None:
    torch.save(
        {
            "architecture": "vit",
            "model_state_dict": model.cpu().state_dict(),
            "input_shape": list(model.input_shape),
            "latent_dim": latent_dim,
            "patch_size": [model.patch_vars, model.patch_steps],
            "embed_dim": model.embed_dim,
            "depth": len(model.encoder_blocks),
            "num_heads": model.encoder_blocks[0].attn.num_heads if model.encoder_blocks else 1,
            "preprocessor": {key: value.tolist() for key, value in preprocessor.items()},
        },
        path,
    )


def _load_autoencoder_artifact(path: str | Path, *, device: str) -> tuple[MapAutoencoder, dict[str, np.ndarray], int]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if checkpoint.get("architecture") != "vit":
        raise ValueError(f"Autoencoder artifact at {path} is not a ViT autoencoder. Remove it and retrain.")
    latent_dim = int(checkpoint["latent_dim"])
    model = MapAutoencoder(
        input_shape=tuple(int(dim) for dim in checkpoint["input_shape"]),
        latent_dim=latent_dim,
        patch_size=tuple(int(dim) for dim in checkpoint.get("patch_size", [1, 4])),
        embed_dim=int(checkpoint.get("embed_dim", max(32, min(128, latent_dim * 4)))),
        depth=int(checkpoint.get("depth", 2)),
        num_heads=int(checkpoint.get("num_heads", 4)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolve_device(device))
    preprocessor = {key: np.asarray(value, dtype=np.float64) for key, value in checkpoint["preprocessor"].items()}
    return model, preprocessor, latent_dim


def _load_cached_autoencoder(output_dir: Path | None, *, device: str) -> dict[str, object] | None:
    if output_dir is None:
        return None
    model_path = output_dir / "autoencoder.pt"
    metrics_path = output_dir / "autoencoder_metrics.json"
    metadata_path = output_dir / "autoencoder_training_metadata.json"
    if not (model_path.exists() and metrics_path.exists() and metadata_path.exists()):
        return None
    try:
        model, preprocessor, _ = _load_autoencoder_artifact(model_path, device=device)
    except ValueError:
        return None
    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
        "loaded_from_cache": True,
    }


def _load_cached_embeddings(output_dir: Path | None) -> dict[str, object] | None:
    if output_dir is None:
        return None
    embeddings_path = output_dir / "autoencoder_embeddings.csv"
    metadata_path = output_dir / "autoencoder_embedding_metadata.json"
    if not embeddings_path.exists():
        return None
    frame = pd.read_csv(embeddings_path, dtype={"patient_id": str})
    embedding_cols = [col for col in frame.columns if col.startswith("embedding_")]
    metadata = {}
    loss = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("autoencoder_architecture") != "vit":
            return None
        loss = metadata.get("cluster_loss")
    else:
        return None
    return {
        "patient_ids": frame["patient_id"].astype(str).tolist(),
        "embeddings": frame[embedding_cols].to_numpy(dtype=float),
        "embedding_frame": frame,
        "explanations": {},
        "metadata": metadata.get("cluster", {}),
        "tensors": np.empty((len(frame), 0), dtype=np.float32),
        "loss": loss,
        "loaded_from_cache": True,
    }


def _write_cluster_outputs(
    output_dir: Path,
    assignments: pd.DataFrame,
    method: str,
    n_clusters: int,
    autoencoder_metrics: dict[str, object],
    autoencoder_metadata: dict[str, object],
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    assignments[assignments["is_centroid"]].to_csv(output_dir / "cluster_centroids.csv", index=False)
    metadata = {
        "feature_mode": "autoencoder",
        "class_specific": "predicted_label" in assignments.columns,
        "clustering_method": method,
        "centroid_definition": "nearest patient to cluster centroid in scaled autoencoder latent space",
        "n_clusters_requested": n_clusters,
        "n_clusters_used": _clusters_used(assignments),
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
        **autoencoder_metadata,
    }
    if autoencoder_metrics:
        metadata.update(
            {
                "autoencoder_train_loss": autoencoder_metrics.get("train_loss"),
                "autoencoder_val_loss": autoencoder_metrics.get("val_loss"),
                "autoencoder_cluster_loss": autoencoder_metrics.get("cluster_loss"),
                "autoencoder_best_epoch": autoencoder_metrics.get("best_epoch"),
                "autoencoder_final_loss": autoencoder_metrics.get("cluster_loss"),
            }
        )
        (output_dir / "autoencoder_metrics.json").write_text(json.dumps(autoencoder_metrics, indent=2), encoding="utf-8")
    (output_dir / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _clusters_used(assignments: pd.DataFrame) -> int | dict[str, int]:
    def count_clusters(frame: pd.DataFrame) -> int:
        return int(frame.loc[frame["cluster"] >= 0, "cluster"].nunique())

    if "predicted_label" not in assignments.columns:
        return count_clusters(assignments)
    return {str(label): count_clusters(group) for label, group in assignments.groupby("predicted_label", sort=True)}


def _write_outputs(
    output_dir: Path,
    patient_ids: list[str],
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    explanations: Mapping[str, np.ndarray],
    model: MapAutoencoder,
    preprocessor: dict[str, np.ndarray],
    n_clusters: int,
    method: str,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    early_stopping_patience: int | None,
    metrics: dict[str, object],
    tensor_metadata: dict[str, object],
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_dim": model.decoder[-1].out_features,
            "latent_dim": latent_dim,
            "preprocessor": {key: value.tolist() for key, value in preprocessor.items()},
        },
        output_dir / "autoencoder.pt",
    )
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
    assignments[assignments["is_centroid"]].to_csv(output_dir / "cluster_centroids.csv", index=False)
    embedding_frame = pd.DataFrame(embeddings, columns=[f"embedding_{idx}" for idx in range(embeddings.shape[1])])
    embedding_frame.insert(0, "patient_id", patient_ids)
    embedding_frame.to_csv(output_dir / "autoencoder_embeddings.csv", index=False)
    metadata = {
        "feature_mode": "autoencoder",
        "class_specific": "predicted_label" in assignments.columns,
        "clustering_method": method,
        "n_clusters_requested": n_clusters,
        "latent_dim": latent_dim,
        "autoencoder_epochs": epochs,
        "autoencoder_learning_rate": learning_rate,
        "autoencoder_batch_size": batch_size,
        "autoencoder_early_stopping_patience": early_stopping_patience,
        "autoencoder_train_loss": metrics["train_loss"],
        "autoencoder_val_loss": metrics["val_loss"],
        "autoencoder_cluster_loss": metrics["cluster_loss"],
        "autoencoder_best_epoch": metrics["best_epoch"],
        "autoencoder_final_loss": metrics["cluster_loss"],
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
        **tensor_metadata,
    }
    (output_dir / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "autoencoder_losses.json").write_text(json.dumps(metrics["history"], indent=2), encoding="utf-8")
    (output_dir / "autoencoder_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_explanation_aggregates(output_dir, assignments, explanations)


def _load_cached_outputs(output_dir: Path | None) -> dict[str, object] | None:
    if output_dir is None:
        return None
    required = [
        output_dir / "autoencoder.pt",
        output_dir / "autoencoder_embeddings.csv",
        output_dir / "autoencoder_metrics.json",
        output_dir / "cluster_assignments.csv",
        output_dir / "cluster_centroids.csv",
    ]
    if not all(path.exists() for path in required):
        return None
    try:
        checkpoint = torch.load(output_dir / "autoencoder.pt", map_location="cpu")
    except Exception:
        return None
    if checkpoint.get("architecture") != "vit":
        return None
    embeddings_frame = pd.read_csv(output_dir / "autoencoder_embeddings.csv", dtype={"patient_id": str})
    embedding_cols = [col for col in embeddings_frame.columns if col.startswith("embedding_")]
    assignments = pd.read_csv(output_dir / "cluster_assignments.csv", dtype={"patient_id": str})
    centroids = pd.read_csv(output_dir / "cluster_centroids.csv", dtype={"patient_id": str}).to_dict("records")
    metrics = json.loads((output_dir / "autoencoder_metrics.json").read_text(encoding="utf-8"))
    return {
        "assignments": assignments,
        "centroids": centroids,
        "embeddings": embeddings_frame[embedding_cols].to_numpy(dtype=float),
        "metrics": metrics,
        "loaded_from_cache": True,
    }


def _write_explanation_aggregates(
    output_dir: Path,
    assignments: pd.DataFrame,
    explanations: Mapping[str, np.ndarray],
) -> None:
    group_columns = ["predicted_label", "cluster"] if "predicted_label" in assignments.columns else ["cluster"]
    for group_key, group in assignments.groupby(group_columns, sort=True):
        matrices = [explanations[str(patient_id)] for patient_id in group["patient_id"] if str(patient_id) in explanations]
        if not matrices:
            continue
        aggregate = np.mean(matrices, axis=0)
        if group_columns == ["cluster"]:
            if isinstance(group_key, tuple):
                group_key = group_key[0]
            path = output_dir / f"cluster_{int(group_key)}.npy"
        else:
            predicted_label, cluster = group_key
            path = output_dir / _safe_path_component(str(predicted_label)) / f"cluster_{int(cluster)}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, aggregate)


def _safe_path_component(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "class"
