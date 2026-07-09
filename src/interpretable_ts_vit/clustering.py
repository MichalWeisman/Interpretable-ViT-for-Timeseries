"""Cluster autoencoder latent vectors and choose representative patients."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from hdbscan import HDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    HDBSCAN = None


def cluster_latent_vectors(
    embeddings: pd.DataFrame | Mapping[str, np.ndarray] | np.ndarray,
    patient_ids: Sequence[str] | None = None,
    predictions: pd.DataFrame | Mapping[str, str] | str | Path | None = None,
    n_clusters: int = 8,
    method: str = "kmeans",
    output_dir: str | Path | None = None,
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int | None = None,
) -> dict[str, object]:
    """Cluster autoencoder embeddings, optionally within predicted classes."""
    if method not in {"kmeans", "hdbscan"}:
        raise ValueError("Only kmeans and hdbscan clustering are currently supported.")
    ids, x = _embedding_matrix(embeddings, patient_ids)
    if len(ids) == 0:
        raise ValueError("No embeddings were provided.")
    predicted_classes = _load_predictions(predictions) if predictions is not None else None
    if predicted_classes is not None:
        return _cluster_by_predicted_class(
            ids,
            x,
            predicted_classes,
            n_clusters,
            method,
            output_dir,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
        )
    labels, k, scaled, centers = _fit_embedding_clusters(x, n_clusters, method, hdbscan_min_cluster_size, hdbscan_min_samples)
    assignments = _embedding_assignment_frame(ids, labels, scaled, centers)
    if output_dir is not None:
        _write_outputs(
            Path(output_dir),
            assignments,
            method,
            n_clusters,
            k,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
            class_specific=False,
        )
    return {"assignments": assignments, "centroids": _centroid_records(assignments)}


def _cluster_by_predicted_class(
    ids: list[str],
    x: np.ndarray,
    predicted_classes: Mapping[str, str],
    n_clusters: int,
    method: str,
    output_dir: str | Path | None,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
) -> dict[str, object]:
    grouped: dict[str, list[int]] = {}
    for idx, patient_id in enumerate(ids):
        predicted_class = predicted_classes.get(str(patient_id))
        if predicted_class is not None:
            grouped.setdefault(str(predicted_class), []).append(idx)
    if not grouped:
        raise ValueError("No autoencoder embeddings matched the patient ids in the predictions.")

    assignment_frames = []
    clusters_used: dict[str, int] = {}
    for predicted_class, indices in sorted(grouped.items()):
        class_ids = [ids[idx] for idx in indices]
        labels, k, scaled, centers = _fit_embedding_clusters(
            x[indices],
            n_clusters,
            method,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
        )
        clusters_used[predicted_class] = k
        frame = _embedding_assignment_frame(class_ids, labels, scaled, centers)
        frame.insert(1, "predicted_label", predicted_class)
        assignment_frames.append(frame)

    assignments = pd.concat(assignment_frames, ignore_index=True)
    if output_dir is not None:
        _write_outputs(
            Path(output_dir),
            assignments,
            method,
            n_clusters,
            clusters_used,
            hdbscan_min_cluster_size,
            hdbscan_min_samples,
            class_specific=True,
        )
    return {"assignments": assignments, "centroids": _centroid_records(assignments)}


def _fit_embedding_clusters(
    x: np.ndarray,
    n_clusters: int,
    method: str,
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray | None]:
    scaled = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))
    if method == "kmeans":
        k = max(1, min(n_clusters, len(scaled)))
        clusterer = KMeans(n_clusters=k, random_state=13, n_init=10)
        labels = clusterer.fit_predict(scaled).astype(int)
        return labels, k, scaled, clusterer.cluster_centers_
    labels, k = _fit_hdbscan(scaled, len(scaled), hdbscan_min_cluster_size, hdbscan_min_samples)
    return labels, k, scaled, None


def _fit_hdbscan(
    x: np.ndarray,
    n_patients: int,
    min_cluster_size: int | None,
    min_samples: int | None,
) -> tuple[np.ndarray, int]:
    if HDBSCAN is None:
        raise ImportError("hdbscan is required for method='hdbscan'. Install it with `pip install hdbscan`.")
    resolved_min_cluster_size = _resolve_hdbscan_min_cluster_size(min_cluster_size)
    if resolved_min_cluster_size < 2:
        raise ValueError("hdbscan_min_cluster_size must be at least 2.")
    if min_samples is not None and int(min_samples) < 1:
        raise ValueError("hdbscan_min_samples must be at least 1.")
    if n_patients < resolved_min_cluster_size:
        return np.full(n_patients, -1, dtype=int), 0

    labels = HDBSCAN(
        min_cluster_size=resolved_min_cluster_size,
        min_samples=None if min_samples is None else int(min_samples),
        metric="euclidean",
    ).fit_predict(x).astype(int)
    return labels, int(np.unique(labels[labels >= 0]).size)


def _embedding_matrix(
    embeddings: pd.DataFrame | Mapping[str, np.ndarray] | np.ndarray,
    patient_ids: Sequence[str] | None,
) -> tuple[list[str], np.ndarray]:
    if isinstance(embeddings, pd.DataFrame):
        frame = embeddings.copy()
        if "patient_id" not in frame.columns:
            raise ValueError("Embedding dataframe must include a patient_id column.")
        ids = frame["patient_id"].astype(str).tolist()
        feature_cols = [col for col in frame.columns if col != "patient_id"]
        if not feature_cols:
            raise ValueError("Embedding dataframe does not contain feature columns.")
        return ids, frame[feature_cols].to_numpy(dtype=np.float64)
    if isinstance(embeddings, Mapping):
        ids = [str(patient_id) for patient_id in embeddings]
        x = np.stack([np.asarray(embeddings[patient_id], dtype=np.float64).reshape(-1) for patient_id in embeddings])
        return ids, x
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Embedding array must have shape [patients, features].")
    ids = [str(i) for i in range(x.shape[0])] if patient_ids is None else [str(patient_id) for patient_id in patient_ids]
    if len(ids) != x.shape[0]:
        raise ValueError("patient_ids length must match embedding rows.")
    return ids, x


def _embedding_assignment_frame(
    patient_ids: list[str],
    labels: np.ndarray,
    scaled: np.ndarray,
    centers: np.ndarray | None,
) -> pd.DataFrame:
    distances = np.full(len(patient_ids), np.nan, dtype=np.float64)
    if centers is not None:
        for idx, label in enumerate(labels):
            distances[idx] = float(np.linalg.norm(scaled[idx] - centers[int(label)]))
    else:
        for label in sorted(set(labels)):
            indices = np.where(labels == label)[0]
            if label == -1 or len(indices) == 0:
                continue
            center = scaled[indices].mean(axis=0)
            distances[indices] = np.linalg.norm(scaled[indices] - center, axis=1)
    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "cluster": labels.astype(int),
            "distance_to_centroid": distances,
            "is_centroid": _centroid_flags(labels, distances),
        }
    )


def _centroid_flags(labels: np.ndarray, distances: np.ndarray) -> list[bool]:
    flags = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels)):
        if label == -1:
            continue
        indices = np.where(labels == label)[0]
        finite_indices = indices[np.isfinite(distances[indices])]
        if len(finite_indices):
            flags[finite_indices[np.argmin(distances[finite_indices])]] = True
    return flags.tolist()


def _centroid_records(assignments: pd.DataFrame) -> pd.DataFrame:
    cols = ["patient_id", "cluster", "distance_to_centroid"]
    if "predicted_label" in assignments.columns:
        cols.insert(1, "predicted_label")
    return assignments[assignments["is_centroid"]].loc[:, cols].reset_index(drop=True)


def _write_outputs(
    out: Path,
    assignments: pd.DataFrame,
    method: str,
    n_clusters: int,
    clusters_used: int | dict[str, int],
    hdbscan_min_cluster_size: int | None,
    hdbscan_min_samples: int | None,
    class_specific: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out / "cluster_assignments.csv", index=False)
    _centroid_records(assignments).to_csv(out / "cluster_centroids.csv", index=False)
    metadata: dict[str, object] = {
        "feature_mode": "autoencoder",
        "class_specific": class_specific,
        "clustering_method": method,
        "centroid_definition": "nearest patient to cluster centroid in scaled autoencoder latent space",
    }
    if class_specific:
        metadata["n_clusters_used_by_class"] = clusters_used
    else:
        metadata["n_clusters_used"] = clusters_used
    if method == "kmeans":
        key = "n_clusters_requested_per_class" if class_specific else "n_clusters_requested"
        metadata[key] = n_clusters
    else:
        metadata["hdbscan_min_cluster_size"] = _resolve_hdbscan_min_cluster_size(hdbscan_min_cluster_size)
        metadata["hdbscan_min_samples"] = hdbscan_min_samples
    (out / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _resolve_hdbscan_min_cluster_size(min_cluster_size: int | None) -> int:
    return 5 if min_cluster_size is None else int(min_cluster_size)


def _load_predictions(predictions: pd.DataFrame | Mapping[str, str] | str | Path) -> dict[str, str]:
    if isinstance(predictions, Mapping):
        return {str(patient_id): str(label) for patient_id, label in predictions.items()}
    if isinstance(predictions, pd.DataFrame):
        frame = predictions.copy()
    else:
        frame = pd.read_csv(predictions)
    required = {"patient_id", "predicted_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    return dict(zip(frame["patient_id"].astype(str), frame["predicted_label"].astype(str)))


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "class"
