"""Cluster patients by flattened model-importance maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_explanations(
    explanations: Mapping[str, np.ndarray] | str | Path,
    predictions: pd.DataFrame | Mapping[str, str] | str | Path | None = None,
    n_clusters: int = 8,
    method: str = "kmeans",
    aggregate: str = "mean",
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Cluster flattened explanation maps and return assignments/averages.

    Parameters can be a mapping of patient id to matrix or a directory of
    `.npy` explanation maps produced by `explain_model`. When `predictions`
    is provided, explanations are clustered separately inside each predicted
    class, so cluster ids represent class-specific patterns.
    """
    if method != "kmeans":
        raise ValueError("Only kmeans clustering is currently supported.")
    if aggregate != "mean":
        raise ValueError("Only mean aggregation is currently supported.")
    maps = _load_explanations(explanations)
    if not maps:
        raise ValueError("No explanation maps were found to cluster.")
    predicted_classes = _load_predictions(predictions) if predictions is not None else None
    if predicted_classes is not None:
        return _cluster_by_predicted_class(maps, predicted_classes, n_clusters, output_dir)
    return _cluster_patient_maps(maps, n_clusters, output_dir)


def _cluster_patient_maps(
    maps: Mapping[str, np.ndarray],
    n_clusters: int,
    output_dir: str | Path | None,
) -> dict[str, object]:
    patient_ids = list(maps.keys())
    labels, k = _fit_kmeans(maps, patient_ids, n_clusters)
    assignments = pd.DataFrame({"patient_id": patient_ids, "cluster": labels})
    aggregates = {
        int(cluster): np.mean([maps[pid] for pid, label in zip(patient_ids, labels) if label == cluster], axis=0)
        for cluster in sorted(set(labels))
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(out / "cluster_assignments.csv", index=False)
        metadata = {
            "feature_mode": "explanation",
            "class_specific": False,
            "n_clusters_requested": n_clusters,
            "n_clusters_used": k,
        }
        (out / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for cluster, matrix in aggregates.items():
            np.save(out / f"cluster_{cluster}.npy", matrix)
    return {"assignments": assignments, "aggregates": aggregates}


def _cluster_by_predicted_class(
    maps: Mapping[str, np.ndarray],
    predicted_classes: Mapping[str, str],
    n_clusters: int,
    output_dir: str | Path | None,
) -> dict[str, object]:
    grouped: dict[str, list[str]] = {}
    for patient_id in maps:
        predicted_class = predicted_classes.get(str(patient_id))
        if predicted_class is not None:
            grouped.setdefault(str(predicted_class), []).append(patient_id)
    if not grouped:
        raise ValueError("No explanation maps matched the patient ids in the predictions.")

    assignment_frames: list[pd.DataFrame] = []
    aggregates: dict[str, dict[int, np.ndarray]] = {}
    clusters_used: dict[str, int] = {}
    for predicted_class, patient_ids in sorted(grouped.items()):
        labels, k = _fit_kmeans(maps, patient_ids, n_clusters)
        clusters_used[predicted_class] = k
        assignment_frames.append(
            pd.DataFrame(
                {
                    "patient_id": patient_ids,
                    "predicted_label": predicted_class,
                    "cluster": labels,
                }
            )
        )
        aggregates[predicted_class] = {
            int(cluster): np.mean([maps[pid] for pid, label in zip(patient_ids, labels) if label == cluster], axis=0)
            for cluster in sorted(set(labels))
        }

    assignments = pd.concat(assignment_frames, ignore_index=True)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(out / "cluster_assignments.csv", index=False)
        metadata = {
            "feature_mode": "explanation",
            "class_specific": True,
            "n_clusters_requested_per_class": n_clusters,
            "n_clusters_used_by_class": clusters_used,
        }
        (out / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for predicted_class, cluster_matrices in aggregates.items():
            class_dir = out / _safe_path_component(predicted_class)
            class_dir.mkdir(parents=True, exist_ok=True)
            for cluster, matrix in cluster_matrices.items():
                np.save(class_dir / f"cluster_{cluster}.npy", matrix)
    return {"assignments": assignments, "aggregates": aggregates}


def _fit_kmeans(maps: Mapping[str, np.ndarray], patient_ids: list[str], n_clusters: int) -> tuple[np.ndarray, int]:
    x = np.stack([maps[pid].reshape(-1) for pid in patient_ids])
    x_scaled = StandardScaler().fit_transform(x)
    k = max(1, min(n_clusters, len(patient_ids)))
    labels = KMeans(n_clusters=k, random_state=13, n_init=10).fit_predict(x_scaled)
    return labels, k


def _load_explanations(explanations: Mapping[str, np.ndarray] | str | Path) -> dict[str, np.ndarray]:
    if isinstance(explanations, Mapping):
        return {str(k): np.asarray(v) for k, v in explanations.items()}
    path = Path(explanations)
    return {p.stem: np.load(p) for p in sorted(path.glob("*.npy"))}


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
