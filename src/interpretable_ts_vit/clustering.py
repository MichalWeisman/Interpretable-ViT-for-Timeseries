"""Cluster patients using explanation, value, or combined feature vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import BinnedTimeSeriesDataset


CLUSTER_FEATURE_MODES = {"explanation", "value", "combined"}


def cluster_explanations(
    explanations: Mapping[str, np.ndarray] | str | Path,
    n_clusters: int = 8,
    method: str = "kmeans",
    aggregate: str = "mean",
    output_dir: str | Path | None = None,
    dataset: BinnedTimeSeriesDataset | None = None,
    feature_mode: str = "explanation",
    value_weight: float = 1.0,
    explanation_weight: float = 1.0,
    mask_weight: float = 0.25,
) -> dict[str, object]:
    """Cluster patient-level vectors and return assignments/importance averages.

    Parameters can be a mapping of patient id to matrix or a directory of
    `.npy` explanation maps produced by `explain_model`.

    `feature_mode` controls what KMeans sees:

    - `"explanation"`: flattened explanation maps only.
    - `"value"`: flattened normalized value channel plus mask channel.
    - `"combined"`: value, mask, and explanation features concatenated.

    The returned `aggregates` are always cluster-level mean explanation maps.
    They are useful as the importance overlay when the displayed heatmap color
    represents actual clinical values.
    """
    if method != "kmeans":
        raise ValueError("Only kmeans clustering is currently supported.")
    if aggregate != "mean":
        raise ValueError("Only mean aggregation is currently supported.")
    if feature_mode not in CLUSTER_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}. Expected one of {sorted(CLUSTER_FEATURE_MODES)}.")
    maps = _load_explanations(explanations)
    patient_ids = _patient_ids_for_mode(maps, dataset, feature_mode)
    x_scaled = _build_cluster_features(
        maps,
        patient_ids,
        dataset=dataset,
        feature_mode=feature_mode,
        value_weight=value_weight,
        explanation_weight=explanation_weight,
        mask_weight=mask_weight,
    )
    k = max(1, min(n_clusters, len(patient_ids)))
    labels = KMeans(n_clusters=k, random_state=13, n_init=10).fit_predict(x_scaled)
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
            "feature_mode": feature_mode,
            "value_weight": value_weight,
            "explanation_weight": explanation_weight,
            "mask_weight": mask_weight,
            "n_clusters_requested": n_clusters,
            "n_clusters_used": k,
        }
        (out / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for cluster, matrix in aggregates.items():
            np.save(out / f"cluster_{cluster}.npy", matrix)
    return {"assignments": assignments, "aggregates": aggregates}


def _load_explanations(explanations: Mapping[str, np.ndarray] | str | Path) -> dict[str, np.ndarray]:
    if isinstance(explanations, Mapping):
        return {str(k): np.asarray(v) for k, v in explanations.items()}
    path = Path(explanations)
    return {p.stem: np.load(p) for p in sorted(path.glob("*.npy"))}


def _patient_ids_for_mode(
    maps: dict[str, np.ndarray],
    dataset: BinnedTimeSeriesDataset | None,
    feature_mode: str,
) -> list[str]:
    if feature_mode == "explanation":
        return list(maps.keys())
    if dataset is None or dataset.patient_ids is None:
        raise ValueError(f"feature_mode={feature_mode!r} requires a dataset with patient_ids.")
    available = set(maps)
    patient_ids = [str(patient_id) for patient_id in dataset.patient_ids if str(patient_id) in available]
    if not patient_ids:
        raise ValueError("No overlapping patient IDs between explanations and dataset.")
    return patient_ids


def _build_cluster_features(
    maps: dict[str, np.ndarray],
    patient_ids: list[str],
    dataset: BinnedTimeSeriesDataset | None,
    feature_mode: str,
    value_weight: float,
    explanation_weight: float,
    mask_weight: float,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    if feature_mode in {"value", "combined"}:
        if dataset is None or dataset.patient_ids is None:
            raise ValueError(f"feature_mode={feature_mode!r} requires a dataset with patient_ids.")
        patient_to_index = {str(patient_id): idx for idx, patient_id in enumerate(dataset.patient_ids)}
        x = dataset.x.detach().cpu().numpy()
        values = np.stack([x[patient_to_index[pid], 0].reshape(-1) for pid in patient_ids])
        masks = np.stack([x[patient_to_index[pid], 1].reshape(-1) for pid in patient_ids])
        blocks.append(value_weight * _standardize_block(values))
        if mask_weight:
            blocks.append(mask_weight * _standardize_block(masks))
    if feature_mode in {"explanation", "combined"}:
        explanations = np.stack([maps[pid].reshape(-1) for pid in patient_ids])
        blocks.append(explanation_weight * _standardize_block(explanations))
    return np.concatenate(blocks, axis=1)


def _standardize_block(block: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(block)
