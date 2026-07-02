"""Cluster patients by flattened model-importance maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_explanations(
    explanations: Mapping[str, np.ndarray] | str | Path,
    n_clusters: int = 8,
    method: str = "kmeans",
    aggregate: str = "mean",
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Cluster flattened explanation maps and return assignments/averages.

    Parameters can be a mapping of patient id to matrix or a directory of
    `.npy` explanation maps produced by `explain_model`.
    """
    if method != "kmeans":
        raise ValueError("Only kmeans clustering is currently supported.")
    if aggregate != "mean":
        raise ValueError("Only mean aggregation is currently supported.")
    maps = _load_explanations(explanations)
    patient_ids = list(maps.keys())
    x = np.stack([maps[pid].reshape(-1) for pid in patient_ids])
    x_scaled = StandardScaler().fit_transform(x)
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
            "feature_mode": "explanation",
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
