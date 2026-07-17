"""Static HTML reports for experiment result review and comparison."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from .binning import TimeSeriesBinner
from .io import load_split
from .visualization import load_normal_ranges, normal_range_status_matrix


DEFAULT_TOP_IMPORTANCE_FRACTION = 0.10
DISPLAY_METADATA_NAME = "variable_display_metadata.json"
GROUP_ORDER = {
    "measurements": 0,
    "lab_tests": 1,
    "inputs": 2,
    "other": 3,
}
GROUP_LABELS = {
    "measurements": "Measurements",
    "lab_tests": "Lab Tests",
    "inputs": "Inputs / Medications",
    "other": "Other",
}


@dataclass(frozen=True)
class ExperimentReportSpec:
    """Input paths for one experiment in a static report."""

    run_dir: str | Path
    dataset_dir: str | Path | None = None
    name: str | None = None


def discover_experiment_specs(
    runs_root: str | Path,
    *,
    dataset_root: str | Path | None = None,
    split: str = "test",
) -> list[ExperimentReportSpec]:
    """Find reportable run directories under a parent directory."""
    root = Path(runs_root)
    if not root.exists():
        raise FileNotFoundError(root)
    dataset_root_path = Path(dataset_root) if dataset_root is not None else None
    specs: list[ExperimentReportSpec] = []
    for run_dir in sorted(path.parent for path in root.rglob("binner.json")):
        if not _is_reportable_run_dir(run_dir, split):
            continue
        dataset_dir = _infer_dataset_dir_for_run(root, run_dir, dataset_root_path)
        specs.append(
            ExperimentReportSpec(
                run_dir=run_dir,
                dataset_dir=dataset_dir,
                name=_relative_run_name(root, run_dir),
            )
        )
    return specs


def build_experiment_report(
    experiments: Sequence[ExperimentReportSpec | Mapping[str, object] | tuple[object, ...] | str | Path],
    output_path: str | Path,
    *,
    split: str = "test",
    top_importance_fraction: float = DEFAULT_TOP_IMPORTANCE_FRACTION,
    mimic_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a self-contained HTML report and return its embedded payload."""
    specs = [_coerce_experiment_spec(experiment) for experiment in experiments]
    if not specs:
        raise ValueError("At least one experiment is required.")
    if mimic_path is not None:
        write_mimic_display_metadata_for_specs(specs, mimic_path)
    payload = {
        "split": split,
        "top_importance_fraction": float(top_importance_fraction),
        "experiments": [
            experiment_report_payload(spec, split=split, top_importance_fraction=top_importance_fraction)
            for spec in specs
        ],
    }
    html_text = render_report_html(payload)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return payload


def experiment_report_payload(
    spec: ExperimentReportSpec,
    *,
    split: str = "test",
    top_importance_fraction: float = DEFAULT_TOP_IMPORTANCE_FRACTION,
) -> dict[str, Any]:
    """Create the serializable report payload for one experiment."""
    run_dir = Path(spec.run_dir)
    dataset_dir = Path(spec.dataset_dir) if spec.dataset_dir is not None else None
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    binner = TimeSeriesBinner.load(run_dir / "binner.json")
    dataset = load_split(run_dir / f"{split}.npz")
    dataset_metadata = _load_json(dataset_dir / "dataset_metadata.json") if dataset_dir else {}
    display_metadata = load_variable_display_metadata(dataset_dir, dataset_metadata, binner.variable_vocab_)
    variables = order_variables(binner.variable_vocab_, display_metadata)
    normal_ranges = load_normal_ranges()
    assignments = _load_assignments(run_dir, split, dataset)
    patterns = class_pattern_payloads(
        dataset,
        binner,
        assignments,
        run_dir / "explanations" / split,
        variables,
        display_metadata,
        normal_ranges,
        top_importance_fraction=top_importance_fraction,
    )
    return {
        "name": spec.name or run_dir.name,
        "run_dir": str(run_dir),
        "dataset_dir": None if dataset_dir is None else str(dataset_dir),
        "configuration": configuration_summary(run_dir, binner, dataset_metadata, split),
        "metrics": metrics_summary(run_dir, split),
        "statistics": statistics_summary(run_dir, split),
        "variable_details": [variable_detail_payload(variable, display_metadata, normal_ranges) for variable in variables],
        "class_patterns": patterns,
    }


def class_pattern_payloads(
    dataset,
    binner: TimeSeriesBinner,
    assignments: pd.DataFrame,
    explanations_dir: str | Path,
    ordered_variables: Sequence[str] | None = None,
    display_metadata: Mapping[str, Mapping[str, object]] | None = None,
    normal_ranges: Mapping[str, object] | None = None,
    *,
    top_importance_fraction: float = DEFAULT_TOP_IMPORTANCE_FRACTION,
) -> list[dict[str, Any]]:
    """Return class-level mean value and importance patterns."""
    if dataset.patient_ids is None:
        raise ValueError("Dataset must include patient_ids for class pattern reports.")
    top_importance_fraction = _validate_top_fraction(top_importance_fraction)
    patient_ids = [str(patient_id) for patient_id in dataset.patient_ids]
    patient_to_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    values = _denormalized_value_tensor(dataset, binner)
    explanations_dir = Path(explanations_dir)
    assignment_frame = assignments.copy()
    assignment_frame["patient_id"] = assignment_frame["patient_id"].astype(str)
    if "predicted_label" not in assignment_frame.columns:
        raise ValueError("Assignments or predictions must include predicted_label.")
    ordered_variables = list(ordered_variables or binner.variable_vocab_)
    row_indices = [binner.variable_vocab_.index(variable) for variable in ordered_variables]
    normal_ranges = normal_ranges or {}
    display_metadata = display_metadata or {}

    patterns: list[dict[str, Any]] = []
    for class_label, group in assignment_frame.groupby("predicted_label", sort=True):
        class_patient_ids = [patient_id for patient_id in group["patient_id"].astype(str) if patient_id in patient_to_index]
        if not class_patient_ids:
            continue
        indices = [patient_to_index[patient_id] for patient_id in class_patient_ids]
        value_matrix = nanmean_without_warning(values[indices], axis=0)
        importance_matrix = mean_importance_for_patient_ids(explanations_dir, class_patient_ids)
        sparse_values, top_mask = top_importance_sparse_values(value_matrix, importance_matrix, top_importance_fraction)
        ordered_values = sparse_values[row_indices]
        ordered_importance = importance_matrix[row_indices]
        ordered_mask = top_mask[row_indices]
        range_status = normal_range_status_matrix(ordered_values, ordered_variables, normal_ranges)
        patterns.append(
            {
                "class_label": str(class_label),
                "n_patients": int(len(class_patient_ids)),
                "variables": [
                    variable_detail_payload(variable, display_metadata, normal_ranges)
                    for variable in ordered_variables
                ],
                "time_bins": [str(value) for value in binner.time_bins_],
                "values": _matrix_to_json(ordered_values),
                "importance": _matrix_to_json(ordered_importance),
                "range_status": _matrix_to_json(range_status),
                "top_mask": ordered_mask.astype(bool).tolist(),
            }
        )
    return patterns


def dataset_value_tensor_for_similarity(dataset, binner: TimeSeriesBinner, value_scale: str = "raw") -> np.ndarray:
    """Return observed value tensor with missing cells represented as NaN."""
    if value_scale == "raw":
        return _denormalized_value_tensor(dataset, binner)
    if value_scale == "z_score":
        x = dataset.x.detach().cpu().numpy()
        values = x[:, 0].astype(np.float64)
        observed = x[:, 1].astype(bool)
        return np.where(observed, values, np.nan)
    raise ValueError("value_scale must be 'raw' or 'z_score'.")


def mean_importance_for_patient_ids(explanations_dir: str | Path, patient_ids: Sequence[str]) -> np.ndarray:
    """Mean explanation matrix for patients with available `.npy` explanations."""
    total = None
    count = 0
    explanations_dir = Path(explanations_dir)
    for patient_id in patient_ids:
        path = explanations_dir / f"{patient_id}.npy"
        if not path.exists():
            continue
        matrix = np.load(path).astype(np.float64)
        total = matrix if total is None else total + matrix
        count += 1
    if count == 0:
        raise ValueError("No explanation matrices found for the requested patients.")
    return total / count


def nanmean_without_warning(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """NaN-aware mean without all-NaN slice warnings."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    totals = np.nansum(values, axis=axis)
    result = np.full_like(totals, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def top_importance_sparse_values(
    value_matrix: np.ndarray,
    importance_matrix: np.ndarray,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep values only in cells whose importance is in the top fraction."""
    top_fraction = _validate_top_fraction(top_fraction)
    value_matrix = np.asarray(value_matrix, dtype=np.float64)
    importance_matrix = np.asarray(importance_matrix, dtype=np.float64)
    if value_matrix.shape != importance_matrix.shape:
        raise ValueError("value_matrix and importance_matrix must have the same shape.")
    finite_mask = np.isfinite(value_matrix) & np.isfinite(importance_matrix)
    finite_importance = importance_matrix[finite_mask]
    if finite_importance.size == 0:
        return np.full_like(value_matrix, np.nan, dtype=np.float64), np.zeros_like(value_matrix, dtype=bool)
    cutoff = float(np.quantile(finite_importance, 1.0 - top_fraction))
    top_mask = finite_mask & (importance_matrix >= cutoff)
    sparse = np.full_like(value_matrix, np.nan, dtype=np.float64)
    sparse[top_mask] = value_matrix[top_mask]
    return sparse, top_mask


def load_variable_display_metadata(
    dataset_dir: str | Path | None,
    dataset_metadata: Mapping[str, object] | None,
    variables: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Load or infer dataset-specific variable display metadata."""
    explicit: dict[str, dict[str, object]] = {}
    if dataset_dir is not None:
        metadata_path = Path(dataset_dir) / DISPLAY_METADATA_NAME
        if metadata_path.exists():
            raw = _load_json(metadata_path)
            explicit = {str(key): dict(value) for key, value in raw.get("variables", raw).items()}
    inferred = infer_variable_display_metadata(dataset_metadata or {}, variables)
    for variable, spec in explicit.items():
        inferred.setdefault(variable, {}).update(spec)
    return {variable: _normalized_variable_metadata(variable, inferred.get(variable, {})) for variable in variables}


def infer_variable_display_metadata(
    dataset_metadata: Mapping[str, object],
    variables: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Infer generic display metadata from prepared dataset metadata."""
    mappings = dataset_metadata.get("variable_mappings", {}) if isinstance(dataset_metadata, Mapping) else {}
    mappings = mappings if isinstance(mappings, Mapping) else {}
    source_by_variable: dict[str, str] = {}
    itemids_by_variable: dict[str, list[object]] = {}
    for source_name, group in [
        ("lab_tests", mappings.get("lab_itemids", {})),
        ("measurements", mappings.get("chart_itemids", {})),
        ("inputs", mappings.get("inputevent_itemids", {})),
        ("inputs", mappings.get("drug_regexes", {})),
    ]:
        if not isinstance(group, Mapping):
            continue
        for variable, items in group.items():
            canonical = _canonical_variable_name(str(variable))
            source_by_variable[canonical] = source_name
            source_by_variable[str(variable)] = source_name
            itemids_by_variable[canonical] = list(items if isinstance(items, list) else [items])
            itemids_by_variable[str(variable)] = list(items if isinstance(items, list) else [items])

    metadata: dict[str, dict[str, object]] = {}
    for variable in variables:
        group = source_by_variable.get(variable, "other")
        items = itemids_by_variable.get(variable, [])
        metadata[variable] = {
            "display_name": readable_variable_name(variable),
            "group": group,
            "items": [_metadata_item_payload(item) for item in items],
        }
    conversions = dataset_metadata.get("unit_conversions", {}) if isinstance(dataset_metadata, Mapping) else {}
    if isinstance(conversions, Mapping):
        for variable, note in conversions.items():
            canonical = _canonical_variable_name(str(variable))
            if canonical in metadata:
                metadata[canonical]["note"] = str(note)
    return metadata


def order_variables(
    variables: Sequence[str],
    display_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    """Order variables by display group while preserving per-group input order."""
    display_metadata = display_metadata or {}
    indexed = list(enumerate(variables))
    indexed.sort(key=lambda item: (GROUP_ORDER.get(str(display_metadata.get(item[1], {}).get("group", "other")), 99), item[0]))
    return [variable for _, variable in indexed]


def variable_detail_payload(
    variable: str,
    display_metadata: Mapping[str, Mapping[str, object]] | None,
    normal_ranges: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Serializable display details for one variable row."""
    spec = _normalized_variable_metadata(variable, (display_metadata or {}).get(variable, {}))
    range_spec = (normal_ranges or {}).get(variable)
    if isinstance(range_spec, Mapping):
        spec["normal_range"] = {
            "low": _json_number_or_none(range_spec.get("low")),
            "high": _json_number_or_none(range_spec.get("high")),
            "unit": range_spec.get("unit"),
            "note": range_spec.get("note"),
        }
    else:
        spec["normal_range"] = None
    return spec


def configuration_summary(run_dir: Path, binner: TimeSeriesBinner, dataset_metadata: Mapping[str, object], split: str) -> dict[str, Any]:
    """Small set of reportable configuration values."""
    model_config = _load_model_config(run_dir)
    cluster_metadata = _load_json(run_dir / "clusters" / split / "cluster_metadata.json")
    window = dataset_metadata.get("window", {}) if isinstance(dataset_metadata, Mapping) else {}
    target_metadata = dataset_metadata.get("target_metadata", {}) if isinstance(dataset_metadata, Mapping) else {}
    return {
        "target": dataset_metadata.get("target", "not recorded") if isinstance(dataset_metadata, Mapping) else "not recorded",
        "target_definition": target_metadata.get("target_definition", "not recorded") if isinstance(target_metadata, Mapping) else "not recorded",
        "observation_hours": window.get("observation_hours", "not recorded") if isinstance(window, Mapping) else "not recorded",
        "gap_hours": window.get("gap_hours", "not recorded") if isinstance(window, Mapping) else "not recorded",
        "prediction_hours": window.get("prediction_hours", "not recorded") if isinstance(window, Mapping) else "not recorded",
        "binning_interval": binner.config.granularity,
        "time_bins": len(binner.time_bins_),
        "variables": len(binner.variable_vocab_),
        "patch_size": model_config.get("patch_size", "not recorded"),
        "embed_dim": model_config.get("embed_dim", "not recorded"),
        "depth": model_config.get("depth", "not recorded"),
        "num_heads": model_config.get("num_heads", "not recorded"),
        "cluster_method": cluster_metadata.get("clustering_method", "not recorded"),
        "cluster_feature_mode": cluster_metadata.get("feature_mode", "not recorded"),
        "clusters_used": cluster_metadata.get("n_clusters_used", "not recorded"),
    }


def metrics_summary(run_dir: Path, split: str) -> dict[str, Any]:
    """Load report metrics, preferring split-specific test metrics."""
    metrics = _load_json(run_dir / f"{split}_evaluation_metrics.json")
    train_metrics = _load_json(run_dir / "metrics.json")
    if not metrics and train_metrics:
        metrics = {key: value for key, value in train_metrics.items() if key != "history"}
    return {
        "accuracy": _json_number_or_none(metrics.get("accuracy")),
        "macro_f1": _json_number_or_none(metrics.get("macro_f1")),
        "auroc": _json_number_or_none(metrics.get("auroc", metrics.get("auc"))),
        "auc": _json_number_or_none(metrics.get("auc", metrics.get("auroc"))),
        "tpr": _json_number_or_none(metrics.get("tpr")),
        "fpr": _json_number_or_none(metrics.get("fpr")),
        "tnr": _json_number_or_none(metrics.get("tnr")),
        "fnr": _json_number_or_none(metrics.get("fnr")),
        "ppv": _json_number_or_none(metrics.get("ppv")),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "best_epoch": train_metrics.get("best_epoch"),
        "epochs_ran": train_metrics.get("epochs_ran"),
    }


def statistics_summary(run_dir: Path, split: str) -> dict[str, Any]:
    """Load statistical-test artifacts for the selected split."""
    class_tests_path = run_dir / f"{split}_class_similarity_tests.csv"
    pattern_similarity_path = run_dir / f"{split}_pattern_similarity.csv"
    summary: dict[str, Any] = {
        "class_similarity_tests": _csv_records(class_tests_path),
        "pattern_similarity_summary": [],
    }
    if pattern_similarity_path.exists():
        frame = pd.read_csv(pattern_similarity_path)
        similarity_columns = [column for column in frame.columns if str(column).startswith("similarity_to_")]
        if similarity_columns:
            described = frame[similarity_columns].describe().transpose().reset_index(names="metric")
            summary["pattern_similarity_summary"] = _json_records(described)
    return summary


def write_mimic_display_metadata_for_specs(
    specs: Sequence[ExperimentReportSpec],
    mimic_path: str | Path,
) -> None:
    """Generate dictionary-backed display metadata for datasets used by specs."""
    dictionaries = _load_mimic_item_dictionaries(mimic_path)
    for spec in specs:
        if spec.dataset_dir is None:
            continue
        write_mimic_variable_display_metadata(spec.dataset_dir, dictionaries)


def write_mimic_variable_display_metadata(
    dataset_dir: str | Path,
    dictionaries: Mapping[str, Mapping[str, str]] | str | Path,
) -> Path:
    """Write variable display metadata with MIMIC item names when available."""
    dataset_dir = Path(dataset_dir)
    metadata = _load_json(dataset_dir / "dataset_metadata.json")
    if not metadata:
        raise FileNotFoundError(dataset_dir / "dataset_metadata.json")
    if not isinstance(dictionaries, Mapping):
        dictionaries = _load_mimic_item_dictionaries(dictionaries)
    variable_metadata = infer_variable_display_metadata(metadata, _variables_from_metadata(metadata))
    mappings = metadata.get("variable_mappings", {}) if isinstance(metadata, Mapping) else {}
    mappings = mappings if isinstance(mappings, Mapping) else {}
    sources = [
        ("lab_itemids", "lab_tests", dictionaries.get("lab", {})),
        ("chart_itemids", "measurements", dictionaries.get("item", {})),
        ("inputevent_itemids", "inputs", dictionaries.get("item", {})),
    ]
    for mapping_key, group, name_by_id in sources:
        group_mapping = mappings.get(mapping_key, {})
        if not isinstance(group_mapping, Mapping):
            continue
        for raw_variable, raw_items in group_mapping.items():
            variable = _canonical_variable_name(str(raw_variable))
            items = list(raw_items if isinstance(raw_items, list) else [raw_items])
            variable_metadata.setdefault(variable, {"display_name": readable_variable_name(variable)})
            variable_metadata[variable]["group"] = group
            variable_metadata[variable]["items"] = [_mimic_item_payload(item, name_by_id) for item in items]
    conversions = metadata.get("unit_conversions", {}) if isinstance(metadata, Mapping) else {}
    if isinstance(conversions, Mapping):
        for variable, note in conversions.items():
            canonical = _canonical_variable_name(str(variable))
            variable_metadata.setdefault(canonical, {"display_name": readable_variable_name(canonical)})
            variable_metadata[canonical]["note"] = str(note)
    output = dataset_dir / DISPLAY_METADATA_NAME
    output.write_text(json.dumps({"variables": variable_metadata}, indent=2, sort_keys=True), encoding="utf-8")
    return output


def render_report_html(payload: Mapping[str, object]) -> str:
    """Render the static report HTML document."""
    payload_json = _json_script_escape(json.dumps(payload, allow_nan=False))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Results Report</title>
  <style>{_REPORT_CSS}</style>
</head>
<body>
  <header>
    <div>
      <h1>Experiment Results</h1>
      <p id="subtitle"></p>
    </div>
    <div class="header-controls">
      <select id="experimentSelect" aria-label="Experiment"></select>
      <button id="compareButton" type="button">Compare</button>
    </div>
  </header>
  <main>
    <section id="experimentView" class="experiment-view"></section>
    <section id="compareView" class="compare-view hidden">
      <div class="compare-toolbar">
        <select id="leftExperiment"></select>
        <select id="rightExperiment"></select>
      </div>
      <div class="compare-grid">
        <div id="leftPane"></div>
        <div id="rightPane"></div>
      </div>
    </section>
  </main>
  <script id="report-data" type="application/json">{payload_json}</script>
  <script>{_REPORT_JS}</script>
</body>
</html>
"""


def _coerce_experiment_spec(experiment: ExperimentReportSpec | Mapping[str, object] | tuple[object, ...] | str | Path) -> ExperimentReportSpec:
    if isinstance(experiment, ExperimentReportSpec):
        return experiment
    if isinstance(experiment, Mapping):
        return ExperimentReportSpec(
            run_dir=experiment["run_dir"],
            dataset_dir=experiment.get("dataset_dir"),
            name=experiment.get("name"),
        )
    if isinstance(experiment, tuple):
        if len(experiment) == 2:
            return ExperimentReportSpec(run_dir=experiment[0], dataset_dir=experiment[1])
        if len(experiment) == 3:
            return ExperimentReportSpec(run_dir=experiment[0], dataset_dir=experiment[1], name=str(experiment[2]))
    return ExperimentReportSpec(run_dir=experiment)


def _json_script_escape(value: str) -> str:
    return value.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _is_reportable_run_dir(run_dir: Path, split: str) -> bool:
    required = [
        run_dir / "binner.json",
        run_dir / "variable_vocab.json",
        run_dir / f"{split}.npz",
        run_dir / f"{split}_evaluation_metrics.json",
        run_dir / "explanations" / split,
    ]
    return all(path.exists() for path in required)


def _infer_dataset_dir_for_run(runs_root: Path, run_dir: Path, dataset_root: Path | None) -> Path | None:
    if dataset_root is None:
        return None
    try:
        relative = run_dir.relative_to(runs_root)
    except ValueError:
        return None
    candidate = dataset_root / relative
    return candidate if (candidate / "dataset_metadata.json").exists() else None


def _relative_run_name(runs_root: Path, run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(runs_root))
    except ValueError:
        return run_dir.name


def _load_assignments(run_dir: Path, split: str, dataset) -> pd.DataFrame:
    assignments_path = run_dir / "clusters" / split / "cluster_assignments.csv"
    if assignments_path.exists():
        return pd.read_csv(assignments_path)
    predictions_path = run_dir / f"{split}_predictions.csv"
    if predictions_path.exists():
        return pd.read_csv(predictions_path)
    legacy_predictions = run_dir / "predictions.csv"
    if legacy_predictions.exists():
        return pd.read_csv(legacy_predictions)
    if dataset.patient_ids is None:
        raise FileNotFoundError(assignments_path)
    labels = ["all"] * len(dataset.patient_ids)
    return pd.DataFrame({"patient_id": [str(patient_id) for patient_id in dataset.patient_ids], "predicted_label": labels})


def _denormalized_value_tensor(dataset, binner: TimeSeriesBinner) -> np.ndarray:
    x = _as_numpy(dataset.x)
    values = x[:, 0].astype(np.float64)
    mask = x[:, 1].astype(bool)
    means = np.array([binner.means_.get(variable, 0.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    stds = np.array([binner.stds_.get(variable, 1.0) for variable in binner.variable_vocab_], dtype=np.float64)[:, None]
    raw_values = values * stds[None, :, :] + means[None, :, :]
    return np.where(mask, raw_values, np.nan)


def _as_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _matrix_to_json(matrix: np.ndarray) -> list[list[float | None]]:
    out = []
    for row in np.asarray(matrix, dtype=np.float64):
        out.append([None if not np.isfinite(value) else float(value) for value in row])
    return out


def _variables_from_metadata(dataset_metadata: Mapping[str, object]) -> list[str]:
    mappings = dataset_metadata.get("variable_mappings", {}) if isinstance(dataset_metadata, Mapping) else {}
    mappings = mappings if isinstance(mappings, Mapping) else {}
    variables: list[str] = []
    for key in ["chart_itemids", "lab_itemids", "inputevent_itemids", "drug_regexes"]:
        group = mappings.get(key, {})
        if not isinstance(group, Mapping):
            continue
        for variable in group:
            canonical = _canonical_variable_name(str(variable))
            if canonical not in variables:
                variables.append(canonical)
    return variables


def _mimic_item_payload(item: object, name_by_id: Mapping[str, str]) -> dict[str, object]:
    if isinstance(item, Mapping):
        item_id = str(item.get("id", item.get("itemid", "")))
        item_name = item.get("name", item.get("label"))
        if item_name:
            return {"id": item_id, "name": str(item_name)}
    else:
        item_id = str(item)
    name = name_by_id.get(item_id)
    if name:
        return {"id": item_id, "name": name}
    return {"id": item_id, "name": "name unavailable", "missing_name": True}


def _metadata_item_payload(item: object) -> dict[str, object]:
    if isinstance(item, Mapping):
        item_id = str(item.get("id", item.get("itemid", "")))
        name = item.get("name", item.get("label"))
        if name:
            return {"id": item_id, "name": str(name)}
        return {"id": item_id, "name": "name unavailable", "missing_name": True}
    return {"id": str(item), "name": "name unavailable", "missing_name": True}


def _load_mimic_item_dictionaries(mimic_path: str | Path) -> dict[str, dict[str, str]]:
    path = Path(mimic_path)
    if not path.exists():
        raise FileNotFoundError(path)
    lab_frame = _read_mimic_dictionary_csv(path, "hosp/d_labitems.csv.gz")
    item_frame = _read_mimic_dictionary_csv(path, "icu/d_items.csv.gz")
    return {
        "lab": _item_name_map(lab_frame),
        "item": _item_name_map(item_frame),
    }


def _read_mimic_dictionary_csv(path: Path, suffix: str) -> pd.DataFrame:
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.endswith(suffix)]
            if not candidates:
                raise FileNotFoundError(f"{suffix} was not found in {path}")
            with archive.open(candidates[0]) as fh:
                return pd.read_csv(fh, compression="gzip")
    direct = path / suffix
    if direct.exists():
        return pd.read_csv(direct)
    matches = sorted(path.rglob(Path(suffix).name))
    matches = [match for match in matches if str(match).endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"{suffix} was not found under {path}")
    return pd.read_csv(matches[0])


def _item_name_map(frame: pd.DataFrame) -> dict[str, str]:
    if "itemid" not in frame.columns:
        return {}
    name_column = "label" if "label" in frame.columns else None
    if name_column is None:
        return {}
    names: dict[str, str] = {}
    for row in frame[["itemid", name_column]].to_dict("records"):
        item_id = str(row["itemid"])
        label = row.get(name_column)
        if pd.notna(label) and str(label).strip():
            names[item_id] = str(label).strip()
    return names


def _csv_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return _json_records(pd.read_csv(path))


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for raw in frame.to_dict("records"):
        records.append({str(key): _json_scalar(value) for key, value in raw.items()})
    return records


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if pd.isna(value):
        return None
    return value


def _normalized_variable_metadata(variable: str, spec: Mapping[str, object]) -> dict[str, object]:
    group = str(spec.get("group", "other"))
    if group not in GROUP_LABELS:
        group = "other"
    items = spec.get("items", [])
    normalized_items = []
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        for item in items:
            if isinstance(item, Mapping):
                item_id = item.get("id", item.get("itemid", ""))
                name = item.get("name", item.get("label", item_id))
                missing_name = bool(item.get("missing_name", False))
            else:
                item_id = item
                name = "name unavailable"
                missing_name = True
            normalized_items.append({"id": str(item_id), "name": str(name), "missing_name": missing_name})
    return {
        "variable": variable,
        "display_name": str(spec.get("display_name", readable_variable_name(variable))),
        "group": group,
        "group_label": GROUP_LABELS[group],
        "items": normalized_items,
        "note": spec.get("note"),
    }


def readable_variable_name(variable: str) -> str:
    """Convert machine variable names into report labels."""
    acronyms = {"bp": "BP", "wbc": "WBC", "bun": "BUN", "pco2": "PCO2", "po2": "PO2", "spo2": "SpO2", "o2": "O2", "iv": "IV"}
    words = re.split(r"[_\s]+", str(variable).strip())
    return " ".join(acronyms.get(word.lower(), word.capitalize()) for word in words if word)


def _canonical_variable_name(variable: str) -> str:
    if variable in {"temperature_c", "temperature_f"}:
        return "temperature"
    return variable


def _validate_top_fraction(top_fraction: float) -> float:
    top_fraction = float(top_fraction)
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    return top_fraction


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh) or {}


def _load_model_config(run_dir: Path) -> dict[str, Any]:
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        return {}
    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception:
        return {}
    config = checkpoint.get("config", {}) if isinstance(checkpoint, Mapping) else {}
    return dict(config) if isinstance(config, Mapping) else {}


def _json_number_or_none(value: object) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


_REPORT_CSS = """
:root { color-scheme: light; --ink: #18212f; --muted: #667085; --line: #d8dee8; --panel: #ffffff; --soft: #f5f7fa; --accent: #0f766e; --low: #1e88e5; --normal: #f7f7f7; --high: #ff0051; }
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #eef2f6; }
header { position: sticky; top: 0; z-index: 5; display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 18px 24px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.94); backdrop-filter: blur(10px); }
h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
h2 { margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }
h3 { margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }
p { margin: 0; color: var(--muted); }
button, select { border: 1px solid var(--line); background: var(--panel); color: var(--ink); border-radius: 6px; padding: 9px 12px; font: inherit; }
button { cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); font-weight: 650; }
main { padding: 22px; }
.hidden { display: none !important; }
.header-controls { display: flex; gap: 10px; align-items: center; min-width: min(620px, 52vw); }
.header-controls select { flex: 1; min-width: 260px; }
.experiment-view { display: block; }
.experiment-card, .compare-pane { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.table-grid { display: grid; gap: 14px; margin: 8px 0 16px; }
.info-table, .stats-table { width: 100%; border-collapse: collapse; border: 1px solid var(--line); background: white; font-size: 12px; }
.info-table caption, .stats-table caption { text-align: left; font-weight: 800; font-size: 14px; padding: 0 0 7px; color: var(--ink); }
.info-table th, .info-table td, .stats-table th, .stats-table td { border-top: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
.info-table th { color: var(--muted); font-weight: 650; background: var(--soft); }
.stats-table th { color: var(--muted); font-weight: 750; background: var(--soft); white-space: nowrap; }
.stats-section { display: grid; gap: 12px; margin: 6px 0 16px; }
.empty-note { padding: 9px 10px; border: 1px solid var(--line); background: var(--soft); border-radius: 6px; color: var(--muted); font-size: 12px; }
.pattern-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px; }
.pattern-tabs button { background: white; color: var(--ink); border-color: var(--line); font-weight: 600; }
.pattern-tabs button.active { background: var(--ink); color: white; border-color: var(--ink); }
.pattern-layout { display: grid; gap: 10px; align-items: start; }
.heatmap-row { display: flex; gap: 10px; align-items: stretch; min-width: 0; }
.heatmap-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; max-height: 58vh; }
.heatmap { display: grid; padding: 8px; gap: 1px; width: max-content; min-width: 100%; }
.corner, .time-label, .row-label, .cell, .group-row { min-height: 16px; }
.corner, .time-label { color: var(--muted); font-size: 11px; display: flex; align-items: end; justify-content: center; padding: 2px; }
.row-label { position: sticky; left: 0; z-index: 2; background: #fff; display: flex; align-items: center; padding: 0 6px; border-right: 1px solid var(--line); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.group-row { grid-column: 1 / -1; display: flex; align-items: center; padding: 6px 4px 2px; color: var(--accent); font-size: 11px; font-weight: 800; border-bottom: 2px solid color-mix(in srgb, var(--accent), white 72%); }
.cell { position: relative; border: 1px solid rgba(255,255,255,.72); min-width: 12px; border-radius: 2px; background: #d9dee7; }
.cell[data-value]:hover::after { content: attr(data-value); position: absolute; left: 50%; bottom: calc(100% + 4px); transform: translateX(-50%); z-index: 10; padding: 3px 5px; border: 1px solid var(--line); border-radius: 4px; background: var(--ink); color: #fff; font-size: 10px; white-space: nowrap; pointer-events: none; box-shadow: 0 2px 6px rgba(16,24,40,.18); }
.cell.empty { background: repeating-linear-gradient(135deg, #eef1f5, #eef1f5 4px, #dde3eb 4px, #dde3eb 8px); }
.clinical-legend { width: 58px; display: grid; grid-template-rows: auto 1fr auto; gap: 6px; justify-items: center; color: var(--muted); font-size: 11px; }
.legend-title { writing-mode: vertical-rl; transform: rotate(180deg); font-weight: 800; color: var(--ink); text-align: center; }
.legend-bar { width: 16px; min-height: 150px; border: 1px solid var(--line); border-radius: 999px; background: linear-gradient(to top, var(--low), #f7f7f7 50%, var(--high)); }
.legend-labels { display: grid; gap: 47px; justify-items: center; }
.variable-description-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; margin-top: 8px; }
.detail-row { padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
.detail-title { font-weight: 750; }
.detail-meta { margin-top: 3px; color: var(--muted); font-size: 12px; }
.items { margin: 6px 0 0; padding-left: 18px; color: #344054; font-size: 12px; line-height: 1.35; }
.missing-name { color: #9a3412; font-weight: 650; }
.compare-toolbar { display: flex; gap: 12px; margin-bottom: 14px; }
.compare-toolbar select { flex: 1; }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.compare-pane .table-grid { grid-template-columns: 1fr; }
.compare-pane .heatmap-wrap { max-height: 48vh; }
.compare-pane .cell { min-width: 10px; }
.compare-pane .row-label { font-size: 10px; }
@media (max-width: 1100px) { .table-grid, .compare-grid { grid-template-columns: 1fr; } main { padding: 14px; } header { align-items: flex-start; flex-direction: column; } .header-controls { width: 100%; min-width: 0; } .header-controls select { min-width: 0; } }
"""


_REPORT_JS = r"""
const payload = JSON.parse(document.getElementById('report-data').textContent);
const fmt = (value, digits = 3) => value === null || value === undefined ? 'not recorded' : (typeof value === 'number' ? value.toFixed(digits).replace(/\.?0+$/, '') : String(value));
const pct = value => value === null || value === undefined ? 'not recorded' : `${(value * 100).toFixed(1)}%`;
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));

document.getElementById('subtitle').textContent = `${payload.experiments.length} experiment${payload.experiments.length === 1 ? '' : 's'} · ${payload.split} split · top ${(payload.top_importance_fraction * 100).toFixed(0)}% importance cells`;

function renderSingleRowTable(title, columns, row, className = 'info-table') {
  return `<table class="${className}"><caption>${escapeHtml(title)}</caption><thead><tr>${columns.map(col => `<th>${escapeHtml(col.label)}</th>`).join('')}</tr></thead><tbody><tr>${columns.map(col => `<td>${escapeHtml(formatCell(row[col.key]))}</td>`).join('')}</tr></tbody></table>`;
}

function configTable(config) {
  const row = {
    target: config.target,
    target_definition: config.target_definition,
    observation: `${fmt(config.observation_hours, 0)}h`,
    gap: Number(config.gap_hours) === 0 ? 'None' : `${fmt(config.gap_hours, 0)}h`,
    binning_interval: config.binning_interval,
    patch_size: Array.isArray(config.patch_size) ? config.patch_size.join(' x ') : config.patch_size,
  };
  const columns = [
    {key: 'target', label: 'Target'},
    {key: 'target_definition', label: 'Target definition'},
    {key: 'observation', label: 'Observation'},
    {key: 'gap', label: 'gap'},
    {key: 'binning_interval', label: 'Binning interval'},
    {key: 'patch_size', label: 'Patch size'},
  ];
  return renderSingleRowTable('Configuration', columns, row);
}

function metricsTable(metrics) {
  const row = {
    auc: metrics.auc ?? metrics.auroc,
    tpr: metrics.tpr,
    fpr: metrics.fpr,
    tnr: metrics.tnr,
    fnr: metrics.fnr,
    ppv: metrics.ppv,
    macro_f1: metrics.macro_f1,
    best_epoch: metrics.best_epoch,
    epochs_ran: metrics.epochs_ran,
  };
  const columns = [
    {key: 'auc', label: 'AUC'},
    {key: 'tpr', label: 'TPR'},
    {key: 'fpr', label: 'FPR'},
    {key: 'tnr', label: 'TNR'},
    {key: 'fnr', label: 'FNR'},
    {key: 'ppv', label: 'PPV'},
    {key: 'macro_f1', label: 'Macro F1'},
    {key: 'best_epoch', label: 'Best epoch'},
    {key: 'epochs_ran', label: 'Epochs ran'},
  ];
  return renderSingleRowTable('Test Results', columns, row);
}

function renderTables(experiment) {
  return `<div class="table-grid">${configTable(experiment.configuration)}${metricsTable(experiment.metrics)}</div>`;
}

function renderRecordTable(title, rows, columns, className = '') {
  if (!rows || rows.length === 0) return `<div class="empty-note">${escapeHtml(title)}: not recorded.</div>`;
  return `<table class="stats-table ${className}"><caption>${escapeHtml(title)}</caption><thead><tr>${columns.map(col => `<th>${escapeHtml(col.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${columns.map(col => `<td>${escapeHtml(formatCell(row[col.key]))}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

function formatCell(value) {
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return fmt(value, Math.abs(value) < 0.001 && value !== 0 ? 4 : 3);
  return value ?? 'not recorded';
}

function renderStatistics(experiment) {
  const stats = experiment.statistics || {};
  const rows = (stats.class_similarity_tests || []).map(row => ({
    pattern_class: row.pattern,
    most_similar_class: row.more_similar_class,
    mean_difference: row.mean_difference,
    p_value: row.p_value,
    is_significant: row.is_significant,
  }));
  const columns = [
    {key: 'pattern_class', label: 'Pattern of class'},
    {key: 'most_similar_class', label: 'Most similar class'},
    {key: 'mean_difference', label: 'similarity mean diff'},
    {key: 'p_value', label: 'p-value'},
    {key: 'is_significant', label: 'Significant'},
  ];
  return `<section class="stats-section">${renderRecordTable('Pattern similarity statistical test', rows, columns, 'statistical-tests')}</section>`;
}

function lerp(a, b, t) { return Math.round(a + (b - a) * t); }
function hexToRgb(hex) {
  const v = hex.replace('#', '');
  return [parseInt(v.slice(0,2), 16), parseInt(v.slice(2,4), 16), parseInt(v.slice(4,6), 16)];
}
function mix(c1, c2, t) {
  const a = hexToRgb(c1), b = hexToRgb(c2);
  return `rgb(${lerp(a[0], b[0], t)}, ${lerp(a[1], b[1], t)}, ${lerp(a[2], b[2], t)})`;
}
function clinicalColor(status, value) {
  if (value === null || value === undefined) return null;
  if (typeof status !== 'number') return '#f7f7f7';
  if (status < 0) return mix('#1e88e5', '#f7f7f7', Math.max(0, Math.min(1, status + 1)));
  return mix('#f7f7f7', '#ff0051', Math.max(0, Math.min(1, status)));
}

function cellTitle(pattern, variable, row, col) {
  const value = pattern.values[row][col];
  const importance = pattern.importance[row][col];
  const status = pattern.range_status ? pattern.range_status[row][col] : null;
  const range = variable.normal_range;
  const rangeText = range ? `${fmt(range.low)}-${fmt(range.high)}${range.unit ? ' ' + range.unit : ''}` : 'not recorded';
  return `${variable.display_name}\nTime: ${pattern.time_bins[col]}\nValue: ${fmt(value)}\nImportance: ${fmt(importance)}\nClinical status: ${fmt(status)}\nNormal range: ${rangeText}\nGroup: ${variable.group_label}`;
}

function renderHeatmap(pattern) {
  const times = pattern.time_bins;
  const variables = pattern.variables;
  const columns = `142px repeat(${times.length}, 12px)`;
  let html = `<div class="heatmap" style="grid-template-columns:${columns}"><div class="corner"></div>`;
  times.forEach((time, idx) => html += `<div class="time-label" title="${escapeHtml(time)}">${idx % Math.ceil(times.length / 6 || 1) === 0 ? idx : ''}</div>`);
  let lastGroup = null;
  variables.forEach((variable, row) => {
    if (variable.group !== lastGroup) {
      lastGroup = variable.group;
      html += `<div class="group-row">${escapeHtml(variable.group_label)}</div>`;
    }
    html += `<div class="row-label" title="${escapeHtml(variable.variable)}">${escapeHtml(variable.display_name)}</div>`;
    times.forEach((_, col) => {
      const value = pattern.values[row][col];
      const status = pattern.range_status ? pattern.range_status[row][col] : null;
      const color = clinicalColor(status, value);
      const style = color ? `background:${color}` : '';
      html += `<div class="cell ${value === null ? 'empty' : ''}" style="${style}" ${value === null ? '' : `data-value="${escapeHtml(fmt(value))}" title="${escapeHtml(cellTitle(pattern, variable, row, col))}"`}></div>`;
    });
  });
  html += '</div>';
  const legend = `<aside class="clinical-legend" aria-label="Clinical range status"><div class="legend-title">Clinical range status</div><div class="legend-labels"><span>High</span><span>Normal</span><span>Low</span></div><div class="legend-bar"></div></aside>`;
  return `<div class="heatmap-row"><div class="heatmap-wrap">${html}</div>${legend}</div>`;
}

function renderDetails(variables) {
  return `<div class="variable-description-grid">${variables.map(variable => {
    const range = variable.normal_range;
    const rangeText = range ? `${fmt(range.low)}-${fmt(range.high)}${range.unit ? ' ' + escapeHtml(range.unit) : ''}` : 'not recorded';
    const items = (variable.items || []).map(item => {
      const name = item.missing_name ? `<span class="missing-name">name unavailable</span>` : escapeHtml(item.name);
      return `<li>${name}${item.id ? ` (${escapeHtml(item.id)})` : ''}</li>`;
    }).join('');
    return `<div class="detail-row"><div class="detail-title">${escapeHtml(variable.display_name)}</div><div class="detail-meta">${escapeHtml(variable.group_label)} · normal: ${rangeText}</div>${variable.note ? `<div class="detail-meta">${escapeHtml(variable.note)}</div>` : ''}${items ? `<ul class="items">${items}</ul>` : ''}</div>`;
  }).join('')}</div>`;
}

function renderPatterns(experiment, idPrefix) {
  if (!experiment.class_patterns.length) return '<p>No class patterns available.</p>';
  const tabs = experiment.class_patterns.map((pattern, index) => `<button type="button" data-pattern="${index}" class="${index === 0 ? 'active' : ''}">${escapeHtml(pattern.class_label)} · n=${pattern.n_patients}</button>`).join('');
  const bodies = experiment.class_patterns.map((pattern, index) => `<div class="pattern-body ${index === 0 ? '' : 'hidden'}" data-pattern-body="${index}"><div class="pattern-layout">${renderHeatmap(pattern)}${renderDetails(pattern.variables)}</div></div>`).join('');
  return `<div class="pattern-block" id="${idPrefix}"><h3>Class Pattern Visualization</h3><div class="pattern-tabs">${tabs}</div>${bodies}</div>`;
}

function attachTabs(root) {
  root.querySelectorAll('.pattern-block').forEach(block => {
    block.querySelectorAll('[data-pattern]').forEach(button => {
      button.addEventListener('click', () => {
        block.querySelectorAll('[data-pattern]').forEach(b => b.classList.remove('active'));
        block.querySelectorAll('[data-pattern-body]').forEach(body => body.classList.add('hidden'));
        button.classList.add('active');
        block.querySelector(`[data-pattern-body="${button.dataset.pattern}"]`).classList.remove('hidden');
      });
    });
  });
}

function renderExperiment(experiment, index, compact = false) {
  return `<article class="${compact ? 'compare-pane' : 'experiment-card'}"><h2>${escapeHtml(experiment.name)}</h2>${renderTables(experiment)}${renderStatistics(experiment)}${renderPatterns(experiment, `${compact ? 'cmp' : 'exp'}-${index}`)}</article>`;
}

const experimentView = document.getElementById('experimentView');
const experimentSelect = document.getElementById('experimentSelect');

const compareButton = document.getElementById('compareButton');
const compareView = document.getElementById('compareView');
const leftSelect = document.getElementById('leftExperiment');
const rightSelect = document.getElementById('rightExperiment');
const leftPane = document.getElementById('leftPane');
const rightPane = document.getElementById('rightPane');
payload.experiments.forEach((experiment, index) => {
  experimentSelect.add(new Option(experiment.name, index));
  leftSelect.add(new Option(experiment.name, index));
  rightSelect.add(new Option(experiment.name, index));
});
rightSelect.value = String(Math.min(1, payload.experiments.length - 1));

function renderSelectedExperiment() {
  const index = Number(experimentSelect.value || 0);
  experimentView.innerHTML = renderExperiment(payload.experiments[index], index, false);
  attachTabs(experimentView);
}
experimentSelect.addEventListener('change', renderSelectedExperiment);
renderSelectedExperiment();

function renderCompare() {
  leftPane.innerHTML = renderExperiment(payload.experiments[Number(leftSelect.value)], Number(leftSelect.value), true);
  rightPane.innerHTML = renderExperiment(payload.experiments[Number(rightSelect.value)], Number(rightSelect.value), true);
  attachTabs(compareView);
}
leftSelect.addEventListener('change', renderCompare);
rightSelect.addEventListener('change', renderCompare);
compareButton.addEventListener('click', () => {
  const showing = !compareView.classList.contains('hidden');
  compareView.classList.toggle('hidden', showing);
  experimentView.classList.toggle('hidden', !showing);
  experimentSelect.classList.toggle('hidden', !showing);
  compareButton.textContent = showing ? 'Compare' : 'Single View';
  if (!showing) renderCompare();
});
"""
