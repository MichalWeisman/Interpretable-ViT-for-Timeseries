# Interpretable Time-Series ViT

Research baseline for classifying irregularly sampled, multivariable time series, clustering class-specific explanation maps, and visualizing the clinical value patterns associated with those clusters.

## Input Shape

Records table columns default to:

- `patient_id`
- `variable`
- `value`
- `timestamp`

Labels table columns default to:

- `patient_id`
- `label`

The binner converts each patient into a two-channel tensor `[2, variables, timesteps]`:

- channel 0: normalized values
- channel 1: observation mask

## CLI

```powershell
tsvit prepare-mimic-hypotension --mimic-path mimic-iv-3.1.zip --out data/hypotension/mimic_hypotension
tsvit prepare-mimic-targets --config configs/mimic_targets.yaml
tsvit prepare-data --records records.csv --labels labels.csv --out data/hypotension/processed --config config.yaml
tsvit train --data data/hypotension/processed --out runs/example --config config.yaml
tsvit explain --run runs/example --split test
tsvit cluster --run runs/example --split test
tsvit plot --run runs/example
```

The same functionality is available from Python through `interpretable_ts_vit`.

## Autoencoder-Clustered Value Heatmaps

The current notebook and pipeline use one interpretation path:

1. Generate per-patient model explanations with `grad_attention_rollout` only.
2. Pair each explanation map with the patient's denormalized clinical value map.
3. Train a small autoencoder on train `[explanation, value]` maps and validate it on validation maps.
4. Encode the selected split with that autoencoder and cluster the latent vectors within each predicted class.
5. Plot cluster-level mean clinical value heatmaps, with opacity or border width showing mean rollout importance.

Saved artifacts are reused when present:

```text
data/hypotension/processed_full_hypotension_importance_values/{train,val,test}.npz
runs/full_hypotension_importance_values/model.pt
runs/full_hypotension_importance_values/explanations/<split>/*.npy
runs/full_hypotension_importance_values/clusters/<split>/autoencoder.pt
runs/full_hypotension_importance_values/clusters/<split>/autoencoder_embeddings.csv
runs/full_hypotension_importance_values/clusters/<split>/autoencoder_metrics.json
```

Configure clustering with:

```yaml
cluster:
  feature_mode: autoencoder
  method: kmeans
  n_clusters: 8
  autoencoder_latent_dim: 16
  autoencoder_epochs: 50
  autoencoder_learning_rate: 0.001
  autoencoder_batch_size: 32
  autoencoder_early_stopping_patience: 10
  plot_mode: value_with_importance_opacity
  importance_threshold: null
  show_values: true
```

Use `method: hdbscan` when you want HDBSCAN to infer the number of clusters from density. In that mode, `n_clusters` is ignored; tune `hdbscan_min_cluster_size` and `hdbscan_min_samples` instead. HDBSCAN noise points are kept as cluster `-1`.

Visual encoding:

- color = mean observed clinical value
- opacity or border width = mean `grad_attention_rollout` importance
- gray = no observations for that variable/time cell
- optional cell label = mean observed clinical value

### How Cluster Values Are Computed

Prepared tensors have shape `[patients, 2, variables, timesteps]`:

- channel 0, `D`: normalized values
- channel 1, `M`: observed-value mask

Missing cells are stored as `D=0, M=0`. Cluster value matrices are computed from denormalized observed values only:

```text
raw_value = normalized_value * training_std(variable) + training_mean(variable)
cluster_value(variable, time) = sum(raw_value * mask) / sum(mask)
```

If `sum(mask) == 0`, the output matrix stores `NaN` and the plot renders that cell as gray.

## MIMIC-IV Hypotension Dataset

The package includes a plug-in dataset adapter for MIMIC-IV v3.1 ICU data. It
reads either the original PhysioNet zip archive or an extracted MIMIC-IV
directory, then writes the generic `records.csv` and `labels.csv` files used by
the rest of the pipeline.

Dataset-specific files live under `data/<dataset>/`; notebooks follow the same
layout under `notebooks/<dataset>/`. For example, the hypotension assets are in
`data/hypotension/` and `notebooks/hypotension/`.

```powershell
tsvit prepare-mimic-hypotension `
  --mimic-path mimic-iv-3.1.zip `
  --out data/hypotension/mimic_hypotension `
  --observation-hours 24 `
  --prediction-hours 6 `
  --threshold 65 `
  --chunk-size 1000000 `
  --cache-dir data/hypotension/mimic_cache
```

Each row in `labels.csv` corresponds to an ICU `stay_id`. By default, the label
is `true` if mean arterial pressure, using MIMIC item IDs `220052` or `220181`,
is less than or equal to 65 mmHg during the 6 hours after the 24-hour
observation window.

MIMIC-IV dates are deidentified with patient-specific shifts, so the adapter
exports timestamps as relative ICU time anchored at `2000-01-01 00:00:00`.
That makes columns in the ViT input represent time since ICU admission rather
than incomparable absolute calendar dates.

### MIMIC Cache

The cache is an intermediate speed-up layer. It is **not** the model input and
it is **not** the tensor dataset. You can delete it and regenerate it from the
MIMIC zip.

By default, `--cache-dir data/hypotension/mimic_cache` contains two kinds of files:

```text
data/hypotension/mimic_cache/
  extracted/
    icu/
      icustays.csv.gz
      chartevents.csv.gz
  chartevents_filtered_<hash>.parquet
```

`extracted/` contains selected raw MIMIC `.csv.gz` files copied out of the big
zip. This avoids repeatedly reading compressed files through the zip container.
It does **not** extract the whole MIMIC archive.

`chartevents_filtered_<hash>.parquet` is a filtered cache built from
`chartevents.csv.gz`. It contains only:

- eligible ICU `stay_id`s for the configured observation/prediction windows
- relevant MIMIC `itemid`s used as model variables or hypotension outcomes
- numeric chart values from `valuenum`
- the raw `charttime`, `stay_id`, `itemid`, and `valuenum` columns

The hash in the filename changes when the eligible cohort or selected item IDs
change, so incompatible runs do not accidentally reuse the wrong filtered
events.

While scanning raw `chartevents.csv.gz`, the adapter prints a progress bar with
percentage and ETA. The percentage is estimated from compressed bytes consumed,
not from a pre-counted number of rows, so it avoids an extra full scan while
still giving a practical time-left estimate.

The portable pre-tensor output is still:

```text
data/hypotension/mimic_hypotension/
  records.csv
  labels.csv
  dataset_metadata.json
```

If you create data on one computer and train on another, copy
`data/hypotension/mimic_hypotension/`. You usually do not need to copy
`data/hypotension/mimic_cache/`.

To disable cache behavior:

```powershell
tsvit prepare-mimic-hypotension `
  --mimic-path mimic-iv-3.1.zip `
  --out data/hypotension/mimic_hypotension `
  --read-zip-directly `
  --no-filtered-cache
```

After creating the MIMIC records and labels, run:

```powershell
tsvit prepare-data --records data/hypotension/mimic_hypotension/records.csv --labels data/hypotension/mimic_hypotension/labels.csv --out data/hypotension/processed --config config.yaml
```

To add another dataset, implement `DatasetAdapter.prepare()` and return a
`PreparedDataset` with the same generic records/labels schema.

## MIMIC-IV Multi-Target Dataset Creation

Use the config-driven endpoint to create one pre-tensor dataset per prediction
target and window configuration from the zipped MIMIC-IV archive:

```powershell
tsvit prepare-mimic-targets --config configs/mimic_targets.yaml
```

The default config creates two window variants:

- `obs48_target24_gap0`: 48-hour observation, 24-hour prediction, no gap
- `obs48_target24_gap8`: 48-hour observation, 24-hour prediction, 8-hour gap

For each window it writes target-specific folders under `data/mimic_targets/`,
for example:

```text
data/mimic_targets/obs48_target24_gap0/hypoglycemia/
  records.csv
  labels.csv
  dataset_metadata.json
```

These are source CSVs only, not tensors. Run `tsvit prepare-data` separately
when you want to bin one of the generated target datasets.

The current targets are:

- `cardiovascular_event`
- `nosocomial_infection`
- `hypoglycemia`
- `prolonged_hyperglycemia`
- `in_hospital_mortality`

Each row in `labels.csv` corresponds to a hospital admission (`hadm_id`).
Timestamps are relative to hospital admission and anchored at
`2000-01-01 00:00:00`. The adapter uses filtered Parquet caches under the
configured `cache_dir` so repeated runs with new windows or targets do not need
to rescan every raw MIMIC table from the zip.

Use `notebooks/mimic_targets/mimic_general_item_exploration.ipynb` to inspect
the selected lab/chart/input item mappings before reading patient-level event
tables. After generating datasets, use
`notebooks/mimic_targets/mimic_target_dataset_exploration.ipynb` to inspect
label balance, variable coverage, value distributions, event timing, and
positive/negative summaries.

## One-File Endpoint

Use [main.py](main.py) when you want to run the whole workflow without the
`tsvit` CLI. Edit the `SETTINGS`, `MIMIC_SETTINGS`, and `PIPELINE_CONFIG`
objects at the top of the file, then run:

```powershell
python main.py
```

The endpoint performs:

- MIMIC-IV hypotension records/labels creation, or generic records/labels reuse
- tensor preparation with the fitted binner
- model training and saving
- model loading and test-set evaluation
- explanation map generation
- explanation clustering
- clinical value heatmap rendering for the clustered patients

For programmatic use from another script or notebook, call:

```python
from interpretable_ts_vit.pipeline import PipelineRunConfig, run_pipeline

result = run_pipeline(PipelineRunConfig())
print(result.artifacts)
```

## Configuration

Configuration files can be YAML or JSON. YAML requires `PyYAML`; JSON works
with the standard library.

```yaml
data:
  granularity: 30min
  time_start: "2026-01-01 00:00:00"
  time_end: "2026-01-03 00:00:00"
  aggregation: mean
  val_fraction: 0.2
  test_fraction: 0.2
model:
  patch_size: [1, 4]
  embed_dim: 64
  depth: 2
  num_heads: 4
train:
  batch_size: 16
  epochs: 10
  learning_rate: 0.001
  early_stopping_patience: 3
  early_stopping_monitor: val_loss
  early_stopping_min_delta: 0.0
  early_stopping_mode: auto
  restore_best_model: true
  verbose: true
cluster:
  method: kmeans
  feature_mode: autoencoder
  n_clusters: 8
  autoencoder_latent_dim: 16
  autoencoder_epochs: 50
  autoencoder_learning_rate: 0.001
  autoencoder_batch_size: 32
  autoencoder_early_stopping_patience: 10
  hdbscan_min_cluster_size: 5
  hdbscan_min_samples: null
```

Early stopping is optional. Leave `early_stopping_patience` as `null` to train
for the full number of epochs. When a validation split is available, each epoch
prints progress and records `val_loss` plus validation metrics in `metrics.json`. With
`restore_best_model: true`, the saved `model.pt` uses the best validation
checkpoint according to `early_stopping_monitor`; `val_loss` is minimized, while
metrics such as `val_macro_f1`, `val_accuracy`, `val_auc`, and `val_auroc` are maximized
when `early_stopping_mode: auto`.

## Python Usage

For notebook experiments, use a data module plus a model module so each stage
can be rerun independently:

```python
from interpretable_ts_vit.config import ClusterConfig, DataConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.data_modules import MIMICHypotensionDataModule
from interpretable_ts_vit.model_modules import ViTTimeSeriesModule

data = MIMICHypotensionDataModule(
    records_path="data/hypotension/mimic_hypotension/records.csv",
    labels_path="data/hypotension/mimic_hypotension/labels.csv",
    processed_dir="data/hypotension/processed",
    data_config=DataConfig(granularity="30min"),
)

model = ViTTimeSeriesModule(
    run_dir="runs/hypotension_v1",
    model_config=ModelConfig(patch_size=(1, 4)),
    train_config=TrainConfig(epochs=3, verbose=True),
    cluster_config=ClusterConfig(n_clusters=4, show_values=True),
)

data.prepare()
model.fit(data)
model.evaluate(data, split="test")
model.explain(data, split="test")
model.cluster_explanations(data, split="test")
model.plot_cluster_values(data, split="test")
model.display_cluster_heatmaps(split="test")
```

The lower-level API remains available when you want to manage tensors and model
objects yourself:

```python
import pandas as pd
from interpretable_ts_vit import TimeSeriesBinner, ViTConfig, ViTTimeSeriesClassifier
from interpretable_ts_vit.datasets import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import train_model

prepared = MIMICIVHypotensionAdapter(
    MIMICHypotensionConfig(mimic_path="mimic-iv-3.1.zip")
).prepare()
records = prepared.records
labels = prepared.labels

binner = TimeSeriesBinner(granularity="30min")
binned = binner.fit_transform(records, labels)
dataset = BinnedTimeSeriesDataset(binned.x, binned.y, binned.patient_ids)

model = ViTTimeSeriesClassifier(
    ViTConfig(
        num_variables=binned.x.shape[2],
        num_timesteps=binned.x.shape[3],
        num_classes=len(binner.index_to_label_),
    )
)
metrics = train_model(model, dataset)
```

## Artifacts

`prepare-data` writes train/validation/test `.npz` files plus `binner.json`
and `variable_vocab.json`. `train` writes `model.pt`, `metrics.json`, and
`predictions.csv`. Evaluation metrics include `auc`/`auroc`, `tpr`, `fpr`,
`tnr`, `fnr`, `ppv`, accuracy, macro F1, and the confusion matrix. `explain`
writes per-patient explanation maps. `cluster`
writes cluster assignments and explanation-space cluster averages. `plot`
writes `cluster_values/*.npy` and PNG value heatmaps whose colors and cell
labels represent mean observed clinical values, not explanation scores.

## Notes

- Fit the binner on training data only to avoid leaking variable vocab,
  normalization statistics, or inferred time bounds from validation/test data.
- Missing observations are represented by `D=0` and `M=0`; observed values use
  `M=1` after per-variable normalization.
- Unknown variables at transform time are ignored so deployed tensors keep the
  same shape learned during training.
