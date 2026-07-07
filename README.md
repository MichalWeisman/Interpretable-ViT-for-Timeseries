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
tsvit prepare-mimic-hypotension --mimic-path mimic-iv-3.1.zip --out data/mimic_hypotension
tsvit prepare-data --records records.csv --labels labels.csv --out data/processed --config config.yaml
tsvit train --data data/processed --out runs/example --config config.yaml
tsvit explain --run runs/example --split test
tsvit cluster --run runs/example --split test
tsvit plot --run runs/example
```

The same functionality is available from Python through `interpretable_ts_vit`.

## Importance-Clustered Value Heatmaps

The default plotted heatmaps use one interpretation path:

1. Split patients by the model's predicted class.
2. Within each predicted class, cluster patients by their model-importance maps.
3. For each class-specific cluster, plot the patients' mean observed clinical values.
4. Use opacity or border width to show where the model was most important.

The resulting patterns are organized by predicted class:

```text
Class True:
  cluster_0
  cluster_1
  ...
Class False:
  cluster_0
  cluster_1
  ...
```

Each PNG under:

```text
runs/<run_name>/cluster_heatmaps/<split>/<predicted_class>/
```

shows the mean observed clinical value for each variable/time bin among the
patients assigned to that predicted class and importance-derived cluster. Rows
are variables in the persisted training order. Columns are relative time bins;
the x-axis label shows the inferred bin granularity, such as `30min bins`, and
each tick is elapsed time from the first bin.

Visual encoding:

- color = mean observed clinical value
- blue = lower value
- red = higher value
- opacity or border width = mean model importance
- optional importance threshold = only the most important cells are emphasized
- gray = no observations for that variable/time cell

The colorbar is labeled `Mean Observed Value`. It intentionally has no numeric
ticks: the bottom is labeled `Low` and the top is labeled `High`. Different
rows can represent different clinical units, so the colorbar should be read as
low-to-high within the plotted value scale, not as a single shared clinical
unit.

The pipeline still may generate explanation maps in:

```text
runs/<run_name>/explanations/<split>/
```

Those maps are used to group similar model-reasoning patterns during the
`cluster` step and to control opacity or border width in the heatmap. When a
predictions CSV is available, clustering is done separately for each
`predicted_label`, so `cluster_0` under `True` is unrelated to `cluster_0`
under `False`. The explanation maps are **not** what the heatmap color
represents.

Configure the default behavior with:

```yaml
cluster:
  method: kmeans
  feature_mode: explanation
  explanation_weight: 1.0
  value_weight: 1.0
  n_clusters: 8
  hdbscan_min_cluster_size: 5
  hdbscan_min_samples: null
  plot_mode: value_with_importance_opacity
  importance_threshold: null
  show_values: false
```

Use `method: hdbscan` when you want HDBSCAN to infer the number of clusters from density. In that mode, `n_clusters` is ignored; tune `hdbscan_min_cluster_size` and `hdbscan_min_samples` instead. HDBSCAN noise points are kept as cluster `-1`.

Use `feature_mode: combined` to cluster from both model explanations and
clinical value trajectories. The two feature blocks are standardized separately
before `explanation_weight` and `value_weight` are applied, so increasing
`value_weight` makes value trajectories matter more without removing the
explanation signal.

Set `plot_mode: value_with_importance_border` if you prefer border width,
instead of opacity, to encode importance.

Set `importance_threshold` to a quantile between `0` and `1` to emphasize only
the most important regions. For example, `importance_threshold: 0.8` keeps
visual emphasis only for cells at or above the 80th percentile of the cluster's
finite importance scores. In opacity mode, cells below the threshold fade to
the minimum opacity; in border mode, cells below the threshold do not receive an
importance border. The threshold does not change the clinical value colors or
the clustering itself.

Set `show_values: true` to print the mean observed value inside each heatmap
cell. Missing cells are left unannotated.

The same options are available from the CLI:

```powershell
tsvit plot --run runs/example --importance-threshold 0.8
tsvit plot --run runs/example --show-values
```

This default answers:

> Among patients predicted as the same class, which patients caused the model
> to focus on similar variable/time regions, and what were their actual
> measurements in those regions?

### How Cluster Values Are Computed

Prepared tensors have shape `[patients, 2, variables, timesteps]`:

- channel 0, `D`: normalized values
- channel 1, `M`: observed-value mask

Missing cells are stored as `D=0, M=0`. Because zero is also a possible numeric
value after normalization, missing cells must not be averaged directly. For a
cluster, the heatmap matrix is computed as:

```text
raw_value = normalized_value * training_std(variable) + training_mean(variable)
cluster_value(variable, time) =
    sum(raw_value * mask) / sum(mask)
```

If `sum(mask) == 0` for a variable/time cell, no patient in that cluster has an
observation there. The output matrix stores `NaN` for that cell and the plot
renders it as gray.

The numeric cluster matrices are saved separately in:

```text
runs/<run_name>/cluster_values/<split>/<predicted_class>/cluster_<id>.npy
```

The PNGs are saved in:

```text
runs/<run_name>/cluster_heatmaps/<split>/<predicted_class>/cluster_<id>.png
```

### Important Interpretation Note

Rows can represent different clinical units, such as heart rate, blood
pressure, respiratory rate, or oxygen saturation. A single color scale across
all rows means the colorbar is literal numeric value, but the clinical meaning
of "high" still depends on the row. For example, `100` means something
different for heart rate than for oxygen saturation.

Use these heatmaps to inspect temporal value patterns inside clusters, not as a
unit-normalized severity scale across unrelated variables.

## MIMIC-IV Hypotension Dataset

The package includes a plug-in dataset adapter for MIMIC-IV v3.1 ICU data. It
reads either the original PhysioNet zip archive or an extracted MIMIC-IV
directory, then writes the generic `records.csv` and `labels.csv` files used by
the rest of the pipeline.

```powershell
tsvit prepare-mimic-hypotension `
  --mimic-path mimic-iv-3.1.zip `
  --out data/mimic_hypotension `
  --observation-hours 24 `
  --prediction-hours 6 `
  --threshold 65 `
  --chunk-size 1000000 `
  --cache-dir data/mimic_cache
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

By default, `--cache-dir data/mimic_cache` contains two kinds of files:

```text
data/mimic_cache/
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
data/mimic_hypotension/
  records.csv
  labels.csv
  dataset_metadata.json
```

If you create data on one computer and train on another, copy
`data/mimic_hypotension/`. You usually do not need to copy `data/mimic_cache/`.

To disable cache behavior:

```powershell
tsvit prepare-mimic-hypotension `
  --mimic-path mimic-iv-3.1.zip `
  --out data/mimic_hypotension `
  --read-zip-directly `
  --no-filtered-cache
```

After creating the MIMIC records and labels, run:

```powershell
tsvit prepare-data --records data/mimic_hypotension/records.csv --labels data/mimic_hypotension/labels.csv --out data/processed --config config.yaml
```

To add another dataset, implement `DatasetAdapter.prepare()` and return a
`PreparedDataset` with the same generic records/labels schema.

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
  feature_mode: explanation
  explanation_weight: 1.0
  value_weight: 1.0
  n_clusters: 8
  hdbscan_min_cluster_size: 5
  hdbscan_min_samples: null
```

Early stopping is optional. Leave `early_stopping_patience` as `null` to train
for the full number of epochs. When a validation split is available, each epoch
prints progress and records `val_loss` plus validation metrics in `metrics.json`. With
`restore_best_model: true`, the saved `model.pt` uses the best validation
checkpoint according to `early_stopping_monitor`; `val_loss` is minimized, while
metrics such as `val_macro_f1`, `val_accuracy`, and `val_auroc` are maximized
when `early_stopping_mode: auto`.

## Python Usage

For notebook experiments, use a data module plus a model module so each stage
can be rerun independently:

```python
from interpretable_ts_vit.config import ClusterConfig, DataConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.data_modules import MIMICHypotensionDataModule
from interpretable_ts_vit.model_modules import ViTTimeSeriesModule

data = MIMICHypotensionDataModule(
    records_path="data/mimic_hypotension/records.csv",
    labels_path="data/mimic_hypotension/labels.csv",
    processed_dir="data/processed",
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
`predictions.csv`. `explain` writes per-patient explanation maps. `cluster`
writes cluster assignments and explanation-space cluster averages. `plot`
writes `cluster_values/*.npy` and PNG value heatmaps whose colors represent
mean observed clinical values, not explanation scores.

## Notes

- Fit the binner on training data only to avoid leaking variable vocab,
  normalization statistics, or inferred time bounds from validation/test data.
- Missing observations are represented by `D=0` and `M=0`; observed values use
  `M=1` after per-variable normalization.
- Unknown variables at transform time are ignored so deployed tensors keep the
  same shape learned during training.
