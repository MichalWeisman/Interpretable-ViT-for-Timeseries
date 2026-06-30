# Interpretable Time-Series ViT

Research baseline for classifying irregularly sampled, multivariable time series and clustering class-specific explanation maps.

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
  --threshold 65
```

Each row in `labels.csv` corresponds to an ICU `stay_id`. By default, the label
is `true` if mean arterial pressure, using MIMIC item IDs `220052` or `220181`,
is less than or equal to 65 mmHg during the 6 hours after the 24-hour
observation window.

MIMIC-IV dates are deidentified with patient-specific shifts, so the adapter
exports timestamps as relative ICU time anchored at `2000-01-01 00:00:00`.
That makes columns in the ViT input represent time since ICU admission rather
than incomparable absolute calendar dates.

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
- heatmap rendering

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
cluster:
  n_clusters: 8
```

## Python Usage

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
`predictions.csv`. `explain`, `cluster`, and `plot` add per-patient
explanation maps, cluster assignments, cluster averages, and PNG heatmaps.

## Notes

- Fit the binner on training data only to avoid leaking variable vocab,
  normalization statistics, or inferred time bounds from validation/test data.
- Missing observations are represented by `D=0` and `M=0`; observed values use
  `M=1` after per-variable normalization.
- Unknown variables at transform time are ignored so deployed tensors keep the
  same shape learned during training.
