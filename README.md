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
tsvit prepare-data --records records.csv --labels labels.csv --out data/processed --config config.yaml
tsvit train --data data/processed --out runs/example --config config.yaml
tsvit explain --run runs/example --split test
tsvit cluster --run runs/example --split test
tsvit plot --run runs/example
```

The same functionality is available from Python through `interpretable_ts_vit`.

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
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import train_model

records = pd.read_csv("records.csv")
labels = pd.read_csv("labels.csv")

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
