# Interpretable Time-Series ViT

Research code for classifying irregular, multivariable clinical time series with a Vision Transformer, then explaining predictions by clustering patient-level explanation maps and rendering the clinical value patterns behind those clusters.

The project is organized around one portable data contract:

```text
dataset adapter output or generic CSVs
  -> records.csv + labels.csv
  -> binned tensor splits
  -> ViT model
  -> explanation maps
  -> explanation/value clusters
  -> clinical value heatmaps
```

## What This Project Does

The package turns patient event tables into fixed-shape tensors, trains a small ViT classifier, generates per-patient importance maps with `grad_attention_rollout`, clusters patients using an autoencoder over `[explanation, value]` maps, and plots cluster-level heatmaps where color shows observed clinical values and opacity or borders show model importance.

It supports two input paths:

- Any already prepared `records.csv` and `labels.csv` pair with the generic schema below.
- Dataset-specific adapters that export that same schema. MIMIC-IV v3.1 target generation is the current concrete adapter example.

## Repository Structure

```text
.
|-- main.py                         # one-file end-to-end workflow for IDE/script use
|-- pyproject.toml                  # package metadata, dependencies, and tsvit CLI entrypoint
|-- requirements.txt                # dependency fallback for non-editable installs
|-- configs/
|   |-- datasets/
|   |   `-- mimic/
|   |       `-- targets.yaml        # canonical MIMIC-IV target-generation config
|   `-- mimic_targets.yaml          # legacy compatibility config path
|-- data/                           # local generated data, caches, and prepared CSVs
|-- notebooks/
|   `-- mimic_targets/              # target/item exploration and analysis notebooks
|-- src/interpretable_ts_vit/
|   |-- cli.py                      # tsvit prepare/train/explain/cluster/plot commands
|   |-- config.py                   # dataclass configs loaded from YAML/JSON
|   |-- pipeline.py                 # programmatic end-to-end pipeline
|   |-- binning.py                  # records/labels -> tensor binner
|   |-- model.py                    # ViT classifier wrapper
|   |-- training.py                 # training, prediction, and evaluation loops
|   |-- explain.py                  # explanation-map generation
|   |-- autoencoder.py              # explanation/value embedding and clustering
|   |-- visualization.py            # value aggregation and heatmap rendering
|   |-- datasets/
|   |   |-- base.py                 # PreparedDataset, TargetWindowConfig, adapter base types
|   |   |-- mimic/                  # canonical MIMIC-IV adapter package
|   |   |-- mimic_targets.py        # legacy compatibility import wrapper
|   |   `-- mimic_iv.py             # legacy compatibility import wrapper
|   |-- data_modules/               # reusable data preparation modules
|   `-- model_modules/              # reusable model workflow modules
`-- tests/                          # unit and end-to-end smoke tests
```

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,explain]"
```

If you do not want an editable install, use the requirements file:

```powershell
pip install -r requirements.txt
```

The editable install exposes the CLI command:

```powershell
tsvit --help
```

## Data Format

All dataset adapters must produce the same generic CSV schema. MIMIC-IV is one source adapter; downstream tensor preparation, training, explanation, clustering, and plotting are dataset-agnostic once these files exist.

`records.csv`:

| Column | Meaning |
| --- | --- |
| `patient_id` | Stable example id. For MIMIC targets this is either `hadm_id` or `stay_id`, stored as text. |
| `variable` | Clinical variable name, for example `heart_rate`, `blood_glucose`, or `mean_bp`. |
| `value` | Numeric observed value. Binary event features use `1.0`. |
| `timestamp` | Timestamp used for binning. MIMIC adapters export relative time anchored at `2000-01-01 00:00:00`. |

`labels.csv`:

| Column | Meaning |
| --- | --- |
| `patient_id` | Must match ids in `records.csv`. |
| `label` | Class label, usually `true` or `false`. |

`prepare-data` converts these tables into tensors with shape `[patients, 2, variables, timesteps]`:

- channel `0`: normalized values
- channel `1`: observation mask

Missing cells are represented as `value = 0` and `mask = 0`. The binner is fitted on the training split only, so validation and test data do not leak variable vocabularies, normalization statistics, or inferred time bounds.

## Running the Pipeline

The common command sequence starts with either your own compatible CSVs or a dataset adapter. For the current MIMIC-IV adapter:

```powershell
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml

tsvit prepare-data `
  --records data/mimic_targets/<window>/<target>/records.csv `
  --labels data/mimic_targets/<window>/<target>/labels.csv `
  --out data/mimic_targets/processed/<window>/<target> `
  --config config.yaml

tsvit train `
  --data data/mimic_targets/processed/<window>/<target> `
  --out runs/mimic_targets/<window>/<target> `
  --config config.yaml

tsvit explain --run runs/mimic_targets/<window>/<target> --split test
tsvit cluster --run runs/mimic_targets/<window>/<target> --split test --config config.yaml
tsvit plot --run runs/mimic_targets/<window>/<target> --split test --config config.yaml
```

Use the same `prepare-data`, `train`, `explain`, `cluster`, and `plot` commands for any non-MIMIC dataset once you have compatible `records.csv` and `labels.csv` files.

## MIMIC-IV Target Workflows

MIMIC-IV is implemented as a dataset-specific adapter under `interpretable_ts_vit.datasets.mimic`. It is an example source database, not the pipeline interface itself.

The current MIMIC-IV entrypoint is:

```powershell
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml
```

Optional overrides:

```powershell
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml --out data/mimic_targets_custom
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml --cohort-level admission
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml --cohort-level icu
```

The legacy path `configs/mimic_targets.yaml` remains supported for existing commands.

`cohort_level` controls the prediction unit:

- `admission`: one example per hospital admission, with `patient_id = hadm_id`; timestamps are relative to hospital admission.
- `icu`: one example per ICU stay, with `patient_id = stay_id`; timestamps are relative to ICU admission.

Both modes anchor exported timestamps at `2000-01-01 00:00:00`, because MIMIC dates are deidentified with patient-specific shifts.

The current targets are:

- `hypoglycemia`
- `hypokalemia`
- `hypotension`

To generate a single target from the CLI, choose the cohort/window settings in the config and pass the target as an override. For example:

```yaml
cohort_level: icu
windows:
  - name: obs24_target8_gap0
    observation_hours: 24
    prediction_hours: 8
    gap_hours: 0
  - name: obs24_target8_gap2
    observation_hours: 24
    prediction_hours: 8
    gap_hours: 2
```

Then run:

```powershell
tsvit prepare-mimic-targets --config configs/datasets/mimic/targets.yaml
```

For each configured window and target, the adapter writes portable pre-tensor files:

```text
data/mimic_targets/<window>/<target>/
  records.csv
  labels.csv
  dataset_metadata.json
```

These are source CSVs, not tensors. Run `tsvit prepare-data` before training.

## MIMIC-IV Target Configuration

`configs/datasets/mimic/targets.yaml` controls MIMIC dataset creation. The legacy `configs/mimic_targets.yaml` file is kept as a compatibility copy. The most important fields are:

| Field | Purpose |
| --- | --- |
| `mimic_path` | Path to the MIMIC-IV zip archive or extracted directory. |
| `output_dir` | Root directory for prepared target folders. |
| `cache_dir` | Disposable cache for extracted raw tables and filtered Parquet files. |
| `logs_dir` | Root for timestamped process logs; dataset creation logs are stored below `<logs_dir>/dataset_creation/<date>/`. Set to `null` to disable file logging. |
| `cohort_level` | `admission` for `hadm_id` examples or `icu` for `stay_id` examples. |
| `windows` | Observation, optional gap, and prediction windows in hours. |
| `timeline_horizon_hours` | Optional elapsed-time horizon for generating repeated complete instances. |
| `window_stride_hours` | Optional interval between repeated observation-window starts; must be used with `timeline_horizon_hours`. |
| `chunk_size` | Number of rows per raw MIMIC chunk read. |
| `use_extracted_files` | Copies selected `.csv.gz` files out of the zip for faster repeated reads. |
| `use_filtered_cache` | Reuses filtered Parquet event tables when compatible. |
| `require_full_window` | Drops stays/admissions without the full prediction window when `true`. |
| `require_outcome_measurement` | For targets that depend on prediction-window outcome measurements, drops examples without those measurements when `true`. |
| `max_admissions` | Optional admission cap for smoke runs. |
| `max_stays` | Optional ICU-stay cap for smoke runs. |

Prediction timing uses the generic `TargetWindowConfig` shape and is defined by `windows`, not by the model-training config:

```yaml
windows:
  - name: obs24_target8_gap0
    observation_hours: 24
    prediction_hours: 8
    gap_hours: 0
  - name: obs24_target8_gap2
    observation_hours: 24
    prediction_hours: 8
    gap_hours: 2
  - name: rolling_obs4_target2_horizon24_stride4
    observation_hours: 4
    prediction_hours: 2
    gap_hours: 0
    timeline_horizon_hours: 24
    window_stride_hours: 4
```

For each window:

- `observation_hours` is the history written into `records.csv`.
- `gap_hours` is the delay between the observation window and the prediction window.
- `prediction_hours` is the future interval used to assign the target label.
- When `timeline_horizon_hours` and `window_stride_hours` are present, the adapter creates repeated fixed-length instances. Only instances whose complete observation, gap, and prediction sequence fits inside the horizon are created. For the rolling example above, starts at hours 0, 4, 8, 12, and 16 create five instances.
- Repeated instances use IDs such as `12345__window_0`. Their labels also include `source_cohort_id`, `window_index`, and `window_start_hours`; tensor preparation keeps all instances from one source admission or ICU stay in the same split.

MIMIC target names and target thresholds live in `src/interpretable_ts_vit/datasets/mimic/mimic_targets.py`. Per-target variable mappings live under `target_variables` in `configs/datasets/mimic/targets.yaml`; use `notebooks/mimic_targets/mimic_general_item_exploration.ipynb` to review candidates and write the selected IDs into the config.

The adapter resolves dictionaries from MIMIC metadata tables when available. Generated `dataset_metadata.json` records the resolved mappings, source tables, target definition, label counts, and cohort-level details.

Each dataset-creation run writes a separate timestamped log, for example `logs/dataset_creation/2026-07-18/dataset_creation_20260718T143012_123456+0300.log`. The process-first directory layout is shared infrastructure: future training, explanation, or reporting logs can use sibling directories without mixing unrelated runs.

## One-File Workflow

Use [main.py](main.py) when you want to run the whole workflow from an IDE, scheduler, or notebook without manually calling each CLI command.

```powershell
python main.py
```

`main.py` currently builds a `MIMICTargetsConfig` and writes into the `data/mimic_targets` and `runs/mimic_targets` layout. Edit `SETTINGS`, `MIMIC_SETTINGS`, and `PIPELINE_CONFIG` at the top of the file to change paths, targets, cohort level, target windows, model size, training options, or which stages run.

The one-file endpoint performs:

- MIMIC target records/labels creation, or generic records/labels reuse
- tensor preparation
- model training
- model loading and evaluation
- explanation generation
- explanation/value clustering
- clinical heatmap rendering

## Configuration

Training and interpretation configs can be YAML or JSON. YAML requires `PyYAML`, which is included in the project dependencies. This config controls tensor binning, model architecture, training, explanations, and clustering after a target dataset has already been created. Dataset-adapter observation, gap, and prediction window sizes live in that adapter's config, for example `configs/datasets/mimic/targets.yaml` for MIMIC-IV.

```yaml
data:
  granularity: 30min
  aggregation: mean
  val_fraction: 0.2
  test_fraction: 0.2
  random_state: 13
model:
  patch_size: [1, 4]
  embed_dim: 64
  depth: 2
  num_heads: 4
  mlp_ratio: 2.0
  dropout: 0.1
train:
  batch_size: 16
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  device: auto
  early_stopping_patience: 3
  early_stopping_monitor: val_loss
  early_stopping_min_delta: 0.0
  early_stopping_mode: auto
  restore_best_model: true
  verbose: true
explain:
  method: grad_attention_rollout
  target_class: null
  batch_size: 16
cluster:
  feature_mode: autoencoder
  method: kmeans
  n_clusters: 8
  autoencoder_latent_dim: 16
  autoencoder_epochs: 50
  autoencoder_learning_rate: 0.001
  autoencoder_batch_size: 32
  autoencoder_early_stopping_patience: 10
  hdbscan_min_cluster_size: 5
  hdbscan_min_samples: null
  plot_mode: value_with_importance_opacity
  importance_threshold: null
  show_values: true
  use_normal_ranges: false
```

Early stopping is optional. Leave `early_stopping_patience` as `null` to train for the full number of epochs. With `restore_best_model: true`, `model.pt` is restored to the best validation checkpoint according to `early_stopping_monitor`.

`time_start` and `time_end` are optional lower-level binning controls. For normal MIMIC target runs, prefer setting `observation_hours`, `prediction_hours`, and `gap_hours` in the target config, then let the generated relative timestamps flow into `prepare-data`.

## Outputs and Artifacts

Prepared source datasets:

```text
data/mimic_targets/<window>/<target>/
  records.csv
  labels.csv
  dataset_metadata.json
```

Processed tensor datasets:

```text
data/mimic_targets/processed/<window>/<target>/
  train.npz
  val.npz
  test.npz
  binner.json
  variable_vocab.json
  splits.json
```

Model run outputs:

```text
runs/mimic_targets/<window>/<target>/
  model.pt
  metrics.json
  predictions.csv
  <split>_predictions.csv
  <split>_evaluation_metrics.json
  explanations/<split>/*.npy
  clusters/<split>/cluster_assignments.csv
  clusters/<split>/autoencoder.pt
  clusters/<split>/autoencoder_embeddings.csv
  clusters/<split>/autoencoder_metrics.json
  cluster_values/<split>/*.npy
  cluster_heatmaps/<split>/*.png
  cluster_centroid_heatmaps/<split>/**/*.png
```

MIMIC caches under `cache_dir` are only speed-up artifacts. They are not model inputs and do not need to be copied when moving a prepared dataset to another machine. The portable files are `records.csv`, `labels.csv`, and `dataset_metadata.json`.

## Explanation and Heatmap Flow

The interpretation path is:

1. Generate per-patient `grad_attention_rollout` maps.
2. Pair each explanation map with the patient's denormalized clinical value map.
3. Train an autoencoder on train `[explanation, value]` maps and validate on validation maps.
4. Encode the selected split into latent vectors.
5. Cluster latent vectors with `kmeans` or `hdbscan`.
6. Plot cluster-level clinical values with optional importance opacity or borders.

Visual encoding:

- color = mean observed clinical value
- opacity or border width = mean rollout importance
- gray = no observed values for that variable/time cell
- optional cell label = mean observed clinical value

Cluster values are computed from denormalized observed cells only:

```text
raw_value = normalized_value * training_std(variable) + training_mean(variable)
cluster_value(variable, time) = sum(raw_value * mask) / sum(mask)
```

If `sum(mask) == 0`, the output stores `NaN` and the plot renders the cell as gray.

## Python API Usage

For notebook experiments, use `GenericCSVDataModule` plus `ViTTimeSeriesModule` so stages can be rerun independently:

```python
from interpretable_ts_vit.config import ClusterConfig, DataConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.data_modules import GenericCSVDataModule
from interpretable_ts_vit.model_modules import ViTTimeSeriesModule

data = GenericCSVDataModule(
    records_path="data/mimic_targets/obs24_target8_gap0/hypoglycemia/records.csv",
    labels_path="data/mimic_targets/obs24_target8_gap0/hypoglycemia/labels.csv",
    processed_dir="data/mimic_targets/processed/obs24_target8_gap0/hypoglycemia",
    data_config=DataConfig(granularity="30min"),
)

model = ViTTimeSeriesModule(
    run_dir="runs/mimic_targets/obs24_target8_gap0/hypoglycemia",
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

For full programmatic orchestration:

```python
from interpretable_ts_vit.pipeline import PipelineRunConfig, run_pipeline

result = run_pipeline(PipelineRunConfig())
print(result.artifacts)
```

The lower-level API remains available when you want to manage tensors and model objects yourself:

```python
import pandas as pd

from interpretable_ts_vit import TimeSeriesBinner, ViTConfig, ViTTimeSeriesClassifier
from interpretable_ts_vit.data import BinnedTimeSeriesDataset
from interpretable_ts_vit.training import train_model

records = pd.read_csv("data/mimic_targets/obs24_target8_gap0/hypoglycemia/records.csv")
labels = pd.read_csv("data/mimic_targets/obs24_target8_gap0/hypoglycemia/labels.csv")

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

## Notebooks

The MIMIC target notebooks live under `notebooks/mimic_targets/`:

- `mimic_general_item_exploration.ipynb`: inspect lab, chart, prescription, and input-event mappings before large dataset generation.
- `mimic_target_dataset_exploration.ipynb`: inspect generated label balance, variable coverage, value distributions, event timing, and positive/negative summaries.
- `hypotension_importance_values.ipynb`: local analysis around hypotension explanation/value outputs.

## Testing

Run the test suite with:

```powershell
python -m pytest
```

Useful focused tests while working on the target adapter:

```powershell
python -m pytest tests/test_mimic_targets_adapter.py
python -m pytest tests/test_pipeline.py
```

## Practical Notes

- Do not use `tsvit prepare-mimic-hypotension`; the current CLI uses `prepare-mimic-targets` for MIMIC target creation.
- Use `GenericCSVDataModule` for prepared records/labels datasets.
- Unknown variables at transform time are ignored so deployed tensors keep the same shape learned during training.
- MIMIC caches can be deleted and regenerated from the original zip or extracted MIMIC directory.
- HDBSCAN is available with `cluster.method: hdbscan`; in that mode `n_clusters` is ignored and noise points are assigned cluster `-1`.
