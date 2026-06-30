"""One-file endpoint for running the full interpretable ViT workflow.

Edit the settings below, then run:

    python main.py

This file intentionally avoids the `tsvit` CLI. It calls the package's Python
API directly so the same workflow is easy to run from an IDE, scheduler, or
notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interpretable_ts_vit.config import ClusterConfig, Config, DataConfig, ExplainConfig, ModelConfig, TrainConfig
from interpretable_ts_vit.datasets import MIMICHypotensionConfig
from interpretable_ts_vit.pipeline import PipelinePaths, PipelineRunConfig, run_pipeline


SETTINGS = {
    # Use MIMIC-IV directly. Set this to False and fill records_path/labels_path
    # if you already have generic records.csv and labels.csv files.
    "prepare_mimic": True,
    "mimic_path": ROOT / "mimic-iv-3.1.zip",
    "records_path": None,
    "labels_path": None,
    "dataset_dir": ROOT / "data" / "mimic_hypotension",
    "processed_dir": ROOT / "data" / "processed",
    "run_dir": ROOT / "runs" / "hypotension_v1",
    "split": "test",
    "render_instance_heatmaps": False,
}


MIMIC_SETTINGS = {
    "observation_hours": 24.0,
    "prediction_hours": 6.0,
    "hypotension_threshold": 65.0,
    "chunk_size": 1_000_000,
    # Keep None for the full dataset. Set a small integer, e.g. 1000, for a
    # quick local smoke run.
    "max_stays": None,
    "min_observations": 1,
    "require_full_prediction_window": True,
    "require_outcome_measurement": True,
}


PIPELINE_CONFIG = Config(
    data=DataConfig(
        granularity="30min",
        # The MIMIC adapter exports relative ICU time anchored at this date.
        time_start="2000-01-01 00:00:00",
        time_end="2000-01-02 00:00:00",
        aggregation="mean",
        val_fraction=0.2,
        test_fraction=0.2,
        random_state=13,
    ),
    model=ModelConfig(
        patch_size=(1, 4),
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
    ),
    train=TrainConfig(
        batch_size=16,
        epochs=10,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="auto",
    ),
    explain=ExplainConfig(
        method="grad_attention_rollout",
        target_class=None,
    ),
    cluster=ClusterConfig(
        n_clusters=8,
        method="kmeans",
        aggregate="mean",
    ),
)


def build_run_config() -> PipelineRunConfig:
    """Build the object passed into the programmatic pipeline."""
    paths = PipelinePaths(
        mimic_path=SETTINGS["mimic_path"],
        records_path=SETTINGS["records_path"],
        labels_path=SETTINGS["labels_path"],
        dataset_dir=SETTINGS["dataset_dir"],
        processed_dir=SETTINGS["processed_dir"],
        run_dir=SETTINGS["run_dir"],
    )
    mimic_config = None
    if SETTINGS["prepare_mimic"]:
        mimic_config = MIMICHypotensionConfig(
            mimic_path=SETTINGS["mimic_path"],
            **MIMIC_SETTINGS,
        )
    return PipelineRunConfig(
        paths=paths,
        config=PIPELINE_CONFIG,
        mimic_config=mimic_config,
        prepare_mimic=SETTINGS["prepare_mimic"],
        prepare_tensors=True,
        train=True,
        evaluate=True,
        explain=True,
        cluster=True,
        plot=True,
        split=SETTINGS["split"],
        render_instance_heatmaps=SETTINGS["render_instance_heatmaps"],
    )


def main() -> None:
    """Run the complete workflow and print a compact artifact summary."""
    result = run_pipeline(build_run_config())
    print(json.dumps({"artifacts": result.artifacts, "evaluation_metrics": result.evaluation_metrics}, indent=2))


if __name__ == "__main__":
    main()
