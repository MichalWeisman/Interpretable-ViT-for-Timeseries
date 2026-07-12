"""Dataset adapters that convert source datasets into the generic pipeline schema."""

from .base import DatasetAdapter, PreparedDataset, TargetWindowConfig, get_dataset_adapter, register_dataset_adapter
from .mimic import MIMICIVMultiTargetAdapter, MIMICTargetVariableConfig, MIMICTargetsConfig, MIMICTargetWindowConfig, configured_variables_for_target, load_mimic_targets_config

__all__ = [
    "DatasetAdapter",
    "MIMICIVMultiTargetAdapter",
    "MIMICTargetVariableConfig",
    "MIMICTargetsConfig",
    "MIMICTargetWindowConfig",
    "PreparedDataset",
    "TargetWindowConfig",
    "configured_variables_for_target",
    "get_dataset_adapter",
    "load_mimic_targets_config",
    "register_dataset_adapter",
]
