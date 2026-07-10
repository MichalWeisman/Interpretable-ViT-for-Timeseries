"""Dataset adapters that convert source datasets into the generic pipeline schema."""

from .base import DatasetAdapter, PreparedDataset, get_dataset_adapter, register_dataset_adapter
from .mimic_iv import MIMICIVHypotensionAdapter, MIMICHypotensionConfig
from .mimic_targets import MIMICIVMultiTargetAdapter, MIMICTargetsConfig, MIMICTargetWindowConfig, load_mimic_targets_config

__all__ = [
    "DatasetAdapter",
    "MIMICIVHypotensionAdapter",
    "MIMICIVMultiTargetAdapter",
    "MIMICHypotensionConfig",
    "MIMICTargetsConfig",
    "MIMICTargetWindowConfig",
    "PreparedDataset",
    "get_dataset_adapter",
    "load_mimic_targets_config",
    "register_dataset_adapter",
]
