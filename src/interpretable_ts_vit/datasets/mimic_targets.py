"""Compatibility exports for the MIMIC-IV target adapter.

New code should import from `interpretable_ts_vit.datasets.mimic`.
"""

from .mimic.mimic_targets import (
    COHORT_LEVELS,
    DEFAULT_TARGETS,
    REQUIRED_CONFIG_FIELDS,
    TARGET_NAMES,
    MIMICIVMultiTargetAdapter,
    MIMICTargetVariableConfig,
    MIMICTargetWindowConfig,
    MIMICTargetsConfig,
    _match_patterns,
    load_mimic_targets_config,
)

__all__ = [
    "COHORT_LEVELS",
    "DEFAULT_TARGETS",
    "REQUIRED_CONFIG_FIELDS",
    "TARGET_NAMES",
    "MIMICIVMultiTargetAdapter",
    "MIMICTargetVariableConfig",
    "MIMICTargetWindowConfig",
    "MIMICTargetsConfig",
    "_match_patterns",
    "load_mimic_targets_config",
]
