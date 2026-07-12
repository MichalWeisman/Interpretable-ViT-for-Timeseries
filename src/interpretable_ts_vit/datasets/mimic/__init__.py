"""MIMIC-IV dataset adapter implementation.

MIMIC-IV is one concrete dataset source that can be converted into the
project's generic records/labels interface.
"""

from .mimic_targets import (
    MIMICIVMultiTargetAdapter,
    MIMICTargetVariableConfig,
    MIMICTargetsConfig,
    MIMICTargetWindowConfig,
    configured_variables_for_target,
    load_mimic_targets_config,
)

__all__ = [
    "MIMICIVMultiTargetAdapter",
    "MIMICTargetVariableConfig",
    "MIMICTargetsConfig",
    "MIMICTargetWindowConfig",
    "configured_variables_for_target",
    "load_mimic_targets_config",
]
