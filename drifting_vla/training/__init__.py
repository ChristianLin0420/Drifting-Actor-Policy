"""
Drifting-VLA Training Module
================================

Core training components:
    - drifting_field: Drifting field V_{p,q}(x) computation
    - losses: Drifting loss + hybrid MSE loss
    - ema: Exponential moving average
"""

from drifting_vla.training.drifting_field import (
    compute_drifting_field,
    compute_feature_normalization_scale,
    normalize_drift_field,
)
from drifting_vla.training.losses import DriftingLoss
from drifting_vla.training.ema import EMA

__all__ = [
    "compute_drifting_field",
    "compute_feature_normalization_scale",
    "normalize_drift_field",
    "DriftingLoss",
    "EMA",
]
