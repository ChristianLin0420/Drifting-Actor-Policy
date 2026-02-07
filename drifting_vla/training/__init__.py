"""
Drifting-VLA Training Module
============================

This module contains the core training components for Drifting-VLA:
    - drifting_field: Algorithm for computing the drifting field V_{p,q}(x)
    - losses: Drifting loss functions (action-space and feature-space)
    - trainer: FSDP-enabled training loop
    - optimizer: AdamW with learning rate schedulers
    - ema: Exponential moving average for model parameters
"""

from drifting_vla.training.drifting_field import (
    compute_drifting_field,
    compute_feature_normalization_scale,
    normalize_drift_field,
)
from drifting_vla.training.losses import DriftingLoss, FeatureSpaceLoss
from drifting_vla.training.trainer import DriftingVLATrainer
from drifting_vla.training.ema import EMA

__all__ = [
    "compute_drifting_field",
    "compute_feature_normalization_scale",
    "normalize_drift_field",
    "DriftingLoss",
    "FeatureSpaceLoss",
    "DriftingVLATrainer",
    "EMA",
]


