"""
Drifting-VLA: VLM-Conditioned One-Step Action Generation
=============================================================

A unified foundation model for multi-embodiment robotic manipulation
using Drifting models for one-step action generation conditioned on
Vision-Language Model (VLM) features.

Main Components:
    - models: DriftingVLA, VLM backbone, DiT transformer
    - training: Drifting loss, EMA, drifting field computation
    - data: Unified multi-dataset loader, action mapping
"""

__version__ = "2.0.0"
__author__ = "Drifting-VLA Team"

from drifting_vla.models.drifting_vla import DriftingVLA, DriftingVLAConfig
from drifting_vla.training.losses import DriftingLoss

__all__ = [
    "DriftingVLA",
    "DriftingVLAConfig",
    "DriftingLoss",
]
