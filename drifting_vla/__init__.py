"""
Drifting-VLA: Vision-Language-Action Policy with Drifting Model
================================================================

A PyTorch implementation of Drifting-VLA, which leverages the training-time
distribution evolution paradigm of Drifting models to learn multimodal action
policies conditioned on vision and language, achieving one-step inference
without iterative sampling.

Main Components:
    - models: Neural network architectures (DiT, encoders, decoders)
    - training: Drifting field computation and training utilities
    - data: Dataset loaders and data collection
    - envs: Environment wrappers (RLBench, LIBERO, CALVIN)
    - inference: One-step policy inference
    - logging: WandB integration and visualization

Example:
    >>> from drifting_vla.models import DriftingVLA
    >>> from drifting_vla.training import DriftingVLATrainer
    >>> 
    >>> model = DriftingVLA(config)
    >>> trainer = DriftingVLATrainer(model, train_loader, val_loader, config)
    >>> trainer.train()
"""

__version__ = "0.1.0"
__author__ = "Drifting-VLA Team"

from drifting_vla.models.drifting_vla import DriftingVLA
from drifting_vla.training.trainer import DriftingVLATrainer
from drifting_vla.training.losses import DriftingLoss, FeatureSpaceLoss

__all__ = [
    "DriftingVLA",
    "DriftingVLATrainer", 
    "DriftingLoss",
    "FeatureSpaceLoss",
]


