"""
Drifting-VLA Policy Wrapper
===========================

Inference wrapper for trained Drifting-VLA models with:
- One-step action generation
- Action denormalization
- Temporal smoothing (optional)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """
    Configuration for inference policy.
    
    Attributes:
        cfg_scale: Classifier-free guidance scale.
        action_horizon: Number of action steps to predict.
        use_ema: Use EMA model weights.
        device: Device for inference.
        temporal_ensemble: Use temporal ensemble averaging.
        ensemble_weights: Weights for temporal ensemble.
    """
    cfg_scale: float = 2.0
    action_horizon: int = 16
    use_ema: bool = True
    device: str = 'cuda'
    temporal_ensemble: bool = True
    ensemble_weights: Optional[list[float]] = None


class DriftingVLAPolicy:
    """
    Policy wrapper for Drifting-VLA inference.
    
    Handles preprocessing, model inference, and postprocessing
    for generating robot actions from observations.
    
    Args:
        model: Trained DriftingVLA model.
        config: PolicyConfig with inference settings.
        action_normalizer: Optional action normalizer for denormalization.
    
    Example:
        >>> model = DriftingVLA.load('checkpoint.pt')
        >>> policy = DriftingVLAPolicy(model, config)
        >>> 
        >>> obs = env.reset()
        >>> while not done:
        ...     action = policy.get_action(obs.images, obs.language)
        ...     obs = env.step(action)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[PolicyConfig] = None,
        action_normalizer: Optional['ActionNormalization'] = None,
    ):
        self.config = config or PolicyConfig()
        self.action_normalizer = action_normalizer
        
        # Setup device
        self.device = torch.device(self.config.device)
        
        # Setup model
        self.model = model.to(self.device)
        self.model.eval()
        
        # Temporal ensemble buffer
        self.action_buffer = []
        self.buffer_idx = 0
        
        # Ensemble weights (exponential decay)
        if self.config.ensemble_weights is None:
            weights = [0.5 ** i for i in range(self.config.action_horizon)]
            self.ensemble_weights = np.array(weights) / sum(weights)
        else:
            self.ensemble_weights = np.array(self.config.ensemble_weights)
        
        logger.info(f"Initialized policy with CFG scale={self.config.cfg_scale}")
    
    @torch.no_grad()
    def get_action(
        self,
        images: Union[np.ndarray, torch.Tensor],
        language: str,
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get single action for current observation.
        
        Args:
            images: Visual observation [C, H, W] or [V, C, H, W].
            language: Task description string.
            proprio: Optional proprioceptive state.
        
        Returns:
            Action array [D_a].
        """
        # Get full action sequence
        actions = self.get_action_sequence(images, language, proprio)
        
        if self.config.temporal_ensemble and len(self.action_buffer) > 0:
            # Temporal ensemble
            action = self._temporal_ensemble()
        else:
            # Use first action
            action = actions[0]
        
        return action
    
    @torch.no_grad()
    def get_action_sequence(
        self,
        images: Union[np.ndarray, torch.Tensor],
        language: str,
        proprio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get full action sequence prediction.
        
        Args:
            images: Visual observation [C, H, W] or [V, C, H, W].
            language: Task description string.
            proprio: Optional proprioceptive state.
        
        Returns:
            Action sequence [T, D_a].
        """
        # Preprocess images
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        
        images = images.to(self.device)
        
        # Add batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)  # [1, C, H, W]
        elif images.dim() == 4 and images.shape[0] != 1:
            # Multi-view: [V, C, H, W] -> [1, V, C, H, W]
            images = images.unsqueeze(0)
        
        # Generate actions with CFG
        actions = self.model.generate(
            images=images,
            language=[language],
            cfg_scale=self.config.cfg_scale,
            num_samples=1,
        )
        
        # Remove batch dimension
        actions = actions.squeeze(0).cpu().numpy()  # [T, D_a]
        
        # Denormalize if normalizer provided
        if self.action_normalizer is not None:
            actions = self.action_normalizer.denormalize(
                torch.from_numpy(actions)
            ).numpy()
        
        # Update action buffer for temporal ensemble
        if self.config.temporal_ensemble:
            self.action_buffer.append(actions)
            if len(self.action_buffer) > self.config.action_horizon:
                self.action_buffer.pop(0)
        
        return actions
    
    def _temporal_ensemble(self) -> np.ndarray:
        """
        Compute temporally ensembled action.
        
        Uses overlapping action predictions to smooth output.
        """
        if len(self.action_buffer) == 0:
            return np.zeros(self.action_buffer[0].shape[-1])
        
        # Collect actions at current timestep from all predictions
        actions = []
        weights = []
        
        for i, seq in enumerate(self.action_buffer):
            # Index into sequence based on when it was predicted
            idx = len(self.action_buffer) - 1 - i
            if idx < len(seq):
                actions.append(seq[idx])
                weights.append(self.ensemble_weights[i])
        
        if not actions:
            return self.action_buffer[-1][0]
        
        # Weighted average
        actions = np.stack(actions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.average(actions, axis=0, weights=weights)
    
    def reset(self) -> None:
        """Reset policy state (clear action buffer)."""
        self.action_buffer = []
        self.buffer_idx = 0
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[PolicyConfig] = None,
        model_config: Optional[dict] = None,
    ) -> 'DriftingVLAPolicy':
        """
        Load policy from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            config: PolicyConfig for inference.
            model_config: Optional model configuration override.
        
        Returns:
            Loaded policy.
        """
        from drifting_vla.models import DriftingVLA, DriftingVLAConfig
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config
        if model_config is None:
            model_config = checkpoint.get('config', {})
        
        # Create model
        if isinstance(model_config, DriftingVLAConfig):
            model = DriftingVLA(model_config)
        else:
            model = DriftingVLA(DriftingVLAConfig(**model_config))
        
        # Load weights (prefer EMA if available and requested)
        config = config or PolicyConfig()
        if config.use_ema and 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict']['ema_model'])
            logger.info("Loaded EMA weights")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model weights")
        
        return cls(model, config)


