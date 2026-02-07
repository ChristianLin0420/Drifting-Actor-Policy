"""
Data Transforms and Augmentations
=================================

Image and action transforms for VLA training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from dataclasses import dataclass
import random


@dataclass
class TransformConfig:
    """
    Configuration for data transforms.
    
    Attributes:
        image_size: Target image size.
        random_crop: Enable random cropping.
        crop_scale: Scale range for random crop [min, max].
        color_jitter: Enable color jittering.
        jitter_brightness: Brightness jitter factor.
        jitter_contrast: Contrast jitter factor.
        jitter_saturation: Saturation jitter factor.
        random_flip: Enable random horizontal flip.
        normalize: Normalize images to [0, 1] range.
        action_noise: Std of noise to add to actions.
    """
    image_size: int = 224
    random_crop: bool = True
    crop_scale: tuple[float, float] = (0.8, 1.0)
    color_jitter: bool = True
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    random_flip: bool = False  # Often not applicable for robotics
    normalize: bool = True
    action_noise: float = 0.0


class VLATransforms:
    """
    Transforms for VLA data augmentation.
    
    Applies image augmentations and optional action perturbations
    for training robustness.
    
    Args:
        config: TransformConfig with augmentation settings.
        training: Whether in training mode (enables augmentations).
    
    Example:
        >>> config = TransformConfig(random_crop=True, color_jitter=True)
        >>> transform = VLATransforms(config, training=True)
        >>> sample = transform(sample)
    """
    
    def __init__(
        self,
        config: Optional[TransformConfig] = None,
        training: bool = True,
    ):
        self.config = config or TransformConfig()
        self.training = training
    
    def __call__(self, sample: dict) -> dict:
        """
        Apply transforms to a sample.
        
        Args:
            sample: Dict with 'images', 'actions', etc.
        
        Returns:
            Transformed sample.
        """
        sample = sample.copy()
        
        # Transform images
        if 'images' in sample:
            sample['images'] = self._transform_images(sample['images'])
        
        # Transform actions (add noise)
        if 'actions' in sample and self.training and self.config.action_noise > 0:
            sample['actions'] = self._add_action_noise(sample['actions'])
        
        return sample
    
    def _transform_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply image transforms."""
        # Handle multi-view: [V, C, H, W] or single view: [C, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        
        transformed = []
        
        for img in images:
            if self.training:
                # Random crop
                if self.config.random_crop:
                    img = self._random_crop(img)
                
                # Color jitter
                if self.config.color_jitter:
                    img = self._color_jitter(img)
                
                # Random flip
                if self.config.random_flip and random.random() > 0.5:
                    img = torch.flip(img, dims=[-1])
            
            # Resize to target size
            if img.shape[-1] != self.config.image_size:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.config.image_size, self.config.image_size),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            
            transformed.append(img)
        
        images = torch.stack(transformed)
        
        if squeeze_back:
            images = images.squeeze(0)
        
        return images
    
    def _random_crop(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random resized crop."""
        _, H, W = img.shape
        
        # Sample scale
        scale = random.uniform(*self.config.crop_scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        
        # Sample position
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        
        # Crop
        img = img[:, top:top+crop_h, left:left+crop_w]
        
        return img
    
    def _color_jitter(self, img: torch.Tensor) -> torch.Tensor:
        """Apply color jittering."""
        # Brightness
        if self.config.jitter_brightness > 0:
            factor = 1 + random.uniform(
                -self.config.jitter_brightness,
                self.config.jitter_brightness
            )
            img = img * factor
        
        # Contrast
        if self.config.jitter_contrast > 0:
            factor = 1 + random.uniform(
                -self.config.jitter_contrast,
                self.config.jitter_contrast
            )
            mean = img.mean()
            img = (img - mean) * factor + mean
        
        # Saturation (only for RGB)
        if img.shape[0] == 3 and self.config.jitter_saturation > 0:
            factor = 1 + random.uniform(
                -self.config.jitter_saturation,
                self.config.jitter_saturation
            )
            gray = img.mean(dim=0, keepdim=True)
            img = (img - gray) * factor + gray
        
        # Clamp to valid range
        img = img.clamp(0, 1)
        
        return img
    
    def _add_action_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to actions."""
        noise = torch.randn_like(actions) * self.config.action_noise
        return actions + noise


class ActionNormalization:
    """
    Action normalization and denormalization.
    
    Normalizes actions to have zero mean and unit variance,
    which improves training stability.
    
    Args:
        mean: Action mean [D_a].
        std: Action std [D_a].
        clip_range: Optional clip range after normalization.
    """
    
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        clip_range: Optional[float] = 5.0,
    ):
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.eps = 1e-8
    
    def fit(self, actions: np.ndarray) -> 'ActionNormalization':
        """
        Compute normalization statistics from data.
        
        Args:
            actions: Action array [N, D_a] or [N, T, D_a].
        
        Returns:
            Self with fitted statistics.
        """
        if actions.ndim == 3:
            actions = actions.reshape(-1, actions.shape[-1])
        
        self.mean = np.mean(actions, axis=0)
        self.std = np.std(actions, axis=0)
        
        return self
    
    def normalize(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize actions."""
        if self.mean is None or self.std is None:
            return actions
        
        mean = torch.as_tensor(self.mean, device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(self.std, device=actions.device, dtype=actions.dtype)
        
        normalized = (actions - mean) / (std + self.eps)
        
        if self.clip_range is not None:
            normalized = normalized.clamp(-self.clip_range, self.clip_range)
        
        return normalized
    
    def denormalize(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Denormalize actions to original scale."""
        if self.mean is None or self.std is None:
            return actions
        
        mean = torch.as_tensor(self.mean, device=actions.device, dtype=actions.dtype)
        std = torch.as_tensor(self.std, device=actions.device, dtype=actions.dtype)
        
        return actions * (std + self.eps) + mean
    
    def save(self, path: str) -> None:
        """Save normalization statistics."""
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            clip_range=self.clip_range,
        )
    
    @classmethod
    def load(cls, path: str) -> 'ActionNormalization':
        """Load normalization statistics."""
        data = np.load(path)
        return cls(
            mean=data['mean'],
            std=data['std'],
            clip_range=float(data['clip_range']),
        )


