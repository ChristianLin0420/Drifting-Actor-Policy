"""
Vision Encoder: DINOv2
======================

This module implements the vision encoder using DINOv2, which provides
strong visual representations learned through self-supervised learning.

DINOv2 advantages for VLA:
- No language supervision bias (unlike CLIP)
- Better spatial understanding
- Robust to domain shift
- Multi-scale features available
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisionEncoderConfig:
    """
    Configuration for vision encoder.
    
    Attributes:
        model_name: DINOv2 model variant.
            Options: 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
        image_size: Input image resolution (must be divisible by patch_size).
        patch_size: ViT patch size (14 for DINOv2).
        hidden_dim: Output feature dimension.
        freeze: Whether to freeze encoder weights.
        use_registers: Use register tokens (DINOv2 v2 feature).
        output_tokens: Return all patch tokens or just CLS.
    """
    model_name: str = 'dinov2_vitl14'
    image_size: int = 224
    patch_size: int = 14
    hidden_dim: int = 1024
    freeze: bool = True
    use_registers: bool = False
    output_tokens: str = 'all'  # 'cls', 'all', 'mean'
    pretrained: bool = True
    

class DINOv2Encoder(nn.Module):
    """
    DINOv2 Vision Encoder for extracting visual features.
    
    Wraps the DINOv2 model from torch.hub or transformers and provides
    a consistent interface for the Drifting-VLA pipeline.
    
    Args:
        config: VisionEncoderConfig with model settings.
    
    Input:
        images: RGB images normalized to [0, 1].
            Shape: [B, C, H, W] where C=3, H=W=image_size
    
    Output:
        features: Visual features.
            Shape depends on output_tokens:
            - 'cls': [B, hidden_dim]
            - 'all': [B, num_patches + 1, hidden_dim]
            - 'mean': [B, hidden_dim]
    
    Example:
        >>> config = VisionEncoderConfig(model_name='dinov2_vitl14', freeze=True)
        >>> encoder = DINOv2Encoder(config)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = encoder(images)  # [4, 257, 1024]
    """
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load DINOv2 model
        self.encoder = self._load_model()
        
        # Get output dimension from model
        self.encoder_dim = self._get_encoder_dim()
        
        # Projection layer if dimensions don't match
        if self.encoder_dim != config.hidden_dim:
            self.projection = nn.Linear(self.encoder_dim, config.hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if specified
        if config.freeze:
            self._freeze_encoder()
        
        # Compute number of patches
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        logger.info(
            f"Initialized DINOv2 encoder: {config.model_name}, "
            f"freeze={config.freeze}, output_tokens={config.output_tokens}"
        )
    
    def _load_model(self) -> nn.Module:
        """Load DINOv2 model from torch.hub."""
        model_name = self.config.model_name
        
        try:
            # Try loading from torch.hub (Facebook Research)
            model = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=self.config.pretrained,
            )
            logger.info(f"Loaded {model_name} from torch.hub")
        except Exception as e:
            logger.warning(f"Failed to load from torch.hub: {e}")
            # Fallback to transformers
            try:
                from transformers import AutoModel
                model_map = {
                    'dinov2_vits14': 'facebook/dinov2-small',
                    'dinov2_vitb14': 'facebook/dinov2-base',
                    'dinov2_vitl14': 'facebook/dinov2-large',
                    'dinov2_vitg14': 'facebook/dinov2-giant',
                }
                hf_name = model_map.get(model_name, 'facebook/dinov2-large')
                model = AutoModel.from_pretrained(hf_name)
                logger.info(f"Loaded {model_name} from transformers as {hf_name}")
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise RuntimeError(f"Could not load DINOv2 model {model_name}")
        
        return model
    
    def _get_encoder_dim(self) -> int:
        """Get the output dimension of the encoder."""
        dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        return dim_map.get(self.config.model_name, 1024)
    
    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        logger.info("Frozen DINOv2 encoder parameters")
    
    def forward(
        self,
        images: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            images: Input images [B, 3, H, W], normalized to [0, 1].
            return_all_tokens: Override config to return all tokens.
        
        Returns:
            Visual features with shape determined by output_tokens config.
        """
        # Normalize images (DINOv2 expects ImageNet normalization)
        images = self._normalize(images)
        
        # Forward through encoder
        if self.config.freeze:
            with torch.no_grad():
                features = self._encode(images)
        else:
            features = self._encode(images)
        
        # Project to target dimension
        features = self.projection(features)
        
        # Select output format
        output_tokens = 'all' if return_all_tokens else self.config.output_tokens
        
        if output_tokens == 'cls':
            return features[:, 0]  # [B, D]
        elif output_tokens == 'mean':
            return features[:, 1:].mean(dim=1)  # [B, D] - exclude CLS, mean over patches
        else:  # 'all'
            return features  # [B, N+1, D]
    
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Run forward pass through encoder."""
        # Handle different DINOv2 API versions
        if hasattr(self.encoder, 'forward_features'):
            # torch.hub version
            features = self.encoder.forward_features(images)
            if isinstance(features, dict):
                features = features['x_norm_patchtokens']
                # Add CLS token
                cls_token = self.encoder.forward_features(images)['x_norm_clstoken']
                features = torch.cat([cls_token.unsqueeze(1), features], dim=1)
        else:
            # transformers version
            outputs = self.encoder(images)
            features = outputs.last_hidden_state
        
        return features
    
    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std
    
    def get_num_tokens(self) -> int:
        """Get the number of output tokens (including CLS)."""
        return self.num_patches + 1


class MultiViewDINOv2Encoder(nn.Module):
    """
    Multi-view vision encoder using DINOv2.
    
    Processes multiple camera views and combines their features.
    Useful for robotic manipulation with multiple cameras (wrist, front, etc.)
    
    Args:
        config: VisionEncoderConfig
        num_views: Number of camera views
        fusion_method: How to combine views ('concat', 'mean', 'attention')
    """
    
    def __init__(
        self,
        config: VisionEncoderConfig,
        num_views: int = 2,
        fusion_method: str = 'concat',
    ):
        super().__init__()
        self.config = config
        self.num_views = num_views
        self.fusion_method = fusion_method
        
        # Shared encoder for all views
        self.encoder = DINOv2Encoder(config)
        
        # View fusion
        if fusion_method == 'concat':
            self.fusion = nn.Linear(
                config.hidden_dim * num_views,
                config.hidden_dim
            )
        elif fusion_method == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=8,
                batch_first=True,
            )
        else:
            self.fusion = None  # mean fusion doesn't need extra params
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process multi-view images.
        
        Args:
            images: Multi-view images [B, V, C, H, W] where V is num_views
        
        Returns:
            Fused features [B, hidden_dim] or [B, N, hidden_dim]
        """
        B, V, C, H, W = images.shape
        
        # Process all views
        images_flat = images.view(B * V, C, H, W)
        features = self.encoder(images_flat, return_all_tokens=True)  # [B*V, N, D]
        
        # Reshape to [B, V, N, D]
        N, D = features.shape[1], features.shape[2]
        features = features.view(B, V, N, D)
        
        # Fuse views
        if self.fusion_method == 'concat':
            # Concatenate view features and project
            features = features.view(B, V * N, D)[:, :N]  # Take first N tokens
            cls_features = features[:, 0]  # [B, V, D]
            cls_features = cls_features.view(B, -1)  # [B, V*D]
            return self.fusion(cls_features)  # [B, D]
        
        elif self.fusion_method == 'mean':
            return features.mean(dim=1)  # [B, N, D]
        
        elif self.fusion_method == 'attention':
            # Cross-view attention
            features_flat = features.view(B, V * N, D)
            attn_out, _ = self.fusion(features_flat, features_flat, features_flat)
            return attn_out[:, :N]  # [B, N, D]
        
        return features[:, 0]  # Default: first view


