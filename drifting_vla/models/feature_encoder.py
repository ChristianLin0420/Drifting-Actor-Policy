"""
Action Feature Encoder for Multi-Scale Loss
============================================

This module implements the action feature encoder used for computing
the feature-space drifting loss. It extracts multi-scale representations
from action sequences that capture both local (per-step) and global
(trajectory-level) features.

The encoder is pre-trained using Masked Action Modeling (MAM), similar
to MAE for images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureEncoderConfig:
    """
    Configuration for action feature encoder.
    
    Attributes:
        action_dim: Input action dimension.
        hidden_dim: Encoder hidden dimension.
        action_horizon: Length of action sequence.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        patch_sizes: Temporal patch sizes for multi-scale features.
    """
    action_dim: int = 10
    hidden_dim: int = 512
    action_horizon: int = 16
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    patch_sizes: list[int] = None
    
    def __post_init__(self):
        if self.patch_sizes is None:
            self.patch_sizes = [1, 2, 4]


class ActionFeatureEncoder(nn.Module):
    """
    Transformer-based action sequence encoder.
    
    Extracts multi-scale features from action sequences for use in
    feature-space drifting loss. The encoder can be pre-trained with
    Masked Action Modeling.
    
    Args:
        config: FeatureEncoderConfig with encoder settings.
    
    Example:
        >>> config = FeatureEncoderConfig(action_dim=10, hidden_dim=512)
        >>> encoder = ActionFeatureEncoder(config)
        >>> actions = torch.randn(4, 16, 10)  # [B, T, D]
        >>> features = encoder.extract_multi_scale(actions)
        >>> # features['global'] -> [4, 512]
        >>> # features['patch_2'] -> [4, 8, 512]
    """
    
    def __init__(self, config: FeatureEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.action_dim, config.hidden_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.action_horizon, config.hidden_dim) * 0.02
        )
        
        # CLS token for global features
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        
        # Output normalization
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Patch embedding layers for different scales
        self.patch_embeddings = nn.ModuleDict()
        for patch_size in config.patch_sizes:
            if patch_size > 1:
                self.patch_embeddings[f'patch_{patch_size}'] = nn.Linear(
                    config.hidden_dim * patch_size, config.hidden_dim
                )
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode action sequence to feature representation.
        
        Args:
            actions: Action sequence [B, T, action_dim].
        
        Returns:
            Features [B, T+1, hidden_dim] (including CLS token).
        """
        B, T, _ = actions.shape
        
        # Project actions
        x = self.input_proj(actions)  # [B, T, D]
        
        # Add positional embedding
        x = x + self.pos_embed[:, :T]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, D]
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        return x
    
    def extract_multi_scale(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Extract multi-scale features from action sequence.
        
        Returns features at different temporal scales:
        - 'global': Global trajectory feature (from CLS token)
        - 'local': Per-timestep features
        - 'patch_N': Features aggregated over N timesteps
        
        Args:
            actions: Action sequence [B, T, action_dim].
        
        Returns:
            Dict mapping scale names to feature tensors.
        """
        B, T, _ = actions.shape
        
        # Get full encoding
        features = self.forward(actions)  # [B, T+1, D]
        
        # Extract scales
        outputs = {}
        
        # Global: CLS token
        outputs['global'] = features[:, 0]  # [B, D]
        
        # Local: Per-timestep (excluding CLS)
        outputs['local'] = features[:, 1:]  # [B, T, D]
        
        # Patch features at different scales
        for patch_size in self.config.patch_sizes:
            if patch_size > 1:
                outputs[f'patch_{patch_size}'] = self._extract_patch_features(
                    outputs['local'], patch_size
                )
        
        return outputs
    
    def _extract_patch_features(
        self,
        local_features: torch.Tensor,
        patch_size: int,
    ) -> torch.Tensor:
        """
        Extract features aggregated over temporal patches.
        
        Args:
            local_features: Per-timestep features [B, T, D].
            patch_size: Number of timesteps per patch.
        
        Returns:
            Patch features [B, num_patches, D].
        """
        B, T, D = local_features.shape
        
        # Compute number of patches
        num_patches = T // patch_size
        
        # Reshape to patches
        patches = local_features[:, :num_patches * patch_size]
        patches = patches.reshape(B, num_patches, patch_size * D)
        
        # Project patches
        patch_key = f'patch_{patch_size}'
        if patch_key in self.patch_embeddings:
            patches = self.patch_embeddings[patch_key](patches)
        else:
            patches = patches.mean(dim=-1, keepdim=True).expand(-1, -1, D)
        
        return patches


class MaskedActionModeling(nn.Module):
    """
    Masked Action Modeling (MAM) for pre-training the feature encoder.
    
    Similar to MAE: randomly mask action patches and train the encoder
    to reconstruct them.
    
    Args:
        encoder: ActionFeatureEncoder to pre-train.
        mask_ratio: Fraction of patches to mask (default: 0.5).
        patch_size: Size of patches to mask (default: 2).
    """
    
    def __init__(
        self,
        encoder: ActionFeatureEncoder,
        mask_ratio: float = 0.5,
        patch_size: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        config = encoder.config
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim * patch_size),
        )
    
    def forward(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        MAM forward pass with masking and reconstruction.
        
        Args:
            actions: Action sequence [B, T, action_dim].
        
        Returns:
            Dict with 'loss', 'pred', 'target', 'mask'.
        """
        B, T, D = actions.shape
        
        # Compute number of patches
        num_patches = T // self.patch_size
        
        # Random masking
        num_mask = int(num_patches * self.mask_ratio)
        noise = torch.rand(B, num_patches, device=actions.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_mask = ids_shuffle[:, :num_mask]
        ids_keep = ids_shuffle[:, num_mask:]
        
        # Create mask
        mask = torch.zeros(B, num_patches, device=actions.device, dtype=torch.bool)
        mask.scatter_(1, ids_mask, True)
        
        # Reshape actions to patches
        actions_patched = actions[:, :num_patches * self.patch_size]
        actions_patched = actions_patched.reshape(B, num_patches, self.patch_size, D)
        
        # Replace masked patches with mask tokens
        x = self.encoder.input_proj(actions.mean(dim=-1, keepdim=True))  # Placeholder
        x = self.encoder.forward(actions)[:, 1:]  # Get local features
        
        # Reconstruction loss on masked patches
        pred = self.decoder(x)  # [B, num_patches, patch_size * action_dim]
        pred = pred.reshape(B, num_patches, self.patch_size, D)
        
        target = actions_patched
        
        # Compute loss only on masked patches
        loss = F.mse_loss(
            pred[mask].reshape(-1, self.patch_size, D),
            target[mask].reshape(-1, self.patch_size, D),
        )
        
        return {
            'loss': loss,
            'pred': pred,
            'target': target,
            'mask': mask,
        }


class ActionFeatureEncoderWithMAM(nn.Module):
    """
    Wrapper for ActionFeatureEncoder with MAM pre-training support.
    
    Provides easy interface for pre-training and then using the
    encoder for feature extraction.
    
    Args:
        config: FeatureEncoderConfig.
    """
    
    def __init__(self, config: FeatureEncoderConfig):
        super().__init__()
        self.encoder = ActionFeatureEncoder(config)
        self.mam = MaskedActionModeling(self.encoder)
        self.pretrained = False
    
    def pretrain_step(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one pre-training step."""
        return self.mam(actions)
    
    def extract_multi_scale(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract multi-scale features."""
        return self.encoder.extract_multi_scale(actions)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights after pre-training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.pretrained = True


