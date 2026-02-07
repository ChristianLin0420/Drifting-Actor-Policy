"""
Action Decoder Module
=====================

This module implements the action decoder for Drifting-VLA, which
transforms transformer outputs into robot actions.

Action representation:
- Position: 3D Cartesian coordinates (x, y, z)
- Orientation: 6D rotation representation (more stable than quaternions)
- Gripper: Binary or continuous gripper state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionDecoderConfig:
    """
    Configuration for action decoder.
    
    Attributes:
        hidden_dim: Input feature dimension.
        action_horizon: Number of action steps to predict.
        position_dim: Position dimensionality (usually 3).
        rotation_dim: Rotation dimensionality (6 for 6D representation).
        gripper_dim: Gripper dimensionality (1 for binary, 2 for open/close).
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
        action_scale: Scale factor for action outputs.
    """
    hidden_dim: int = 1024
    action_horizon: int = 16
    position_dim: int = 3
    rotation_dim: int = 6  # 6D rotation representation
    gripper_dim: int = 1
    num_layers: int = 2
    dropout: float = 0.0
    action_scale: float = 1.0


class ActionDecoder(nn.Module):
    """
    MLP-based action decoder for robot control.
    
    Decodes transformer output tokens into action sequences including
    position, rotation, and gripper commands.
    
    Args:
        config: ActionDecoderConfig with decoder settings.
    
    Input:
        features: Transformer output features.
            Shape: [B, T, D] where T is action horizon
    
    Output:
        actions: Predicted actions.
            Shape: [B, T, action_dim] where action_dim = pos + rot + gripper
    
    Example:
        >>> config = ActionDecoderConfig(hidden_dim=1024, action_horizon=16)
        >>> decoder = ActionDecoder(config)
        >>> features = torch.randn(4, 16, 1024)
        >>> actions = decoder(features)  # [4, 16, 10]
    """
    
    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.config = config
        
        # Total action dimension
        self.action_dim = config.position_dim + config.rotation_dim + config.gripper_dim
        
        # Build MLP layers
        layers = []
        in_dim = config.hidden_dim
        hidden_dim = config.hidden_dim // 2
        
        for i in range(config.num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Separate heads for position, rotation, gripper
        self.position_head = nn.Linear(in_dim, config.position_dim)
        self.rotation_head = nn.Linear(in_dim, config.rotation_dim)
        self.gripper_head = nn.Linear(in_dim, config.gripper_dim)
        
        # Initialize output layers to small values
        self._init_output_heads()
    
    def _init_output_heads(self) -> None:
        """Initialize output heads with small weights."""
        for head in [self.position_head, self.rotation_head, self.gripper_head]:
            nn.init.trunc_normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Decode features into actions.
        
        Args:
            features: Transformer output [B, T, D].
            return_dict: If True, return separate action components.
        
        Returns:
            actions: [B, T, action_dim] or dict with separate components.
        """
        # Pass through MLP
        h = self.mlp(features)  # [B, T, hidden_dim]
        
        # Predict action components
        position = self.position_head(h) * self.config.action_scale  # [B, T, pos_dim]
        rotation = self.rotation_head(h)  # [B, T, rot_dim]
        gripper = self.gripper_head(h)  # [B, T, gripper_dim]
        
        # Normalize rotation based on representation
        if self.config.rotation_dim == 6:
            # 6D rotation representation - orthonormalize
            rotation = self._normalize_rotation_6d(rotation)
        elif self.config.rotation_dim == 4:
            # Quaternion - normalize to unit quaternion
            rotation = F.normalize(rotation, dim=-1)
        
        if return_dict:
            return {
                'position': position,
                'rotation': rotation,
                'gripper': gripper,
                'actions': torch.cat([position, rotation, gripper], dim=-1),
            }
        
        # Concatenate all action components
        actions = torch.cat([position, rotation, gripper], dim=-1)
        return actions
    
    def _normalize_rotation_6d(self, rotation: torch.Tensor) -> torch.Tensor:
        """
        Normalize 6D rotation to valid rotation matrix columns.
        
        The 6D rotation representation consists of the first two columns
        of a rotation matrix. We orthonormalize them here.
        
        Args:
            rotation: Raw 6D rotation [B, T, 6].
        
        Returns:
            Normalized 6D rotation [B, T, 6].
        """
        # Split into two 3D vectors
        a1 = rotation[..., :3]  # First column
        a2 = rotation[..., 3:]  # Second column
        
        # Normalize first vector
        b1 = F.normalize(a1, dim=-1)
        
        # Make second vector orthogonal to first
        dot = (b1 * a2).sum(dim=-1, keepdim=True)
        b2 = a2 - dot * b1
        b2 = F.normalize(b2, dim=-1)
        
        return torch.cat([b1, b2], dim=-1)
    
    def get_action_dim(self) -> int:
        """Get the total action dimension."""
        return self.action_dim


class ActionTokenizer(nn.Module):
    """
    Convert actions to tokens and vice versa.
    
    Used for tokenizing continuous actions into discrete tokens
    for certain architectures or analysis.
    
    Args:
        action_dim: Dimension of continuous actions.
        hidden_dim: Token embedding dimension.
        num_bins: Number of discretization bins per dimension.
    """
    
    def __init__(
        self,
        action_dim: int = 10,
        hidden_dim: int = 1024,
        num_bins: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        
        # Embedding for each action dimension
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_bins, hidden_dim // action_dim)
            for _ in range(action_dim)
        ])
        
        # Projection to full hidden dim
        self.projection = nn.Linear(hidden_dim // action_dim * action_dim, hidden_dim)
    
    def discretize(
        self,
        actions: torch.Tensor,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> torch.Tensor:
        """
        Discretize continuous actions to bins.
        
        Args:
            actions: Continuous actions [B, T, D].
            min_val: Minimum action value.
            max_val: Maximum action value.
        
        Returns:
            Discretized actions [B, T, D] as long tensor.
        """
        # Clamp to range
        actions = actions.clamp(min_val, max_val)
        
        # Normalize to [0, 1]
        actions = (actions - min_val) / (max_val - min_val)
        
        # Convert to bin indices
        bins = (actions * (self.num_bins - 1)).long()
        
        return bins
    
    def undiscretize(
        self,
        bins: torch.Tensor,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> torch.Tensor:
        """
        Convert discrete bins back to continuous actions.
        
        Args:
            bins: Discretized actions [B, T, D].
            min_val: Minimum action value.
            max_val: Maximum action value.
        
        Returns:
            Continuous actions [B, T, D].
        """
        # Convert to normalized [0, 1]
        actions = bins.float() / (self.num_bins - 1)
        
        # Scale to original range
        actions = actions * (max_val - min_val) + min_val
        
        return actions
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Tokenize continuous actions.
        
        Args:
            actions: Continuous actions [B, T, D].
        
        Returns:
            Action tokens [B, T, hidden_dim].
        """
        B, T, D = actions.shape
        
        # Discretize
        bins = self.discretize(actions)  # [B, T, D]
        
        # Embed each dimension
        embeddings = []
        for d in range(D):
            emb = self.embeddings[d](bins[..., d])  # [B, T, hidden_dim // D]
            embeddings.append(emb)
        
        # Concatenate and project
        tokens = torch.cat(embeddings, dim=-1)  # [B, T, hidden_dim]
        tokens = self.projection(tokens)
        
        return tokens


class NoiseTokenizer(nn.Module):
    """
    Convert noise vectors to tokens for transformer input.
    
    Maps random noise to initial tokens that will be transformed
    into action predictions by the DiT.
    
    Args:
        noise_dim: Dimension of input noise.
        hidden_dim: Output token dimension.
        action_horizon: Number of action tokens to generate.
    """
    
    def __init__(
        self,
        noise_dim: int = 64,
        hidden_dim: int = 1024,
        action_horizon: int = 16,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.action_horizon = action_horizon
        
        # Projection from noise to tokens
        self.projection = nn.Linear(noise_dim, hidden_dim)
        
        # Learnable position embeddings for action sequence
        self.position_embedding = nn.Parameter(
            torch.randn(1, action_horizon, hidden_dim) * 0.02
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Convert noise to action tokens.
        
        Args:
            noise: Random noise [B, T, noise_dim] or [B, noise_dim].
        
        Returns:
            Action tokens [B, T, hidden_dim].
        """
        # Handle different input shapes
        if noise.dim() == 2:
            B = noise.shape[0]
            noise = noise.unsqueeze(1).expand(-1, self.action_horizon, -1)
        
        # Project to hidden dim
        tokens = self.projection(noise)  # [B, T, hidden_dim]
        
        # Add positional embedding
        tokens = tokens + self.position_embedding
        
        return tokens


