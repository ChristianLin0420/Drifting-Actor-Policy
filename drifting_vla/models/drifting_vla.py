"""
Drifting-VLA: Main Model Assembly
=================================

This module assembles all components into the complete Drifting-VLA model
for Vision-Language-Action policy learning.

Architecture:
1. Vision Encoder (DINOv2) -> Visual features
2. Language Encoder (CLIP) -> Language features
3. Cross-Attention Fusion -> Fused context
4. DiT Transformer -> Processed action tokens
5. Action Decoder -> Robot actions
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union
import logging

from drifting_vla.models.vision_encoder import DINOv2Encoder, VisionEncoderConfig
from drifting_vla.models.language_encoder import CLIPLanguageEncoder, LanguageEncoderConfig
from drifting_vla.models.fusion import CrossAttentionFusion, FusionConfig
from drifting_vla.models.dit import DiTTransformer, DiTConfig
from drifting_vla.models.action_decoder import ActionDecoder, ActionDecoderConfig, NoiseTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DriftingVLAConfig:
    """
    Complete configuration for Drifting-VLA model.
    
    Attributes:
        vision: VisionEncoderConfig for the vision encoder.
        language: LanguageEncoderConfig for the language encoder.
        fusion: FusionConfig for cross-attention fusion.
        transformer: DiTConfig for the DiT transformer.
        action_decoder: ActionDecoderConfig for action prediction.
        
        # Model-level settings
        hidden_dim: Global hidden dimension (used if sub-configs don't specify).
        action_horizon: Number of action steps to predict.
        noise_dim: Dimension of input noise for action generation.
        cfg_scale_range: Range of CFG scales for training [min, max].
        cfg_dropout: Probability of dropping conditioning for CFG.
    """
    # Sub-module configs (will be created with defaults if None)
    vision: Optional[VisionEncoderConfig] = None
    language: Optional[LanguageEncoderConfig] = None
    fusion: Optional[FusionConfig] = None
    transformer: Optional[DiTConfig] = None
    action_decoder: Optional[ActionDecoderConfig] = None
    
    # Global settings
    hidden_dim: int = 1024
    action_horizon: int = 16
    noise_dim: int = 64
    cfg_scale_range: tuple[float, float] = (1.0, 4.0)
    cfg_dropout: float = 0.1
    
    def __post_init__(self):
        """Create default sub-configs if not provided."""
        if self.vision is None:
            self.vision = VisionEncoderConfig(hidden_dim=self.hidden_dim)
        if self.language is None:
            self.language = LanguageEncoderConfig(hidden_dim=self.hidden_dim)
        if self.fusion is None:
            self.fusion = FusionConfig(hidden_dim=self.hidden_dim)
        if self.transformer is None:
            self.transformer = DiTConfig(
                hidden_dim=self.hidden_dim,
                conditioning_dim=self.hidden_dim,
            )
        if self.action_decoder is None:
            self.action_decoder = ActionDecoderConfig(
                hidden_dim=self.hidden_dim,
                action_horizon=self.action_horizon,
            )


class DriftingVLA(nn.Module):
    """
    Drifting-VLA: Vision-Language-Action Policy with Drifting Model.
    
    This model generates robot actions conditioned on visual observations
    and language instructions using the Drifting paradigm for one-step
    inference.
    
    Args:
        config: DriftingVLAConfig with all model settings.
    
    Example:
        >>> config = DriftingVLAConfig(hidden_dim=1024, action_horizon=16)
        >>> model = DriftingVLA(config)
        >>> 
        >>> # Inputs
        >>> images = torch.randn(4, 3, 224, 224)
        >>> language = ["pick up the red cup", "open the drawer"] * 2
        >>> noise = torch.randn(4, 16, 64)
        >>> 
        >>> # Generate actions
        >>> actions = model(images, language, noise)  # [4, 16, 10]
    
    Note:
        During training, use forward() with return_features=True to get
        intermediate features for the drifting loss computation.
    """
    
    def __init__(self, config: DriftingVLAConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = DINOv2Encoder(config.vision)
        logger.info(f"Initialized vision encoder: {config.vision.model_name}")
        
        # Language encoder
        self.language_encoder = CLIPLanguageEncoder(config.language)
        logger.info(f"Initialized language encoder: {config.language.model_name}")
        
        # Project vision features to fusion hidden dim
        vision_hidden_dim = config.vision.hidden_dim
        fusion_hidden_dim = config.fusion.hidden_dim
        self.vision_proj = nn.Linear(vision_hidden_dim, fusion_hidden_dim)
        
        # Project language features to fusion hidden dim  
        lang_hidden_dim = config.language.hidden_dim
        self.language_proj = nn.Linear(lang_hidden_dim, fusion_hidden_dim)
        
        # Noise tokenizer
        self.noise_tokenizer = NoiseTokenizer(
            noise_dim=config.noise_dim,
            hidden_dim=fusion_hidden_dim,
            action_horizon=config.action_horizon,
        )
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(config.fusion)
        
        # Conditioning projection (from language encoder dim to transformer conditioning dim)
        transformer_hidden_dim = config.transformer.hidden_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(lang_hidden_dim, transformer_hidden_dim),
            nn.SiLU(),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim),
        )
        
        # CFG scale embedding
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, transformer_hidden_dim),
            nn.SiLU(),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim),
        )
        
        # DiT transformer
        self.transformer = DiTTransformer(config.transformer)
        
        # Action decoder
        self.action_decoder = ActionDecoder(config.action_decoder)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized DriftingVLA with {self.get_num_params():,} parameters")
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Apply to learnable components (not frozen encoders)
        self.vision_proj.apply(_init_module)
        self.language_proj.apply(_init_module)
        self.noise_tokenizer.apply(_init_module)
        self.fusion.apply(_init_module)
        self.cond_proj.apply(_init_module)
        self.cfg_embed.apply(_init_module)
    
    def forward(
        self,
        images: torch.Tensor,
        language: Union[str, list[str], torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass to generate actions.
        
        Args:
            images: RGB images [B, C, H, W] normalized to [0, 1].
            language: Task descriptions (strings) or pre-tokenized [B, L].
            noise: Random noise [B, T, noise_dim]. If None, sampled.
            cfg_scale: CFG scale [B] or scalar. If None, set to 1.0.
            return_features: If True, return intermediate features.
        
        Returns:
            If return_features=False:
                actions: Predicted actions [B, T, action_dim]
            If return_features=True:
                Dict with 'actions', 'visual_features', 'language_features',
                'fused_features', 'transformer_features'
        """
        B = images.shape[0]
        device = images.device
        
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn(
                B, self.config.action_horizon, self.config.noise_dim,
                device=device
            )
        
        # Set default CFG scale
        if cfg_scale is None:
            cfg_scale = torch.ones(B, device=device)
        elif isinstance(cfg_scale, (int, float)):
            cfg_scale = torch.full((B,), cfg_scale, device=device)
        
        # Encode visual features (supports multi-view [B, V, C, H, W])
        if images.dim() == 5:
            # Multi-view: process each view through shared DINOv2, concat tokens
            B_img, V, C, H, W = images.shape
            images_flat = images.view(B_img * V, C, H, W)
            feats = self.vision_encoder(images_flat, return_all_tokens=True)  # [B*V, N, D_v]
            N_tok, D_v = feats.shape[1], feats.shape[2]
            visual_features = feats.view(B_img, V * N_tok, D_v)  # [B, V*N, D_v]
        else:
            visual_features = self.vision_encoder(images, return_all_tokens=True)  # [B, N, D_v]
        visual_features = self.vision_proj(visual_features)  # [B, ?, D_f]
        
        # Encode language features
        language_features = self.language_encoder(
            language, return_all_tokens=True
        )  # [B, Nl, D_l]
        language_features = self.language_proj(language_features)  # [B, Nl, D_f]
        
        # Get language embedding for conditioning (CLS/pooled)
        language_pooled = self.language_encoder(language)  # [B, D_l]
        
        # Project to conditioning (use original lang dim)
        cond = self.cond_proj(language_pooled)  # [B, D]
        
        # Add CFG scale embedding
        cfg_embed = self.cfg_embed(cfg_scale.unsqueeze(-1))  # [B, D]
        cond = cond + cfg_embed
        
        # Convert noise to tokens
        noise_tokens = self.noise_tokenizer(noise)  # [B, T, D]
        
        # Fuse noise tokens with visual-language context
        fused_tokens = self.fusion(
            noise_tokens, visual_features, language_features
        )  # [B, T, D]
        
        # Process through DiT transformer
        transformer_out = self.transformer(fused_tokens, cond)  # [B, T, D]
        
        # Decode to actions
        actions = self.action_decoder(transformer_out)  # [B, T, action_dim]
        
        if return_features:
            return {
                'actions': actions,
                'visual_features': visual_features,
                'language_features': language_features,
                'fused_features': fused_tokens,
                'transformer_features': transformer_out,
                'conditioning': cond,
            }
        
        return actions
    
    def generate(
        self,
        images: torch.Tensor,
        language: Union[str, list[str]],
        cfg_scale: float = 2.0,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate action samples (inference mode).
        
        Args:
            images: Input images [B, C, H, W].
            language: Task descriptions.
            cfg_scale: Classifier-free guidance scale.
            num_samples: Number of samples to generate per input.
        
        Returns:
            actions: Generated actions [B * num_samples, T, action_dim].
        """
        B = images.shape[0]
        device = images.device
        
        # Expand inputs for multiple samples
        if num_samples > 1:
            images = images.repeat_interleave(num_samples, dim=0)
            if isinstance(language, list):
                language = [l for l in language for _ in range(num_samples)]
        
        # Generate with CFG
        with torch.no_grad():
            if cfg_scale > 1.0:
                # Generate conditioned output
                actions_cond = self.forward(
                    images, language,
                    cfg_scale=torch.full((B * num_samples,), cfg_scale, device=device)
                )
                
                # Generate unconditioned output
                actions_uncond = self.forward(
                    images, [""] * (B * num_samples),
                    cfg_scale=torch.ones(B * num_samples, device=device)
                )
                
                # Apply CFG
                actions = actions_uncond + cfg_scale * (actions_cond - actions_uncond)
            else:
                actions = self.forward(images, language)
        
        return actions
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_encoders(self) -> None:
        """Freeze vision and language encoders."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.language_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen vision and language encoders")
    
    def unfreeze_encoders(self) -> None:
        """Unfreeze vision and language encoders for fine-tuning."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.language_encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen vision and language encoders")


def create_drifting_vla(
    model_size: str = 'base',
    action_horizon: int = 16,
    action_dim: int = 10,
    **kwargs,
) -> DriftingVLA:
    """
    Factory function to create Drifting-VLA models.
    
    Args:
        model_size: Model size ('small', 'base', 'large').
        action_horizon: Number of action steps.
        action_dim: Action dimension.
        **kwargs: Additional config overrides.
    
    Returns:
        Configured DriftingVLA model.
    """
    # Model size configurations
    size_configs = {
        'small': {
            'hidden_dim': 512,
            'num_layers': 12,
            'num_heads': 8,
        },
        'base': {
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
        },
        'large': {
            'hidden_dim': 1024,
            'num_layers': 24,
            'num_heads': 16,
        },
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    size_cfg = size_configs[model_size]
    
    # Create config
    config = DriftingVLAConfig(
        hidden_dim=size_cfg['hidden_dim'],
        action_horizon=action_horizon,
        transformer=DiTConfig(
            hidden_dim=size_cfg['hidden_dim'],
            num_layers=size_cfg['num_layers'],
            num_heads=size_cfg['num_heads'],
            conditioning_dim=size_cfg['hidden_dim'],
        ),
        action_decoder=ActionDecoderConfig(
            hidden_dim=size_cfg['hidden_dim'],
            action_horizon=action_horizon,
            position_dim=3,
            rotation_dim=6,
            gripper_dim=1,
        ),
    )
    
    return DriftingVLA(config)


