"""
Drifting-VLA: VLM-Conditioned Drifting DiT
==============================================

Main model assembly:
  System 1: VLM Backbone (Qwen3-VL-2B) — frozen or LoRA fine-tuned
  System 2: Drifting DiT — cross-attention + self-attention action generator

    Supports pre-computed VLM features for training efficiency
  - Embodiment embedding for multi-embodiment conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
import logging

from drifting_vla.models.dit import DiTTransformer, DiTConfig
from drifting_vla.models.fusion import CrossAttentionFusion, FusionConfig
from drifting_vla.models.action_decoder import NoiseTokenizer
from drifting_vla.models.vlm_backbone import VLMBackbone, VLMConfig, VLM_SPECS
from drifting_vla.data.action_mapping import (
    UNIFIED_ACTION_DIM, get_action_mask_tensor,
    normalize_quaternion_in_unified,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftingVLAConfig:
    """
    Configuration for Drifting-VLA.
    
    Attributes:
        vlm: VLM backbone config.
        hidden_dim: DiT hidden dimension.
        num_layers: DiT transformer layers.
        num_heads: DiT attention heads.
        mlp_ratio: DiT MLP expansion ratio.
        num_cross_attn_layers: Cross-attention layers for VLM fusion.
        action_horizon: Number of action steps to predict.
        noise_dim: Input noise dimension per timestep.
        num_embodiments: Number of embodiment types (3: gripper, bimanual, dex).
        cfg_dropout: Probability of dropping conditioning for CFG training.
        use_flash_attn: Use Flash Attention.
        use_rope: Use Rotary Position Embedding.
        use_qk_norm: Use QK-Norm in attention.
        use_swiglu: Use SwiGLU in FFN.
        dropout: Dropout rate.
    """
    # VLM backbone
    vlm_model_key: str = 'qwen3vl'
    vlm_freeze: bool = True
    vlm_use_lora: bool = False
    
    # DiT architecture
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_cross_attn_layers: int = 2
    
    # Action generation
    action_horizon: int = 16
    noise_dim: int = 64
    action_dim: int = UNIFIED_ACTION_DIM  # 128
    
    # Multi-embodiment
    num_embodiments: int = 3
    
    # Training
    cfg_dropout: float = 0.1
    cfg_scale_range: Tuple[float, float] = (1.0, 4.0)
    
    # Architecture options
    use_flash_attn: bool = True
    use_rope: bool = True
    use_qk_norm: bool = True
    use_swiglu: bool = True
    dropout: float = 0.0


class DriftingVLA(nn.Module):
    """
    Drifting-VLA: VLM-conditioned one-step action generation.
    
    Architecture:
        1. VLM backbone extracts c_seq [B, L, vlm_dim] and c_pool [B, vlm_dim]
        2. Feature projectors map to DiT dim: [B, L, hidden_dim]
        3. Noise tokenizer creates [B, T, hidden_dim] from noise
        4. Cross-attention fuses noise tokens with VLM context
        5. DiT transformer refines tokens with adaLN conditioning
        6. Unified action head outputs [B, T, 128]
    
    Training modes:
        - Precomputed: Use pre-extracted VLM features (fast, no VLM in GPU)
        - Live: Run VLM forward during training (for LoRA fine-tuning)
    """
    
    def __init__(self, config: DriftingVLAConfig):
        super().__init__()
        self.config = config
        
        # VLM backbone (loaded lazily)
        vlm_spec = VLM_SPECS[config.vlm_model_key]
        self.vlm_hidden_dim = vlm_spec['hidden_dim']
        
        vlm_config = VLMConfig(
            model_key=config.vlm_model_key,
            dit_hidden_dim=config.hidden_dim,
            freeze_base=config.vlm_freeze,
            use_lora=config.vlm_use_lora,
        )
        self.vlm_backbone = VLMBackbone(vlm_config)
        
        # Noise tokenizer: [noise_dim] → [hidden_dim]
        self.noise_tokenizer = NoiseTokenizer(
            noise_dim=config.noise_dim,
            hidden_dim=config.hidden_dim,
            action_horizon=config.action_horizon,
        )
        
        # Embodiment embedding
        self.embodiment_embed = nn.Embedding(
            config.num_embodiments, config.hidden_dim
        )
        
        # CFG scale embedding
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        
        # Cross-attention fusion (noise attends to VLM features)
        fusion_config = FusionConfig(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_cross_attn_layers,
            dropout=config.dropout,
            use_rope=config.use_rope,
            qk_norm=config.use_qk_norm,
        )
        self.cross_attn_fusion = CrossAttentionFusion(fusion_config)
        
        # DiT transformer (self-attention with adaLN conditioning)
        dit_config = DiTConfig(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            use_rope=config.use_rope,
            use_qk_norm=config.use_qk_norm,
            use_swiglu=config.use_swiglu,
            use_flash_attn=config.use_flash_attn,
            conditioning_dim=config.hidden_dim,
        )
        self.transformer = DiTTransformer(dit_config)
        
        # Unified action head: [hidden_dim] → [128]
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        
        # Initialize
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DriftingVLA: {total_params:,} total params, "
            f"{trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)"
        )
    
    def _init_weights(self):
        """Initialize weights for trainable components."""
        # Initialize action head output to small values
        nn.init.trunc_normal_(self.action_head[-1].weight, std=0.01)
        nn.init.zeros_(self.action_head[-1].bias)
        
        # Initialize embeddings
        nn.init.normal_(self.embodiment_embed.weight, std=0.02)
    
    def forward(
        self,
        # VLM features (pre-computed or from live forward)
        vlm_features: Optional[torch.Tensor] = None,
        vlm_pooled: Optional[torch.Tensor] = None,
        # Or raw inputs for live VLM forward
        images: Optional[torch.Tensor] = None,
        language: Optional[List[str]] = None,
        # Action generation inputs
        noise: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        # Optional attention mask for variable-length VLM features
        vlm_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            vlm_features: [B, L, vlm_dim] pre-computed VLM hidden states.
            vlm_pooled: [B, vlm_dim] pre-computed pooled features.
            images: [B, V, 3, H, W] raw images (for live mode).
            language: List of B strings (for live mode).
            noise: [B, T, noise_dim] random noise. If None, sampled.
            embodiment_id: [B] embodiment type (0/1/2). Default: 0 (gripper).
            cfg_scale: [B] CFG scale. Default: 1.0.
            vlm_attn_mask: [B, L] attention mask for VLM features.
        
        Returns:
            actions: [B, T, 128] predicted actions in unified space.
        """
        # Determine batch size and device
        if vlm_features is not None:
            B = vlm_features.shape[0]
            device = vlm_features.device
        elif images is not None:
            B = images.shape[0]
            device = images.device
        else:
            raise ValueError("Either vlm_features or images must be provided")
        
        # --- Get VLM context ---
        if vlm_features is not None:
            # Pre-computed path (training)
            c_seq, c_pool = self.vlm_backbone.forward_precomputed(vlm_features, vlm_pooled)
        else:
            # Live path (inference)
            c_seq, c_pool = self.vlm_backbone.forward_live(images, language)
        
        # --- Sample noise ---
        if noise is None:
            noise = torch.randn(
                B, self.config.action_horizon, self.config.noise_dim,
                device=device,
            )
        
        # --- Default embodiment and CFG ---
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        if cfg_scale is None:
            cfg_scale = torch.ones(B, device=device)
        
        # --- Build global conditioning vector ---
        # c_global = c_pool + embodiment_emb + cfg_emb
        emb_embed = self.embodiment_embed(embodiment_id)  # [B, D]
        cfg_embed = self.cfg_embed(cfg_scale.unsqueeze(-1))  # [B, D]
        c_global = c_pool + emb_embed + cfg_embed  # [B, D]
        
        # --- Noise to tokens ---
        noise_tokens = self.noise_tokenizer(noise)  # [B, T, D]
        
        # --- Cross-attention: noise attends to VLM context ---
        # CrossAttentionFusion expects (noise_tokens, visual_features, language_features)
        # We pass c_seq as visual_features and empty as language (already fused in VLM)
        # Note: We don't pass vlm_attn_mask to fusion because the zero-padded
        # features already have near-zero values that contribute minimally to attention.
        empty_lang = torch.zeros(B, 0, self.config.hidden_dim, device=device)
        fused_tokens = self.cross_attn_fusion(
            noise_tokens, c_seq, empty_lang, None
        )  # [B, T, D]
        
        # --- DiT transformer with adaLN ---
        transformer_out = self.transformer(fused_tokens, c_global)  # [B, T, D]
        
        # --- Action head ---
        actions = self.action_head(transformer_out)  # [B, T, 128]
        
        return actions
    
    def forward_with_features(
        self,
        vlm_features: torch.Tensor,
        vlm_pooled: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward with intermediate features (for debugging/visualization).
        
        Returns dict with 'actions', 'c_seq', 'c_pool', 'c_global', etc.
        """
        B = vlm_features.shape[0]
        device = vlm_features.device
        
        c_seq, c_pool = self.vlm_backbone.forward_precomputed(vlm_features, vlm_pooled)
        
        if noise is None:
            noise = torch.randn(B, self.config.action_horizon, self.config.noise_dim, device=device)
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        if cfg_scale is None:
            cfg_scale = torch.ones(B, device=device)
        
        emb_embed = self.embodiment_embed(embodiment_id)
        cfg_embed = self.cfg_embed(cfg_scale.unsqueeze(-1))
        c_global = c_pool + emb_embed + cfg_embed
        
        noise_tokens = self.noise_tokenizer(noise)
        empty_lang = torch.zeros(B, 0, self.config.hidden_dim, device=device)
        fused_tokens = self.cross_attn_fusion(noise_tokens, c_seq, empty_lang, vlm_attn_mask)
        transformer_out = self.transformer(fused_tokens, c_global)
        actions = self.action_head(transformer_out)
        
        return {
            'actions': actions,
            'c_seq': c_seq,
            'c_pool': c_pool,
            'c_global': c_global,
            'noise_tokens': noise_tokens,
            'fused_tokens': fused_tokens,
            'transformer_out': transformer_out,
        }
    
    @torch.no_grad()
    def generate(
        self,
        vlm_features: Optional[torch.Tensor] = None,
        vlm_pooled: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        language: Optional[List[str]] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate actions (inference mode, 1-NFE).
        
        Args:
            vlm_features: [B, L, D] pre-computed features (optional).
            vlm_pooled: [B, D] pre-computed pooled (optional).
            images: [B, V, 3, H, W] raw images (optional).
            language: List of B strings (optional).
            embodiment_id: [B] embodiment type.
            cfg_scale: CFG guidance scale.
            action_mask: [128] bool mask for active dims.
        
        Returns:
            actions: [B, T, 128] generated actions.
        """
        if vlm_features is not None:
            B = vlm_features.shape[0]
            device = vlm_features.device
        else:
            B = images.shape[0]
            device = images.device
        
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        
        cfg_tensor = torch.full((B,), cfg_scale, device=device)
        
        actions = self.forward(
            vlm_features=vlm_features,
            vlm_pooled=vlm_pooled,
            images=images,
            language=language,
            embodiment_id=embodiment_id,
            cfg_scale=cfg_tensor,
        )
        
        # Apply action mask if provided
        if action_mask is not None:
            mask = action_mask.to(device).float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 128]
            actions = actions * mask
        
        return actions
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_drifting_vla(
    model_size: str = 'base',
    vlm_model_key: str = 'qwen3vl',
    action_horizon: int = 16,
    **kwargs,
) -> DriftingVLA:
    """
    Factory function for DriftingVLA.
    
    Args:
        model_size: 'small', 'base', or 'large'.
        vlm_model_key: 'qwen3vl' or 'paligemma2'.
        action_horizon: Number of action timesteps.
    
    Returns:
        Configured DriftingVLA model.
    """
    size_configs = {
        'small': dict(hidden_dim=512, num_layers=8, num_heads=8),
        'base': dict(hidden_dim=768, num_layers=12, num_heads=12),
        'large': dict(hidden_dim=1024, num_layers=24, num_heads=16),
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    cfg = size_configs[model_size]
    cfg.update(kwargs)
    
    config = DriftingVLAConfig(
        vlm_model_key=vlm_model_key,
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        action_horizon=action_horizon,
    )
    
    return DriftingVLA(config)


