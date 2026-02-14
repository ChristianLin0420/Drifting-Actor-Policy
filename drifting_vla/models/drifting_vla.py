"""
Drifting-VLA: VLM-Conditioned Drifting DiT
==============================================

Main model assembly:
  System 1: VLM Backbone (Qwen3-VL-2B) — frozen or LoRA fine-tuned
  System 2: Drifting DiT — cross-attention + self-attention action generator

Features (aligned with RDT-1B):
  - Proprioception input z_t (robot state conditioning)
  - Multi-frame history (t-1, t, t+1) via VLM multi-image
  - Camera + timestep positional embeddings on VLM features
  - Embodiment embedding for multi-embodiment conditioning
  - 128-dim unified action space with per-embodiment masking
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
)

logger = logging.getLogger(__name__)


@dataclass
class DriftingVLAConfig:
    """Configuration for Drifting-VLA."""
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

    # Multi-embodiment (0=abs_eef, 1=joints, 2=bimanual, 3=dex_hand, 4=delta_eef, 5=bimanual_mobile)
    num_embodiments: int = 6

    # Proprioception (RDT-1B style)
    use_proprio: bool = True
    proprio_dim: int = UNIFIED_ACTION_DIM  # 128-dim unified proprio

    # Multi-frame history (RDT-1B style)
    num_history_frames: int = 3  # t-1, t, t+1

    # Camera + timestep embeddings (RDT-1B style)
    max_cameras: int = 8
    max_history_steps: int = 4   # supports up to 4 history frames

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

    Architecture (aligned with RDT-1B):
        1. VLM backbone extracts c_seq, c_pool from multi-frame multi-view images + language
        2. Camera+timestep embeddings added to VLM tokens
        3. Proprioception z_t embedded and added to global conditioning
        4. Noise tokenizer creates action tokens from random noise
        5. Cross-attention fuses noise tokens with VLM context
        6. DiT transformer refines with adaLN conditioning (c_global)
        7. Action head outputs [B, T, 128] unified actions
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
        self.embodiment_embed = nn.Embedding(config.num_embodiments, config.hidden_dim)

        # CFG scale embedding
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # ── NEW: Proprioception embedding (RDT-1B z_t) ──
        if config.use_proprio:
            self.proprio_embed = nn.Sequential(
                nn.Linear(config.proprio_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
        else:
            self.proprio_embed = None

        # ── NEW: Camera ID embedding (which camera view) ──
        self.camera_embed = nn.Embedding(config.max_cameras, config.hidden_dim)

        # ── NEW: Timestep embedding (which frame in history) ──
        self.history_time_embed = nn.Embedding(config.max_history_steps, config.hidden_dim)

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
        """Initialize weights."""
        nn.init.trunc_normal_(self.action_head[-1].weight, std=0.01)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.normal_(self.embodiment_embed.weight, std=0.02)
        nn.init.normal_(self.camera_embed.weight, std=0.02)
        nn.init.normal_(self.history_time_embed.weight, std=0.02)

    def _add_camera_time_embeddings(
        self,
        c_seq: torch.Tensor,
        num_views: int = 1,
        num_frames: int = 1,
    ) -> torch.Tensor:
        """
        Add camera ID + timestep positional embeddings to VLM feature tokens.

        When VLM processes multi-frame multi-view inputs, the output c_seq contains
        tokens from all (frame, camera) combinations. We add learned embeddings so
        the DiT can distinguish which tokens came from which camera and timestep.

        Args:
            c_seq: [B, L, D] projected VLM features
            num_views: Number of camera views per frame
            num_frames: Number of history frames (1 or 3 typically)
        Returns:
            c_seq with camera+time embeddings added
        """
        B, L, D = c_seq.shape
        device = c_seq.device

        if num_views <= 1 and num_frames <= 1:
            # Single view, single frame — add default camera=0, time=0
            c_seq = c_seq + self.camera_embed(torch.zeros(1, dtype=torch.long, device=device))
            c_seq = c_seq + self.history_time_embed(torch.zeros(1, dtype=torch.long, device=device))
            return c_seq

        # Estimate tokens per view-frame group
        total_groups = num_views * num_frames
        tokens_per_group = L // max(total_groups, 1)

        if tokens_per_group == 0:
            return c_seq

        for group_idx in range(min(total_groups, L // max(tokens_per_group, 1))):
            frame_idx = group_idx // num_views
            cam_idx = group_idx % num_views

            start = group_idx * tokens_per_group
            end = start + tokens_per_group
            if end > L:
                end = L

            cam_id = torch.tensor(min(cam_idx, self.config.max_cameras - 1), device=device)
            time_id = torch.tensor(min(frame_idx, self.config.max_history_steps - 1), device=device)

            c_seq[:, start:end, :] = c_seq[:, start:end, :] + self.camera_embed(cam_id)
            c_seq[:, start:end, :] = c_seq[:, start:end, :] + self.history_time_embed(time_id)

        return c_seq

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
        # NEW: Proprioception (robot state)
        proprio: Optional[torch.Tensor] = None,
        # Metadata for camera+time embeddings
        num_views: int = 1,
        num_frames: int = 1,
        # Optional attention mask
        vlm_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            vlm_features: [B, L, vlm_dim] pre-computed VLM hidden states.
            vlm_pooled: [B, vlm_dim] pre-computed pooled features.
            images: [B, V, 3, H, W] or [B, T_hist*V, 3, H, W] raw images.
            language: List of B strings.
            noise: [B, T, noise_dim] random noise.
            embodiment_id: [B] embodiment type (0/1/2).
            cfg_scale: [B] CFG scale.
            proprio: [B, 128] proprioception in unified space (robot state).
            num_views: Number of camera views per frame.
            num_frames: Number of history frames.
            vlm_attn_mask: [B, L] attention mask.

        Returns:
            actions: [B, T, 128] predicted actions.
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
            c_seq, c_pool = self.vlm_backbone.forward_precomputed(vlm_features, vlm_pooled)
        else:
            c_seq, c_pool = self.vlm_backbone.forward_live(images, language)

        # --- NEW: Add camera + timestep positional embeddings ---
        c_seq = self._add_camera_time_embeddings(c_seq, num_views, num_frames)

        # --- Sample noise ---
        if noise is None:
            noise = torch.randn(B, self.config.action_horizon, self.config.noise_dim, device=device)

        # --- Default embodiment and CFG ---
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)
        if cfg_scale is None:
            cfg_scale = torch.ones(B, device=device)

        # --- Build global conditioning vector ---
        emb_embed = self.embodiment_embed(embodiment_id)
        cfg_emb = self.cfg_embed(cfg_scale.unsqueeze(-1))
        c_global = c_pool + emb_embed + cfg_emb

        # --- NEW: Add proprioception to global conditioning ---
        if proprio is not None and self.proprio_embed is not None:
            proprio_emb = self.proprio_embed(proprio.float())
            c_global = c_global + proprio_emb

        # --- Noise to tokens ---
        noise_tokens = self.noise_tokenizer(noise)

        # --- Cross-attention: noise attends to VLM context ---
        empty_lang = torch.zeros(B, 0, self.config.hidden_dim, device=device)
        fused_tokens = self.cross_attn_fusion(noise_tokens, c_seq, empty_lang, None)

        # --- DiT transformer with adaLN ---
        transformer_out = self.transformer(fused_tokens, c_global)

        # --- Action head ---
        actions = self.action_head(transformer_out)

        return actions

    @torch.no_grad()
    def generate(
        self,
        vlm_features: Optional[torch.Tensor] = None,
        vlm_pooled: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        language: Optional[List[str]] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        cfg_scale: float = 2.0,
        proprio: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate actions (inference mode, 1-NFE)."""
        if vlm_features is not None:
            B, device = vlm_features.shape[0], vlm_features.device
        else:
            B, device = images.shape[0], images.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        actions = self.forward(
            vlm_features=vlm_features, vlm_pooled=vlm_pooled,
            images=images, language=language,
            embodiment_id=embodiment_id,
            cfg_scale=torch.full((B,), cfg_scale, device=device),
            proprio=proprio,
        )

        if action_mask is not None:
            mask = action_mask.to(device).float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).unsqueeze(0)
            actions = actions * mask

        return actions

    def get_num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_drifting_vla(
    model_size: str = 'base',
    vlm_model_key: str = 'qwen3vl',
    action_horizon: int = 16,
    **kwargs,
) -> DriftingVLA:
    """Factory function for DriftingVLA."""
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
