"""
VLM Backbone: Vision-Encoder-Only Architecture (Pi0 Style)
============================================================

Extracts visual features from Qwen3-VL's ViT encoder and text embeddings
from the LLM's embedding layer — WITHOUT running the expensive 28-layer
autoregressive LLM decoder.

Architecture:
  images → Qwen3-VL ViT encoder (model.visual, ~407M) → [N_vis, 2048]
  text   → embed_tokens (lookup, ~311M)                → [L_text, 2048]
  [visual ⊕ text] → projector → [B, L, dit_dim]       → DiT cross-attention

This is how Pi0 uses PaliGemma: vision encoder + text embeddings only.
The full LLM decoder (28 layers, 1.4B params) is never run during training.

Training modes:
  - frozen:       ViT encoder frozen, fast (default pre-training)
  - encoder_lora: LoRA on ViT encoder only (~5M trainable)
  - precomputed:  No VLM forward at all (features loaded from HDF5)

Performance (batch=4, 3 images each, A40):
  - ViT encoder only: ~150ms  (this implementation)
  - Full LLM forward: ~300ms+ (old implementation, deleted)
  - Memory: ~6GB/GPU (vs 42GB+ with full LLM)
"""

import torch
import torch.nn as nn
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM backbone."""
    model_key: str = 'qwen3vl'           # 'qwen3vl' or 'paligemma2'
    dit_hidden_dim: int = 768             # Output projection dimension
    max_views: int = 8                    # Maximum camera views

    # ── Training mode ──
    freeze_base: bool = True              # Freeze VLM base weights
    use_lora: bool = False                # Apply LoRA to vision encoder
    lora_r: int = 16                      # LoRA rank
    lora_alpha: int = 32                  # LoRA scaling factor
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "qkv", "proj",                    # Qwen3-VL ViT uses fused qkv + proj
    ])


# VLM model specifications
VLM_SPECS = {
    'qwen3vl': {
        'hf_name': 'Qwen/Qwen3-VL-2B-Instruct',
        'hidden_dim': 2048,
        'image_size': 448,
        'vis_tokens_per_image': 196,      # 448/16=28, 28*28/4=196 (patch merge)
        'params_visual': '407M',
        'params_total': '2.1B',
    },
    'paligemma2': {
        'hf_name': 'google/paligemma2-3b-mix-448',
        'hidden_dim': 2048,
        'image_size': 448,
        'vis_tokens_per_image': 256,
        'params_visual': '400M',
        'params_total': '3.0B',
    },
}


class VLMBackbone(nn.Module):
    """
    Vision-Encoder-Only VLM backbone for Drifting-VLA.
    
    Key design (Pi0 style):
      - Run ONLY the ViT vision encoder (~407M) — NOT the 28-layer LLM decoder
      - Text features from embedding lookup (instant, <1ms)
      - Single batched forward for all images in the batch
      - LoRA applied to ViT only (not LLM), fits on A40 easily
    
    Args:
        config: VLMConfig with backbone settings.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        super().__init__()
        self.config = config or VLMConfig()
        
        spec = VLM_SPECS[self.config.model_key]
        self.vlm_hidden_dim = spec['hidden_dim']
        self.hf_name = spec['hf_name']
        
        # VLM will be loaded lazily or via load_vlm()
        self.vlm = None
        self._loaded = False
        
        # Visual dim projection (Qwen3-VL ViT outputs 1024, LLM embed_tokens is 2048)
        # This handles the dim mismatch when using vision-encoder-only architecture
        vis_encoder_dim = {'qwen3vl': 1024, 'paligemma2': 2048}.get(self.config.model_key, self.vlm_hidden_dim)
        if vis_encoder_dim != self.vlm_hidden_dim:
            self.vis_proj = nn.Linear(vis_encoder_dim, self.vlm_hidden_dim)
        else:
            self.vis_proj = None
        
        # Projection: vlm_hidden_dim → dit_hidden_dim
        self.proj_seq = nn.Sequential(
            nn.Linear(self.vlm_hidden_dim, self.config.dit_hidden_dim),
            nn.LayerNorm(self.config.dit_hidden_dim),
        )
        self.proj_pool = nn.Sequential(
            nn.Linear(self.vlm_hidden_dim, self.config.dit_hidden_dim),
            nn.LayerNorm(self.config.dit_hidden_dim),
        )
        
        logger.info(
            f"VLMBackbone initialized: {self.config.model_key} "
            f"(visual={spec['params_visual']}), vlm_dim={self.vlm_hidden_dim} → "
            f"dit_dim={self.config.dit_hidden_dim}, "
            f"freeze={self.config.freeze_base}, lora={self.config.use_lora}"
        )
    
    # ──────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────
    
    def load_vlm(self, device: torch.device = torch.device('cuda')):
        """Load the VLM model (only visual encoder + embed_tokens are used)."""
        if self._loaded:
            return
        
        if self.config.model_key == 'qwen3vl':
            self._load_qwen3vl(device)
        elif self.config.model_key == 'paligemma2':
            self._load_paligemma2(device)
        else:
            raise ValueError(f"Unknown model_key: {self.config.model_key}")
        
        # Freeze all weights first
        if self.config.freeze_base and self.vlm is not None:
            for param in self.vlm.parameters():
                param.requires_grad = False
            logger.info("Frozen VLM weights (visual encoder + text embeddings)")
        
        # Apply LoRA to vision encoder only (if configured)
        if self.config.use_lora and self.vlm is not None:
            self._apply_lora()
        
        self._loaded = True
    
    def _load_qwen3vl(self, device):
        """Load Qwen3-VL-2B-Instruct."""
        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                self.hf_name, dtype=torch.bfloat16,
            ).to(device)
            logger.info(f"Loaded Qwen3-VL-2B from {self.hf_name}")
        except ImportError as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            raise
    
    def _load_paligemma2(self, device):
        """Load PaliGemma2-3B."""
        try:
            from transformers import PaliGemmaForConditionalGeneration
            self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                self.hf_name, dtype=torch.bfloat16,
            ).to(device)
            logger.info(f"Loaded PaliGemma2-3B from {self.hf_name}")
        except ImportError as e:
            logger.error(f"Failed to load PaliGemma2: {e}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA to the vision encoder only (not the LLM decoder)."""
        try:
            from peft import LoraConfig, get_peft_model
            
            # Target only the vision encoder modules
            if self.config.model_key == 'qwen3vl':
                # Qwen3-VL vision encoder is at model.model.visual
                visual_module = self.vlm.model.visual
            else:
                # PaliGemma vision encoder
                visual_module = self.vlm.model.vision_tower
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
            )
            
            # Apply LoRA only to the visual encoder
            self.vlm.model.visual = get_peft_model(visual_module, lora_config)

            # Enable gradient checkpointing on visual encoder
            try:
                self.vlm.model.visual.enable_input_require_grads()
                self.vlm.model.visual.gradient_checkpointing_enable()
                logger.info("ViT gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable ViT gradient checkpointing: {e}")

            trainable = sum(p.numel() for p in self.vlm.model.visual.parameters() if p.requires_grad)
            total_vis = sum(p.numel() for p in self.vlm.model.visual.parameters())
            logger.info(
                f"LoRA applied to ViT encoder: {trainable:,} trainable / {total_vis:,} total "
                f"({trainable/total_vis*100:.2f}%) | r={self.config.lora_r}"
            )
            
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install: pip install peft")
    
    def _has_trainable_vlm_params(self) -> bool:
        """Check if VLM has any trainable params (LoRA or unfrozen)."""
        if self.vlm is None:
            return False
        return any(p.requires_grad for p in self.vlm.parameters())
        
    # ──────────────────────────────────────────────────────────────
    # Pre-computed features path (fastest, no VLM forward)
    # ──────────────────────────────────────────────────────────────
    
    def forward_precomputed(
        self,
        vlm_features: torch.Tensor,
        vlm_pooled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project pre-computed VLM features to DiT dimension.
        
        Args:
            vlm_features: [B, L, vlm_hidden_dim]
            vlm_pooled: [B, vlm_hidden_dim]
        
        Returns:
            c_seq: [B, L, dit_hidden_dim]
            c_pool: [B, dit_hidden_dim]
        """
        c_seq = self.proj_seq(vlm_features.float())
        c_pool = self.proj_pool(vlm_pooled.float())
        return c_seq, c_pool
    
    # ──────────────────────────────────────────────────────────────
    # Vision-Encoder-Only forward (Pi0 style) — MAIN PATH
    # ──────────────────────────────────────────────────────────────

    def forward_encoder_only(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vision-encoder-only forward: ViT + text embeddings, skip LLM decoder.

        This runs:
          1. ViT encoder (~407M params, ~150ms for batch=4×3 images)
          2. Text embedding lookup (<1ms)
          3. Concatenate + project to DiT dimension

        The 28-layer LLM decoder (~1.4B params) is NEVER run.
        
        Args:
            pixel_values:   [N_total_patches, C_patch] — all patches, all images
            image_grid_thw: [N_images, 3] — (temporal, height, width) per image
            input_ids:      [B, L_text] — tokenized text (padded)
            attention_mask: [B, L_text] — text attention mask (1=real, 0=pad)
        
        Returns:
            c_seq:  [B, L_total, dit_hidden_dim] — visual + text features
            c_pool: [B, dit_hidden_dim]          — pooled features
            seq_mask: [B, L_total] bool           — True=real, False=padding
        """
        if not self._loaded:
            self.load_vlm(pixel_values.device)

        device = pixel_values.device
        B = input_ids.shape[0]

        # Determine gradient context
        grad_ctx = nullcontext() if self._has_trainable_vlm_params() else torch.no_grad()

        with grad_ctx:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # ── Step 1: Vision encoder forward (~150ms for 12 images) ──
                if self.config.model_key == 'qwen3vl':
                    visual_tokens = self._forward_qwen3vl_visual(pixel_values, image_grid_thw)
                else:
                    visual_tokens = self._forward_paligemma_visual(pixel_values)

                # ── Step 2: Text embedding lookup (<1ms) ──
                if self.config.model_key == 'qwen3vl':
                    text_embeds = self.vlm.model.language_model.embed_tokens(input_ids)
                else:
                    text_embeds = self.vlm.model.embed_tokens(input_ids)

        # visual_tokens: [N_total_vis_tokens, hidden_dim] — flat across all images
        # text_embeds:   [B, L_text, hidden_dim]

        # ── Step 3: Split visual tokens per sample and concatenate with text ──
        vis_tokens_per_image = VLM_SPECS[self.config.model_key]['vis_tokens_per_image']
        n_images = image_grid_thw.shape[0]
        images_per_sample = max(n_images // B, 1)

        # Reshape visual tokens: [N_total, D] → [B, N_per_sample, D]
        expected_total = images_per_sample * vis_tokens_per_image * B
        actual_total = visual_tokens.shape[0]

        if actual_total == expected_total:
            # Uniform images per sample — fast reshape
            vis_per_sample = visual_tokens.reshape(B, images_per_sample * vis_tokens_per_image, -1)
        else:
            # Variable images per sample — split based on grid_thw
            vis_per_sample = self._split_visual_tokens(
                visual_tokens, image_grid_thw, B, device
            )

        # Concatenate: [visual_tokens | text_tokens]
        D = self.vlm_hidden_dim
        L_vis = vis_per_sample.shape[1]
        L_text = text_embeds.shape[1]
        L_total = L_vis + L_text

        combined = torch.cat([vis_per_sample.float(), text_embeds.float()], dim=1)  # [B, L_total, D]

        # Build attention mask: all visual tokens are real, text follows input mask
        vis_mask = torch.ones(B, L_vis, dtype=torch.bool, device=device)
        text_mask = attention_mask.bool()  # [B, L_text]
        seq_mask = torch.cat([vis_mask, text_mask], dim=1)  # [B, L_total]

        # ── Step 4: Masked pooling ──
        mask_f = seq_mask.unsqueeze(-1).float()  # [B, L_total, 1]
        pooled = (combined * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # [B, D]

        # ── Step 5: Project to DiT dimension ──
        c_seq = self.proj_seq(combined)      # [B, L_total, dit_dim]
        c_pool = self.proj_pool(pooled)      # [B, dit_dim]

        return c_seq, c_pool, seq_mask

    def _forward_qwen3vl_visual(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Run Qwen3-VL vision encoder only.

        Args:
            pixel_values: [N_total_patches, C_patch] (from processor)
            image_grid_thw: [N_images, 3]

        Returns:
            visual_tokens: [N_total_merged_tokens, 2048]
        """
        visual_out = self.vlm.model.visual(pixel_values, image_grid_thw)
        # visual_out can be:
        #   - tuple: (hidden_states, deepstack_list)
        #   - BaseModelOutputWithDeepstackFeatures: .last_hidden_state
        #   - plain tensor
        if isinstance(visual_out, tuple):
            hidden = visual_out[0]
        elif hasattr(visual_out, 'last_hidden_state'):
            hidden = visual_out.last_hidden_state
        elif hasattr(visual_out, 'hidden_states'):
            hidden = visual_out.hidden_states
        else:
            hidden = visual_out

        # Qwen3-VL ViT outputs 1024-dim but LLM embed_tokens is 2048-dim.
        # Use the registered vis_proj to match dimensions.
        if self.vis_proj is not None and hidden.shape[-1] != self.vlm_hidden_dim:
            hidden = self.vis_proj(hidden.to(self.vis_proj.weight.dtype))

        return hidden

    def _forward_paligemma_visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run PaliGemma2 vision tower only.

        Args:
            pixel_values: [B, C, H, W] or [N, C, H, W]

        Returns:
            visual_tokens: [N_total_tokens, 2048]
        """
        vision_out = self.vlm.model.vision_tower(pixel_values)
        hidden = vision_out.last_hidden_state  # [B_img, L_per_img, D]
        return hidden.reshape(-1, hidden.shape[-1])  # [N_total, D]

    def _split_visual_tokens(
        self,
        visual_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Split flat visual tokens [N_total, D] into [B, L_vis_max, D] with padding.

        Handles variable number of images per sample.
        """
        D = visual_tokens.shape[-1]

        # Compute tokens per image from grid_thw
        # Each image's merged tokens = (h/2) * (w/2) = h*w/4 for Qwen3-VL patch merger
        tokens_per_img = []
        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            # After patch merger: spatial dims are halved
            n_tokens = (h // 2) * (w // 2) if h > 1 else h * w
            tokens_per_img.append(max(n_tokens, 1))

        # Group images by sample
        n_images = len(tokens_per_img)
        imgs_per_sample = n_images // B

        per_sample_lens = []
        offset = 0
        for b in range(B):
            sample_len = sum(tokens_per_img[b * imgs_per_sample:(b + 1) * imgs_per_sample])
            per_sample_lens.append(sample_len)

        max_len = max(per_sample_lens)
        padded = torch.zeros(B, max_len, D, device=device, dtype=visual_tokens.dtype)

        offset = 0
        for b in range(B):
            L = per_sample_lens[b]
            padded[b, :L] = visual_tokens[offset:offset + L]
            offset += L

        return padded

    # ──────────────────────────────────────────────────────────────
    # Offline feature extraction (for pre-computing)
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract VLM features for offline pre-computation.

        Returns:
            hidden: [L, vlm_hidden_dim] concatenated visual + text features
            pooled: [vlm_hidden_dim] mean-pooled features
        """
        if not self._loaded:
            self.load_vlm(pixel_values.device)

        c_seq, c_pool, _ = self.forward_encoder_only(
            pixel_values, image_grid_thw, input_ids, attention_mask
        )
        # Return unprojected features for storage (raw vlm_hidden_dim)
        # Re-project at training time
        return c_seq.squeeze(0), c_pool.squeeze(0)
    
    def get_vlm_hidden_dim(self) -> int:
        """Return VLM's hidden dimension."""
        return self.vlm_hidden_dim
