"""
VLM Backbone: Qwen3-VL-2B and PaliGemma2-3B
=============================================

Unified VLM backbone for extracting vision-language features.
Supports multiple pre-trained VLMs with a consistent interface.

Multi-view processing:
- Each camera view is processed through the vision encoder independently
- Visual tokens from all views are concatenated
- Language tokens are appended ONCE (not per view)
- The language model performs joint multi-view + language fusion
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM backbone."""
    model_key: str = 'qwen3vl'           # 'qwen3vl' or 'paligemma2'
    dit_hidden_dim: int = 768             # Output projection dimension
    max_views: int = 4                    # Maximum camera views
    max_vlm_tokens: int = 4200            # Max tokens (4 views × ~1025 + lang)
    freeze_base: bool = True              # Freeze VLM base weights
    use_lora: bool = False                # Enable LoRA fine-tuning
    lora_r: int = 16                      # LoRA rank
    lora_alpha: int = 32                  # LoRA scaling
    lora_target_layers: int = 4           # Number of last layers to apply LoRA


# VLM model specifications
VLM_SPECS = {
    'qwen3vl': {
        'hf_name': 'Qwen/Qwen3-VL-2B-Instruct',
        'hidden_dim': 2048,
        'image_size': 448,
        'multi_image_native': True,
        'params': '2.1B',
    },
    'paligemma2': {
        'hf_name': 'google/paligemma2-3b-mix-448',
        'hidden_dim': 2048,
        'image_size': 448,
        'multi_image_native': False,
        'params': '3.0B',
    },
}


class VLMBackbone(nn.Module):
    """
    Unified VLM backbone for Drifting-VLA.
    
    Supports Qwen3-VL-2B (default) and PaliGemma2-3B.
    
    Multi-view processing:
    - Vision encoder runs per-view independently
    - Visual features from all views are concatenated
    - Language is appended once after all visual tokens
    - Language model fuses multi-view + language jointly
    
    Args:
        config: VLMConfig with backbone settings.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        super().__init__()
        self.config = config or VLMConfig()
        
        spec = VLM_SPECS[self.config.model_key]
        self.vlm_hidden_dim = spec['hidden_dim']
        self.hf_name = spec['hf_name']
        self.multi_image_native = spec['multi_image_native']
        
        # VLM will be loaded lazily or via load_vlm()
        self.vlm = None
        self.processor = None
        self._loaded = False
        
        # Projection layers (always trainable)
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
            f"({spec['params']}), vlm_dim={self.vlm_hidden_dim} → dit_dim={self.config.dit_hidden_dim}"
        )
    
    def load_vlm(self, device: torch.device = torch.device('cuda')):
        """
        Load the VLM model and processor.
        
        Called lazily — VLM is NOT loaded during __init__ to save memory
        when using pre-computed features.
        """
        if self._loaded:
            return
        
        if self.config.model_key == 'qwen3vl':
            self._load_qwen3vl(device)
        elif self.config.model_key == 'paligemma2':
            self._load_paligemma2(device)
        else:
            raise ValueError(f"Unknown model_key: {self.config.model_key}")
        
        # Freeze base weights
        if self.config.freeze_base and self.vlm is not None:
            for param in self.vlm.parameters():
                param.requires_grad = False
            logger.info(f"Frozen VLM base weights")
        
        # Apply LoRA if configured
        if self.config.use_lora and self.vlm is not None:
            self._apply_lora()
        
        self._loaded = True
    
    def _load_qwen3vl(self, device):
        """Load Qwen3-VL-2B-Instruct."""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.hf_name)
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                self.hf_name,
                dtype=torch.bfloat16,
            ).to(device)
            self.vlm.eval()
            
            logger.info(f"Loaded Qwen3-VL-2B from {self.hf_name}")
            
        except ImportError as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            logger.error("Install: pip install 'transformers>=4.57' qwen-vl-utils")
            raise
    
    def _load_paligemma2(self, device):
        """Load PaliGemma2-3B."""
        try:
            from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(self.hf_name)
            self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
                self.hf_name,
                dtype=torch.bfloat16,
            ).to(device)
            self.vlm.eval()
            
            logger.info(f"Loaded PaliGemma2-3B from {self.hf_name}")
            
        except ImportError as e:
            logger.error(f"Failed to load PaliGemma2: {e}")
            raise
    
    def _apply_lora(self):
        """Apply LoRA adapters to VLM."""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            
            self.vlm = get_peft_model(self.vlm, lora_config)
            trainable = sum(p.numel() for p in self.vlm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.vlm.parameters())
            logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({trainable/total*100:.3f}%)")
            
        except ImportError:
            logger.warning("peft not installed, skipping LoRA. Install: pip install peft")
    
    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        language: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract VLM features from multi-view images + language.
        
        This is used for OFFLINE pre-computation (not during training).
        
        Args:
            images: [V, 3, H, W] multi-view images, float32 [0,1]
            language: Task description string
        
        Returns:
            hidden: [L, vlm_hidden_dim] last hidden states
            pooled: [vlm_hidden_dim] mean-pooled features
        """
        if not self._loaded:
            self.load_vlm(images.device)
        
        if self.config.model_key == 'qwen3vl':
            return self._extract_qwen3vl(images, language)
        else:
            return self._extract_paligemma2(images, language)
    
    def _extract_qwen3vl(self, images, language):
        """
        Extract features from Qwen3-VL (native multi-image).
        
        Uses the Qwen3-VL chat template to jointly encode images + language,
        then extracts the last hidden states as features.
        """
        V = images.shape[0]
        device = images.device
        
        # Build message content: multiple images + text
        content = []
        pil_images = []
        for v in range(V):
            img_pil = self._tensor_to_pil(images[v])
            pil_images.append(img_pil)
            content.append({"type": "image", "image": img_pil})
        content.append({"type": "text", "text": language or "describe the scene"})
        
        messages = [{"role": "user", "content": content}]
        
        # Use processor's apply_chat_template (Qwen3-VL style)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        
        # Forward pass to get hidden states
        outputs = self.vlm(
            **inputs,
            output_hidden_states=True,
        )
        
        hidden = outputs.hidden_states[-1].squeeze(0)  # [L, hidden_dim]
        pooled = hidden.mean(dim=0)                      # [hidden_dim]
        
        return hidden, pooled
    
    def _extract_paligemma2(self, images, language):
        """
        Extract features from PaliGemma2.
        
        Vision per view independently, language appended once.
        """
        V = images.shape[0]
        device = images.device
        
        # Step 1: Vision encoder per view
        visual_tokens_list = []
        for v in range(V):
            img_pil = self._tensor_to_pil(images[v])
            inputs = self.processor(
                text="",  # Empty text for vision-only
                images=img_pil,
                return_tensors="pt",
            ).to(device)
            
            # Extract vision features only
            vision_out = self.vlm.model.vision_tower(inputs['pixel_values'])
            visual_tokens_list.append(vision_out.last_hidden_state.squeeze(0))
        
        # Step 2: Concatenate visual tokens from all views
        visual_tokens = torch.cat(visual_tokens_list, dim=0)  # [V*N_vis, 2048]
        
        # Step 3: Append language tokens
        lang_inputs = self.processor.tokenizer(
            language, return_tensors="pt", padding=True
        ).to(device)
        lang_embeds = self.vlm.model.embed_tokens(lang_inputs['input_ids']).squeeze(0)
        
        combined = torch.cat([visual_tokens, lang_embeds], dim=0)
        
        # Step 4: Through language model for joint fusion
        outputs = self.vlm.model(
            inputs_embeds=combined.unsqueeze(0),
            output_hidden_states=True,
        )
        
        hidden = outputs.hidden_states[-1].squeeze(0)
        pooled = hidden.mean(dim=0)
        
        return hidden, pooled
    
    def forward_precomputed(
        self,
        vlm_features: torch.Tensor,
        vlm_pooled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project pre-computed VLM features to DiT dimension.
        
        Used DURING TRAINING with pre-computed features from disk.
        
        Args:
            vlm_features: [B, L, vlm_hidden_dim] pre-computed hidden states
            vlm_pooled: [B, vlm_hidden_dim] pre-computed pooled features
        
        Returns:
            c_seq: [B, L, dit_hidden_dim] projected sequence features
            c_pool: [B, dit_hidden_dim] projected pooled features
        """
        # Ensure float32 for projection layers (VLM may output bf16)
        vlm_features = vlm_features.float()
        vlm_pooled = vlm_pooled.float()
        c_seq = self.proj_seq(vlm_features)
        c_pool = self.proj_pool(vlm_pooled)
        return c_seq, c_pool
    
    def forward_live(
        self,
        images: torch.Tensor,
        language: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and project VLM features live (for inference).
        
        Args:
            images: [B, V, 3, H, W] multi-view images
            language: List of B task descriptions
        
        Returns:
            c_seq: [B, L, dit_hidden_dim]
            c_pool: [B, dit_hidden_dim]
        """
        if not self._loaded:
            self.load_vlm(images.device)
        
        B = images.shape[0]
        all_hidden = []
        all_pooled = []
        
        for b in range(B):
            hidden, pooled = self.extract_features(images[b], language[b])
            all_hidden.append(hidden)
            all_pooled.append(pooled)
        
        # Pad to max length and stack
        max_len = max(h.shape[0] for h in all_hidden)
        padded_hidden = []
        for h in all_hidden:
            if h.shape[0] < max_len:
                pad = torch.zeros(max_len - h.shape[0], h.shape[1], device=h.device, dtype=h.dtype)
                h = torch.cat([h, pad], dim=0)
            padded_hidden.append(h)
        
        vlm_features = torch.stack(padded_hidden)  # [B, L_max, D]
        vlm_pooled = torch.stack(all_pooled)        # [B, D]
        
        return self.forward_precomputed(vlm_features, vlm_pooled)
    
    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor):
        """Convert [3, H, W] float tensor to PIL Image."""
        from PIL import Image
        import numpy as np
        
        img = tensor.cpu().float()
        if img.max() <= 1.0:
            img = img * 255
        img = img.byte().permute(1, 2, 0).numpy()
        return Image.fromarray(img)
    
    def get_vlm_hidden_dim(self) -> int:
        """Return VLM's hidden dimension."""
        return self.vlm_hidden_dim

