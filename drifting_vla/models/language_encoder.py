"""
Language Encoder: CLIP Text Encoder
====================================

This module implements the language encoder using CLIP's text encoder,
which provides robust language understanding for task descriptions.

CLIP text encoder advantages:
- Pre-trained on large-scale image-text pairs
- Good at understanding object and action concepts
- Efficient (smaller than LLMs)
- Well-aligned with visual features
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class LanguageEncoderConfig:
    """
    Configuration for language encoder.
    
    Attributes:
        model_name: CLIP model variant or HuggingFace model name.
        hidden_dim: Output feature dimension.
        max_length: Maximum token sequence length.
        freeze: Whether to freeze encoder weights.
        pooling: How to pool token embeddings ('cls', 'mean', 'eos').
    """
    model_name: str = 'openai/clip-vit-large-patch14'
    hidden_dim: int = 1024
    max_length: int = 77
    freeze: bool = True
    pooling: str = 'eos'  # CLIP uses end-of-sequence token
    pretrained: bool = True


class CLIPLanguageEncoder(nn.Module):
    """
    CLIP Text Encoder for language understanding.
    
    Wraps the CLIP text encoder and provides a consistent interface
    for encoding task descriptions in Drifting-VLA.
    
    Args:
        config: LanguageEncoderConfig with model settings.
    
    Input:
        text: Either tokenized input_ids [B, L] or raw text strings.
        attention_mask: Optional attention mask [B, L].
    
    Output:
        embeddings: Language embeddings.
            Shape: [B, hidden_dim] for pooled output
            Shape: [B, L, hidden_dim] for all tokens
    
    Example:
        >>> config = LanguageEncoderConfig(freeze=True)
        >>> encoder = CLIPLanguageEncoder(config)
        >>> embeddings = encoder(["pick up the red cup", "open the drawer"])
    """
    
    def __init__(self, config: LanguageEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load model and tokenizer
        self.encoder, self.tokenizer = self._load_model()
        
        # Get encoder dimension
        self.encoder_dim = self._get_encoder_dim()
        
        # Projection layer if dimensions don't match
        if self.encoder_dim != config.hidden_dim:
            self.projection = nn.Linear(self.encoder_dim, config.hidden_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze if specified
        if config.freeze:
            self._freeze_encoder()
        
        logger.info(
            f"Initialized CLIP language encoder: {config.model_name}, "
            f"freeze={config.freeze}"
        )
    
    def _load_model(self):
        """Load CLIP text encoder and tokenizer."""
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            
            encoder = CLIPTextModel.from_pretrained(
                self.config.model_name,
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model_name,
            )
            logger.info(f"Loaded CLIP from {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Could not load CLIP model: {e}")
        
        return encoder, tokenizer
    
    def _get_encoder_dim(self) -> int:
        """Get the output dimension of the encoder."""
        # CLIP model dimensions
        if 'large' in self.config.model_name.lower():
            return 768
        elif 'huge' in self.config.model_name.lower():
            return 1024
        else:  # base
            return 512
    
    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        logger.info("Frozen CLIP language encoder parameters")
    
    def tokenize(
        self,
        text: Union[str, list[str]],
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize text input.
        
        Args:
            text: Single string or list of strings.
            device: Target device for tensors.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        if isinstance(text, str):
            text = [text]
        
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt',
        )
        
        if device is not None:
            tokens = {k: v.to(device) for k, v in tokens.items()}
        
        return tokens
    
    def forward(
        self,
        text: Union[str, list[str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Raw text strings or pre-tokenized input_ids [B, L].
            attention_mask: Attention mask [B, L] if text is pre-tokenized.
            return_all_tokens: Return all token embeddings instead of pooled.
        
        Returns:
            Language embeddings [B, D] or [B, L, D].
        """
        # Handle raw text input
        if isinstance(text, (str, list)):
            device = next(self.encoder.parameters()).device
            tokens = self.tokenize(text, device=device)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
        else:
            input_ids = text
        
        # Forward through encoder
        if self.config.freeze:
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Get embeddings
        hidden_states = outputs.last_hidden_state  # [B, L, D]
        
        if return_all_tokens:
            return self.projection(hidden_states)
        
        # Pool embeddings
        if self.config.pooling == 'cls':
            pooled = hidden_states[:, 0]
        elif self.config.pooling == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)
        else:  # 'eos' - CLIP default
            # Get the last non-padding token (EOS)
            if attention_mask is not None:
                # Find EOS position
                eos_idx = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), eos_idx]
            else:
                pooled = outputs.pooler_output
        
        return self.projection(pooled)
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.config.hidden_dim


class TaskEmbedding(nn.Module):
    """
    Learnable task embedding layer.
    
    Combines CLIP language embeddings with learnable task-specific
    embeddings for better task conditioning.
    
    Args:
        language_encoder: Pre-initialized language encoder.
        num_tasks: Number of discrete tasks (for learnable embeddings).
        hidden_dim: Embedding dimension.
        use_language: Whether to use language encoder (can be disabled).
    """
    
    def __init__(
        self,
        language_encoder: Optional[CLIPLanguageEncoder] = None,
        num_tasks: Optional[int] = None,
        hidden_dim: int = 1024,
        use_language: bool = True,
    ):
        super().__init__()
        self.use_language = use_language and language_encoder is not None
        self.language_encoder = language_encoder
        self.hidden_dim = hidden_dim
        
        # Learnable task embeddings
        if num_tasks is not None:
            self.task_embeddings = nn.Embedding(num_tasks, hidden_dim)
        else:
            self.task_embeddings = None
        
        # Combine language and task embeddings
        if self.use_language and self.task_embeddings is not None:
            self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.combine = None
    
    def forward(
        self,
        text: Optional[Union[str, list[str], torch.Tensor]] = None,
        task_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get task embeddings from text and/or task IDs.
        
        Args:
            text: Task descriptions (for language encoder).
            task_ids: Discrete task IDs [B] (for learnable embeddings).
        
        Returns:
            Task embeddings [B, hidden_dim].
        """
        embeddings = []
        
        # Language embeddings
        if self.use_language and text is not None:
            lang_emb = self.language_encoder(text)
            embeddings.append(lang_emb)
        
        # Learnable task embeddings
        if self.task_embeddings is not None and task_ids is not None:
            task_emb = self.task_embeddings(task_ids)
            embeddings.append(task_emb)
        
        if len(embeddings) == 0:
            raise ValueError("Must provide either text or task_ids")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Combine embeddings
        combined = torch.cat(embeddings, dim=-1)
        return self.combine(combined)


