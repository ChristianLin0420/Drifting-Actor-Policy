"""
Cross-Attention Fusion Module
=============================

This module implements cross-attention fusion for combining visual,
language, and noise embeddings in Drifting-VLA.

The fusion module allows the noise/action tokens to attend to visual
and language context, creating a conditioned representation for the
drifting transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class FusionConfig:
    """
    Configuration for cross-attention fusion.
    
    Attributes:
        hidden_dim: Feature dimension for all modalities.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Dropout probability.
        use_rope: Whether to use Rotary Position Embeddings.
        qk_norm: Whether to apply QK normalization.
    """
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 2
    dropout: float = 0.0
    use_rope: bool = True
    qk_norm: bool = True


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Applies rotary embeddings to queries and keys for position-aware
    attention without explicit positional encodings.
    
    Args:
        dim: Embedding dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor [B, H, L, D]
            k: Key tensor [B, H, L, D]
            offset: Position offset for caching.
        
        Returns:
            Rotated q and k tensors.
        """
        seq_len = q.shape[2]
        
        # Extend cache if needed
        if seq_len + offset > self.cos_cache.shape[0]:
            self._build_cache(seq_len + offset)
        
        cos = self.cos_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        
        q_rot = self._rotate(q, cos, sin)
        k_rot = self._rotate(k, cos, sin)
        
        return q_rot, k_rot
    
    def _rotate(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation to input tensor."""
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class QKNorm(nn.Module):
    """
    Query-Key Normalization for stable attention.
    
    Normalizes queries and keys before computing attention scores,
    which improves training stability especially for large models.
    """
    
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=eps)
        self.k_norm = nn.LayerNorm(head_dim, eps=eps)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize queries and keys.
        
        Args:
            q: Query tensor [B, H, L, D]
            k: Key tensor [B, H, L, D]
        
        Returns:
            Normalized q and k tensors.
        """
        return self.q_norm(q), self.k_norm(k)


class CrossAttention(nn.Module):
    """
    Cross-Attention layer for modality fusion.
    
    Allows query tokens (e.g., noise/action) to attend to context
    tokens (e.g., visual and language features).
    
    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
        use_rope: Whether to use RoPE.
        qk_norm: Whether to use QK normalization.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
        use_rope: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Optional components
        self.rope = RotaryPositionEmbedding(self.head_dim) if use_rope else None
        self.qk_norm = QKNorm(self.head_dim) if qk_norm else None
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query tokens [B, Lq, D] (noise/action tokens).
            context: Context tokens [B, Lc, D] (visual + language).
            attention_mask: Optional key-value mask [B, Lc] bool where
                True=real token, False=padding. Prevents query tokens
                from attending to padding positions in context.
        
        Returns:
            Output tokens [B, Lq, D].
        """
        B, Lq, _ = query.shape
        Lc = context.shape[1]
        
        # Project to Q, K, V
        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, Lc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, Lc, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)
        
        # Apply RoPE (if using)
        if self.rope is not None:
            q, _ = self.rope(q, q)  # Apply to query only for cross-attention
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask: [B, Lc] bool → [B, 1, 1, Lc] for broadcasting
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # KV mask: [B, Lc] → mask out padding positions in context
                kv_mask = ~attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Lc], True=mask
                attn = attn.masked_fill(kv_mask, float('-inf'))
            else:
                # Legacy: [B, Lq, Lc] explicit mask
                attn = attn.masked_fill(attention_mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.hidden_dim)
        
        return self.out_proj(out)


class FusionLayer(nn.Module):
    """
    Single fusion layer with cross-attention and FFN.
    
    Combines cross-attention to context with a feed-forward network,
    using pre-normalization (as in modern transformers).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        
        # Cross-attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            qk_norm=qk_norm,
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fusion layer forward pass.
        
        Args:
            query: Query tokens [B, Lq, D].
            context: Context tokens [B, Lc, D].
            attention_mask: Optional attention mask.
        
        Returns:
            Updated query tokens [B, Lq, D].
        """
        # Cross-attention with residual
        query = query + self.cross_attn(self.norm1(query), context, attention_mask)
        
        # FFN with residual
        query = query + self.ffn(self.norm2(query))
        
        return query


class CrossAttentionFusion(nn.Module):
    """
    Multi-layer cross-attention fusion module.
    
    Fuses noise/action tokens with visual and language context through
    multiple layers of cross-attention.
    
    Args:
        config: FusionConfig with module settings.
    
    Example:
        >>> config = FusionConfig(hidden_dim=1024, num_layers=2)
        >>> fusion = CrossAttentionFusion(config)
        >>> 
        >>> # Inputs
        >>> noise = torch.randn(4, 16, 1024)  # Action tokens
        >>> vision = torch.randn(4, 257, 1024)  # Visual features
        >>> language = torch.randn(4, 77, 1024)  # Language features
        >>> 
        >>> # Fuse
        >>> context = torch.cat([vision, language], dim=1)
        >>> fused = fusion(noise, context)
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # Input projection for noise (if needed)
        self.noise_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Fusion layers
        self.layers = nn.ModuleList([
            FusionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_rope=config.use_rope,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        noise_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse noise tokens with visual and language context.
        
        Args:
            noise_tokens: Noise/action tokens [B, T, D].
            visual_features: Visual features [B, Nv, D].
            language_features: Language features [B, Nl, D].
            attention_mask: Optional attention mask.
        
        Returns:
            Fused tokens [B, T, D].
        """
        # Project noise tokens
        x = self.noise_proj(noise_tokens)
        
        # Concatenate visual and language context
        context = torch.cat([visual_features, language_features], dim=1)
        
        # Apply fusion layers
        for layer in self.layers:
            x = layer(x, context, attention_mask)
        
        return self.norm(x)


