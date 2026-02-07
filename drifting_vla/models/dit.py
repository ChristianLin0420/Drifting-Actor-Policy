"""
Diffusion Transformer (DiT) for Drifting-VLA
=============================================

This module implements DiT-style transformer blocks adapted for the
Drifting-VLA architecture. Key features include:

- adaLN-Zero conditioning (adaptive layer normalization)
- RMSNorm for efficiency
- SwiGLU activation
- RoPE (Rotary Position Embedding)
- QK-Norm for training stability

The DiT transforms noise tokens conditioned on visual and language
features to generate action sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass  
class DiTConfig:
    """
    Configuration for DiT Transformer.
    
    Attributes:
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: MLP hidden dimension multiplier.
        dropout: Dropout probability.
        use_rope: Whether to use Rotary Position Embeddings.
        use_qk_norm: Whether to use Query-Key normalization.
        use_swiglu: Whether to use SwiGLU activation.
        use_flash_attn: Whether to use Flash Attention.
        conditioning_dim: Dimension of conditioning signal (for adaLN).
    """
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_rope: bool = True
    use_qk_norm: bool = True
    use_swiglu: bool = True
    use_flash_attn: bool = True
    conditioning_dim: int = 1024


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm as it doesn't compute mean.
    
    Args:
        dim: Feature dimension.
        eps: Numerical stability constant.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    Combines Swish activation with Gated Linear Unit for better
    performance in transformer FFNs.
    
    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (before gating).
        out_features: Output dimension.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: silu(w1(x)) * w3(x) -> w2."""
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization.
    
    Modulates layer normalization based on conditioning signal,
    with outputs initialized to zero for stable training.
    
    Args:
        hidden_dim: Feature dimension.
        conditioning_dim: Conditioning signal dimension.
    """
    
    def __init__(self, hidden_dim: int, conditioning_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Linear layer to generate scale, shift, and gate
        # Output: [scale1, shift1, gate1, scale2, shift2, gate2]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, 6 * hidden_dim, bias=True),
        )
        
        # Initialize to zero for residual stability
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate modulation parameters from conditioning.
        
        Args:
            x: Input features [B, L, D].
            c: Conditioning signal [B, D_c].
        
        Returns:
            Tuple of (scale1, shift1, gate1, scale2, shift2, gate2).
        """
        # Generate modulation parameters
        modulation = self.adaLN_modulation(c)  # [B, 6*D]
        
        # Split into 6 parts
        scale1, shift1, gate1, scale2, shift2, gate2 = modulation.chunk(6, dim=-1)
        
        # Each parameter: [B, D] -> [B, 1, D] for broadcasting
        scale1 = scale1.unsqueeze(1)
        shift1 = shift1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)
        
        return scale1, shift1, gate1, scale2, shift2, gate2
    
    def modulate(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """Apply modulation: norm(x) * (1 + scale) + shift."""
        return self.norm(x) * (1 + scale) + shift


class DiTAttention(nn.Module):
    """
    Multi-head self-attention for DiT blocks.
    
    Features:
    - Optional RoPE for position encoding
    - Optional QK-Norm for stability
    - Optional Flash Attention for efficiency
    
    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout.
        use_rope: Whether to use RoPE.
        use_qk_norm: Whether to use QK normalization.
        use_flash_attn: Whether to use Flash Attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and self._check_flash_attn()
        
        # QKV projection
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # RoPE
        if use_rope:
            from drifting_vla.models.fusion import RotaryPositionEmbedding
            self.rope = RotaryPositionEmbedding(self.head_dim)
        else:
            self.rope = None
        
        # QK normalization
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None
        
        self.dropout = nn.Dropout(dropout)
    
    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            from flash_attn import flash_attn_func
            return True
        except ImportError:
            return False
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Self-attention forward pass.
        
        Args:
            x: Input features [B, L, D].
            attention_mask: Optional attention mask [B, L] or [B, L, L].
        
        Returns:
            Output features [B, L, D].
        """
        B, L, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # QK normalization
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # RoPE
        if self.rope is not None:
            q, k = self.rope(q, k)
        
        # Attention computation
        if self.use_flash_attn and attention_mask is None:
            # Use Flash Attention (requires specific tensor layout)
            from flash_attn import flash_attn_func
            q = q.transpose(1, 2)  # [B, L, H, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            out = out.reshape(B, L, self.hidden_dim)
        else:
            # Standard attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))
            
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        
        return self.out_proj(out)


class DiTBlock(nn.Module):
    """
    DiT Transformer Block with adaLN-Zero conditioning.
    
    Architecture:
    1. adaLN + Self-Attention + Gate
    2. adaLN + FFN (SwiGLU) + Gate
    
    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout probability.
        conditioning_dim: Conditioning signal dimension.
        use_rope: Whether to use RoPE.
        use_qk_norm: Whether to use QK normalization.
        use_swiglu: Whether to use SwiGLU activation.
        use_flash_attn: Whether to use Flash Attention.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        conditioning_dim: int = 1024,
        use_rope: bool = True,
        use_qk_norm: bool = True,
        use_swiglu: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        
        # adaLN-Zero for conditioning
        self.adaln = AdaLNZero(hidden_dim, conditioning_dim)
        
        # Self-attention
        self.attn = DiTAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            use_qk_norm=use_qk_norm,
            use_flash_attn=use_flash_attn,
        )
        
        # Feed-forward network
        mlp_hidden = int(hidden_dim * mlp_ratio)
        if use_swiglu:
            # SwiGLU uses 2/3 of the hidden dimension for the gated part
            mlp_hidden = int(mlp_hidden * 2 / 3)
            self.ffn = SwiGLU(hidden_dim, mlp_hidden, hidden_dim, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, hidden_dim),
                nn.Dropout(dropout),
            )
        
        # Layer norm for FFN (used with adaLN modulation)
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DiT block forward pass.
        
        Args:
            x: Input features [B, L, D].
            c: Conditioning signal [B, D_c].
            attention_mask: Optional attention mask.
        
        Returns:
            Output features [B, L, D].
        """
        # Get modulation parameters
        scale1, shift1, gate1, scale2, shift2, gate2 = self.adaln(x, c)
        
        # Self-attention with modulation
        x_norm = self.adaln.modulate(x, scale1, shift1)
        x = x + gate1 * self.attn(x_norm, attention_mask)
        
        # FFN with modulation
        x_norm = self.ffn_norm(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(x_norm)
        
        return x


class DiTTransformer(nn.Module):
    """
    Full DiT Transformer for Drifting-VLA.
    
    Processes noise tokens conditioned on fused visual-language features
    to generate action representations.
    
    Args:
        config: DiTConfig with transformer settings.
    
    Example:
        >>> config = DiTConfig(hidden_dim=1024, num_layers=24)
        >>> transformer = DiTTransformer(config)
        >>> 
        >>> # Inputs
        >>> x = torch.randn(4, 16, 1024)  # Noise tokens
        >>> c = torch.randn(4, 1024)  # Conditioning (pooled context)
        >>> 
        >>> # Forward
        >>> out = transformer(x, c)  # [4, 16, 1024]
    """
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                conditioning_dim=config.conditioning_dim,
                use_rope=config.use_rope,
                use_qk_norm=config.use_qk_norm,
                use_swiglu=config.use_swiglu,
                use_flash_attn=config.use_flash_attn,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize transformer weights."""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_init_module)
    
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DiT Transformer forward pass.
        
        Args:
            x: Input tokens [B, L, D].
            c: Conditioning signal [B, D_c].
            attention_mask: Optional attention mask.
        
        Returns:
            Output tokens [B, L, D].
        """
        for block in self.blocks:
            x = block(x, c, attention_mask)
        
        return self.final_norm(x)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


