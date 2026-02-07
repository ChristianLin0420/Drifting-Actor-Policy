"""
Tests for Model Components
==========================
"""

import pytest
import torch
import numpy as np

from drifting_vla.models.dit import DiTBlock, DiTTransformer, DiTConfig
from drifting_vla.models.action_decoder import ActionDecoder, ActionDecoderConfig
from drifting_vla.models.fusion import CrossAttentionFusion, FusionConfig


class TestDiTBlock:
    """Tests for DiT transformer block."""
    
    @pytest.fixture
    def block(self):
        """Create a DiT block for testing."""
        return DiTBlock(
            hidden_dim=256,
            num_heads=8,
            mlp_ratio=4.0,
            conditioning_dim=256,
            use_rope=True,
            use_qk_norm=True,
            use_swiglu=True,
            use_flash_attn=False,  # Disable for testing
        )
    
    def test_forward_shape(self, block):
        """Test that output shape matches input."""
        B, L, D = 4, 16, 256
        x = torch.randn(B, L, D)
        c = torch.randn(B, 256)
        
        out = block(x, c)
        
        assert out.shape == x.shape
    
    def test_conditioning_effect(self, block):
        """Test that conditioning changes output."""
        B, L, D = 4, 16, 256
        x = torch.randn(B, L, D)
        c1 = torch.randn(B, 256)
        c2 = torch.randn(B, 256)
        
        out1 = block(x, c1)
        out2 = block(x, c2)
        
        assert not torch.allclose(out1, out2)


class TestDiTTransformer:
    """Tests for full DiT transformer."""
    
    @pytest.fixture
    def config(self):
        """Create transformer config."""
        return DiTConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            conditioning_dim=256,
            use_flash_attn=False,
        )
    
    @pytest.fixture
    def transformer(self, config):
        """Create transformer."""
        return DiTTransformer(config)
    
    def test_forward_shape(self, transformer):
        """Test output shape."""
        B, L, D = 4, 16, 256
        x = torch.randn(B, L, D)
        c = torch.randn(B, 256)
        
        out = transformer(x, c)
        
        assert out.shape == x.shape
    
    def test_num_params(self, transformer):
        """Test parameter counting."""
        num_params = transformer.get_num_params()
        
        assert num_params > 0
        assert isinstance(num_params, int)


class TestActionDecoder:
    """Tests for action decoder."""
    
    @pytest.fixture
    def config(self):
        """Create decoder config."""
        return ActionDecoderConfig(
            hidden_dim=256,
            action_horizon=16,
            position_dim=3,
            rotation_dim=6,
            gripper_dim=1,
        )
    
    @pytest.fixture
    def decoder(self, config):
        """Create decoder."""
        return ActionDecoder(config)
    
    def test_forward_shape(self, decoder, config):
        """Test output shape."""
        B, T, D = 4, 16, 256
        features = torch.randn(B, T, D)
        
        actions = decoder(features)
        
        expected_dim = config.position_dim + config.rotation_dim + config.gripper_dim
        assert actions.shape == (B, T, expected_dim)
    
    def test_return_dict(self, decoder):
        """Test dict output mode."""
        B, T, D = 4, 16, 256
        features = torch.randn(B, T, D)
        
        output = decoder(features, return_dict=True)
        
        assert 'position' in output
        assert 'rotation' in output
        assert 'gripper' in output
        assert 'actions' in output
    
    def test_rotation_normalization(self, decoder):
        """Test that rotation is normalized to valid 6D representation."""
        B, T, D = 4, 16, 256
        features = torch.randn(B, T, D)
        
        output = decoder(features, return_dict=True)
        rotation = output['rotation']
        
        # First 3 components should be unit vectors
        col1 = rotation[..., :3]
        col1_norm = col1.norm(dim=-1)
        assert torch.allclose(col1_norm, torch.ones_like(col1_norm), atol=1e-5)


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion."""
    
    @pytest.fixture
    def config(self):
        """Create fusion config."""
        return FusionConfig(
            hidden_dim=256,
            num_heads=8,
            num_layers=2,
        )
    
    @pytest.fixture
    def fusion(self, config):
        """Create fusion module."""
        return CrossAttentionFusion(config)
    
    def test_forward_shape(self, fusion):
        """Test output shape."""
        B = 4
        noise_tokens = torch.randn(B, 16, 256)
        visual_features = torch.randn(B, 50, 256)
        language_features = torch.randn(B, 20, 256)
        
        out = fusion(noise_tokens, visual_features, language_features)
        
        assert out.shape == noise_tokens.shape
    
    def test_context_effect(self, fusion):
        """Test that different context produces different output."""
        B = 4
        noise_tokens = torch.randn(B, 16, 256)
        visual1 = torch.randn(B, 50, 256)
        visual2 = torch.randn(B, 50, 256)
        language = torch.randn(B, 20, 256)
        
        out1 = fusion(noise_tokens, visual1, language)
        out2 = fusion(noise_tokens, visual2, language)
        
        assert not torch.allclose(out1, out2)


