"""
Tests for Drifting Field Computation
====================================
"""

import pytest
import torch
import numpy as np

from drifting_vla.training.drifting_field import (
    compute_drifting_field,
    compute_feature_normalization_scale,
    normalize_drift_field,
    compute_kernel_similarities,
)


class TestDriftingField:
    """Tests for drifting field computation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        N, D = 32, 64
        x = torch.randn(N, D)
        y_pos = torch.randn(N // 2, D)
        y_neg = torch.randn(N, D)
        return x, y_pos, y_neg
    
    def test_drifting_field_shape(self, sample_data):
        """Test that output shape matches input."""
        x, y_pos, y_neg = sample_data
        
        V = compute_drifting_field(x, y_pos, y_neg)
        
        assert V.shape == x.shape
    
    def test_drifting_field_dtype(self, sample_data):
        """Test that output dtype matches input."""
        x, y_pos, y_neg = sample_data
        
        V = compute_drifting_field(x, y_pos, y_neg)
        
        assert V.dtype == x.dtype
    
    def test_drifting_field_device(self, sample_data):
        """Test that output device matches input."""
        x, y_pos, y_neg = sample_data
        
        V = compute_drifting_field(x, y_pos, y_neg)
        
        assert V.device == x.device
    
    def test_drifting_field_gpu(self, sample_data):
        """Test on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        x, y_pos, y_neg = sample_data
        x = x.cuda()
        y_pos = y_pos.cuda()
        y_neg = y_neg.cuda()
        
        V = compute_drifting_field(x, y_pos, y_neg)
        
        assert V.device.type == 'cuda'
    
    def test_drifting_field_temperatures(self, sample_data):
        """Test with different temperature values."""
        x, y_pos, y_neg = sample_data
        
        for temps in [[0.1], [0.02, 0.1, 0.5], [0.01, 0.02, 0.05, 0.1, 0.2]]:
            V = compute_drifting_field(x, y_pos, y_neg, temperatures=temps)
            assert V.shape == x.shape
    
    def test_drifting_field_without_normalization(self, sample_data):
        """Test without feature/drift normalization."""
        x, y_pos, y_neg = sample_data
        
        V = compute_drifting_field(
            x, y_pos, y_neg,
            normalize_features=False,
            normalize_drift=False,
        )
        
        assert V.shape == x.shape
    
    def test_equilibrium_at_data(self, sample_data):
        """Test that drift is small when x matches y_pos."""
        x, y_pos, y_neg = sample_data
        
        # Use y_pos as x (at equilibrium)
        V = compute_drifting_field(y_pos, y_pos, y_neg)
        
        # Drift should be relatively small
        drift_norm = V.norm(dim=1).mean()
        assert drift_norm < 10.0  # Reasonable threshold


class TestFeatureNormalization:
    """Tests for feature normalization."""
    
    def test_scale_computation(self):
        """Test that scale normalizes distances."""
        torch.manual_seed(42)
        D = 64
        x = torch.randn(32, D) * 10  # Large scale
        y_pos = torch.randn(16, D) * 10
        y_neg = torch.randn(32, D) * 10
        
        scale = compute_feature_normalization_scale(x, y_pos, y_neg, D)
        
        assert scale > 0
        assert isinstance(scale, torch.Tensor)
    
    def test_scale_no_gradient(self):
        """Test that scale has no gradient."""
        torch.manual_seed(42)
        D = 64
        x = torch.randn(32, D, requires_grad=True)
        y_pos = torch.randn(16, D)
        y_neg = torch.randn(32, D)
        
        scale = compute_feature_normalization_scale(x, y_pos, y_neg, D)
        
        assert not scale.requires_grad


class TestDriftNormalization:
    """Tests for drift normalization."""
    
    def test_normalized_drift_scale(self):
        """Test that normalized drift has expected scale."""
        torch.manual_seed(42)
        D = 64
        V = torch.randn(32, D) * 5  # Variable scale
        
        V_norm = normalize_drift_field(V, D)
        
        # E[||V||^2 / D] should be close to 1
        norm_squared = (V_norm ** 2).sum(dim=1).mean() / D
        assert 0.5 < norm_squared < 2.0  # Allow some variance


class TestKernelSimilarities:
    """Tests for kernel computation."""
    
    def test_kernel_shape(self):
        """Test kernel output shape."""
        torch.manual_seed(42)
        x = torch.randn(10, 32)
        y = torch.randn(20, 32)
        
        K = compute_kernel_similarities(x, y, tau=0.1)
        
        assert K.shape == (10, 20)
    
    def test_kernel_positive(self):
        """Test that kernel values are positive."""
        torch.manual_seed(42)
        x = torch.randn(10, 32)
        y = torch.randn(20, 32)
        
        K = compute_kernel_similarities(x, y, tau=0.1)
        
        assert (K >= 0).all()
    
    def test_kernel_self_similarity(self):
        """Test that self-similarity is 1."""
        torch.manual_seed(42)
        x = torch.randn(10, 32)
        
        K = compute_kernel_similarities(x, x, tau=0.1)
        
        # Diagonal should be 1 (distance 0)
        diagonal = torch.diag(K)
        assert torch.allclose(diagonal, torch.ones_like(diagonal))
    
    def test_kernel_temperature_effect(self):
        """Test that lower temperature makes kernel sharper."""
        torch.manual_seed(42)
        x = torch.randn(10, 32)
        y = torch.randn(20, 32)
        
        K_low = compute_kernel_similarities(x, y, tau=0.01)
        K_high = compute_kernel_similarities(x, y, tau=1.0)
        
        # Low temp should have more variance (sharper)
        assert K_low.std() > K_high.std()


