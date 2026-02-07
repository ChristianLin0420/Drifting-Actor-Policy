"""
Tests for Loss Functions
========================
"""

import pytest
import torch

from drifting_vla.training.losses import DriftingLoss, CombinedDriftingLoss


class TestDriftingLoss:
    """Tests for drifting loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create loss function."""
        return DriftingLoss(
            temperatures=[0.02, 0.05, 0.2],
            normalize_features=True,
            normalize_drift=True,
        )
    
    def test_forward(self, loss_fn):
        """Test forward pass."""
        B, D = 32, 64
        x = torch.randn(B, D, requires_grad=True)
        y_pos = torch.randn(B // 2, D)
        y_neg = torch.randn(B, D)
        
        output = loss_fn(x, y_pos, y_neg)
        
        assert output.loss.shape == ()
        assert output.drift_norm.shape == ()
    
    def test_gradient_flow(self, loss_fn):
        """Test that gradients flow through loss."""
        B, D = 32, 64
        x = torch.randn(B, D, requires_grad=True)
        y_pos = torch.randn(B // 2, D)
        y_neg = x.detach()
        
        output = loss_fn(x, y_pos, y_neg)
        output.loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_loss_reduction(self, loss_fn):
        """Test different reduction modes."""
        B, D = 32, 64
        x = torch.randn(B, D)
        y_pos = torch.randn(B // 2, D)
        y_neg = torch.randn(B, D)
        
        # Mean reduction
        loss_fn.reduction = 'mean'
        out_mean = loss_fn(x, y_pos, y_neg)
        
        # Sum reduction
        loss_fn.reduction = 'sum'
        out_sum = loss_fn(x, y_pos, y_neg)
        
        # Sum should be B times larger than mean
        assert out_sum.loss > out_mean.loss
    
    def test_loss_positive(self, loss_fn):
        """Test that loss is non-negative."""
        B, D = 32, 64
        x = torch.randn(B, D)
        y_pos = torch.randn(B // 2, D)
        y_neg = torch.randn(B, D)
        
        output = loss_fn(x, y_pos, y_neg)
        
        assert output.loss >= 0


class TestCombinedDriftingLoss:
    """Tests for combined loss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create combined loss without feature encoder."""
        return CombinedDriftingLoss(
            feature_encoder=None,
            action_weight=1.0,
            feature_weight=0.5,
        )
    
    def test_forward_without_feature_encoder(self, loss_fn):
        """Test forward without feature encoder."""
        B, T, D = 8, 16, 10
        actions = torch.randn(B, T, D, requires_grad=True)
        actions_pos = torch.randn(B // 2, T, D)
        actions_neg = torch.randn(B, T, D)
        
        output = loss_fn(actions, actions_pos, actions_neg)
        
        assert output.loss.shape == ()
        assert 'loss_action' in output.per_temp_losses
    
    def test_gradient_flow_combined(self, loss_fn):
        """Test gradient flow in combined loss."""
        B, T, D = 8, 16, 10
        actions = torch.randn(B, T, D, requires_grad=True)
        actions_pos = torch.randn(B // 2, T, D)
        actions_neg = actions.detach()
        
        output = loss_fn(actions, actions_pos, actions_neg)
        output.loss.backward()
        
        assert actions.grad is not None


