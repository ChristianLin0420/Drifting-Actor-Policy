"""
Tests for Data Pipeline
=======================
"""

import pytest
import torch
import numpy as np
import tempfile
import h5py
from pathlib import Path

from drifting_vla.data.sample_queue import SampleQueue, GlobalSampleQueue
from drifting_vla.data.transforms import VLATransforms, TransformConfig, ActionNormalization


class TestSampleQueue:
    """Tests for sample queue."""
    
    @pytest.fixture
    def queue(self):
        """Create sample queue."""
        return SampleQueue(
            queue_size=100,
            action_dim=64,
            device=torch.device('cpu'),
        )
    
    def test_add_and_sample(self, queue):
        """Test adding and sampling from queue."""
        # Add samples
        for task_id in range(3):
            actions = torch.randn(10, 64)
            queue.add(task_id, actions)
        
        # Sample
        samples = queue.sample(20)
        
        assert samples.shape == (20, 64)
    
    def test_queue_size_limit(self, queue):
        """Test that queue respects size limit."""
        # Add more than queue size
        for _ in range(20):
            queue.add(0, torch.randn(10, 64))
        
        counts = queue.get_task_counts()
        
        assert counts[0] <= queue.queue_size
    
    def test_empty_queue(self, queue):
        """Test sampling from empty queue."""
        samples = queue.sample(10)
        
        assert samples.shape == (10, 64)
        assert (samples == 0).all()  # Should be zeros
    
    def test_per_task_sampling(self, queue):
        """Test sampling from specific tasks."""
        # Add to specific tasks
        queue.add(0, torch.randn(10, 64))
        queue.add(2, torch.randn(10, 64))
        
        # Sample from task 0 only
        samples = queue.sample(10, task_ids=[0])
        
        assert samples.shape == (10, 64)
    
    def test_clear(self, queue):
        """Test clearing the queue."""
        queue.add(0, torch.randn(10, 64))
        queue.clear()
        
        assert queue.total_samples() == 0


class TestGlobalSampleQueue:
    """Tests for global sample queue."""
    
    @pytest.fixture
    def queue(self):
        """Create global queue."""
        return GlobalSampleQueue(
            queue_size=100,
            action_dim=64,
        )
    
    def test_global_add_sample(self, queue):
        """Test global queue add and sample."""
        queue.add(torch.randn(10, 64))
        queue.add(torch.randn(10, 64))
        
        samples = queue.sample(20)
        
        assert samples.shape == (20, 64)


class TestVLATransforms:
    """Tests for VLA data transforms."""
    
    @pytest.fixture
    def config(self):
        """Create transform config."""
        return TransformConfig(
            image_size=224,
            random_crop=True,
            color_jitter=True,
        )
    
    @pytest.fixture
    def transform(self, config):
        """Create transforms."""
        return VLATransforms(config, training=True)
    
    def test_image_transform(self, transform):
        """Test image transformation."""
        sample = {
            'images': torch.randn(3, 256, 256),
            'actions': torch.randn(16, 10),
        }
        
        transformed = transform(sample)
        
        # Image should be resized to target size
        assert transformed['images'].shape[-1] == 224
    
    def test_training_vs_eval(self, config):
        """Test that training mode enables augmentation."""
        train_transform = VLATransforms(config, training=True)
        eval_transform = VLATransforms(config, training=False)
        
        sample = {
            'images': torch.randn(3, 224, 224),
            'actions': torch.randn(16, 10),
        }
        
        # Same input should produce same output in eval mode
        torch.manual_seed(42)
        out1 = eval_transform(sample.copy())
        torch.manual_seed(42)
        out2 = eval_transform(sample.copy())
        
        assert torch.allclose(out1['images'], out2['images'])


class TestActionNormalization:
    """Tests for action normalization."""
    
    def test_fit(self):
        """Test fitting normalization statistics."""
        actions = np.random.randn(100, 10) * 5 + 3
        
        normalizer = ActionNormalization()
        normalizer.fit(actions)
        
        assert normalizer.mean is not None
        assert normalizer.std is not None
        assert normalizer.mean.shape == (10,)
    
    def test_normalize_denormalize(self):
        """Test that denormalize(normalize(x)) ≈ x."""
        actions = torch.randn(50, 10) * 5 + 3
        
        normalizer = ActionNormalization()
        normalizer.fit(actions.numpy())
        
        normalized = normalizer.normalize(actions)
        recovered = normalizer.denormalize(normalized)
        
        assert torch.allclose(recovered, actions, atol=1e-5)
    
    def test_normalized_statistics(self):
        """Test that normalized data has ~zero mean, ~unit std."""
        actions = torch.randn(1000, 10) * 5 + 3
        
        normalizer = ActionNormalization()
        normalizer.fit(actions.numpy())
        
        normalized = normalizer.normalize(actions)
        
        assert normalized.mean().abs() < 0.1
        assert (normalized.std() - 1.0).abs() < 0.1
    
    def test_save_load(self):
        """Test saving and loading normalization."""
        actions = np.random.randn(100, 10) * 5 + 3
        
        normalizer = ActionNormalization()
        normalizer.fit(actions)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            normalizer.save(f.name)
            loaded = ActionNormalization.load(f.name)
        
        assert np.allclose(normalizer.mean, loaded.mean)
        assert np.allclose(normalizer.std, loaded.std)


