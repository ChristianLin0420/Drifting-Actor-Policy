"""
Unified Multi-Dataset Loader
==============================

Wraps heterogeneous datasets (RLBench, DexGraspNet, Bridge, ALOHA, RT-1)
into a unified format with 128-dim action space and standardized images.

Supports:
  - Pre-computed VLM features from HDF5
  - Action mapping to 128-dim unified space
  - Per-dataset action normalization
  - Weighted sampling across datasets
"""

import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

from drifting_vla.data.action_mapping import (
    UNIFIED_ACTION_DIM,
    DATASET_EMBODIMENT,
    map_to_unified,
    get_action_mask,
)

logger = logging.getLogger(__name__)


class UnifiedSample:
    """A single unified sample from any dataset."""
    __slots__ = [
        'images', 'language', 'actions', 'action_mask',
        'embodiment_id', 'task_id', 'dataset_name', 'sample_id',
        'vlm_features', 'vlm_pooled',
    ]


class UnifiedDataset(Dataset):
    """
    Unified wrapper for multiple robotics datasets.
    
    Each sample is converted to:
        - images: [V, 3, 448, 448] multi-view images
        - language: str
        - actions: [T, 128] unified action vector
        - action_mask: [128] boolean mask
        - embodiment_id: int (0=gripper, 1=bimanual, 2=dex_hand)
        - vlm_features: [L, D] pre-computed VLM features (if available)
        - vlm_pooled: [D] pre-computed pooled features (if available)
    
    Args:
        datasets: Dict of dataset_name → Dataset instances.
        vlm_features_dir: Path to pre-computed VLM features HDF5 files.
        image_size: Target image size (448 for VLM).
        action_horizon: Number of action timesteps.
        normalize_actions: Whether to apply per-dataset normalization.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        vlm_features_dir: Optional[str] = None,
        image_size: int = 448,
        action_horizon: int = 16,
        normalize_actions: bool = True,
    ):
        self.datasets = datasets
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.normalize_actions = normalize_actions
        
        # Load VLM features if available
        self.vlm_h5_files = {}
        if vlm_features_dir:
            vlm_dir = Path(vlm_features_dir)
            for name in datasets:
                h5_path = vlm_dir / f'{name}_vlm_features.h5'
                if h5_path.exists():
                    self.vlm_h5_files[name] = h5py.File(str(h5_path), 'r')
                    logger.info(f"Loaded VLM features for {name}: {h5_path}")
        
        # Build global sample index: (dataset_name, local_idx)
        self.global_index = []
        self.dataset_offsets = {}
        
        offset = 0
        for name, ds in self.datasets.items():
            ds_len = len(ds)
            self.dataset_offsets[name] = offset
            for i in range(ds_len):
                self.global_index.append((name, i))
            offset += ds_len
        
        # Action normalization stats per dataset
        self.action_stats = {}
        if normalize_actions:
            self._compute_all_action_stats()
        
        logger.info(
            f"UnifiedDataset: {len(self.global_index)} total samples, "
            f"{len(self.datasets)} datasets: {list(self.datasets.keys())}"
        )
    
    def _compute_all_action_stats(self):
        """Compute action normalization statistics for each dataset."""
        for name, ds in self.datasets.items():
            if hasattr(ds, 'compute_action_statistics'):
                mean, std = ds.compute_action_statistics()
                
                # Map to unified space
                embodiment_id = DATASET_EMBODIMENT.get(name, 0)
                unified_mean = map_to_unified(mean, embodiment_id)
                unified_std = map_to_unified(std, embodiment_id)
                
                # Set std of inactive dims to 1 (avoid division by zero)
                mask_info = get_action_mask(embodiment_id)
                unified_std[~mask_info.mask] = 1.0
                unified_std = np.maximum(unified_std, 1e-4)
                
                self.action_stats[name] = {
                    'mean': unified_mean,
                    'std': unified_std,
                    'native_mean': mean,
                    'native_std': std,
                }
                logger.info(f"Action stats for {name}: mean_norm={np.linalg.norm(mean):.4f}")
            else:
                logger.warning(f"Dataset {name} doesn't support action stats, using identity")
                self.action_stats[name] = {
                    'mean': np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32),
                    'std': np.ones(UNIFIED_ACTION_DIM, dtype=np.float32),
                }
    
    def __len__(self) -> int:
        return len(self.global_index)
    
    def __getitem__(self, idx: int) -> dict:
        """Load and convert a sample to unified format."""
        dataset_name, local_idx = self.global_index[idx]
        ds = self.datasets[dataset_name]
        
        # Load raw sample from source dataset
        raw_sample = ds[local_idx]
        
        # Get embodiment info
        embodiment_id = DATASET_EMBODIMENT.get(dataset_name, 0)
        mask_info = get_action_mask(embodiment_id)
        
        # --- Convert actions to 128-dim unified space ---
        actions = raw_sample['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
        
        # Map to unified space
        unified_actions = map_to_unified(actions, embodiment_id)
        
        # Normalize
        if self.normalize_actions and dataset_name in self.action_stats:
            stats = self.action_stats[dataset_name]
            unified_actions = (unified_actions - stats['mean']) / stats['std']
        
        # --- Process images ---
        images = raw_sample['images']
        if isinstance(images, torch.Tensor):
            images_np = images.numpy()
        else:
            images_np = images
        
        # Ensure [V, 3, H, W] format
        if images_np.ndim == 3:
            images_np = images_np[np.newaxis]  # [1, 3, H, W]
        
        # Resize to target size if needed
        if images_np.shape[-1] != self.image_size or images_np.shape[-2] != self.image_size:
            import cv2
            V, C, H, W = images_np.shape
            resized = np.zeros((V, C, self.image_size, self.image_size), dtype=np.float32)
            for v in range(V):
                # CHW → HWC for cv2
                img_hwc = images_np[v].transpose(1, 2, 0)
                img_hwc = cv2.resize(img_hwc, (self.image_size, self.image_size))
                resized[v] = img_hwc.transpose(2, 0, 1)
            images_np = resized
        
        # --- Load VLM features if available ---
        vlm_features = None
        vlm_pooled = None
        
        if dataset_name in self.vlm_h5_files:
            h5 = self.vlm_h5_files[dataset_name]
            key = f'sample_{idx}'
            if key in h5:
                vlm_features = torch.from_numpy(h5[key]['hidden'][:]).float()
                vlm_pooled = torch.from_numpy(h5[key]['pooled'][:]).float()
        
        # --- Build output dict ---
        result = {
            'images': torch.from_numpy(images_np).float(),
            'language': raw_sample.get('language', ''),
            'actions': torch.from_numpy(unified_actions).float(),
            'action_mask': torch.from_numpy(mask_info.mask.astype(np.float32)),
            'embodiment_id': embodiment_id,
            'task_id': raw_sample.get('task_id', 0),
            'dataset_name': dataset_name,
            'sample_id': idx,
        }
        
        if vlm_features is not None:
            result['vlm_features'] = vlm_features
            result['vlm_pooled'] = vlm_pooled
        
        return result
    
    def get_action_stats(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get (mean, std) for a dataset in unified space."""
        stats = self.action_stats.get(dataset_name, None)
        if stats is None:
            return np.zeros(UNIFIED_ACTION_DIM), np.ones(UNIFIED_ACTION_DIM)
        return stats['mean'], stats['std']
    
    def get_all_action_stats(self) -> Dict[str, dict]:
        """Get all dataset action stats."""
        return self.action_stats
    
    def close(self):
        """Close HDF5 file handles."""
        for h5 in self.vlm_h5_files.values():
            h5.close()


def create_weighted_sampler(
    unified_dataset: UnifiedDataset,
    weights: Optional[Dict[str, float]] = None,
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for multi-dataset training.
    
    Args:
        unified_dataset: UnifiedDataset instance.
        weights: Per-dataset sampling weights (e.g., {'rlbench': 0.5, 'dexgraspnet': 0.5}).
                 If None, uniform weights.
    
    Returns:
        WeightedRandomSampler for DataLoader.
    """
    if weights is None:
        # Uniform per-sample weights (each dataset proportional to size)
        sample_weights = np.ones(len(unified_dataset), dtype=np.float64)
    else:
        # Assign weight based on source dataset
        sample_weights = np.zeros(len(unified_dataset), dtype=np.float64)
        
        for i, (ds_name, _) in enumerate(unified_dataset.global_index):
            ds_size = sum(1 for n, _ in unified_dataset.global_index if n == ds_name)
            w = weights.get(ds_name, 1.0)
            sample_weights[i] = w / ds_size  # Normalize by dataset size
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(unified_dataset),
        replacement=True,
    )


def collate_unified(batch: List[dict]) -> dict:
    """
    Custom collate function for UnifiedDataset.
    
    Handles variable-length VLM features by padding.
    """
    result = {}
    
    # Pad images to max views and stack
    max_views = max(s['images'].shape[0] for s in batch)
    C, H, W = batch[0]['images'].shape[1], batch[0]['images'].shape[2], batch[0]['images'].shape[3]
    padded_images = []
    for s in batch:
        img = s['images']  # [V, C, H, W]
        if img.shape[0] < max_views:
            pad = torch.zeros(max_views - img.shape[0], C, H, W)
            img = torch.cat([img, pad], dim=0)
        padded_images.append(img)
    result['images'] = torch.stack(padded_images)
    result['actions'] = torch.stack([s['actions'] for s in batch])
    result['action_mask'] = torch.stack([s['action_mask'] for s in batch])
    result['embodiment_id'] = torch.tensor([s['embodiment_id'] for s in batch], dtype=torch.long)
    result['task_id'] = torch.tensor([s['task_id'] for s in batch], dtype=torch.long)
    result['language'] = [s['language'] for s in batch]
    result['dataset_name'] = [s['dataset_name'] for s in batch]
    result['sample_id'] = torch.tensor([s['sample_id'] for s in batch], dtype=torch.long)
    
    # Pad VLM features if present in ANY sample
    has_vlm = any('vlm_features' in s and s.get('vlm_features') is not None for s in batch)
    if has_vlm:
        # Find max length and dim from available features
        available = [s for s in batch if 'vlm_features' in s and s['vlm_features'] is not None]
        max_len = max(s['vlm_features'].shape[0] for s in available)
        dim = available[0]['vlm_features'].shape[1]
        
        padded_features = torch.zeros(len(batch), max_len, dim)
        vlm_pooled_list = []
        vlm_attn_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        
        for i, s in enumerate(batch):
            if 'vlm_features' in s and s['vlm_features'] is not None:
                L = s['vlm_features'].shape[0]
                padded_features[i, :L] = s['vlm_features']
                vlm_attn_mask[i, :L] = True
                vlm_pooled_list.append(s['vlm_pooled'])
            else:
                # No VLM features — fill with zeros (will be handled in trainer)
                vlm_pooled_list.append(torch.zeros(dim))
        
        result['vlm_features'] = padded_features
        result['vlm_pooled'] = torch.stack(vlm_pooled_list)
        result['vlm_attn_mask'] = vlm_attn_mask
    
    return result
