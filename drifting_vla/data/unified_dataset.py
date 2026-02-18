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
    
    Supports two types of underlying datasets:
      1. Legacy datasets (RLBenchDataset, LeRobotDataset, etc.) — actions in native space,
         need map_to_unified() conversion.
      2. EpisodeHDF5Dataset — actions already pre-mapped to 128-dim unified space,
         skip mapping, use directly.
    
    Each sample is converted to:
        - images: [V, 3, 448, 448] multi-view images
        - language: str
        - actions: [T, 128] unified action vector
        - action_mask: [128] boolean mask
        - embodiment_id: int (0-5)
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
        
        # Detect which datasets are pre-mapped (EpisodeHDF5Dataset)
        from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset
        self._premapped = {
            name for name, ds in datasets.items()
            if isinstance(ds, EpisodeHDF5Dataset)
        }
        if self._premapped:
            logger.info(f"Pre-mapped HDF5 datasets (skip map_to_unified): {self._premapped}")
        
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
        """Compute action normalization statistics for each dataset.
        
        Uses RDT-1B style robust [-1, 1] normalization:
          1. Compute mean/std per dataset per dim
          2. Clip outliers at ±5 std
          3. Compute robust min/max from clipped range
          4. Normalize to [-1, 1]: x_norm = (x - center) / half_range
          
        This prevents outlier actions (e.g., behavior1k base/torso) from
        producing normalized values of 100+ which destabilize training.
        """
        for name, ds in self.datasets.items():
            if hasattr(ds, 'compute_action_statistics'):
                mean, std = ds.compute_action_statistics()
                
                from drifting_vla.data.action_mapping import DATASET_NATIVE_ACTION_DIM
                embodiment_id = DATASET_EMBODIMENT.get(name, 0)
                native_dim = DATASET_NATIVE_ACTION_DIM.get(name, None)
                
                if name in self._premapped:
                    # HDF5 datasets: mean/std already in 128-dim unified space
                    unified_mean = mean
                    unified_std = std
                else:
                    # Legacy datasets: need mapping to unified space
                unified_mean = map_to_unified(mean, embodiment_id)
                unified_std = map_to_unified(std, embodiment_id)
                
                mask_info = get_action_mask(embodiment_id, native_dim=native_dim)
                
                # Robust [-1, 1] normalization (RDT-1B style)
                # Clip range: mean ± 5*std → then normalize to [-1, 1]
                clip_std = np.maximum(unified_std, 0.01)  # Min std to avoid zero-range
                action_min = unified_mean - 5 * clip_std
                action_max = unified_mean + 5 * clip_std
                action_center = (action_min + action_max) / 2  # ≈ mean
                action_half_range = (action_max - action_min) / 2  # ≈ 5*std
                action_half_range = np.maximum(action_half_range, 0.01)
                
                # Inactive dims: center=0, half_range=1 (identity normalization)
                action_center[~mask_info.mask] = 0.0
                action_half_range[~mask_info.mask] = 1.0
                
                self.action_stats[name] = {
                    'center': action_center,
                    'half_range': action_half_range,
                    'mean': unified_mean,   # Keep for backward compat
                    'std': unified_std,
                    'native_mean': mean if name not in self._premapped else unified_mean,
                    'native_std': std if name not in self._premapped else unified_std,
                }
                logger.info(
                    f"Action stats for {name}: mean_norm={np.linalg.norm(unified_mean):.4f}, "
                    f"max_half_range={action_half_range[mask_info.mask].max():.4f}"
                )
            else:
                logger.warning(f"Dataset {name} doesn't support action stats, using identity")
                self.action_stats[name] = {
                    'center': np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32),
                    'half_range': np.ones(UNIFIED_ACTION_DIM, dtype=np.float32),
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
        
        # ── Pre-mapped HDF5 datasets: pass through directly ──
        if dataset_name in self._premapped:
            return self._getitem_premapped(raw_sample, dataset_name, idx)
        
        # ── Legacy datasets: map to unified space ──
        return self._getitem_legacy(raw_sample, dataset_name, idx)
    
    def _getitem_premapped(self, raw_sample: dict, dataset_name: str, idx: int) -> dict:
        """Handle EpisodeHDF5Dataset samples (actions already in 128-dim unified space)."""
        # Actions already [T, 128] in unified space
        actions = raw_sample['actions']
        if isinstance(actions, torch.Tensor):
            actions_np = actions.numpy()
        else:
            actions_np = np.array(actions, dtype=np.float32)
        
        # Normalize to [-1, 1]
        if self.normalize_actions and dataset_name in self.action_stats:
            stats = self.action_stats[dataset_name]
            actions_np = (actions_np - stats['center']) / stats['half_range']
            actions_np = np.clip(actions_np, -1.0, 1.0)
        
        # Images already [N, 3, H, W] and correct size
        images = raw_sample['images']
        if isinstance(images, torch.Tensor):
            images_t = images.float()
        else:
            images_t = torch.from_numpy(np.array(images, dtype=np.float32))
        if images_t.ndim == 3:
            images_t = images_t.unsqueeze(0)
        
        # Action mask already [128]
        action_mask = raw_sample['action_mask']
        if isinstance(action_mask, torch.Tensor):
            action_mask_t = action_mask.float()
        else:
            action_mask_t = torch.from_numpy(np.array(action_mask, dtype=np.float32))
        
        # Proprio already [128]
        proprio = raw_sample.get('proprio', torch.zeros(UNIFIED_ACTION_DIM))
        if isinstance(proprio, torch.Tensor):
            proprio_t = proprio.float()
        else:
            proprio_t = torch.from_numpy(np.array(proprio, dtype=np.float32))
        
        result = {
            'images': images_t,
            'language': raw_sample.get('language', ''),
            'actions': torch.from_numpy(actions_np).float(),
            'action_mask': action_mask_t,
            'embodiment_id': int(raw_sample.get('embodiment_id', 0)),
            'task_id': int(raw_sample.get('task_id', 0)),
            'dataset_name': dataset_name,
            'sample_id': idx,
            'proprio': proprio_t,
            'num_views': int(raw_sample.get('num_views', images_t.shape[0])),
            'num_frames': int(raw_sample.get('num_frames', 1)),
        }
        
        # Pass through VLM ViT encoder inputs (from DataLoader worker preprocessing)
        for vlm_key in ('vlm_input_ids', 'vlm_attention_mask', 'vlm_pixel_values', 'vlm_image_grid_thw'):
            if vlm_key in raw_sample:
                result[vlm_key] = raw_sample[vlm_key]
        
        # VLM features (if available)
        if dataset_name in self.vlm_h5_files:
            h5 = self.vlm_h5_files[dataset_name]
            key = f'sample_{idx}'
            if key in h5:
                result['vlm_features'] = torch.from_numpy(h5[key]['hidden'][:]).float()
                result['vlm_pooled'] = torch.from_numpy(h5[key]['pooled'][:]).float()
        
        return result
    
    def _getitem_legacy(self, raw_sample: dict, dataset_name: str, idx: int) -> dict:
        """Handle legacy dataset samples (need map_to_unified conversion)."""
        # Get embodiment info with exact native dim for tight masking
        from drifting_vla.data.action_mapping import DATASET_NATIVE_ACTION_DIM
        embodiment_id = DATASET_EMBODIMENT.get(dataset_name, 0)
        native_dim = DATASET_NATIVE_ACTION_DIM.get(dataset_name, None)
        mask_info = get_action_mask(embodiment_id, native_dim=native_dim)
        
        # --- Convert actions to 128-dim unified space ---
        actions = raw_sample['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
        
        # Map to unified space
        unified_actions = map_to_unified(actions, embodiment_id)
        
        # Normalize to [-1, 1] (RDT-1B style robust normalization)
        if self.normalize_actions and dataset_name in self.action_stats:
            stats = self.action_stats[dataset_name]
            unified_actions = (unified_actions - stats['center']) / stats['half_range']
            # Clip to [-1, 1] to handle outliers beyond 5-sigma
            unified_actions = np.clip(unified_actions, -1.0, 1.0)
        
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

        # Pass through proprioception — pad to 128-dim unified space
        if 'proprio' in raw_sample:
            p = raw_sample['proprio']
            if isinstance(p, torch.Tensor):
                p = p.numpy()
            elif not isinstance(p, np.ndarray):
                p = np.array(p, dtype=np.float32)
            # Pad to 128-dim
            proprio = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)
            n = min(len(p.flatten()), UNIFIED_ACTION_DIM)
            proprio[:n] = p.flatten()[:n]
            result['proprio'] = torch.from_numpy(proprio).float()
        else:
            result['proprio'] = torch.zeros(UNIFIED_ACTION_DIM, dtype=torch.float32)

        # Pass through camera/time metadata
        result['num_views'] = raw_sample.get('num_views', images_np.shape[0])
        result['num_frames'] = raw_sample.get('num_frames', 1)

        if vlm_features is not None:
            result['vlm_features'] = vlm_features
            result['vlm_pooled'] = vlm_pooled

        # Pass through VLM ViT encoder inputs (from DataLoader worker preprocessing)
        for vlm_key in ('vlm_input_ids', 'vlm_attention_mask', 'vlm_pixel_values', 'vlm_image_grid_thw'):
            if vlm_key in raw_sample:
                result[vlm_key] = raw_sample[vlm_key]

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
    temperature: float = 0.5,
) -> WeightedRandomSampler:
    """
    Create a temperature-balanced weighted sampler for multi-dataset training.
    
    RDT-1B style: sample each dataset with probability proportional to
    N_i^temperature where N_i is the dataset size. temperature=0.5 (square root)
    prevents large datasets from dominating while still giving them more weight.
    
    Args:
        unified_dataset: UnifiedDataset instance.
        weights: Optional explicit per-dataset weights (overrides temperature).
        temperature: Sampling temperature. 0=uniform across datasets, 1=proportional
                     to size, 0.5=square root (RDT-1B default).
    
    Returns:
        WeightedRandomSampler for DataLoader.
    """
    # Count samples per dataset
    ds_counts = {}
    for ds_name, _ in unified_dataset.global_index:
        ds_counts[ds_name] = ds_counts.get(ds_name, 0) + 1
    
    if weights is not None:
        # Explicit weights
        ds_weights = weights
    else:
        # Temperature-based balancing (RDT-1B style)
        # p_i ∝ N_i^temperature
        total_temp = sum(n ** temperature for n in ds_counts.values())
        ds_weights = {name: (count ** temperature) / total_temp
                      for name, count in ds_counts.items()}
    
    # Assign per-sample weights
        sample_weights = np.zeros(len(unified_dataset), dtype=np.float64)
        for i, (ds_name, _) in enumerate(unified_dataset.global_index):
        # Weight for this sample = dataset_weight / dataset_size
        # so all samples within a dataset have equal probability
        sample_weights[i] = ds_weights.get(ds_name, 1.0) / ds_counts[ds_name]
    
    # Log sampling distribution
    total_weight = sum(ds_weights.values())
    for name in sorted(ds_counts.keys()):
        pct = ds_weights.get(name, 0) / (total_weight + 1e-8) * 100
        logger.info(f"  Sampling {name}: {ds_counts[name]} samples, weight={pct:.1f}%")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(unified_dataset),
        replacement=True,
    )


def collate_unified(batch: List[dict]) -> dict:
    """
    Custom collate function for UnifiedDataset.
    
    Handles:
      - Variable-length images (different camera counts) by padding
      - VLM ViT encoder inputs (pixel_values, image_grid_thw) by concatenation
      - Text inputs (input_ids, attention_mask) by padding
      - Pre-computed VLM features by padding
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

    # Proprioception
    if 'proprio' in batch[0]:
        result['proprio'] = torch.stack([s['proprio'] for s in batch])

    # Camera/time metadata
    result['num_views'] = max(s.get('num_views', 1) for s in batch)
    result['num_frames'] = max(s.get('num_frames', 1) for s in batch)
    
    # ── VLM ViT encoder inputs (vision-encoder-only, from DataLoader workers) ──
    has_vlm_tokens = 'vlm_input_ids' in batch[0]
    if has_vlm_tokens:
        from torch.nn.utils.rnn import pad_sequence

        # Text: pad input_ids and attention_mask to max length
        result['vlm_input_ids'] = pad_sequence(
            [s['vlm_input_ids'] for s in batch],
            batch_first=True, padding_value=0,
        )
        result['vlm_attention_mask'] = pad_sequence(
            [s['vlm_attention_mask'] for s in batch],
            batch_first=True, padding_value=0,
        )

        # pixel_values: [N_patches, C_patch] per sample — concatenate all
        # The ViT encoder processes ALL patches from ALL samples in one call
        pv_list = [s['vlm_pixel_values'] for s in batch if 'vlm_pixel_values' in s]
        if pv_list:
            result['vlm_pixel_values'] = torch.cat(pv_list, dim=0)  # [N_total_patches, C]

        # image_grid_thw: [N_images, 3] per sample — concatenate all
        if 'vlm_image_grid_thw' in batch[0]:
            thw_list = [s['vlm_image_grid_thw'] for s in batch]
            result['vlm_image_grid_thw'] = torch.cat(thw_list, dim=0)  # [N_total_images, 3]

    # ── Pre-computed VLM features (offline path) ──
    has_vlm = any('vlm_features' in s and s.get('vlm_features') is not None for s in batch)
    if has_vlm and not has_vlm_tokens:
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
                vlm_pooled_list.append(torch.zeros(dim))
        
        result['vlm_features'] = padded_features
        result['vlm_pooled'] = torch.stack(vlm_pooled_list)
        result['vlm_attn_mask'] = vlm_attn_mask
    
    return result
