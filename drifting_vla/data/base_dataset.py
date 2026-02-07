"""
Base Dataset Interface
======================

Abstract base class and data structures for VLA datasets.
Provides a consistent interface across different environments.
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Union
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class VLADataPoint:
    """
    Data point for VLA training.
    
    Attributes:
        images: Visual observations [C, H, W] or [V, C, H, W] for multi-view.
        language: Task description string.
        actions: Action sequence [T, D_a].
        task_id: Optional task identifier.
        episode_id: Optional episode identifier.
        timestep: Optional timestep within episode.
        
        # Optional additional fields
        proprio: Optional proprioceptive state [D_p].
        depth: Optional depth images [H, W] or [V, H, W].
        mask: Optional action mask [T] for variable length.
    """
    images: torch.Tensor
    language: str
    actions: torch.Tensor
    task_id: Optional[int] = None
    episode_id: Optional[int] = None
    timestep: Optional[int] = None
    proprio: Optional[torch.Tensor] = None
    depth: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    
    def to(self, device: torch.device) -> 'VLADataPoint':
        """Move tensors to device."""
        return VLADataPoint(
            images=self.images.to(device),
            language=self.language,
            actions=self.actions.to(device),
            task_id=self.task_id,
            episode_id=self.episode_id,
            timestep=self.timestep,
            proprio=self.proprio.to(device) if self.proprio is not None else None,
            depth=self.depth.to(device) if self.depth is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
        )


class BaseVLADataset(Dataset, ABC):
    """
    Abstract base class for VLA datasets.
    
    Provides common functionality for loading and processing VLA data
    across different environments (RLBench, LIBERO, CALVIN, etc.)
    
    Subclasses must implement:
    - __len__(): Return dataset size
    - __getitem__(idx): Return VLADataPoint for index
    - get_task_names(): Return list of task names
    """
    
    def __init__(
        self,
        data_dir: str,
        action_horizon: int = 16,
        image_size: int = 224,
        use_depth: bool = False,
        transform: Optional[callable] = None,
    ):
        """
        Initialize base dataset.
        
        Args:
            data_dir: Root directory containing data.
            action_horizon: Number of action steps to return.
            image_size: Image resolution (assumes square).
            use_depth: Whether to load depth images.
            transform: Optional transform to apply to data.
        """
        self.data_dir = data_dir
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_depth = use_depth
        self.transform = transform
        
        # Action statistics for normalization
        self.action_mean: Optional[np.ndarray] = None
        self.action_std: Optional[np.ndarray] = None
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """
        Get a data sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Dict with 'images', 'language', 'actions', etc.
        """
        pass
    
    @abstractmethod
    def get_task_names(self) -> list[str]:
        """Return list of task names in the dataset."""
        pass
    
    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Normalize actions using dataset statistics.
        
        Args:
            actions: Raw actions [T, D_a].
        
        Returns:
            Normalized actions [T, D_a].
        """
        if self.action_mean is None or self.action_std is None:
            return actions
        
        return (actions - self.action_mean) / (self.action_std + 1e-8)
    
    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize actions back to original scale.
        
        Args:
            actions: Normalized actions [T, D_a].
        
        Returns:
            Denormalized actions [T, D_a].
        """
        if self.action_mean is None or self.action_std is None:
            return actions
        
        return actions * (self.action_std + 1e-8) + self.action_mean
    
    def compute_action_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute action mean and std from dataset.
        
        Returns:
            Tuple of (mean, std) arrays.
        """
        all_actions = []
        
        for i in range(len(self)):
            sample = self[i]
            actions = sample['actions']
            if isinstance(actions, torch.Tensor):
                actions = actions.numpy()
            all_actions.append(actions)
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
        
        return self.action_mean, self.action_std
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """
        Collate batch of samples.
        
        Args:
            batch: List of sample dicts.
        
        Returns:
            Batched dict with stacked tensors.
        """
        result = {}
        
        # Stack tensor fields
        for key in ['images', 'actions', 'proprio', 'depth', 'mask']:
            if key in batch[0] and batch[0][key] is not None:
                result[key] = torch.stack([s[key] for s in batch])
        
        # List fields (strings, ids)
        for key in ['language']:
            if key in batch[0]:
                result[key] = [s[key] for s in batch]
        
        for key in ['task_id', 'episode_id', 'timestep']:
            if key in batch[0] and batch[0][key] is not None:
                result[key] = torch.tensor([s[key] for s in batch])
        
        return result


class ConcatDataset(Dataset):
    """
    Concatenate multiple VLA datasets.
    
    Useful for combining data from multiple tasks or environments.
    """
    
    def __init__(self, datasets: list[BaseVLADataset]):
        """
        Initialize concatenated dataset.
        
        Args:
            datasets: List of datasets to concatenate.
        """
        self.datasets = datasets
        self.cumulative_sizes = []
        
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
    
    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0
    
    def __getitem__(self, idx: int) -> dict:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        
        # Compute local index
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        return self.datasets[dataset_idx][local_idx]
    
    def get_task_names(self) -> list[str]:
        """Get all task names across datasets."""
        names = []
        for ds in self.datasets:
            names.extend(ds.get_task_names())
        return list(set(names))


