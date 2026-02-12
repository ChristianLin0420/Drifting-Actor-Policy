"""
LeRobot Dataset Adapter
========================

Loads datasets using the `lerobot` package (v0.4+).
Covers: ALOHA, DROID, BC-Z, TACO Play, Stanford HYDRA, Behavior 1K, etc.

Pi0-style variable view handling:
- Accept ALL available RGB views (no hard cap)
- Filter out depth/segmentation channels (RGB only)
- Each view resized to image_size × image_size
- Variable V handled via padding + attention mask in collate/model
"""

import torch
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def _select_rgb_image_keys(all_keys: list, sample: dict) -> list:
    """
    Select RGB image keys from a LeRobot sample (Pi0 style: use all RGB views).
    
    Filters out depth, segmentation, and non-image keys.
    Returns keys sorted for deterministic ordering.
    """
    rgb_keys = []
    for k in sorted(all_keys):
        # Must be a 3-channel image tensor
        v = sample.get(k)
        if not isinstance(v, torch.Tensor) or v.dim() != 3:
            continue
        C = v.shape[0]
        # Skip depth (1-channel), segmentation (1-channel or int)
        if C != 3:
            continue
        # Skip keys explicitly marked as depth or segmentation
        kl = k.lower()
        if any(skip in kl for skip in ['depth', 'seg_', 'segmentation', 'mask', 'normal']):
            continue
        rgb_keys.append(k)
    return rgb_keys


class LeRobotDataset:
    """
    Adapter wrapping a lerobot.LeRobotDataset into our standard format.
    
    Pi0-style: accepts ALL available RGB views per sample — no hard cap.
    The collate function pads views to max_V in the batch.
    
    Output per sample:
        images:   [V, 3, image_size, image_size]  — V varies per dataset
        language:  str                              — from dataset
        actions:   [T, D_native]                   — from dataset
    """
    
    def __init__(
        self,
        repo_id: str,
        image_size: int = 448,
        action_horizon: int = 16,
        max_samples: Optional[int] = None,
    ):
        self.repo_id = repo_id
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.max_samples = max_samples
        
        self.action_mean = None
        self.action_std = None
        self.action_dim = None
        self.image_keys = []
        self.language_key = None
        self.lerobot_ds = None
        self._indices = []
        
        self._load(repo_id)
    
    def _load(self, repo_id: str):
        """Load dataset via lerobot package."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset as LDS
            
            logger.info(f"Loading LeRobot dataset: {repo_id} ...")
            self.lerobot_ds = LDS(repo_id)
            
            # Detect image keys from first sample (RGB only, Pi0 style)
            s0 = self.lerobot_ds[0]
            all_img_candidates = [k for k in s0.keys() if 'image' in k.lower()]
            self.image_keys = _select_rgb_image_keys(all_img_candidates, s0)
            
            # Detect action dim
            if 'action' in s0:
                self.action_dim = s0['action'].shape[0]
            else:
                self.action_dim = 7
            
            # Detect language key
            self.language_key = None
            for candidate in ['task', 'language_instruction', 'language', 'text']:
                if candidate in s0 and s0[candidate]:
                    self.language_key = candidate
                    break
            
            total = len(self.lerobot_ds)
            if self.max_samples and self.max_samples < total:
                self._indices = list(range(self.max_samples))
            else:
                self._indices = list(range(total))
            
            logger.info(
                f"  ✓ {repo_id}: {len(self._indices)} samples, "
                f"action_dim={self.action_dim}, "
                f"views={len(self.image_keys)} {self.image_keys}, "
                f"lang_key={self.language_key}"
            )
            
        except ImportError:
            logger.error("lerobot package not installed. Run: pip install lerobot")
        except Exception as e:
            logger.error(f"Failed to load {repo_id}: {e}")
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> dict:
        """Load a sample in our standard format."""
        if self.lerobot_ds is None:
            return self._empty_sample()
        
        real_idx = self._indices[idx]
        try:
            row = self.lerobot_ds[real_idx]
        except (RuntimeError, Exception):
            # Video decode errors (torchcodec) — skip this sample
            if idx + 1 < len(self._indices):
                return self.__getitem__(idx + 1)
            return self._empty_sample()
        
        # --- Actions ---
        action = row.get('action', torch.zeros(self.action_dim))
        actions = action.unsqueeze(0).repeat(self.action_horizon, 1).numpy()
        
        # --- Images (all RGB views, Pi0 style) ---
        images = self._extract_images(row)
        
        # --- Language ---
        language = self._extract_language(row)
        
        # --- Metadata ---
        episode_idx = row.get('episode_index', torch.tensor(0))
        if isinstance(episode_idx, torch.Tensor):
            episode_idx = episode_idx.item()
        
        return {
            'images': torch.from_numpy(images).float(),
            'language': language,
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'task_id': 0,
            'episode_id': int(episode_idx),
            'timestep': int(row.get('frame_index', torch.tensor(0)).item()) if isinstance(row.get('frame_index'), torch.Tensor) else 0,
        }
    
    def _extract_images(self, row: dict) -> np.ndarray:
        """
        Extract ALL RGB views from a sample (Pi0 style, no hard cap).
        
        Returns: [V, 3, image_size, image_size] numpy array.
        """
        import cv2
        
        images = []
        for key in self.image_keys:
            val = row.get(key)
            if val is None:
                continue
            
            img = val.numpy() if isinstance(val, torch.Tensor) else np.array(val)
            
            # Ensure CHW float
            if img.ndim == 2:
                continue  # Skip grayscale/depth
            if img.shape[0] not in (1, 3, 4):
                # Might be HWC — transpose
                if img.shape[2] in (1, 3, 4):
                    img = img.transpose(2, 0, 1)
            if img.shape[0] != 3:
                continue  # Skip non-RGB
            
            C, H, W = img.shape
            
            # Resize if needed
            if H != self.image_size or W != self.image_size:
                img_hwc = img.transpose(1, 2, 0)
                img_hwc = cv2.resize(img_hwc, (self.image_size, self.image_size))
                img = img_hwc.transpose(2, 0, 1)
            
            images.append(img.astype(np.float32))
        
        if not images:
            # No valid images — return single black frame
            images = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)]
        
        return np.stack(images, axis=0)
    
    def _extract_language(self, row: dict) -> str:
        """Extract language instruction from the actual dataset."""
        if self.language_key and self.language_key in row:
            lang = row[self.language_key]
            if isinstance(lang, str) and lang:
                return lang
            if lang:
                return str(lang)
        
        # Fallback: try all common keys
        for key in ['task', 'language_instruction', 'task_description', 'language', 'text', 'instruction']:
            if key in row and row[key]:
                return str(row[key])
        
        logger.warning(f"No language instruction found in {self.repo_id}")
        return ''
    
    def _empty_sample(self) -> dict:
        return {
            'images': torch.zeros(1, 3, self.image_size, self.image_size),
            'language': '',
            'actions': torch.zeros(self.action_horizon, self.action_dim or 7),
            'task_id': 0,
            'episode_id': 0,
            'timestep': 0,
        }
    
    def get_image_keys(self) -> list:
        """Return the list of RGB image keys detected in this dataset."""
        return list(self.image_keys)
    
    def compute_action_statistics(self) -> tuple:
        """Compute per-dimension action mean/std."""
        if self.lerobot_ds is None or len(self._indices) == 0:
            dim = self.action_dim or 7
            self.action_mean = np.zeros(dim, dtype=np.float32)
            self.action_std = np.ones(dim, dtype=np.float32)
            return self.action_mean, self.action_std
        
        n = min(500, len(self._indices))
        all_actions = []
        for i in range(n):
            try:
                row = self.lerobot_ds[self._indices[i]]
                if 'action' in row:
                    all_actions.append(row['action'].numpy())
            except Exception:
                continue
        
        if not all_actions:
            dim = self.action_dim or 7
            self.action_mean = np.zeros(dim, dtype=np.float32)
            self.action_std = np.ones(dim, dtype=np.float32)
            return self.action_mean, self.action_std
        
        all_actions = np.stack(all_actions)
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32)
        self.action_std = np.maximum(self.action_std, 1e-4)
        
        logger.info(f"  Action stats for {self.repo_id}: {len(all_actions)} samples, dim={self.action_dim}")
        return self.action_mean, self.action_std
