"""
LeRobot Dataset Adapter
========================

Loads datasets from pre-downloaded Arrow data (fast, local) or via lerobot API (fallback).
Covers: ALOHA, DROID, BC-Z, TACO Play, Stanford HYDRA, Behavior 1K, etc.

Loading priority:
  1. Local Arrow data at {data_root}/{ds_name}/arrow_data/ (instant, memory-mapped)
  2. lerobot API fallback (slow, downloads from HuggingFace)

Pi0-style variable view handling:
- Accept ALL available RGB views per sample — no hard cap
- Filter out depth/segmentation channels (RGB only)
- Each view resized to image_size × image_size
- Variable V handled via padding + attention mask in collate/model
"""

import torch
import numpy as np
from pathlib import Path
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


def _detect_image_keys_from_arrow(column_names: list) -> list:
    """Detect RGB image columns from Arrow dataset column names.
    
    Includes columns like 'observation.images.image_with_depth' (which IS RGB),
    but excludes 'observation.images.depth.head' (which is actual depth data).
    """
    img_keys = []
    # Patterns that indicate actual depth/segmentation data (not RGB)
    depth_seg_patterns = [
        'images.depth', 'images.seg_', 'images.segmentation',
        'images.mask', 'images.normal',
    ]
    for col in sorted(column_names):
        cl = col.lower()
        if 'image' not in cl:
            continue
        # Skip actual depth/segmentation columns
        if any(pat in cl for pat in depth_seg_patterns):
            continue
        img_keys.append(col)
    return img_keys


class LeRobotDataset:
    """
    Adapter for LeRobot-format datasets.
    
    Loads from local Arrow data when available (fast), falls back to lerobot API.
    
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
        data_root: Optional[str] = None,
        ds_name: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.max_samples = max_samples
        self.data_root = data_root
        self.ds_name = ds_name
        
        self.action_mean = None
        self.action_std = None
        self.action_dim = None
        self.image_keys = []
        self.language_key = None
        
        # Data backends (one will be active)
        self.lerobot_ds = None     # lerobot API backend
        self._arrow_ds = None      # Arrow backend (fast)
        self._backend = None       # 'arrow' or 'lerobot'
        self._indices = []
        
        self._load(repo_id)
    
    def _load(self, repo_id: str):
        """Load dataset: try Arrow first, fall back to lerobot API."""
        # --- Strategy 1: Local Arrow data (instant, no network) ---
        if self.data_root and self.ds_name:
            arrow_path = Path(self.data_root) / self.ds_name / 'arrow_data'
            if arrow_path.exists():
                if self._load_from_arrow(arrow_path):
                    return
        
        # --- Strategy 2: lerobot API (slow, downloads from HF) ---
        self._load_from_lerobot(repo_id)
    
    def _load_from_arrow(self, arrow_path: Path) -> bool:
        """Load from pre-downloaded Arrow data. Returns True on success."""
        try:
            from datasets import load_from_disk
            
            logger.info(f"Loading from Arrow: {arrow_path} ...")
            self._arrow_ds = load_from_disk(str(arrow_path))
            self._backend = 'arrow'
            
            cols = self._arrow_ds.column_names
            
            # Detect image keys
            self.image_keys = _detect_image_keys_from_arrow(cols)
            
            # Detect action dim
            if 'action' in cols:
                row0 = self._arrow_ds[0]
                action_val = row0['action']
                if isinstance(action_val, list):
                    self.action_dim = len(action_val)
                elif hasattr(action_val, 'shape'):
                    self.action_dim = action_val.shape[0]
                else:
                    self.action_dim = 7
            else:
                self.action_dim = 7
            
            # Detect language key
            self.language_key = None
            for candidate in ['task', 'language_instruction', 'language', 'text']:
                if candidate in cols:
                    self.language_key = candidate
                    break
            
            total = len(self._arrow_ds)
            if self.max_samples and self.max_samples < total:
                self._indices = list(range(self.max_samples))
            else:
                self._indices = list(range(total))
            
            logger.info(
                f"  ✓ {self.ds_name or self.repo_id}: {len(self._indices)} samples (Arrow), "
                f"action_dim={self.action_dim}, "
                f"views={len(self.image_keys)} {self.image_keys}, "
                f"lang_key={self.language_key}"
            )
            return True
            
        except Exception as e:
            logger.warning(f"  Arrow load failed: {e}, falling back to lerobot API")
            self._arrow_ds = None
            self._backend = None
            return False
    
    def _load_from_lerobot(self, repo_id: str):
        """Load via lerobot API (slow, downloads from HF)."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset as LDS
            
            logger.info(f"Loading LeRobot dataset: {repo_id} ...")
            self.lerobot_ds = LDS(repo_id)
            self._backend = 'lerobot'
            
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
                f"  ✓ {repo_id}: {len(self._indices)} samples (lerobot API), "
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
        """Load a sample with proprioception and multi-frame history."""
        if self._backend == 'arrow':
            return self._getitem_arrow(idx)
        elif self._backend == 'lerobot':
            return self._getitem_lerobot(idx)
        else:
            return self._empty_sample()
    
    # ──────────────────────────────────────────────────────────────
    # Arrow backend (fast path)
    # ──────────────────────────────────────────────────────────────
    
    def _getitem_arrow(self, idx: int) -> dict:
        """Load sample from Arrow dataset."""
        real_idx = self._indices[idx]
        try:
            row = self._arrow_ds[real_idx]
        except Exception:
            if idx + 1 < len(self._indices):
                return self._getitem_arrow(idx + 1)
            return self._empty_sample()
        
        # --- Actions ---
        action = self._to_tensor(row.get('action'), self.action_dim)
        actions = action.unsqueeze(0).repeat(self.action_horizon, 1).numpy()
        
        # --- Images ---
        images, num_views, num_frames = self._extract_images_arrow(real_idx, row)
        
        # --- Language ---
        language = self._extract_language_generic(row)
        
        # --- Proprioception ---
        proprio = self._extract_proprio_generic(row)
        
        # --- Metadata ---
        episode_idx = row.get('episode_index', 0)
        frame_idx = row.get('frame_index', 0)
        
        return {
            'images': torch.from_numpy(images).float(),
            'language': language,
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'proprio': torch.from_numpy(proprio).float(),
            'task_id': 0,
            'episode_id': int(episode_idx),
            'timestep': int(frame_idx),
            'num_views': num_views,
            'num_frames': num_frames,
        }
    
    def _extract_images_arrow(self, real_idx: int, current_row: dict) -> tuple:
        """Extract multi-frame images from Arrow data."""
        import cv2
        
        if not self.image_keys:
            # No images in this dataset — return black frame
            img = np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
            return img, 1, 1
        
        current_ep = current_row.get('episode_index', -1)
        
        # Collect frames: t-1, t, t+1
        frame_offsets = [-1, 0, 1]
        all_frame_images = []
        
        for dt in frame_offsets:
            target_idx = real_idx + dt
            
            if 0 <= target_idx < len(self._arrow_ds):
                try:
                    target_row = self._arrow_ds[target_idx]
                    target_ep = target_row.get('episode_index', -2)
                    if target_ep != current_ep:
                        target_row = current_row
                except Exception:
                    target_row = current_row
            else:
                target_row = current_row
            
            # Extract all views for this frame
            view_imgs = []
            for key in self.image_keys:
                val = target_row.get(key)
                if val is None:
                    continue
                img = self._pil_or_array_to_chw(val, cv2)
                if img is not None:
                    view_imgs.append(img)
            
            if not view_imgs:
                view_imgs = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)]
            
            all_frame_images.append(np.stack(view_imgs, axis=0))  # [V, 3, H, W]
        
        num_views = all_frame_images[0].shape[0]
        num_frames = len(frame_offsets)
        images = np.concatenate(all_frame_images, axis=0)  # [T*V, 3, H, W]
        return images, num_views, num_frames
    
    def _pil_or_array_to_chw(self, val, cv2) -> Optional[np.ndarray]:
        """Convert PIL Image or array to [3, H, W] float32 numpy."""
        # Handle PIL Image
        from PIL import Image
        if isinstance(val, Image.Image):
            img = np.array(val.convert('RGB'))  # [H, W, 3] uint8
            img = img.transpose(2, 0, 1).astype(np.float32)  # [3, H, W]
        elif isinstance(val, torch.Tensor):
            img = val.numpy().astype(np.float32)
        elif isinstance(val, np.ndarray):
            img = val.astype(np.float32)
        else:
            return None
        
        if img.ndim != 3:
            return None
        
        # Ensure CHW
        if img.shape[0] not in (1, 3, 4) and img.shape[2] in (1, 3, 4):
            img = img.transpose(2, 0, 1)
        if img.shape[0] != 3:
            return None
        
        C, H, W = img.shape
        if H != self.image_size or W != self.image_size:
            img_hwc = img.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (self.image_size, self.image_size))
            img = img_hwc.transpose(2, 0, 1)
        
        return img
    
    # ──────────────────────────────────────────────────────────────
    # lerobot API backend (original slow path)
    # ──────────────────────────────────────────────────────────────
    
    def _getitem_lerobot(self, idx: int) -> dict:
        """Load sample from lerobot API."""
        if self.lerobot_ds is None:
            return self._empty_sample()

        real_idx = self._indices[idx]
        try:
            row = self.lerobot_ds[real_idx]
        except (RuntimeError, Exception):
            if idx + 1 < len(self._indices):
                return self._getitem_lerobot(idx + 1)
            return self._empty_sample()

        # --- Actions ---
        action = row.get('action', torch.zeros(self.action_dim))
        actions = action.unsqueeze(0).repeat(self.action_horizon, 1).numpy()

        # --- Multi-frame images (t-1, t, t+1) × all RGB views ---
        images, num_views, num_frames = self._extract_multiframe_images_lerobot(real_idx, row)

        # --- Language ---
        language = self._extract_language_generic(row)

        # --- Proprioception (robot state) ---
        proprio = self._extract_proprio_generic(row)

        # --- Metadata ---
        episode_idx = row.get('episode_index', torch.tensor(0))
        if isinstance(episode_idx, torch.Tensor):
            episode_idx = episode_idx.item()

        return {
            'images': torch.from_numpy(images).float(),
            'language': language,
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'proprio': torch.from_numpy(proprio).float(),
            'task_id': 0,
            'episode_id': int(episode_idx),
            'timestep': int(row.get('frame_index', torch.tensor(0)).item()) if isinstance(row.get('frame_index'), torch.Tensor) else 0,
            'num_views': num_views,
            'num_frames': num_frames,
        }

    def _extract_multiframe_images_lerobot(self, real_idx: int, current_row: dict) -> tuple:
        """Extract multi-frame (t-1, t, t+1) multi-view images via lerobot API."""
        current_ep = current_row.get('episode_index', torch.tensor(-1))
        if isinstance(current_ep, torch.Tensor):
            current_ep = current_ep.item()

        frame_offsets = [-1, 0, 1]
        all_frame_images = []

        for dt in frame_offsets:
            target_idx = real_idx + dt
            if 0 <= target_idx < len(self.lerobot_ds):
                try:
                    target_row = self.lerobot_ds[target_idx]
                    target_ep = target_row.get('episode_index', torch.tensor(-2))
                    if isinstance(target_ep, torch.Tensor):
                        target_ep = target_ep.item()
                    if target_ep != current_ep:
                        target_row = current_row
                except (RuntimeError, Exception):
                    target_row = current_row
            else:
                target_row = current_row

            frame_imgs = self._extract_images_lerobot(target_row)
            all_frame_images.append(frame_imgs)

        num_views = all_frame_images[0].shape[0]
        num_frames = len(frame_offsets)
        images = np.concatenate(all_frame_images, axis=0)
        return images, num_views, num_frames

    def _extract_images_lerobot(self, row: dict) -> np.ndarray:
        """Extract ALL RGB views from a lerobot sample."""
        import cv2
        
        images = []
        for key in self.image_keys:
            val = row.get(key)
            if val is None:
                continue
            
            img = val.numpy() if isinstance(val, torch.Tensor) else np.array(val)
            
            if img.ndim == 2:
                continue
            if img.shape[0] not in (1, 3, 4):
                if img.shape[2] in (1, 3, 4):
                    img = img.transpose(2, 0, 1)
            if img.shape[0] != 3:
                continue
            
            C, H, W = img.shape
            if H != self.image_size or W != self.image_size:
                img_hwc = img.transpose(1, 2, 0)
                img_hwc = cv2.resize(img_hwc, (self.image_size, self.image_size))
                img = img_hwc.transpose(2, 0, 1)
            
            images.append(img.astype(np.float32))
        
        if not images:
            images = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)]
        
        return np.stack(images, axis=0)
    
    # ──────────────────────────────────────────────────────────────
    # Shared helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(val, expected_dim: int) -> torch.Tensor:
        """Convert list/array/tensor to a 1D torch tensor."""
        if val is None:
            return torch.zeros(expected_dim)
        if isinstance(val, torch.Tensor):
            return val.float()
        if isinstance(val, list):
            return torch.tensor(val, dtype=torch.float32)
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val).float()
        return torch.zeros(expected_dim)

    def _extract_proprio_generic(self, row: dict) -> np.ndarray:
        """Extract proprioception from either Arrow or lerobot row."""
        for key in ['observation.state', 'state', 'observation.proprio']:
            if key in row:
                val = row[key]
                if isinstance(val, torch.Tensor):
                    val = val.numpy()
                elif isinstance(val, list):
                    val = np.array(val, dtype=np.float32)
                elif not isinstance(val, np.ndarray):
                    continue

                proprio = np.zeros(128, dtype=np.float32)
                n = min(len(val), 128)
                proprio[:n] = np.array(val).flatten()[:n]
                return proprio

        return np.zeros(128, dtype=np.float32)
    
    def _extract_language_generic(self, row: dict) -> str:
        """Extract language instruction from either Arrow or lerobot row."""
        if self.language_key and self.language_key in row:
            lang = row[self.language_key]
            if isinstance(lang, str) and lang:
                return lang
            if lang:
                return str(lang)
        
        for key in ['task', 'language_instruction', 'task_description', 'language', 'text', 'instruction']:
            if key in row and row[key]:
                return str(row[key])
        
        return ''
    
    def _empty_sample(self) -> dict:
        return {
            'images': torch.zeros(1, 3, self.image_size, self.image_size),
            'language': '',
            'actions': torch.zeros(self.action_horizon, self.action_dim or 7),
            'proprio': torch.zeros(128),
            'task_id': 0,
            'episode_id': 0,
            'timestep': 0,
            'num_views': 1,
            'num_frames': 1,
        }
    
    def get_image_keys(self) -> list:
        """Return the list of RGB image keys detected in this dataset."""
        return list(self.image_keys)
    
    def compute_action_statistics(self) -> tuple:
        """Compute per-dimension action mean/std.
        
        For Arrow backend: reads only the 'action' column (avoids deserializing images).
        """
        dim = self.action_dim or 7
        
        if len(self._indices) == 0:
            self.action_mean = np.zeros(dim, dtype=np.float32)
            self.action_std = np.ones(dim, dtype=np.float32)
            return self.action_mean, self.action_std
        
        n = min(500, len(self._indices))
        all_actions = []
        
        if self._backend == 'arrow' and 'action' in self._arrow_ds.column_names:
            # Fast path: read only 'action' column (no image deserialization)
            indices = self._indices[:n]
            action_col = self._arrow_ds.select_columns(['action'])
            for i in indices:
                try:
                    action_val = action_col[i]['action']
                    if isinstance(action_val, list):
                        all_actions.append(np.array(action_val, dtype=np.float32))
                    elif isinstance(action_val, np.ndarray):
                        all_actions.append(action_val.astype(np.float32))
                except Exception:
                    continue
        elif self._backend == 'lerobot':
            for i in range(n):
                try:
                    row = self.lerobot_ds[self._indices[i]]
                    if 'action' in row:
                        all_actions.append(row['action'].numpy())
                except Exception:
                    continue
        
        if not all_actions:
            self.action_mean = np.zeros(dim, dtype=np.float32)
            self.action_std = np.ones(dim, dtype=np.float32)
            return self.action_mean, self.action_std
        
        all_actions = np.stack(all_actions)
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32)
        self.action_std = np.maximum(self.action_std, 1e-4)
        
        logger.info(f"  Action stats for {self.repo_id}: {len(all_actions)} samples, dim={self.action_dim}")
        return self.action_mean, self.action_std
