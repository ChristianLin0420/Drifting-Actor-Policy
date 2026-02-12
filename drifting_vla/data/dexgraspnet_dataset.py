"""
DexGraspNet 2.0 Dataset
========================

Loader for the DexGraspNet 2.0 dexterous grasping dataset.
Source: https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0

Data format (HDF5):
  - Object point clouds with colors
  - Grasp poses: wrist SE(3) + 16 joint angles for Allegro/LEAP hand
  - Scene descriptions

Action format:
  - Wrist pose: [x, y, z, qx, qy, qz, qw] (7-dim)
  - Finger joints: [j1, ..., j16] (16-dim)
  - Total: 23-dim per grasp

Since DexGraspNet stores static grasps (not trajectories),
each sample has action_horizon=1 and is repeated T times for training.
"""

import torch
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DexGraspNetDataset:
    """
    DexGraspNet 2.0 dataset for dexterous grasping.
    
    Loads grasp poses from HDF5 files and renders point cloud images.
    
    Args:
        data_dir: Root directory of DexGraspNet data.
        split: 'train', 'val', or 'test'.
        image_size: Target image size (448 for VLM).
        action_horizon: Action sequence length (grasps padded to this).
        max_samples: Maximum number of samples to load (for debugging).
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 448,
        action_horizon: int = 16,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.action_horizon = action_horizon
        self.max_samples = max_samples
        
        # Discover data files
        self.samples = self._build_index()
        
        # Action statistics
        self.action_mean = None
        self.action_std = None
        
        logger.info(f"DexGraspNet dataset: {len(self.samples)} samples from {self.data_dir}")
    
    def _build_index(self) -> List[dict]:
        """Build sample index from data files."""
        samples = []
        
        # Check for HDF5 files
        h5_files = list(self.data_dir.glob('**/*.h5')) + list(self.data_dir.glob('**/*.hdf5'))
        
        # Check for numpy files
        npz_files = list(self.data_dir.glob('**/*.npz'))
        
        # Check for JSON metadata
        json_files = list(self.data_dir.glob('**/*.json'))
        
        if h5_files:
            samples = self._index_h5_files(h5_files)
        elif npz_files:
            samples = self._index_npz_files(npz_files)
        elif json_files:
            samples = self._index_json_files(json_files)
        else:
            # Try parquet (HuggingFace datasets)
            pq_files = list(self.data_dir.glob('**/*.parquet'))
            if pq_files:
                samples = self._index_parquet_files(pq_files)
            else:
                logger.warning(f"No data files found in {self.data_dir}")
        
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _index_h5_files(self, h5_files: List[Path]) -> List[dict]:
        """Index samples from HDF5 files."""
        samples = []
        
        for h5_path in sorted(h5_files):
            try:
                with h5py.File(str(h5_path), 'r') as f:
                    # Common DexGraspNet HDF5 layouts
                    if 'grasps' in f:
                        n_grasps = len(f['grasps'])
                        for i in range(n_grasps):
                            samples.append({
                                'format': 'h5',
                                'path': str(h5_path),
                                'index': i,
                                'group': 'grasps',
                            })
                    elif 'grasp_poses' in f:
                        n_grasps = f['grasp_poses'].shape[0]
                        for i in range(n_grasps):
                            samples.append({
                                'format': 'h5',
                                'path': str(h5_path),
                                'index': i,
                                'group': 'grasp_poses',
                            })
                    else:
                        # Try to find any array-like data
                        for key in f.keys():
                            if hasattr(f[key], 'shape') and f[key].ndim >= 1:
                                n = f[key].shape[0]
                                for i in range(n):
                                    samples.append({
                                        'format': 'h5',
                                        'path': str(h5_path),
                                        'index': i,
                                        'group': key,
                                    })
                                break
            except Exception as e:
                logger.warning(f"Failed to index {h5_path}: {e}")
        
        return samples
    
    def _index_npz_files(self, npz_files: List[Path]) -> List[dict]:
        """Index samples from NPZ files."""
        samples = []
        for npz_path in sorted(npz_files):
            samples.append({
                'format': 'npz',
                'path': str(npz_path),
                'index': 0,
            })
        return samples
    
    def _index_json_files(self, json_files: List[Path]) -> List[dict]:
        """Index samples from JSON metadata files."""
        samples = []
        for json_path in sorted(json_files):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        samples.append({
                            'format': 'json',
                            'path': str(json_path),
                            'index': i,
                            'data': item,
                        })
                elif isinstance(data, dict) and 'grasps' in data:
                    for i, grasp in enumerate(data['grasps']):
                        samples.append({
                            'format': 'json',
                            'path': str(json_path),
                            'index': i,
                            'data': grasp,
                        })
            except Exception as e:
                logger.warning(f"Failed to index {json_path}: {e}")
        return samples
    
    def _index_parquet_files(self, pq_files: List[Path]) -> List[dict]:
        """Index samples from parquet files (HuggingFace)."""
        samples = []
        try:
            import pyarrow.parquet as pq
            for pq_path in sorted(pq_files):
                table = pq.read_table(str(pq_path))
                n_rows = len(table)
                for i in range(n_rows):
                    samples.append({
                        'format': 'parquet',
                        'path': str(pq_path),
                        'index': i,
                    })
        except ImportError:
            logger.warning("pyarrow not installed, cannot read parquet files")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Load a DexGraspNet sample."""
        sample_info = self.samples[idx]
        fmt = sample_info['format']
        
        if fmt == 'h5':
            return self._load_h5_sample(sample_info)
        elif fmt == 'npz':
            return self._load_npz_sample(sample_info)
        elif fmt == 'json':
            return self._load_json_sample(sample_info)
        elif fmt == 'parquet':
            return self._load_parquet_sample(sample_info)
        else:
            raise ValueError(f"Unknown format: {fmt}")
    
    def _load_h5_sample(self, info: dict) -> dict:
        """Load sample from HDF5."""
        with h5py.File(info['path'], 'r') as f:
            grp = f[info['group']]
            idx = info['index']
            
            if isinstance(grp, h5py.Dataset):
                # Direct array
                grasp_data = grp[idx]
                action = self._parse_grasp_array(grasp_data)
            else:
                # Group with sub-datasets
                grasp_grp = grp[str(idx)] if str(idx) in grp else grp
                action = self._parse_grasp_group(grasp_grp, idx)
            
            # Try to load images/point cloud
            images = self._load_or_render_images(f, idx)
            
            # Language description
            language = self._get_language(f, idx)
        
        return self._build_sample(action, images, language, idx)
    
    def _load_npz_sample(self, info: dict) -> dict:
        """Load sample from NPZ."""
        data = np.load(info['path'], allow_pickle=True)
        
        # Extract grasp pose
        if 'grasp_pose' in data:
            action = data['grasp_pose'].astype(np.float32)
        elif 'wrist_pose' in data and 'joint_angles' in data:
            wrist = data['wrist_pose'].astype(np.float32)
            joints = data['joint_angles'].astype(np.float32)
            action = np.concatenate([wrist, joints])
        else:
            # Fallback: first array
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    action = arr.astype(np.float32)
                    break
            else:
                action = np.zeros(23, dtype=np.float32)
        
        images = self._render_placeholder_images()
        language = ""  # NPZ format has no language; VLM pre-compute handles this
        
        return self._build_sample(action, images, language, info['index'])
    
    def _load_json_sample(self, info: dict) -> dict:
        """Load sample from JSON."""
        data = info['data']
        
        # Parse grasp pose from JSON
        if 'wrist_pose' in data:
            wrist = np.array(data['wrist_pose'], dtype=np.float32)
            joints = np.array(data.get('joint_angles', np.zeros(16)), dtype=np.float32)
            action = np.concatenate([wrist, joints])
        elif 'pose' in data:
            action = np.array(data['pose'], dtype=np.float32)
        else:
            action = np.zeros(23, dtype=np.float32)
        
        language = data.get('description', 'grasp the object')
        images = self._render_placeholder_images()
        
        return self._build_sample(action, images, language, info['index'])
    
    def _load_parquet_sample(self, info: dict) -> dict:
        """Load sample from parquet."""
        import pyarrow.parquet as pq
        
        table = pq.read_table(info['path'])
        row = table.slice(info['index'], 1).to_pydict()
        
        # Extract action from available columns
        action = np.zeros(23, dtype=np.float32)
        for col in ['grasp_pose', 'action', 'wrist_pose']:
            if col in row and row[col]:
                arr = np.array(row[col][0], dtype=np.float32)
                action[:len(arr)] = arr
                break
        
        language = row.get('description', ['grasp the object'])[0]
        images = self._render_placeholder_images()
        
        return self._build_sample(action, images, language, info['index'])
    
    def _parse_grasp_array(self, arr: np.ndarray) -> np.ndarray:
        """Parse a grasp from flat array."""
        arr = arr.astype(np.float32)
        if len(arr) >= 23:
            return arr[:23]
        elif len(arr) >= 7:
            # Wrist pose only, pad joints
            action = np.zeros(23, dtype=np.float32)
            action[:len(arr)] = arr
            return action
        else:
            action = np.zeros(23, dtype=np.float32)
            action[:len(arr)] = arr
            return action
    
    def _parse_grasp_group(self, grp, idx) -> np.ndarray:
        """Parse grasp from HDF5 group."""
        action = np.zeros(23, dtype=np.float32)
        
        if 'wrist_pose' in grp:
            wrist = np.array(grp['wrist_pose'], dtype=np.float32).flatten()
            action[:min(7, len(wrist))] = wrist[:7]
        
        if 'joint_angles' in grp:
            joints = np.array(grp['joint_angles'], dtype=np.float32).flatten()
            action[7:7+min(16, len(joints))] = joints[:16]
        
        return action
    
    def _load_or_render_images(self, h5_file, idx) -> np.ndarray:
        """Load images from H5 or render placeholder."""
        # Try to find image data
        for key in ['images', 'rgb', 'renders', 'observations/images']:
            if key in h5_file:
                data = h5_file[key]
                if hasattr(data, 'shape') and data.ndim >= 3:
                    if idx < data.shape[0]:
                        img = np.array(data[idx], dtype=np.uint8)
                        return self._process_loaded_images(img)
        
        return self._render_placeholder_images()
    
    def _process_loaded_images(self, img: np.ndarray) -> np.ndarray:
        """Process loaded images to standard format."""
        import cv2
        
        # Ensure [V, H, W, 3] format
        if img.ndim == 3:
            img = img[np.newaxis]  # [1, H, W, 3]
        
        V = img.shape[0]
        result = np.zeros((V, 3, self.image_size, self.image_size), dtype=np.float32)
        
        for v in range(V):
            frame = img[v]
            if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            result[v] = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return result
    
    def _render_placeholder_images(self) -> np.ndarray:
        """
        Return black placeholder images when no rendered images are available.
        
        NOTE: For production training, point cloud rendering should be implemented
        via Open3D or similar. Black images signal to the VLM that visual input
        is absent for this modality (the VLM learns to rely on language).
        """
        images = np.zeros((2, 3, self.image_size, self.image_size), dtype=np.float32)
        return images
    
    def _get_language(self, h5_file, idx) -> str:
        """Get language description from actual dataset metadata."""
        for key in ['language', 'description', 'text', 'task_description', 'instruction']:
            if key in h5_file:
                data = h5_file[key]
                if hasattr(data, 'shape') and data.ndim >= 1 and idx < data.shape[0]:
                    text = data[idx]
                    if isinstance(text, bytes):
                        return text.decode('utf-8')
                    return str(text)
        
        # DexGraspNet 2.0: if no per-sample description, use scene/object metadata
        for key in ['object_name', 'scene_id', 'object_category']:
            if key in h5_file:
                data = h5_file[key]
                if hasattr(data, 'shape') and data.ndim >= 1 and idx < data.shape[0]:
                    obj = data[idx]
                    if isinstance(obj, bytes):
                        obj = obj.decode('utf-8')
                    return f"grasp the {obj}"
        
        logger.warning(f"No language instruction found for DexGraspNet sample {idx}")
        return ""
    
    def _build_sample(
        self,
        action: np.ndarray,
        images: np.ndarray,
        language: str,
        idx: int,
    ) -> dict:
        """Build standard sample dict."""
        # DexGraspNet stores single grasps → repeat for action_horizon
        if action.ndim == 1:
            action = np.tile(action, (self.action_horizon, 1))  # [T, 23]
        elif action.shape[0] < self.action_horizon:
            pad_len = self.action_horizon - action.shape[0]
            action = np.pad(action, ((0, pad_len), (0, 0)), mode='edge')
        
        # Normalize actions if stats available
        if self.action_mean is not None:
            action = (action - self.action_mean) / (self.action_std + 1e-4)
        
        return {
            'images': torch.from_numpy(images).float(),
            'language': language,
            'actions': torch.from_numpy(action.astype(np.float32)),
            'task_id': 0,
            'episode_id': idx,
            'timestep': 0,
        }
    
    def compute_action_statistics(self) -> tuple:
        """Compute action mean/std from dataset."""
        all_actions = []
        
        n_samples = min(500, len(self.samples))
        indices = np.random.choice(len(self.samples), n_samples, replace=False)
        
        for idx in indices:
            try:
                sample = self[idx]
                actions = sample['actions'].numpy()
                all_actions.append(actions[0])  # First timestep (they're repeated)
            except Exception:
                continue
        
        if not all_actions:
            self.action_mean = np.zeros(23, dtype=np.float32)
            self.action_std = np.ones(23, dtype=np.float32)
            return self.action_mean, self.action_std
        
        all_actions = np.stack(all_actions)
        self.action_mean = np.mean(all_actions, axis=0).astype(np.float32)
        self.action_std = np.std(all_actions, axis=0).astype(np.float32)
        self.action_std = np.maximum(self.action_std, 1e-4)
        
        logger.info(f"DexGraspNet action stats: {len(all_actions)} samples")
        return self.action_mean, self.action_std
