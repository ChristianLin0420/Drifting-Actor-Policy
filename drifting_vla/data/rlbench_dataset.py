"""
RLBench Dataset
===============

Dataset class for loading RLBench demonstrations.
Supports both HDF5 format and PerAct format (from Hugging Face dataset).

PerAct format (hqfang/rlbench-18-tasks):
    train/task_name/all_variations/episodes/episode{N}/
        - front_rgb/*.png
        - wrist_rgb/*.png  
        - low_dim_obs.pkl
        - variation_descriptions.pkl
"""

import pickle
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Union
import logging

from drifting_vla.data.base_dataset import BaseVLADataset

logger = logging.getLogger(__name__)


class RLBenchDataset(BaseVLADataset):
    """
    Dataset for RLBench demonstrations.
    
    Supports two formats:
    1. HDF5 format (custom collected)
    2. PerAct format (from Hugging Face: hqfang/rlbench-18-tasks)
    
    Args:
        data_dir: Directory containing dataset (train/val/test splits).
        split: Which split to use ('train', 'val', 'test').
        tasks: List of tasks to load (None = all).
        action_horizon: Number of action steps to predict.
        image_size: Image resolution (resized if different).
        cameras: List of camera names to use.
        use_depth: Whether to load depth images.
        transform: Optional data transform.
        max_episodes_per_task: Maximum episodes per task (for debugging).
    
    Example:
        >>> dataset = RLBenchDataset(
        ...     data_dir='./data/rlbench',
        ...     split='train',
        ...     tasks=['stack_blocks', 'turn_tap'],
        ...     action_horizon=16,
        ... )
        >>> sample = dataset[0]
        >>> print(sample['images'].shape)  # [3, 224, 224]
    """
    
    # Camera name mappings between formats
    CAMERA_MAP = {
        'front': 'front_rgb',
        'wrist': 'wrist_rgb', 
        'left_shoulder': 'left_shoulder_rgb',
        'right_shoulder': 'right_shoulder_rgb',
        'overhead': 'overhead_rgb',
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        tasks: Optional[list[str]] = None,
        action_horizon: int = 16,
        image_size: int = 224,
        cameras: list[str] = ['front'],
        use_depth: bool = False,
        transform: Optional[callable] = None,
        max_episodes_per_task: Optional[int] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            action_horizon=action_horizon,
            image_size=image_size,
            use_depth=use_depth,
            transform=transform,
        )
        
        self.split = split
        self.cameras = cameras
        self.max_episodes_per_task = max_episodes_per_task
        
        # Determine data root
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        
        if not self.split_dir.exists():
            # Fallback: data_dir is the split directory itself
            self.split_dir = self.data_dir
        
        # Detect format and discover tasks
        self.format = self._detect_format()
        self.available_tasks = self._discover_tasks()
        
        # Filter to requested tasks
        if tasks is not None:
            self.tasks = [t for t in tasks if t in self.available_tasks]
        else:
            self.tasks = list(self.available_tasks.keys())
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        logger.info(
            f"Initialized RLBench dataset ({self.format} format): "
            f"{len(self.tasks)} tasks, {len(self.samples)} samples"
        )
    
    def _detect_format(self) -> str:
        """Detect dataset format (peract or hdf5)."""
        # Check for PerAct format indicators
        for task_dir in self.split_dir.iterdir():
            if task_dir.is_dir():
                variations_dir = task_dir / 'all_variations'
                if variations_dir.exists():
                    return 'peract'
                # Check for HDF5 files
                if list(task_dir.glob('*.hdf5')) or list(task_dir.glob('*.h5')):
                    return 'hdf5'
        
        # Check root for HDF5
        if list(self.split_dir.glob('*.hdf5')):
            return 'hdf5'
        
        # Default to peract
        return 'peract'
    
    def _discover_tasks(self) -> dict[str, Path]:
        """Discover available tasks from directory structure."""
        tasks = {}
        
        if self.format == 'peract':
            # PerAct format: split/task_name/all_variations/episodes/
            for path in self.split_dir.iterdir():
                if path.is_dir():
                    episodes_dir = path / 'all_variations' / 'episodes'
                    if episodes_dir.exists():
                        tasks[path.name] = episodes_dir
        else:
            # HDF5 format
            for path in self.split_dir.iterdir():
                if path.is_dir():
                    hdf5_files = list(path.glob('*.hdf5')) + list(path.glob('*.h5'))
                    if hdf5_files:
                        tasks[path.name] = path
            
            # Also check for single HDF5 files in root
            for hdf5_file in self.split_dir.glob('*.hdf5'):
                task_name = hdf5_file.stem
                if task_name not in tasks:
                    tasks[task_name] = hdf5_file
        
        return tasks
    
    def _build_sample_index(self) -> list[dict]:
        """Build index of all samples across tasks."""
        if self.format == 'peract':
            return self._build_peract_index()
        else:
            return self._build_hdf5_index()
    
    def _build_peract_index(self) -> list[dict]:
        """Build sample index for PerAct format."""
        samples = []
        task_id = 0
        
        for task_name in self.tasks:
            episodes_dir = self.available_tasks[task_name]
            episode_dirs = sorted(
                [d for d in episodes_dir.iterdir() if d.is_dir()],
                key=lambda x: int(x.name.replace('episode', ''))
            )
            
            for ep_idx, ep_dir in enumerate(episode_dirs):
                if (self.max_episodes_per_task and 
                    ep_idx >= self.max_episodes_per_task):
                    break
                
                # Load low_dim_obs to get episode length
                low_dim_path = ep_dir / 'low_dim_obs.pkl'
                if not low_dim_path.exists():
                    continue
                
                try:
                    with open(low_dim_path, 'rb') as f:
                        low_dim_obs = pickle.load(f)
                    
                    ep_length = len(low_dim_obs)
                    
                    # Get language description
                    desc_path = ep_dir / 'variation_descriptions.pkl'
                    if desc_path.exists():
                        with open(desc_path, 'rb') as f:
                            descriptions = pickle.load(f)
                        language = descriptions[0] if descriptions else task_name.replace('_', ' ')
                    else:
                        language = task_name.replace('_', ' ')
                    
                    # Create samples with sliding window
                    for t in range(0, max(1, ep_length - self.action_horizon + 1)):
                        samples.append({
                            'format': 'peract',
                            'episode_dir': str(ep_dir),
                            'timestep': t,
                            'task_name': task_name,
                            'task_id': task_id,
                            'language': language,
                            'episode_length': ep_length,
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to index {ep_dir}: {e}")
            
            task_id += 1
        
        return samples
    
    def _build_hdf5_index(self) -> list[dict]:
        """Build sample index for HDF5 format."""
        samples = []
        task_id = 0
        
        for task_name in self.tasks:
            task_path = self.available_tasks[task_name]
            
            if task_path.is_file():
                samples.extend(
                    self._index_hdf5_file(task_path, task_name, task_id)
                )
            else:
                for hdf5_path in sorted(task_path.glob('*.hdf5')):
                    samples.extend(
                        self._index_hdf5_file(hdf5_path, task_name, task_id)
                    )
            
            task_id += 1
        
        return samples
    
    def _index_hdf5_file(
        self,
        hdf5_path: Path,
        task_name: str,
        task_id: int,
    ) -> list[dict]:
        """Index samples from a single HDF5 file."""
        samples = []
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                episode_keys = [k for k in f.keys() if k.startswith('episode')]
                
                for ep_idx, ep_key in enumerate(sorted(episode_keys)):
                    if (self.max_episodes_per_task and 
                        ep_idx >= self.max_episodes_per_task):
                        break
                    
                    episode = f[ep_key]
                    
                    if 'actions' in episode:
                        ep_length = episode['actions'].shape[0]
                    else:
                        continue
                    
                    if 'language' in episode:
                        language = episode['language'][()]
                        if isinstance(language, bytes):
                            language = language.decode('utf-8')
                    else:
                        language = task_name.replace('_', ' ')
                    
                    for t in range(0, max(1, ep_length - self.action_horizon + 1)):
                        samples.append({
                            'format': 'hdf5',
                            'hdf5_path': str(hdf5_path),
                            'episode_key': ep_key,
                            'timestep': t,
                            'task_name': task_name,
                            'task_id': task_id,
                            'language': language,
                            'episode_length': ep_length,
                        })
        
        except Exception as e:
            logger.warning(f"Failed to index {hdf5_path}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Load a sample."""
        sample_info = self.samples[idx]
        
        if sample_info.get('format') == 'peract':
            return self._load_peract_sample(sample_info, idx)
        else:
            return self._load_hdf5_sample(sample_info, idx)
    
    def _load_peract_sample(self, sample_info: dict, idx: int) -> dict:
        """Load a sample from PerAct format."""
        import cv2
        
        ep_dir = Path(sample_info['episode_dir'])
        t = sample_info['timestep']
        ep_length = sample_info['episode_length']
        T = min(t + self.action_horizon, ep_length)
        
        # Load low_dim_obs for actions and proprio
        with open(ep_dir / 'low_dim_obs.pkl', 'rb') as f:
            low_dim_obs = pickle.load(f)
        
        # Extract actions (gripper pose delta or joint velocities)
        actions = []
        for i in range(t, T):
            obs = low_dim_obs[i]
            # PerAct stores gripper pose as action
            # gripper_pose: [x, y, z, qx, qy, qz, qw]
            # gripper_open: 0 or 1
            action = np.concatenate([
                obs.gripper_pose,
                [obs.gripper_open]
            ])
            actions.append(action)
        
        actions = np.array(actions, dtype=np.float32)
        
        # Pad actions if needed
        if actions.shape[0] < self.action_horizon:
            pad_length = self.action_horizon - actions.shape[0]
            actions = np.pad(
                actions,
                ((0, pad_length), (0, 0)),
                mode='edge'
            )
        
        # Load images
        images = []
        for camera in self.cameras:
            camera_dir = self.CAMERA_MAP.get(camera, f'{camera}_rgb')
            img_dir = ep_dir / camera_dir
            
            if img_dir.exists():
                img_path = img_dir / f'{t}.png'
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Resize if needed
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Convert to CHW and normalize
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(img)
        
        if len(images) == 1:
            images = images[0]
        else:
            images = np.stack(images, axis=0)
        
        # Extract proprioception
        obs = low_dim_obs[t]
        proprio = np.concatenate([
            obs.joint_positions,
            obs.gripper_pose,
            [obs.gripper_open]
        ]).astype(np.float32)
        
        # Normalize actions
        if self.action_mean is not None:
            actions = self.normalize_actions(actions)
        
        sample = {
            'images': torch.from_numpy(images),
            'language': sample_info['language'],
            'actions': torch.from_numpy(actions),
            'task_id': sample_info['task_id'],
            'episode_id': idx,
            'timestep': t,
            'proprio': torch.from_numpy(proprio),
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_hdf5_sample(self, sample_info: dict, idx: int) -> dict:
        """Load a sample from HDF5 format."""
        with h5py.File(sample_info['hdf5_path'], 'r') as f:
            episode = f[sample_info['episode_key']]
            t = sample_info['timestep']
            T = min(t + self.action_horizon, sample_info['episode_length'])
            
            images = self._load_hdf5_images(episode, t)
            
            actions = episode['actions'][t:T]
            actions = np.array(actions, dtype=np.float32)
            
            if actions.shape[0] < self.action_horizon:
                pad_length = self.action_horizon - actions.shape[0]
                actions = np.pad(
                    actions,
                    ((0, pad_length), (0, 0)),
                    mode='edge'
                )
            
            if 'proprio' in episode.get('observations', episode):
                proprio = episode['observations/proprio'][t]
                proprio = np.array(proprio, dtype=np.float32)
            else:
                proprio = None
        
        if self.action_mean is not None:
            actions = self.normalize_actions(actions)
        
        sample = {
            'images': torch.from_numpy(images),
            'language': sample_info['language'],
            'actions': torch.from_numpy(actions),
            'task_id': sample_info['task_id'],
            'episode_id': idx,
            'timestep': t,
        }
        
        if proprio is not None:
            sample['proprio'] = torch.from_numpy(proprio)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_hdf5_images(self, episode, timestep: int) -> np.ndarray:
        """Load and preprocess images from HDF5 episode."""
        import cv2
        
        images = []
        
        for camera in self.cameras:
            for path in [
                f'observations/images/{camera}',
                f'observations/{camera}_rgb',
                f'{camera}',
            ]:
                if path in episode:
                    img = episode[path][timestep]
                    break
            else:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            img = np.array(img, dtype=np.uint8)
            
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))
            
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(img)
        
        if len(images) == 1:
            return images[0]
        else:
            return np.stack(images, axis=0)
    
    def get_task_names(self) -> list[str]:
        """Return list of task names."""
        return self.tasks
    
    def get_task_language_map(self) -> dict[str, list[str]]:
        """Get mapping from task names to language descriptions."""
        task_langs = {}
        
        for sample in self.samples:
            task = sample['task_name']
            lang = sample['language']
            
            if task not in task_langs:
                task_langs[task] = []
            if lang not in task_langs[task]:
                task_langs[task].append(lang)
        
        return task_langs


