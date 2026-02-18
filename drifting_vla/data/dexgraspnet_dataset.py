"""
DexGraspNet 2.0 Dataset
========================

Loader for the DexGraspNet 2.0 dexterous grasping dataset.
Source: https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0

Data layout (after extraction):
  data/dexgraspnet/
    scenes/scene_XXXX/realsense/
      depth_gt/YYYY.png        — 16-bit depth (1280×720)
      label_gt/YYYY.png        — segmentation (uint8 object IDs)
      annotations/YYYY.xml     — object names + poses
      YYYY.npz                 — dexterous hand grasps for viewpoint YYYY
    gripper_grasps/scene_XXXX/realsense/
      points.npy               — [N, 3] contact points
      poses.npy                — [N, 17] gripper grasp poses

NPZ grasp format (dexterous hand):
  translation: [K, 3]           — wrist xyz
  rotation:    [K, 3, 3]        — wrist rotation matrix
  j0..j15:     [K] each         — 16 finger joint angles
  point:       [K, 3]           — contact points

Action format: [wrist_xyz(3) + quaternion(4) + finger_joints(16)] = 23-dim
"""

import torch
import numpy as np
import h5py
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def _depth_to_rgb(depth: np.ndarray, colormap='turbo') -> np.ndarray:
    """Convert 16-bit depth to 3-channel pseudo-RGB using colormap.
    
    Returns: [H, W, 3] uint8 array.
    """
    import cv2
    d = depth.astype(np.float32)
    if d.max() > d.min():
        d = (d - d.min()) / (d.max() - d.min())
    else:
        d = np.zeros_like(d)
    d_uint8 = (d * 255).clip(0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(d_uint8, cv2.COLORMAP_TURBO)  # [H, W, 3] BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)


class DexGraspNetDataset:
    """
    DexGraspNet 2.0 dataset for dexterous grasping.
    
    Loads scene depth images + dexterous grasp poses + object names.
    
    Each sample provides:
      images:   [V, 3, H, W]  — depth pseudo-RGB (V=1 or 2 for depth+label)
      actions:  [T, 23]       — wrist pose (7) + finger joints (16)
      language: str            — "grasp the {object_name}"
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
        
        # Discover samples
        self.samples = self._build_index()
        
        # Action statistics
        self.action_mean = None
        self.action_std = None
        
        logger.info(f"DexGraspNet dataset: {len(self.samples)} samples from {self.data_dir}")
    
    def _build_index(self) -> List[dict]:
        """Build sample index from scene directories and grasp files."""
        samples = []
        
        # --- Strategy 1: Scene NPZ grasps (dexterous hand, with depth images) ---
        scenes_dir = self.data_dir / 'scenes'
        if scenes_dir.exists():
            samples.extend(self._index_scene_npz_grasps(scenes_dir))
        
        # --- Strategy 2: HDF5 files (pre-processed format) ---
        if not samples:
            h5_files = list(self.data_dir.rglob('*.h5')) + list(self.data_dir.rglob('*.hdf5'))
        if h5_files:
                samples.extend(self._index_h5_files(h5_files))
        
        # --- Strategy 3: Gripper grasps (simpler format, no images) ---
        if not samples:
            gripper_dir = self.data_dir / 'gripper_grasps'
            if gripper_dir.exists():
                samples.extend(self._index_gripper_grasps(gripper_dir))
        
        # --- Strategy 4: Loose NPZ/parquet files ---
        if not samples:
            npz_files = list(self.data_dir.rglob('*.npz'))
            pq_files = list(self.data_dir.rglob('*.parquet'))
            if npz_files:
                for p in sorted(npz_files):
                    samples.append({'format': 'npz', 'path': str(p), 'index': 0})
            elif pq_files:
                samples.extend(self._index_parquet_files(pq_files))
        
        if not samples:
            logger.warning(f"No DexGraspNet data found in {self.data_dir}")
        
        if self.max_samples and len(samples) > self.max_samples:
            samples = samples[:self.max_samples]
        
        return samples
    
    def _index_scene_npz_grasps(self, scenes_dir: Path) -> List[dict]:
        """Index dexterous hand grasps from scene NPZ files + depth images."""
        samples = []
        
        for scene_dir in sorted(scenes_dir.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.startswith('scene_'):
                continue
            
            rs_dir = scene_dir / 'realsense'
            if not rs_dir.exists():
                continue
            
            # Find NPZ files (each = grasps for one viewpoint)
            npz_files = sorted(rs_dir.glob('*.npz'))
            
            for npz_path in npz_files:
                view_id = npz_path.stem  # e.g., "046"
                
                # Check for matching depth image
                depth_path = rs_dir / 'depth_gt' / f'{view_id.zfill(4)}.png'
                label_path = rs_dir / 'label_gt' / f'{view_id.zfill(4)}.png'
                annot_path = rs_dir / 'annotations' / f'{view_id.zfill(4)}.xml'
                
                try:
                    data = np.load(str(npz_path), allow_pickle=True)
                    n_grasps = data['translation'].shape[0] if 'translation' in data else 0
                except Exception:
                    continue
                
                for g in range(n_grasps):
                    samples.append({
                        'format': 'scene_npz',
                        'npz_path': str(npz_path),
                        'grasp_idx': g,
                        'depth_path': str(depth_path) if depth_path.exists() else None,
                        'label_path': str(label_path) if label_path.exists() else None,
                        'annot_path': str(annot_path) if annot_path.exists() else None,
                        'scene_name': scene_dir.name,
                    })
        
        logger.info(f"  Found {len(samples)} scene NPZ grasps from {scenes_dir}")
        return samples
    
    def _index_gripper_grasps(self, gripper_dir: Path) -> List[dict]:
        """Index gripper grasps (simpler 17-dim format)."""
        samples = []
        
        for scene_dir in sorted(gripper_dir.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.startswith('scene_'):
                continue
            
            rs_dir = scene_dir / 'realsense'
            if not rs_dir.exists():
                continue
            
            poses_path = rs_dir / 'poses.npy'
            points_path = rs_dir / 'points.npy'
            
            if not poses_path.exists():
                continue
            
            try:
                poses = np.load(str(poses_path))
                n_grasps = poses.shape[0]
            except Exception:
                continue
            
            # Check for scene depth images in the scenes/ directory
            scene_name = scene_dir.name
            scene_imgs_dir = self.data_dir / 'scenes' / scene_name / 'realsense'
            
            for g in range(min(n_grasps, 100)):  # Limit per scene
                samples.append({
                    'format': 'gripper',
                    'poses_path': str(poses_path),
                    'points_path': str(points_path) if points_path.exists() else None,
                    'grasp_idx': g,
                    'scene_name': scene_name,
                    'scene_imgs_dir': str(scene_imgs_dir) if scene_imgs_dir.exists() else None,
                })
        
        logger.info(f"  Found {len(samples)} gripper grasps from {gripper_dir}")
        return samples
    
    def _index_h5_files(self, h5_files: List[Path]) -> List[dict]:
        """Index samples from HDF5 files."""
        samples = []
        for h5_path in sorted(h5_files):
            try:
                with h5py.File(str(h5_path), 'r') as f:
                    for key in ['grasp_poses', 'grasps']:
                        if key in f:
                                n = f[key].shape[0]
                                for i in range(n):
                                    samples.append({
                                    'format': 'h5', 'path': str(h5_path),
                                    'index': i, 'group': key,
                                    })
                                break
            except Exception as e:
                logger.warning(f"Failed to index {h5_path}: {e}")
        return samples
    
    def _index_parquet_files(self, pq_files: List[Path]) -> List[dict]:
        """Index samples from parquet files."""
        samples = []
        try:
            import pyarrow.parquet as pq
            for pq_path in sorted(pq_files):
                table = pq.read_table(str(pq_path))
                for i in range(len(table)):
                    samples.append({'format': 'parquet', 'path': str(pq_path), 'index': i})
        except ImportError:
            pass
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Load a DexGraspNet sample."""
        info = self.samples[idx]
        fmt = info['format']
        
        if fmt == 'scene_npz':
            return self._load_scene_npz(info)
        elif fmt == 'gripper':
            return self._load_gripper(info)
        elif fmt == 'h5':
            return self._load_h5(info)
        else:
            return self._empty_sample()
    
    # ──────────────────────────────────────────────────────────────
    # Loaders
    # ──────────────────────────────────────────────────────────────
    
    def _load_scene_npz(self, info: dict) -> dict:
        """Load dexterous hand grasp from scene NPZ + rendered RGB / depth images."""
        import cv2
        
        g = info['grasp_idx']
        
        # --- Action: wrist xyz(3) + quat(4) + finger joints(16) = 23 ---
        try:
            data = np.load(info['npz_path'], allow_pickle=True)
            trans = data['translation'][g]  # [3]
            rot = data['rotation'][g]       # [3, 3]
            quat = _rotation_matrix_to_quaternion(rot)  # [4]
            
            joints = np.zeros(16, dtype=np.float32)
            for j in range(16):
                key = f'j{j}'
                if key in data:
                    joints[j] = float(data[key][g])
            
            action = np.concatenate([trans, quat, joints]).astype(np.float32)
        except Exception:
            action = np.zeros(23, dtype=np.float32)
        
        # --- Images: prefer rendered RGB (8 views), fallback to depth ---
        images = self._load_rendered_views(info['scene_name'])
        
        if len(images) == 0:
            # Fallback: depth pseudo-RGB + segmentation
            images = []
            if info['depth_path'] and Path(info['depth_path']).exists():
                try:
                    from PIL import Image
                    depth_pil = Image.open(info['depth_path'])
                    depth_arr = np.array(depth_pil)
                    rgb = _depth_to_rgb(depth_arr)
                    rgb = cv2.resize(rgb, (self.image_size, self.image_size))
                    images.append(rgb.transpose(2, 0, 1).astype(np.float32))
                except Exception:
                    pass
            
            if info['label_path'] and Path(info['label_path']).exists():
                try:
                    from PIL import Image
                    label_pil = Image.open(info['label_path'])
                    label_arr = np.array(label_pil).astype(np.float32)
                    label_rgb = np.stack([
                        ((label_arr * 37) % 256),
                        ((label_arr * 91) % 256),
                        ((label_arr * 159) % 256),
                    ], axis=-1).astype(np.uint8)
                    label_rgb = cv2.resize(label_rgb, (self.image_size, self.image_size))
                    images.append(label_rgb.transpose(2, 0, 1).astype(np.float32))
                except Exception:
                    pass
        
        if not images:
            images = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)]
        
        if isinstance(images, list):
            images = np.stack(images, axis=0)  # [V, 3, H, W]
        
        # --- Language: from XML annotation ---
        language = self._parse_annotation_xml(info.get('annot_path'))
        
        return self._build_sample(action, images, language, info['grasp_idx'])
    
    def _load_rendered_views(self, scene_name: str) -> list:
        """Load pre-rendered 8-view RGB images for a scene."""
        import cv2
        
        rendered_dir = self.data_dir / 'rendered' / scene_name
        if not rendered_dir.exists():
            return []
        
        view_files = sorted(rendered_dir.glob('view_*.png'))
        if not view_files:
            return []
        
        images = []
        for vf in view_files:
            try:
                from PIL import Image
                img = np.array(Image.open(str(vf)).convert('RGB'))  # [H, W, 3]
                if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                    img = cv2.resize(img, (self.image_size, self.image_size))
                images.append(img.transpose(2, 0, 1).astype(np.float32))  # [3, H, W]
            except Exception:
                continue
        
        return images
    
    def _load_gripper(self, info: dict) -> dict:
        """Load gripper grasp."""
        import cv2
        
        g = info['grasp_idx']
        
        try:
            poses = np.load(info['poses_path'])
            pose = poses[g].astype(np.float32)  # [17]
            # Map 17-dim gripper to 23-dim (pad finger joints with zeros)
            action = np.zeros(23, dtype=np.float32)
            action[:min(17, len(pose))] = pose[:17]
        except Exception:
        action = np.zeros(23, dtype=np.float32)
        
        # Try to load scene images
        images = self._load_scene_images(info.get('scene_imgs_dir'))
        language = f"grasp object in {info['scene_name']}"
        
        return self._build_sample(action, images, language, info['grasp_idx'])
    
    def _load_h5(self, info: dict) -> dict:
        """Load from HDF5."""
        with h5py.File(info['path'], 'r') as f:
            grp = f[info['group']]
            i = info['index']
            arr = grp[i].astype(np.float32)
            action = np.zeros(23, dtype=np.float32)
            action[:min(23, len(arr))] = arr[:23]
            
            images = self._load_h5_images(f, i)
            language = self._get_h5_language(f, i)
        
        return self._build_sample(action, images, language, info['index'])
    
    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    
    def _parse_annotation_xml(self, annot_path: Optional[str]) -> str:
        """Parse object names from annotation XML → language instruction."""
        if not annot_path or not Path(annot_path).exists():
            return "grasp the object"
        
        try:
            tree = ET.parse(annot_path)
            root = tree.getroot()
            obj_names = []
            for obj in root.findall('.//obj'):
                name_elem = obj.find('obj_name')
                if name_elem is not None and name_elem.text:
                    # Clean up: "015_peach.ply" → "peach"
                    name = name_elem.text.replace('.ply', '')
                    parts = name.split('_')
                    if len(parts) > 1 and parts[0].isdigit():
                        name = '_'.join(parts[1:])
                    name = name.replace('_', ' ')
                    obj_names.append(name)
            
            if obj_names:
                # Pick a random object for the instruction
                obj = obj_names[np.random.randint(len(obj_names))]
                return f"grasp the {obj}"
            return "grasp the object on the table"
        except Exception:
            return "grasp the object"
    
    def _load_scene_images(self, scene_imgs_dir: Optional[str]) -> np.ndarray:
        """Load depth images from scene directory."""
        import cv2
        
        if not scene_imgs_dir or not Path(scene_imgs_dir).exists():
            return np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
        
        depth_dir = Path(scene_imgs_dir) / 'depth_gt'
        if not depth_dir.exists():
            return np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
        
        # Pick first available depth image
        depth_files = sorted(depth_dir.glob('*.png'))
        if not depth_files:
            return np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
        
        try:
            from PIL import Image
            depth_pil = Image.open(str(depth_files[0]))
            depth_arr = np.array(depth_pil)
            rgb = _depth_to_rgb(depth_arr)
            rgb = cv2.resize(rgb, (self.image_size, self.image_size))
            img = rgb.transpose(2, 0, 1).astype(np.float32)
            return img[np.newaxis]  # [1, 3, H, W]
        except Exception:
            return np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
    
    def _load_h5_images(self, h5_file, idx) -> np.ndarray:
        """Load images from HDF5."""
        for key in ['images', 'rgb', 'renders', 'observations/images']:
            if key in h5_file:
                data = h5_file[key]
                if hasattr(data, 'shape') and data.ndim >= 3 and idx < data.shape[0]:
                    import cv2
                        img = np.array(data[idx], dtype=np.uint8)
        if img.ndim == 3:
                        img = img[np.newaxis]
        V = img.shape[0]
        result = np.zeros((V, 3, self.image_size, self.image_size), dtype=np.float32)
        for v in range(V):
            frame = img[v]
            if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            result[v] = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        return result
    
        return np.zeros((1, 3, self.image_size, self.image_size), dtype=np.float32)
    
    def _get_h5_language(self, h5_file, idx) -> str:
        """Get language from HDF5."""
        for key in ['descriptions', 'language', 'text', 'description']:
            if key in h5_file:
                data = h5_file[key]
                if hasattr(data, 'shape') and data.ndim >= 1 and idx < data.shape[0]:
                    text = data[idx]
                    if isinstance(text, bytes):
                        return text.decode('utf-8')
                    return str(text)
        return "grasp the object"
    
    def _build_sample(self, action: np.ndarray, images: np.ndarray,
                      language: str, idx: int) -> dict:
        """Build standard sample dict."""
        if action.ndim == 1:
            action = np.tile(action, (self.action_horizon, 1))
        elif action.shape[0] < self.action_horizon:
            action = np.pad(action, ((0, self.action_horizon - action.shape[0]), (0, 0)), mode='edge')
        
        return {
            'images': torch.from_numpy(images).float(),
            'language': language,
            'actions': torch.from_numpy(action.astype(np.float32)),
            'proprio': torch.zeros(128),
            'task_id': 0,
            'episode_id': idx,
            'timestep': 0,
            'num_views': images.shape[0],
            'num_frames': 1,
        }
    
    def _empty_sample(self) -> dict:
        return {
            'images': torch.zeros(1, 3, self.image_size, self.image_size),
            'language': '',
            'actions': torch.zeros(self.action_horizon, 23),
            'proprio': torch.zeros(128),
            'task_id': 0,
            'episode_id': 0,
            'timestep': 0,
            'num_views': 1,
            'num_frames': 1,
        }
    
    def compute_action_statistics(self) -> tuple:
        """Compute action mean/std from dataset."""
        all_actions = []
        n = min(500, len(self.samples))
        indices = np.random.choice(len(self.samples), n, replace=False) if n < len(self.samples) else range(n)
        
        for idx in indices:
            try:
                sample = self[idx]
                all_actions.append(sample['actions'].numpy()[0])
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
        
        logger.info(f"DexGraspNet action stats: {len(all_actions)} samples, dim=23")
        return self.action_mean, self.action_std
