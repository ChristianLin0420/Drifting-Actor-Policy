"""
Episode-Centric HDF5 Dataset
==============================

Main data loader for Drifting-VLA training.
Reads from per-episode HDF5 files with pre-mapped 128-dim actions.

Features:
  - Episode-aware temporal sampling (real consecutive action chunks)
  - Multi-frame history (t-1, t, t+1) with guaranteed same-episode frames
  - LRU file handle cache for efficient HDF5 access
  - Supports both temporal episodes and static scenes (DexGraspNet)
  - Vision-encoder-only VLM preprocessing (no chat template, no PIL conversion loops)

VLM Preprocessing (Pi0 style):
  Images are preprocessed for the ViT encoder directly via the processor.
  Text is tokenized via the processor's tokenizer.
  No chat template, no conversation building, no PIL conversion in model forward.
  All CPU work happens in DataLoader workers (parallel).

HDF5 layout (temporal):
  ep_XXXXXX.hdf5
  ├── images/view_0  [T, H, W, 3] uint8
  ├── actions        [T, 128] float32   (pre-mapped to unified space)
  ├── action_mask    [128] bool
  ├── proprio        [T, 128] float32
  ├── language       scalar string
  └── attrs: {dataset_name, embodiment_id, episode_length, n_views, ...}

HDF5 layout (DexGraspNet scene):
  scene_XXXX.hdf5
  ├── images/        [8, H, W, 3] uint8   (8 rendered views)
  ├── grasps/        [K, 128] float32      (K grasps, pre-mapped)
  ├── action_mask    [128] bool
  ├── object_names/  [N] strings
  └── attrs: {n_grasps, n_views}
"""

import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging
import cv2

logger = logging.getLogger(__name__)


class EpisodeHDF5Dataset(Dataset):
    """
    Episode-centric HDF5 dataset for temporal robotics data.

    Each sample = (episode_file, start_timestep).
    Returns real consecutive action chunks [T, 128] and multi-frame images.

    VLM preprocessing: if vlm_processor is provided, images are preprocessed
    for the ViT encoder (pixel_values, image_grid_thw) and text is tokenized
    (input_ids, attention_mask) in DataLoader workers — parallel, no model forward.

    Args:
        episode_dir: Path to data/episodes/{dataset_name}/
        action_horizon: Number of action timesteps per chunk (default 16).
        num_history_frames: Number of image history frames (default 3: t-1, t, t+1).
        image_size: Target image size (default 448).
        stride: Stride between adjacent samples within an episode.
        max_samples: Limit total samples (for debugging).
        vlm_processor: HuggingFace processor for ViT preprocessing + text tokenization.
    """

    def __init__(
        self,
        episode_dir: str,
        action_horizon: int = 16,
        num_history_frames: int = 3,
        image_size: int = 448,
        stride: int = 1,
        max_samples: Optional[int] = None,
        vlm_processor=None,
        image_aug: bool = False,
        cond_mask_prob: float = 0.0,
    ):
        self.episode_dir = Path(episode_dir)
        self.action_horizon = action_horizon
        self.num_history_frames = num_history_frames
        self.image_size = image_size
        self.stride = stride
        self.vlm_processor = vlm_processor
        self.image_aug = image_aug
        self.cond_mask_prob = cond_mask_prob

        # Load metadata
        metadata_path = self.episode_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.episode_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.dataset_name = self.metadata.get('dataset_name', self.episode_dir.name)
        self.embodiment_id = self.metadata.get('embodiment_id', 0)
        self.view_names = self.metadata.get('view_names', ['view_0'])
        self.is_static = self.metadata.get('is_static', False)

        # Build sample index
        self.sample_index = self._build_sample_index()

        if max_samples and len(self.sample_index) > max_samples:
            self.sample_index = self.sample_index[:max_samples]

        # HDF5 file handle cache (LRU)
        self._file_cache: Dict[str, h5py.File] = {}
        self._cache_max = 100

        logger.info(
            f"EpisodeHDF5Dataset({self.dataset_name}): "
            f"{len(self.sample_index)} samples from "
            f"{len(self.metadata.get('episodes', []))} episodes, "
            f"static={self.is_static}"
        )

    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """Build index: each entry = (episode_file_path, start_timestep)."""
        index = []

        if self.is_static:
            # DexGraspNet: each entry = (scene_file, grasp_idx)
            for ep_info in self.metadata.get('episodes', []):
                ep_file = str(self.episode_dir / ep_info['filename'])
                n_grasps = ep_info.get('n_grasps', ep_info.get('length', 1))
                for g in range(n_grasps):
                    index.append((ep_file, g))
        else:
            # Temporal: each entry = (episode_file, start_timestep)
            for ep_info in self.metadata.get('episodes', []):
                ep_file = str(self.episode_dir / ep_info['filename'])
                ep_len = ep_info['length']
                # Valid starting points
                max_start = max(1, ep_len - self.action_horizon + 1)
                for t in range(0, max_start, self.stride):
                    index.append((ep_file, t))

        return index

    def _get_file(self, path: str) -> h5py.File:
        """Get HDF5 file handle with LRU caching."""
        if path not in self._file_cache:
            if len(self._file_cache) >= self._cache_max:
                oldest_key = next(iter(self._file_cache))
                try:
                    self._file_cache[oldest_key].close()
                except Exception:
                    pass
                del self._file_cache[oldest_key]
            self._file_cache[path] = h5py.File(path, 'r')
        return self._file_cache[path]

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict:
        """Load a sample from HDF5 + optionally preprocess for VLM ViT encoder."""
        ep_file, t_start = self.sample_index[idx]

        try:
            f = self._get_file(ep_file)

            if self.is_static:
                sample = self._load_static(f, t_start, idx)
            else:
                sample = self._load_temporal(f, t_start, idx)

            # Preprocess for VLM ViT encoder in DataLoader worker (parallel)
            return self._preprocess_vlm(sample)
        except Exception as e:
            logger.warning(f"Error loading sample {idx} from {ep_file}: {e}")
            return self._empty_sample()

    def _load_temporal(self, f: h5py.File, t_start: int, idx: int) -> dict:
        """Load temporal episode sample."""
        ep_len = f.attrs['episode_length']
        n_views = f.attrs['n_views']
        T = self.action_horizon

        # ── Actions: T consecutive timesteps ──
        t_end = min(t_start + T, ep_len)
        actions = f['actions'][t_start:t_end]  # [t_actual, 128]
        if actions.shape[0] < T:
            pad = np.tile(actions[-1:], (T - actions.shape[0], 1))
            actions = np.concatenate([actions, pad], axis=0)

        action_mask = f['action_mask'][:]  # [128]

        # ── Images: (t-1, t, t+1) × V views ──
        if self.num_history_frames == 3:
            frame_indices = [
                max(0, t_start - 1),
                t_start,
                min(t_start + 1, ep_len - 1),
            ]
        elif self.num_history_frames == 1:
            frame_indices = [t_start]
        else:
            half = self.num_history_frames // 2
            frame_indices = [
                max(0, min(t_start + dt - half, ep_len - 1))
                for dt in range(self.num_history_frames)
            ]

        images = self._load_images(f, frame_indices, n_views)
        num_frames = len(frame_indices)

        # ── Language ──
        language = self._read_language(f)

        # ── Proprioception ──
        if 'proprio' in f:
            proprio = f['proprio'][t_start].astype(np.float32)
        else:
            proprio = np.zeros(128, dtype=np.float32)

        # ── Condition masking for classifier-free guidance (RDT-1B style) ──
        # Randomly drop language, images, or proprio with probability cond_mask_prob
        if self.cond_mask_prob > 0:
            if np.random.random() < self.cond_mask_prob:
                language = ""        # Drop language
            if np.random.random() < self.cond_mask_prob:
                images = np.zeros_like(images)  # Drop images (black)
            if np.random.random() < self.cond_mask_prob:
                proprio = np.zeros(128, dtype=np.float32)  # Drop proprio

        return {
            'images': torch.from_numpy(images).float(),
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'action_mask': torch.from_numpy(action_mask.astype(np.float32)),
            'language': language,
            'proprio': torch.from_numpy(proprio),
            'embodiment_id': int(f.attrs.get('embodiment_id', self.embodiment_id)),
            'dataset_name': self.dataset_name,
            'num_views': int(n_views),
            'num_frames': num_frames,
            'episode_id': idx,
            'timestep': t_start,
            'sample_id': idx,
            'task_id': 0,
        }

    def _load_static(self, f: h5py.File, grasp_idx: int, idx: int) -> dict:
        """Load static scene sample (DexGraspNet)."""
        n_views = f.attrs.get('n_views', 8)
        T = self.action_horizon

        # ── Action: single grasp, tiled T times ──
        grasp = f['grasps'][grasp_idx]  # [128]
        actions = np.tile(grasp[np.newaxis, :], (T, 1))  # [T, 128]
        action_mask = f['action_mask'][:]  # [128]

        # ── Images: all views, 1 frame ──
        images = self._load_images_static(f, n_views)

        # ── Language ──
        language = self._read_language(f)

        # ── No proprio for static grasps ──
        proprio = np.zeros(128, dtype=np.float32)

        return {
            'images': torch.from_numpy(images).float(),
            'actions': torch.from_numpy(actions.astype(np.float32)),
            'action_mask': torch.from_numpy(action_mask.astype(np.float32)),
            'language': language,
            'proprio': torch.from_numpy(proprio),
            'embodiment_id': int(f.attrs.get('embodiment_id', self.embodiment_id)),
            'dataset_name': self.dataset_name,
            'num_views': int(n_views),
            'num_frames': 1,
            'episode_id': idx,
            'timestep': 0,
            'sample_id': idx,
            'task_id': 0,
        }

    def _load_images(
        self, f: h5py.File, frame_indices: List[int], n_views: int
    ) -> np.ndarray:
        """Load multi-frame × multi-view images.

        Returns: [num_frames * num_views, 3, H, W] float32 [0, 1]
        Order: frame-major, view-minor.
        
        When n_views=0 (datasets without images, e.g., bc_z in parquet form),
        returns a single background image per frame (RDT-1B style: replace
        missing cameras with processor-mean-colored background).
        """
        images = []
        if n_views == 0:
            # No cameras — produce 1 background image per frame
            for _ in frame_indices:
                images.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))
        else:
            for fi in frame_indices:
                for v in range(n_views):
                    key = f'images/view_{v}'
                    if key in f:
                        img = f[key][fi]  # [H, W, 3] uint8
                        img = self._preprocess_image(img)
                        images.append(img)
                    else:
                        images.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))

        return np.stack(images, axis=0)  # [F*V, 3, H, W] or [F, 3, H, W] if no views

    def _load_images_static(self, f: h5py.File, n_views: int) -> np.ndarray:
        """Load all views for static scene (1 frame)."""
        images = []
        if 'images' in f and hasattr(f['images'], 'shape'):
            all_imgs = f['images'][:]  # [V, H, W, 3]
            for v in range(min(n_views, all_imgs.shape[0])):
                img = self._preprocess_image(all_imgs[v])
                images.append(img)
        else:
            for v in range(n_views):
                key = f'images/view_{v}'
                if key in f:
                    img = f[key][0] if f[key].ndim == 4 else f[key][:]
                    img = self._preprocess_image(img)
                    images.append(img)
                else:
                    images.append(np.zeros((3, self.image_size, self.image_size), dtype=np.float32))

        if not images:
            images = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)]

        return np.stack(images, axis=0)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Convert [H, W, 3] uint8 → [3, image_size, image_size] float32 [0, 1].

        Pipeline (RDT-1B style):
          1. Pad to square (shorter side padded with black)
          2. Resize to image_size × image_size
          3. Optional ColorJitter augmentation (50% probability)
          4. Normalize to [0, 1] float32
        
        Images are stored at original resolution in HDF5;
        all resizing happens here at training time.
        """
        H, W = img.shape[:2]

        # Step 1: Pad to square (RDT-1B uses processor-mean color; we use black)
        if H != W:
            size = max(H, W)
            padded = np.zeros((size, size, 3), dtype=np.uint8)
            y_off = (size - H) // 2
            x_off = (size - W) // 2
            padded[y_off:y_off + H, x_off:x_off + W] = img
            img = padded

        # Step 2: Resize to target size
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # Step 3: Image augmentation (RDT-1B style)
        if self.image_aug and np.random.random() > 0.5:
            try:
                from PIL import Image as PILImage
                from torchvision import transforms
                pil_img = PILImage.fromarray(img)
                pil_img = transforms.ColorJitter(
                    brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03
                )(pil_img)
                img = np.array(pil_img)
            except Exception:
                pass

        # Step 4: Normalize
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # [3, H, W]
        return img

    def _read_language(self, f: h5py.File) -> str:
        """Read language instruction from HDF5."""
        if 'language' in f:
            val = f['language'][()]
            if isinstance(val, bytes):
                return val.decode('utf-8')
            return str(val)

        if 'object_names' in f:
            try:
                names = [n.decode('utf-8') if isinstance(n, bytes) else str(n)
                         for n in f['object_names'][:]]
                if names:
                    return f"grasp the {names[0]}"
            except Exception:
                pass

        return ""

    def _preprocess_vlm(self, sample: dict) -> dict:
        """Preprocess for VLM ViT encoder in DataLoader worker.

        Vision-encoder-only: just run processor on images to get pixel_values
        and image_grid_thw. Tokenize text with the tokenizer. No chat template,
        no conversation building, no PIL conversion loops in model forward.

        This runs in parallel across DataLoader workers.
        """
        if self.vlm_processor is None:
            return sample

        from PIL import Image

        images_tensor = sample['images']  # [N, 3, H, W] float [0,1]
        language = sample.get('language', '') or 'describe the scene'

        # Convert tensor images → PIL for processor
        pil_images = []
        for i in range(images_tensor.shape[0]):
            img = images_tensor[i]
            if img.abs().sum() < 1.0:
                continue  # skip padded zero-images
            img_np = (img.cpu().float() * 255).byte().permute(1, 2, 0).numpy()
            pil_images.append(Image.fromarray(img_np))

        if not pil_images:
            pil_images = [Image.new('RGB', (self.image_size, self.image_size))]

        # Process images for ViT encoder (pixel_values + image_grid_thw)
        # Use a dummy text to trigger image processing, then separately tokenize
        img_inputs = self.vlm_processor(
            text=[''],  # dummy text, we only want pixel_values
            images=pil_images,
            return_tensors='pt',
        )

        # Tokenize text separately (no images, no chat template)
        text_inputs = self.vlm_processor.tokenizer(
            language,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=128,
        )

        # Store preprocessed VLM inputs (squeeze batch dim — re-batched by collate)
        if 'pixel_values' in img_inputs:
            sample['vlm_pixel_values'] = img_inputs['pixel_values'].squeeze(0)
        if 'image_grid_thw' in img_inputs:
            sample['vlm_image_grid_thw'] = img_inputs['image_grid_thw'].squeeze(0)

        sample['vlm_input_ids'] = text_inputs['input_ids'].squeeze(0)
        sample['vlm_attention_mask'] = text_inputs['attention_mask'].squeeze(0)

        return sample

    def _empty_sample(self) -> dict:
        """Return a valid empty sample for error recovery."""
        return {
            'images': torch.zeros(1, 3, self.image_size, self.image_size),
            'actions': torch.zeros(self.action_horizon, 128),
            'action_mask': torch.zeros(128),
            'language': '',
            'proprio': torch.zeros(128),
            'embodiment_id': self.embodiment_id,
            'dataset_name': self.dataset_name,
            'num_views': 1,
            'num_frames': 1,
            'episode_id': 0,
            'timestep': 0,
            'sample_id': 0,
            'task_id': 0,
        }

    def compute_action_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean/std from metadata (pre-computed during conversion)."""
        stats = self.metadata.get('action_stats', {})
        mean = np.array(stats.get('mean', np.zeros(128)), dtype=np.float32)
        std = np.array(stats.get('std', np.ones(128)), dtype=np.float32)
        return mean, std

    def close(self):
        """Close all cached file handles."""
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_cache.clear()

    def __del__(self):
        self.close()
