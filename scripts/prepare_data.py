#!/usr/bin/env python3
"""
Drifting-VLA Data Preparation — Download + Convert in One Pass
================================================================

Single script that downloads from HuggingFace and converts directly to
Episode-Centric HDF5 format.  **No intermediate Arrow files.**

Replaces the old two-step pipeline:
  OLD:  download_datasets.py → arrow_data/ → convert_to_episodes.py → episodes/
  NEW:  prepare_data.py → episodes/   (stream → HDF5, no temp files)

Supported dataset types:
  - lerobot  : HF datasets API  → stream rows → episode HDF5
  - rlbench  : snapshot_download → read PKL/PNG → episode HDF5
  - dexgrasp : snapshot_download → read NPZ/PNG → scene HDF5
  - dexwild  : snapshot_download → read source HDF5 → episode HDF5

Usage:
    # Prepare all datasets (full)
    python scripts/prepare_data.py --all

    # Prepare specific dataset
    python scripts/prepare_data.py --dataset aloha

    # Quick smoke-test (2 episodes per dataset)
    python scripts/prepare_data.py --all --max-episodes 2

    # Custom sample count
    python scripts/prepare_data.py --dataset droid --max-episodes 10

Output:
    data/episodes/{dataset_name}/
    ├── metadata.json
    ├── ep_000000.hdf5
    └── ...
"""

import argparse
import json
import logging
import sys
import os
import cv2
import h5py
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Silence noisy HuggingFace HTTP request logging
for _noisy in ('httpx', 'huggingface_hub', 'filelock', 'urllib3', 'datasets'):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.data.action_mapping import (
    DATASET_EMBODIMENT, DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM,
    DATASET_FIELD_FORMATS, LEROBOT_DATASETS, UNIFIED_ACTION_DIM,
    map_to_unified, get_action_mask, assemble_state_vec_batch,
)


# =============================================================================
# Dataset Registry
# =============================================================================

DATASETS = {
    'rlbench': {
        'hf_repo': 'hqfang/rlbench-18-tasks',
        'type': 'rlbench',
        'description': 'RLBench 18-task sim (270K, 8-dim gripper)',

    },
    'dexgraspnet': {
        'hf_repo': 'lhrlhr/DexGraspNet2.0',
        'type': 'dexgraspnet',
        'description': 'DexGraspNet 2.0 dexterous grasping (500K, 23-dim dex hand)',
    },
    'aloha': {
        'hf_repo': 'lerobot/aloha_sim_transfer_cube_human_image',
        'type': 'lerobot',
        'description': 'ALOHA sim bimanual cube transfer (20K, 14-dim)',
    },
    'droid': {
        'hf_repo': 'lerobot/droid_1.0.1',
        'type': 'lerobot',
        'description': 'DROID full dataset (25.5M, 7-dim gripper)',
    },
    'bc_z': {
        'hf_repo': 'lerobot/berkeley_autolab_ur5',
        'type': 'lerobot',
        'description': 'Berkeley AutoLab UR5 language-cond (98K, 7-dim gripper)',
    },
    'taco_play': {
        'hf_repo': 'lerobot/taco_play',
        'type': 'lerobot',
        'description': 'TACO Play diverse manipulation (238K, 7-dim gripper)',
    },
    'utaustin_mutex': {
        'hf_repo': 'lerobot/utaustin_mutex',
        'type': 'lerobot',
        'description': 'UT Austin MUTEX language-cond (362K, 7-dim gripper)',
    },
    'cmu_stretch': {
        'hf_repo': 'lerobot/cmu_stretch',
        'type': 'lerobot',
        'description': 'CMU Stretch mobile manip (25K, 8-dim gripper)',
    },
    'nyu_franka': {
        'hf_repo': 'lerobot/nyu_franka_play_dataset',
        'type': 'lerobot',
        'description': 'NYU Franka Play bimanual (45K, 15-dim)',
    },
    'stanford_hydra': {
        'hf_repo': 'lerobot/stanford_hydra_dataset',
        'type': 'lerobot',
        'description': 'Stanford HYDRA diverse manip (358K, 7-dim gripper)',
    },
    'dexwild': {
        'hf_repo': 'YingyangPolarBear/DexWild',
        'type': 'dexwild',
        'description': 'DexWild dexterous LEAP hand (95K, 22-dim dex)',
    },
    'dexora': {
        'hf_repo': 'Dexora/Dexora_Real-World_Dataset',
        'type': 'lerobot',
        'description': 'Dexora bimanual dexterous real-world (2.9M, 39-dim)',
    },
}

# Add Behavior 1K tasks
for _i in range(50):
    DATASETS[f'behavior1k_t{_i:04d}'] = {
        'hf_repo': f'lerobot/behavior1k-task{_i:04d}',
        'type': 'lerobot',
        'description': f'Behavior 1K task {_i:04d} OmniGibson sim (23-dim bimanual)',
    }

# View name registries
DATASET_VIEW_NAMES = {
    'rlbench': ['front_rgb', 'wrist_rgb'],
    'aloha': ['top'],
    'bc_z': ['image', 'hand_image'],
    'taco_play': ['image'],
    'utaustin_mutex': [],
    'cmu_stretch': ['image'],
    'nyu_franka': ['image'],
    'stanford_hydra': ['image'],
    'dexgraspnet': [f'view_{i}' for i in range(8)],
    'dexwild': ['thumb_cam', 'pinky_cam'],
}
for _i in range(50):
    DATASET_VIEW_NAMES[f'behavior1k_t{_i:04d}'] = ['head', 'left_wrist', 'right_wrist']


# =============================================================================
# Image helpers
# =============================================================================

def _extract_image(row: dict, key: str, target_size: int) -> np.ndarray:
    """Extract and resize a single image from a dataset row.

    Returns: [H, W, 3] uint8 or None.
    """
    val = row.get(key)
    if val is None:
        return None

    try:
        if hasattr(val, 'convert'):
            img = np.array(val.convert('RGB'))
        elif hasattr(val, 'numpy'):
            img = val.numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.dtype in (np.float32, np.float64):
                img = (img * 255).clip(0, 255).astype(np.uint8)
        elif isinstance(val, dict) and 'bytes' in val:
            import io
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(val['bytes'])).convert('RGB'))
        elif isinstance(val, np.ndarray):
            img = val
        else:
            return None

        if img.ndim != 3 or img.shape[2] != 3:
            return None
        if img.shape[0] != target_size or img.shape[1] != target_size:
            img = cv2.resize(img, (target_size, target_size))
        return img.astype(np.uint8)
    except Exception:
        return None


def _detect_image_keys(column_names: list) -> list:
    """Detect RGB image columns from HF dataset."""
    img_keys = []
    skip_patterns = [
        'depth', 'seg_', 'segmentation', 'mask', 'normal',
    ]
    for col in sorted(column_names):
        cl = col.lower()
        if 'image' not in cl:
            continue
        if any(pat in cl for pat in skip_patterns):
            continue
        img_keys.append(col)
    return img_keys


# =============================================================================
# LeRobot datasets — stream from HF, write HDF5 directly (NO Arrow)
# =============================================================================

def prepare_lerobot(
    ds_name: str,
    hf_repo: str,
    output_dir: Path,
    image_size: int = 448,
    max_episodes: int = None,
    rgb_only: bool = True,
):
    """Download + convert a LeRobot-format dataset in one pass.

    FAST PATH: reads actions/metadata from parquet (instant), decodes video
    per-episode in batch (50-100× faster than per-frame lerobot_ds[i]).
    
    For parquet-with-images datasets (e.g., aloha), images are read directly
    from the dataset without video decoding.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Loading via LeRobot API: {hf_repo}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        lerobot_ds = LeRobotDataset(hf_repo, video_backend='pyav')
    except Exception as e:
        logger.error(f"  Failed to load {hf_repo} via LeRobot API: {e}")
        import traceback
        traceback.print_exc()
        return False

    hf = lerobot_ds.hf_dataset  # parquet — fast access for actions/metadata
    total_frames = len(hf)
    logger.info(f"  Loaded {total_frames} frames")

    # Detect image keys from a single decoded sample
    sample0 = lerobot_ds[0]
    all_keys = list(sample0.keys())
    _NON_RGB = ('depth', 'seg', 'segmentation', 'mask', 'normal', 'flow', 'pointcloud')
    image_keys = [k for k in all_keys if 'image' in k.lower()]
    if rgb_only:
        image_keys = [k for k in image_keys
                      if not any(s in k.lower() for s in _NON_RGB)]
    logger.info(f"  Image keys (rgb_only={rgb_only}): {image_keys}")

    # Detect video paths for batch decoding
    # LeRobot v2: videos at {root}/videos/{camera_key}/chunk-{chunk}/file-{file}.mp4
    dataset_root = getattr(lerobot_ds, 'root', None)
    videos_dir = None
    chunks_size = 1000  # default
    if dataset_root is not None:
        vd = Path(dataset_root) / 'videos'
        if vd.exists():
            videos_dir = vd
            # Read chunks_size from meta/info.json
            info_path = Path(dataset_root) / 'meta' / 'info.json'
            if info_path.exists():
                import json as _json
                info = _json.load(open(info_path))
                chunks_size = info.get('chunks_size', 1000)
    if videos_dir is not None:
        logger.info(f"  Video dir: {videos_dir} (batch decode, chunks_size={chunks_size})")
    else:
        logger.info(f"  No video dir — using per-frame decode (parquet images)")

    # Detect language key (only in full dataset, not parquet)
    language_key = None
    for k in ['task', 'language_instruction', 'language', 'text']:
        if k in all_keys:
            language_key = k
            break

    # Detect state/proprio key (in parquet)
    parquet_cols = hf.column_names
    proprio_keys = [k for k in parquet_cols
                    if 'state' in k.lower() or 'proprio' in k.lower()]

    # Action info
    embodiment_id = DATASET_EMBODIMENT.get(ds_name, 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name, 7)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)
    field_names = DATASET_FIELD_FORMATS.get(ds_name)

    # Group frames by episode from parquet (fast, no video decode)
    logger.info(f"  Grouping frames by episode...")
    episodes = defaultdict(list)
    # Batch read episode_index column (much faster than per-row)
    ep_col = hf['episode_index']
    for i in range(total_frames):
        ep_idx = int(ep_col[i])
        episodes[ep_idx].append(i)

    ep_ids = sorted(episodes.keys())
    if max_episodes:
        ep_ids = ep_ids[:max_episodes]

    logger.info(f"  {len(ep_ids)} episodes to write")

    all_actions = []
    ep_metadata_list = []

    for ep_num, ep_id in enumerate(tqdm(ep_ids, desc=f"  {ds_name}")):
        frame_indices = sorted(episodes[ep_id])
        ep_len = len(frame_indices)

        ep_filename = f"ep_{ep_num:06d}.hdf5"
        ep_path = output_dir / ep_filename

        with h5py.File(str(ep_path), 'w') as hf_out:
            # ── Images: batch video decode (FAST) ──
            n_views = len(image_keys)
            if n_views > 0:
                for v_idx, img_key in enumerate(image_keys):
                    frames = _load_episode_images(
                        lerobot_ds, videos_dir, img_key, ep_id, ep_len,
                        frame_indices, chunks_size,
                    )
                    if frames is not None and len(frames) > 0:
                        # Force uint8 — some video decoders return object arrays
                        if frames.dtype != np.uint8:
                            try:
                                frames = np.asarray(frames, dtype=np.uint8)
                            except (TypeError, ValueError):
                                logger.warning(f"  Skipping view {img_key} ep {ep_id}: bad dtype {frames.dtype}")
                                n_views = max(n_views - 1, 0)
                                continue
                        hf_out.create_dataset(
                            f'images/view_{v_idx}', data=frames,
                            chunks=(1,) + frames.shape[1:],
                            compression='gzip', compression_opts=1,
                        )
                    else:
                        n_views = max(n_views - 1, 0)

            # ── Actions from parquet (fast, no video) ──
            actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
            for t, fi in enumerate(frame_indices):
                try:
                    act = hf[fi]['action']
                    if act is not None:
                        a = np.asarray(act).flatten().astype(np.float32)
                        actions_native[t, :min(len(a), native_dim)] = a[:native_dim]
                except (TypeError, ValueError):
                    pass  # skip malformed rows, keep zeros

            if field_names:
                actions_unified, _ = assemble_state_vec_batch(actions_native, field_names)
            else:
                actions_unified = map_to_unified(actions_native, embodiment_id)
            hf_out.create_dataset('actions', data=actions_unified)
            all_actions.append(actions_unified)

            # ── Action mask ──
            hf_out.create_dataset('action_mask', data=mask_info.mask)

            # ── Proprioception from parquet (fast) ──
            if proprio_keys:
                proprio_data = np.zeros((ep_len, UNIFIED_ACTION_DIM), dtype=np.float32)
                for t, fi in enumerate(frame_indices):
                    row = hf[fi]
                    for pk in proprio_keys:
                        val = row.get(pk)
                        if val is not None:
                            try:
                                val = np.atleast_1d(np.asarray(val).flatten().astype(np.float32))[:native_dim]
                            except (TypeError, ValueError):
                                continue
                            if field_names:
                                p_unified, _ = assemble_state_vec_batch(val.reshape(1, -1), field_names)
                                proprio_data[t] = p_unified[0]
                            else:
                                proprio_data[t] = map_to_unified(val.reshape(1, -1), embodiment_id)[0]
                            break
                hf_out.create_dataset('proprio', data=proprio_data)

            # ── Language (1 video decode per episode for task text) ──
            lang = ""
            if language_key:
                try:
                    full_row = lerobot_ds[frame_indices[0]]
                    lang_val = full_row.get(language_key, "")
                    if lang_val:
                        lang = str(lang_val)
                except Exception:
                    pass
            hf_out.create_dataset('language', data=lang)

            # ── Attributes ──
            hf_out.attrs['dataset_name'] = ds_name
            hf_out.attrs['embodiment_id'] = embodiment_id
            hf_out.attrs['episode_length'] = ep_len
            hf_out.attrs['n_views'] = n_views
            hf_out.attrs['action_dim'] = native_dim
            hf_out.attrs['image_size'] = 0  # original resolution; resize at training time

        ep_metadata_list.append({'filename': ep_filename, 'length': ep_len, 'n_grasps': 0})

    # Write metadata
    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_cat.mean(axis=0).tolist()
        action_std = all_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    view_names = DATASET_VIEW_NAMES.get(ds_name, [f'view_{v}' for v in range(len(image_keys))])
    metadata = {
        'dataset_name': ds_name,
        'embodiment_id': embodiment_id,
        'is_static': False,
        'view_names': view_names[:len(image_keys)] if image_keys else view_names,
        'total_samples': sum(ep['length'] for ep in ep_metadata_list),
        'total_episodes': len(ep_metadata_list),
        'action_stats': {'mean': action_mean, 'std': action_std},
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"  ✅ {ds_name}: {len(ep_metadata_list)} episodes, "
        f"{metadata['total_samples']} frames → {output_dir}"
    )
    return True


# =============================================================================
# Batch video decoding (50-100× faster than per-frame lerobot_ds[i])
# =============================================================================

def _load_episode_images(
    lerobot_ds, videos_dir, img_key: str, ep_id: int,
    ep_len: int, frame_indices: list, chunks_size: int = 1000,
) -> np.ndarray:
    """Load all images for one episode + one camera view.
    
    Strategy:
      1. Try batch video decode (opens MP4 once, reads ALL frames sequentially)
      2. Fall back to per-frame lerobot_ds[i] if video file not found
    
    Returns: [T, H, W, 3] uint8 or None
    """
    # ── Strategy 1: Batch video decode (FAST) ──
    if videos_dir is not None:
        frames = _decode_episode_video(videos_dir, img_key, ep_id, ep_len, chunks_size)
        if frames is not None:
            return frames

    # ── Strategy 2: Per-frame decode (SLOW fallback for parquet-with-images) ──
    images = []
    for fi in frame_indices:
        try:
            sample = lerobot_ds[fi]
            img_val = sample.get(img_key)
            if img_val is not None:
                img = _lerobot_to_numpy_image(img_val)
                if img is not None:
                    images.append(img)
                    continue
        except Exception:
            pass
        # Append placeholder on failure
        if images:
            images.append(np.zeros_like(images[0]))
        else:
            images.append(None)

    # Remove Nones and ensure all same shape
    images = [img for img in images if img is not None and isinstance(img, np.ndarray)]
    if not images:
        return None
    try:
        return np.stack(images).astype(np.uint8)
    except (ValueError, TypeError):
        return None


def _decode_episode_video(
    videos_dir: Path, img_key: str, ep_id: int, expected_frames: int,
    chunks_size: int = 1000,
) -> np.ndarray:
    """Decode ALL frames from an episode video file at once.
    
    50-100× faster than per-frame lerobot_ds[i] because:
    - Opens the video file ONCE (not per frame)
    - Sequential read (no seeking)
    - Native pyav batch decode
    
    LeRobot v2 video path format:
      {videos_dir}/{camera_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
      where chunk_index = ep_id // chunks_size, file_index = ep_id % chunks_size
    
    Returns: [T, H, W, 3] uint8 or None if file not found
    """
    chunk_idx = ep_id // chunks_size
    file_idx = ep_id % chunks_size
    video_path = Path(videos_dir) / img_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.exists():
        # Fallback: try episode_XXXXXX.mp4 (older format)
        video_path = Path(videos_dir) / img_key / f"episode_{ep_id:06d}.mp4"
    if not video_path.exists():
        return None

    try:
        import av
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        frames = []
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='rgb24')  # [H, W, 3] uint8
            frames.append(img)
            if len(frames) >= expected_frames:
                break

        container.close()

        if not frames:
            return None

        # Pad if video has fewer frames than expected
        while len(frames) < expected_frames:
            frames.append(frames[-1].copy())

        return np.stack(frames[:expected_frames])

    except Exception as e:
        logger.debug(f"  Video decode failed for {video_path}: {e}")
        return None


def _lerobot_to_numpy_image(img_val) -> np.ndarray:
    """Convert a LeRobot image value (PIL, tensor, or numpy) to [H, W, 3] uint8."""
    from PIL import Image as PILImage
    import torch

    try:
        if isinstance(img_val, PILImage.Image):
            img = np.array(img_val.convert('RGB'))
        elif isinstance(img_val, torch.Tensor):
            img = img_val.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)  # CHW → HWC
            if img.dtype in (np.float32, np.float64):
                img = (img * 255).clip(0, 255).astype(np.uint8)
        elif isinstance(img_val, np.ndarray):
            img = img_val
        else:
            return None

        if img.ndim != 3 or img.shape[2] != 3:
            return None
        return img.astype(np.uint8)
    except Exception:
        return None


# =============================================================================
# Generic HF dataset → episode HDF5 converter (reusable)
# =============================================================================

def _convert_hf_dataset_to_episodes(
    hf_ds,
    ds_name: str,
    output_dir: Path,
    image_size: int = 448,
    max_episodes: int = None,
):
    """Generic converter: HF dataset in memory → episode HDF5 files.

    Works for any parquet-based HuggingFace dataset (RLBench, LeRobot, etc).
    """
    all_cols = hf_ds.column_names
    image_keys = _detect_image_keys(all_cols)
    logger.info(f"  Image keys: {image_keys}")

    language_key = None
    for k in ['task', 'language_instruction', 'language', 'text']:
        if k in all_cols:
            language_key = k
            break

    embodiment_id = DATASET_EMBODIMENT.get(ds_name, 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name, 7)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    episodes = defaultdict(list)
    for i in range(len(hf_ds)):
        ep_idx = hf_ds[i].get('episode_index', 0)
        episodes[ep_idx].append(i)

    ep_ids = sorted(episodes.keys())
    if max_episodes:
        ep_ids = ep_ids[:max_episodes]

    logger.info(f"  {len(ep_ids)} episodes to write")

    all_actions = []
    ep_metadata_list = []
    n_views = len(image_keys) if image_keys else 0

    for ep_num, ep_id in enumerate(tqdm(ep_ids, desc=f"  {ds_name}")):
        frame_indices = sorted(episodes[ep_id])
        ep_len = len(frame_indices)

        ep_filename = f"ep_{ep_num:06d}.hdf5"
        ep_path = output_dir / ep_filename

        with h5py.File(str(ep_path), 'w') as hf:
            if n_views > 0:
                first_img = _extract_image(hf_ds[frame_indices[0]], image_keys[0], image_size)
                H, W = (first_img.shape[:2]) if first_img is not None else (image_size, image_size)
                for v_idx, img_key in enumerate(image_keys):
                    view_data = np.zeros((ep_len, H, W, 3), dtype=np.uint8)
                    for t, fi in enumerate(frame_indices):
                        img = _extract_image(hf_ds[fi], img_key, image_size)
                        if img is not None:
                            view_data[t] = img
                    hf.create_dataset(
                        f'images/view_{v_idx}', data=view_data,
                        chunks=(1, H, W, 3), compression='gzip', compression_opts=1,
                    )

            actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
            for t, fi in enumerate(frame_indices):
                act = hf_ds[fi].get('action')
                if act is not None:
                    act = np.array(act, dtype=np.float32)
                    actions_native[t, :min(len(act), native_dim)] = act[:native_dim]

            actions_unified = map_to_unified(actions_native, embodiment_id)
            hf.create_dataset('actions', data=actions_unified)
            all_actions.append(actions_unified)
            hf.create_dataset('action_mask', data=mask_info.mask)

            # Proprioception
            proprio_keys = [k for k in all_cols if 'state' in k.lower() or 'proprio' in k.lower() or k == 'observation.state']
            if proprio_keys:
                proprio_data = np.zeros((ep_len, UNIFIED_ACTION_DIM), dtype=np.float32)
                for t, fi in enumerate(frame_indices):
                    row = hf_ds[fi]
                    for pk in proprio_keys:
                        val = row.get(pk)
                        if val is not None:
                            val = np.atleast_1d(np.array(val, dtype=np.float32))[:native_dim]
                            p_unified = map_to_unified(val.reshape(1, -1), embodiment_id)[0]
                            proprio_data[t] = p_unified
                            break
                hf.create_dataset('proprio', data=proprio_data)

            lang = ""
            if language_key:
                lang_val = hf_ds[frame_indices[0]].get(language_key, "")
                if lang_val:
                    lang = str(lang_val)
            hf.create_dataset('language', data=lang)

            hf.attrs['dataset_name'] = ds_name
            hf.attrs['embodiment_id'] = embodiment_id
            hf.attrs['episode_length'] = ep_len
            hf.attrs['n_views'] = n_views
            hf.attrs['action_dim'] = native_dim
            hf.attrs['image_size'] = 0  # original resolution; resize at training time

        ep_metadata_list.append({'filename': ep_filename, 'length': ep_len, 'n_grasps': 0})

    view_names = DATASET_VIEW_NAMES.get(ds_name, [f'view_{i}' for i in range(n_views)])
    _write_metadata(output_dir, ds_name, embodiment_id, False,
                    view_names[:n_views] if image_keys else view_names,
                    all_actions, ep_metadata_list)
    logger.info(f"  ✅ {ds_name}: {len(ep_metadata_list)} episodes → {output_dir}")
    return True


# =============================================================================
# RLBench — HF datasets API (parquet) with PKL/PNG fallback
# =============================================================================

def prepare_rlbench(
    ds_info: dict,
    output_dir: Path,
    data_root: Path,
    image_size: int = 448,
    max_episodes: int = None,
):
    """Download + convert RLBench in one pass.

    hqfang/rlbench-18-tasks is a parquet-based HF dataset,
    so we use load_dataset() first. Falls back to snapshot_download
    for raw PerAct-format PKL/PNG repos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Primary path: HF datasets API (works for parquet repos) ──
    try:
        from datasets import load_dataset

        logger.info(f"  Streaming {ds_info['hf_repo']} (split=train) ...")
        hf_ds = load_dataset(ds_info['hf_repo'], split='train')
        logger.info(f"  Loaded {len(hf_ds)} samples in memory")

        return _convert_hf_dataset_to_episodes(
            hf_ds, 'rlbench', output_dir, image_size, max_episodes,
        )
    except Exception as e:
        logger.warning(f"  HF datasets load failed: {e}, trying PKL fallback...")

    # ── Fallback: snapshot_download for raw PKL/PNG repos ──
    from huggingface_hub import snapshot_download

    rlbench_dir = data_root / 'rlbench'
    train_dir = rlbench_dir / 'train'
    if not (train_dir.exists() and any(train_dir.rglob('*.pkl'))):
        logger.info(f"  Downloading {ds_info['hf_repo']} via snapshot ...")
        try:
            snapshot_download(
                repo_id=ds_info['hf_repo'], repo_type='dataset',
                local_dir=str(rlbench_dir),
                ignore_patterns=['*.avi', '*.mp4'],
            )
        except Exception as e2:
            logger.error(f"  Download failed: {e2}")
            return False

    ep_dirs = sorted(rlbench_dir.rglob('episode*'))
    ep_dirs = [d for d in ep_dirs if d.is_dir() and (d / 'low_dim_obs.pkl').exists()]
    if not ep_dirs:
        logger.error("  No RLBench episodes found (neither parquet nor PKL)")
        return False

    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]

    logger.info(f"  Converting rlbench (PKL path): {len(ep_dirs)} episodes")

    embodiment_id = DATASET_EMBODIMENT.get('rlbench', 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('rlbench', 8)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)
    cameras = ['front_rgb', 'wrist_rgb']

    all_actions = []
    ep_metadata_list = []

    for ep_num, ep_dir in enumerate(tqdm(ep_dirs, desc="  rlbench")):
        try:
            from drifting_vla.data.rlbench_dataset import _load_pickle
            obs_list = _load_pickle(ep_dir / 'low_dim_obs.pkl')
            ep_len = len(obs_list)
            if ep_len == 0:
                continue

            ep_filename = f"ep_{ep_num:06d}.hdf5"
            ep_path = output_dir / ep_filename

            with h5py.File(str(ep_path), 'w') as hf:
                for v_idx, cam in enumerate(cameras):
                    cam_dir = ep_dir / cam
                    if not cam_dir.exists():
                        continue
                    img_files = sorted(cam_dir.glob('*.png'))
                    if not img_files:
                        continue
                    view_data = np.zeros((ep_len, image_size, image_size, 3), dtype=np.uint8)
                    for t, img_file in enumerate(img_files[:ep_len]):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (image_size, image_size))
                            view_data[t] = img
                    hf.create_dataset(
                        f'images/view_{v_idx}', data=view_data,
                        chunks=(1, image_size, image_size, 3),
                        compression='gzip', compression_opts=1,
                    )

                actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                for t, obs in enumerate(obs_list):
                    try:
                        action = np.concatenate([
                            obs.gripper_pose, [obs.gripper_open],
                        ]).astype(np.float32)
                        actions_native[t, :min(len(action), native_dim)] = action[:native_dim]
                    except (AttributeError, TypeError):
                        pass

                actions_unified = map_to_unified(actions_native, embodiment_id)
                hf.create_dataset('actions', data=actions_unified)
                all_actions.append(actions_unified)
                hf.create_dataset('action_mask', data=mask_info.mask)

                lang = ""
                desc_path = ep_dir / 'variation_descriptions.pkl'
                if desc_path.exists():
                    try:
                        descs = _load_pickle(desc_path)
                        if isinstance(descs, list) and descs:
                            lang = str(descs[0])
                        elif isinstance(descs, str):
                            lang = descs
                    except Exception:
                        pass
                if not lang:
                    for p in ep_dir.parts:
                        if p not in ('train', 'val', 'test', 'all_variations', 'episodes') and not p.startswith('episode'):
                            lang = p.replace('_', ' ')
                hf.create_dataset('language', data=lang)

                hf.attrs['dataset_name'] = 'rlbench'
                hf.attrs['embodiment_id'] = embodiment_id
                hf.attrs['episode_length'] = ep_len
                hf.attrs['n_views'] = len(cameras)
                hf.attrs['action_dim'] = native_dim
                hf.attrs['image_size'] = 0  # original resolution; resize at training time

            ep_metadata_list.append({'filename': ep_filename, 'length': ep_len})
        except Exception as e:
            logger.warning(f"  Error converting {ep_dir.name}: {e}")
            continue

    _write_metadata(output_dir, 'rlbench', embodiment_id, False, cameras, all_actions, ep_metadata_list)
    logger.info(f"  ✅ rlbench: {len(ep_metadata_list)} episodes → {output_dir}")
    return True


# =============================================================================
# DexGraspNet — snapshot_download → read NPZ/PNG → scene HDF5
# =============================================================================

def prepare_dexgraspnet(
    ds_info: dict,
    output_dir: Path,
    data_root: Path,
    image_size: int = 448,
    max_episodes: int = None,
    **kwargs,
):
    """Download + convert DexGraspNet in one pass."""
    from huggingface_hub import snapshot_download
    from scipy.spatial.transform import Rotation

    dex_dir = data_root / 'dexgraspnet'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──
    scenes_dir = dex_dir / 'scenes'
    if not scenes_dir.exists():
        logger.info("  Downloading DexGraspNet ...")
        try:
            snapshot_download(
                repo_id=ds_info['hf_repo'], repo_type='dataset',
                local_dir=str(dex_dir),
                allow_patterns=['*.parquet', '*.json', '*.h5', '*.hdf5', '*.npz', 'README*'],
                ignore_patterns=['*.ply', '*.obj', '*.stl', '*.png', '*.jpg'],
            )
        except Exception as e:
            logger.error(f"  Download failed: {e}")
            return False

    if not scenes_dir.exists():
        logger.error(f"  No scenes/ directory found at {scenes_dir}")
        return False

    # ── Convert ──
    scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and d.name.startswith('scene_')])
    if max_episodes:
        scene_dirs = scene_dirs[:max_episodes]

    rendered_dir = dex_dir / 'rendered'
    embodiment_id = DATASET_EMBODIMENT.get('dexgraspnet', 3)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('dexgraspnet', 23)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    all_actions = []
    ep_metadata_list = []

    for scene_dir in tqdm(scene_dirs, desc="  dexgraspnet"):
        scene_name = scene_dir.name
        npz_files = sorted((scene_dir / 'realsense').glob('*.npz')) if (scene_dir / 'realsense').exists() else []
        if not npz_files:
            continue

        images_list = []
        render_scene_dir = rendered_dir / scene_name if rendered_dir.exists() else None
        if render_scene_dir and render_scene_dir.exists():
            for vf in sorted(render_scene_dir.glob('view_*.png'))[:8]:
                try:
                    img = cv2.imread(str(vf))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (image_size, image_size))
                        images_list.append(img)
                except Exception:
                    pass

        if not images_list:
            continue

        n_views = len(images_list)
        images_arr = np.stack(images_list, axis=0).astype(np.uint8)

        all_grasps = []
        object_names = set()
        for npz_file in npz_files:
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                translations = data.get('translation')
                rotations_data = data.get('rotation')
                if translations is None or rotations_data is None:
                    continue
                K = translations.shape[0]
                for k in range(K):
                    xyz = translations[k].astype(np.float32)
                    try:
                        quat = Rotation.from_matrix(rotations_data[k].astype(np.float32)).as_quat().astype(np.float32)
                    except Exception:
                        quat = np.array([0, 0, 0, 1], dtype=np.float32)
                    fingers = np.zeros(16, dtype=np.float32)
                    for j in range(16):
                        jkey = f'j{j}'
                        if jkey in data and k < len(data[jkey]):
                            fingers[j] = float(data[jkey][k])
                    all_grasps.append(np.concatenate([xyz, quat, fingers]))
            except Exception:
                continue

        if not all_grasps:
            continue

        grasps_native = np.stack(all_grasps, axis=0)
        grasps_unified = map_to_unified(grasps_native, embodiment_id)
        all_actions.append(grasps_unified)

        scene_filename = f"{scene_name}.hdf5"
        with h5py.File(str(output_dir / scene_filename), 'w') as hf:
            hf.create_dataset('images', data=images_arr, compression='gzip', compression_opts=1)
            hf.create_dataset('grasps', data=grasps_unified)
            hf.create_dataset('action_mask', data=mask_info.mask)
            obj_names = sorted(object_names)
            lang = f"grasp the {obj_names[0]}" if obj_names else f"grasp object in {scene_name}"
            hf.create_dataset('language', data=lang)
            hf.attrs['dataset_name'] = 'dexgraspnet'
            hf.attrs['embodiment_id'] = embodiment_id
            hf.attrs['n_grasps'] = len(all_grasps)
            hf.attrs['n_views'] = n_views
            hf.attrs['image_size'] = 0  # original resolution; resize at training time

        ep_metadata_list.append({'filename': scene_filename, 'length': 1, 'n_grasps': len(all_grasps)})

    _write_metadata(output_dir, 'dexgraspnet', embodiment_id, True,
                    [f'view_{i}' for i in range(8)], all_actions, ep_metadata_list)
    total_grasps = sum(ep.get('n_grasps', 0) for ep in ep_metadata_list)
    logger.info(f"  ✅ dexgraspnet: {len(ep_metadata_list)} scenes, {total_grasps} grasps → {output_dir}")
    return True


# =============================================================================
# DexWild — download tar + extract → read source HDF5 → episode HDF5
# =============================================================================

def prepare_dexwild(
    ds_info: dict,
    output_dir: Path,
    data_root: Path,
    image_size: int = 448,
    max_episodes: int = None,
    **kwargs,
):
    """Download + convert DexWild in one pass."""
    from huggingface_hub import snapshot_download

    dex_dir = data_root / 'dexwild'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ──
    src_hdf5_files = sorted(dex_dir.glob('**/*.hdf5'))
    if not src_hdf5_files:
        logger.info("  Downloading DexWild ...")
        try:
            snapshot_download(
                repo_id=ds_info['hf_repo'], repo_type='dataset',
                local_dir=str(dex_dir),
            )
            src_hdf5_files = sorted(dex_dir.glob('**/*.hdf5'))
        except Exception as e:
            logger.error(f"  Download failed: {e}")
            return False

    if not src_hdf5_files:
        logger.error(f"  No HDF5 files found in {dex_dir}")
        return False

    # ── Convert ──
    logger.info(f"  Converting dexwild from {len(src_hdf5_files)} HDF5 source files")

    embodiment_id = DATASET_EMBODIMENT.get('dexwild', 3)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('dexwild', 22)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    all_actions = []
    ep_metadata_list = []
    ep_counter = 0

    for src_path in src_hdf5_files:
        task_name = src_path.stem.replace('_data', '').replace('robot_', '').replace('human_', '')
        logger.info(f"    Reading {src_path.name} (task={task_name})...")

        try:
            with h5py.File(str(src_path), 'r') as src:
                ep_keys = sorted([k for k in src.keys() if k.startswith('ep_')])
                if max_episodes:
                    remaining = max_episodes - ep_counter
                    if remaining <= 0:
                        break
                    ep_keys = ep_keys[:remaining]

                for ep_key in tqdm(ep_keys, desc=f"    {src_path.stem}", leave=False):
                    try:
                        ep = src[ep_key]

                        finger_joints = None
                        for hand_key in ['right_leapv2', 'left_leapv2', 'right_leapv1', 'left_leapv1']:
                            if hand_key in ep and hand_key in ep[hand_key]:
                                raw = ep[hand_key][hand_key][:]
                                finger_joints = raw[:, 1:].astype(np.float32)
                                break
                        if finger_joints is None:
                            continue

                        ep_len = finger_joints.shape[0]
                        n_finger = min(finger_joints.shape[1], 16)

                        wrist_eef = np.zeros((ep_len, 7), dtype=np.float32)
                        for eef_key in ['right_arm_eef', 'left_arm_eef']:
                            if eef_key in ep and eef_key in ep[eef_key]:
                                raw = ep[eef_key][eef_key][:].astype(np.float32)
                                T_w = min(raw.shape[0], ep_len)
                                wrist_eef[:T_w, :7] = raw[:T_w, 1:8]
                                break

                        actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                        actions_native[:, :7] = wrist_eef
                        actions_native[:, 7:7 + n_finger] = finger_joints[:, :n_finger]

                        actions_unified = map_to_unified(actions_native, embodiment_id)
                        all_actions.append(actions_unified)

                        ep_filename = f"ep_{ep_counter:06d}.hdf5"
                        ep_path = output_dir / ep_filename

                        with h5py.File(str(ep_path), 'w') as hf:
                            n_views_written = 0
                            for cam_key in ['right_thumb_cam', 'right_pinky_cam',
                                            'left_thumb_cam', 'left_pinky_cam']:
                                if cam_key not in ep:
                                    continue
                                cam_group = ep[cam_key]
                                img_keys = sorted([k for k in cam_group.keys()
                                                   if k.endswith('.jpg') or k.endswith('.png')])
                                if not img_keys:
                                    continue
                                n_frames = min(len(img_keys), ep_len)
                                view_data = np.zeros((ep_len, image_size, image_size, 3), dtype=np.uint8)
                                for t in range(n_frames):
                                    img = cam_group[img_keys[t]][:]
                                    if img.shape[0] != image_size or img.shape[1] != image_size:
                                        img = cv2.resize(img, (image_size, image_size))
                                    view_data[t] = img
                                if 0 < n_frames < ep_len:
                                    view_data[n_frames:] = view_data[n_frames - 1]
                                hf.create_dataset(
                                    f'images/view_{n_views_written}', data=view_data,
                                    chunks=(1, image_size, image_size, 3),
                                    compression='gzip', compression_opts=1,
                                )
                                n_views_written += 1

                            if n_views_written == 0:
                                ep_path.unlink(missing_ok=True)
                                continue

                            hf.create_dataset('actions', data=actions_unified)
                            hf.create_dataset('action_mask', data=mask_info.mask)
                            hf.create_dataset('language', data="pour the liquid")
                            hf.attrs['dataset_name'] = 'dexwild'
                            hf.attrs['embodiment_id'] = embodiment_id
                            hf.attrs['episode_length'] = ep_len
                            hf.attrs['n_views'] = n_views_written
                            hf.attrs['action_dim'] = native_dim
                            hf.attrs['image_size'] = 0  # original resolution; resize at training time

                        ep_metadata_list.append({'filename': ep_filename, 'length': ep_len})
                        ep_counter += 1
                    except Exception as e:
                        logger.debug(f"      Error in {ep_key}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"    Error reading {src_path.name}: {e}")
            continue

    if not ep_metadata_list:
        logger.error("  No DexWild episodes converted")
        return False

    _write_metadata(output_dir, 'dexwild', embodiment_id, False,
                    ['thumb_cam', 'pinky_cam'], all_actions, ep_metadata_list)
    logger.info(f"  ✅ dexwild: {len(ep_metadata_list)} episodes → {output_dir}")
    return True


# =============================================================================
# Metadata helper
# =============================================================================

def _write_metadata(
    output_dir: Path,
    ds_name: str,
    embodiment_id: int,
    is_static: bool,
    view_names: list,
    all_actions: list,
    ep_metadata_list: list,
):
    """Compute action stats and write metadata.json."""
    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_cat.mean(axis=0).tolist()
        action_std = all_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    metadata = {
        'dataset_name': ds_name,
        'embodiment_id': embodiment_id,
        'is_static': is_static,
        'view_names': view_names,
        'total_samples': sum(ep.get('n_grasps', ep.get('length', 0)) for ep in ep_metadata_list),
        'total_episodes': len(ep_metadata_list),
        'action_stats': {'mean': action_mean, 'std': action_std},
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


# =============================================================================
# Main entry point
# =============================================================================

PREPARE_FN = {
    'lerobot': 'lerobot',
    'rlbench': 'rlbench',
    'dexgraspnet': 'dexgraspnet',
    'dexwild': 'dexwild',
}


def main():
    parser = argparse.ArgumentParser(
        description='Drifting-VLA Data Preparation (download + convert, no intermediate files)',
    )
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset name')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Multiple dataset names (for parallel prep)')
    parser.add_argument('--all', action='store_true', help='Prepare all datasets')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-root', type=str, default='./data/episodes')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Max episodes per dataset (for quick tests)')
    parser.add_argument('--force', action='store_true',
                        help='Re-prepare even if episodes already exist')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove HF cache after each dataset to save disk')
    parser.add_argument('--rgb-only', action='store_true', default=True,
                        help='Only keep RGB image keys, exclude depth/seg/mask (default: True)')
    parser.add_argument('--no-rgb-only', dest='rgb_only', action='store_false',
                        help='Keep ALL image keys including depth/seg')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel processes for dataset preparation (default: 1)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        ds_names = list(DATASETS.keys())
    elif args.datasets:
        ds_names = args.datasets
    elif args.dataset:
        ds_names = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    # Filter to only datasets that need processing
    ds_to_process = []
    ds_skipped = []
    for ds_name in ds_names:
        if ds_name not in DATASETS:
            logger.warning(f"Unknown dataset: {ds_name}, skipping")
            continue
        output_dir = Path(args.output_root) / ds_name
        if not args.force and (output_dir / 'metadata.json').exists():
            ds_skipped.append(ds_name)
        else:
            ds_to_process.append(ds_name)

    logger.info(f"{'=' * 60}")
    logger.info(f"Drifting-VLA Data Preparation")
    logger.info(f"  To process: {len(ds_to_process)} datasets")
    if ds_skipped:
        logger.info(f"  Skipped (already done): {len(ds_skipped)}")
    if args.max_episodes:
        logger.info(f"  Max episodes: {args.max_episodes}")
    logger.info(f"  RGB only: {args.rgb_only}")
    if args.parallel > 1:
        logger.info(f"  Parallel workers: {args.parallel}")
    logger.info(f"{'=' * 60}")

    if not ds_to_process:
        logger.info("All datasets already prepared!")
        sys.exit(0)

    # Prepare datasets (parallel or sequential)
    if args.parallel > 1 and len(ds_to_process) > 1:
        results = _run_parallel(ds_to_process, args)
    else:
        results = {}
        for ds_name in ds_to_process:
            results[ds_name] = _prepare_single(ds_name, args)

    # Include skipped as success
    for ds_name in ds_skipped:
        results[ds_name] = True

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary:")
    for name, ok in sorted(results.items()):
        logger.info(f"  {'✅' if ok else '❌'} {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        logger.error(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info(f"\n✅ All {len(results)} dataset(s) ready!")


def _prepare_single(ds_name: str, args) -> bool:
    """Prepare a single dataset. Returns True on success."""
    ds_info = DATASETS[ds_name]
    data_root = Path(args.data_root)
    output_dir = Path(args.output_root) / ds_name

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Preparing: {ds_name} ({ds_info['description']})")
    logger.info(f"{'=' * 60}")

    try:
        ds_type = ds_info['type']

        if ds_type == 'lerobot':
            ok = prepare_lerobot(
                ds_name, ds_info['hf_repo'], output_dir,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
                rgb_only=args.rgb_only,
            )
        elif ds_type == 'rlbench':
            ok = prepare_rlbench(
                ds_info, output_dir, data_root,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
            )
        elif ds_type == 'dexgraspnet':
            ok = prepare_dexgraspnet(
                ds_info, output_dir, data_root,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
            )
        elif ds_type == 'dexwild':
            ok = prepare_dexwild(
                ds_info, output_dir, data_root,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
            )
        else:
            logger.warning(f"  Unknown type: {ds_type}")
            ok = False

    except Exception as e:
        logger.error(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    # Cleanup HF cache after each dataset to save disk
    if args.cleanup:
        import shutil
        for cache_dir in [
            Path.home() / '.cache' / 'huggingface' / 'datasets',
            Path.home() / '.cache' / 'huggingface' / 'hub',
        ]:
            if cache_dir.exists():
                try:
                    cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
                    if cache_size > 0.1:
                        logger.info(f"  Cleaning {cache_dir.name} cache ({cache_size:.1f} GB)...")
                        shutil.rmtree(str(cache_dir), ignore_errors=True)
                except Exception:
                    pass

    return ok


def _run_parallel(ds_names: list, args) -> dict:
    """Run dataset preparation in parallel using multiprocessing.
    
    Each process handles one dataset independently.
    """
    from multiprocessing import Pool, set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    n_workers = min(args.parallel, len(ds_names))
    logger.info(f"Starting {n_workers} parallel workers for {len(ds_names)} datasets...")

    # Build task list: (ds_name, args) pairs
    # We use subprocess instead of multiprocessing.Pool to avoid
    # pickling issues with the complex args and GPU context
    import subprocess
    
    procs = {}
    results = {}
    running = {}
    remaining = list(ds_names)
    
    while remaining or running:
        # Launch new processes up to n_workers
        while remaining and len(running) < n_workers:
            ds_name = remaining.pop(0)
            cmd = [
                sys.executable, 'scripts/prepare_data.py',
                '--dataset', ds_name,
                '--data-root', args.data_root,
                '--output-root', args.output_root,
                '--image-size', str(args.image_size),
            ]
            if args.max_episodes:
                cmd += ['--max-episodes', str(args.max_episodes)]
            if args.force:
                cmd += ['--force']
            if args.cleanup:
                cmd += ['--cleanup']
            if not args.rgb_only:
                cmd += ['--no-rgb-only']
            
            logger.info(f"  [PARALLEL] Starting: {ds_name}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=str(Path(__file__).parent.parent),
            )
            running[ds_name] = proc
        
        # Check for completed processes
        import time
        time.sleep(2)
        completed = []
        for ds_name, proc in running.items():
            ret = proc.poll()
            if ret is not None:
                output = proc.stdout.read()
                success = (ret == 0)
                results[ds_name] = success
                status = '✅' if success else '❌'
                logger.info(f"  [PARALLEL] {status} {ds_name} (exit={ret})")
                if not success:
                    # Print last few lines of output on failure
                    for line in output.strip().split('\n')[-5:]:
                        logger.error(f"    {line}")
                completed.append(ds_name)
        
        for ds_name in completed:
            del running[ds_name]
    
    return results


if __name__ == '__main__':
    main()

