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
        'hf_repo': 'boardd/dexwild-dataset',
        'type': 'dexwild',
        'description': 'DexWild dexterous LEAP hand (95K, 22-dim dex)',
    },
    'dexora': {
        'hf_repo': 'Dexora/Dexora_Real-World_Dataset',
        'type': 'dexora',
        'description': 'Dexora bimanual dexterous real-world (2.9M, 39-dim)',
    },
    # ── Open X-Embodiment additions (LeRobot format) ──
    'bridgev2': {
        'hf_repo': 'jxie/bridge_data_v2',
        'type': 'hf_generic',
        'description': 'BridgeV2 diverse manip (53K, 8-dim delta EEF)',
    },
    'kuka': {
        'hf_repo': 'lerobot/stanford_kuka_multimodal_dataset',
        'type': 'lerobot',
        'description': 'Stanford Kuka mobile manip (579K, 11-dim EEF+base)',
    },
    'berkeley_fanuc': {
        'hf_repo': 'lerobot/berkeley_fanuc_manipulation',
        'type': 'lerobot',
        'description': 'Berkeley FANUC manip (63K, 7-dim delta EEF)',
    },
    'cmu_play_fusion': {
        'hf_repo': 'lerobot/cmu_play_fusion',
        'type': 'lerobot',
        'description': 'CMU PlayFusion manip (236K, 8-dim EEF)',
    },
    'jaco_play': {
        'hf_repo': 'lerobot/jaco_play',
        'type': 'lerobot',
        'description': 'Jaco Play 3-DOF (78K, 3-dim delta pos)',
    },
    'austin_buds': {
        'hf_repo': 'lerobot/austin_buds_dataset',
        'type': 'lerobot',
        'description': 'UT Austin BUDS diverse manip (34K, 8-dim delta EEF)',
    },
    'austin_sailor': {
        'hf_repo': 'lerobot/austin_sailor_dataset',
        'type': 'lerobot',
        'description': 'UT Austin SAILOR manip (353K, 10-dim abs EEF+6D rot)',
    },
    'austin_sirius': {
        'hf_repo': 'lerobot/austin_sirius_dataset',
        'type': 'lerobot',
        'description': 'UT Austin SIRIUS manip (280K, 10-dim abs EEF+6D rot)',
    },
    'columbia_pusht': {
        'hf_repo': 'lerobot/columbia_cairlab_pusht_real',
        'type': 'lerobot',
        'description': 'Columbia PushT real (28K, 7-dim delta EEF)',
    },
    'nyu_door': {
        'hf_repo': 'lerobot/nyu_door_opening_surprising_effectiveness',
        'type': 'lerobot',
        'description': 'NYU door opening (20K, 7-dim EEF vel+gripper)',
    },
    # ── ALOHA Static variants (LeRobot format, 14-dim bimanual) ──
    'aloha_static_cups_open': {
        'hf_repo': 'lerobot/aloha_static_cups_open',
        'type': 'lerobot',
        'description': 'ALOHA static cups open (20K, 14-dim bimanual)',
    },
    'aloha_static_vinh_cup': {
        'hf_repo': 'lerobot/aloha_static_vinh_cup',
        'type': 'lerobot',
        'description': 'ALOHA static vinh cup (45.5K, 14-dim bimanual)',
    },
    'aloha_static_vinh_cup_left': {
        'hf_repo': 'lerobot/aloha_static_vinh_cup_left',
        'type': 'lerobot',
        'description': 'ALOHA static vinh cup left (50K, 14-dim bimanual)',
    },
    'aloha_static_coffee': {
        'hf_repo': 'lerobot/aloha_static_coffee',
        'type': 'lerobot',
        'description': 'ALOHA static coffee (55K, 14-dim bimanual)',
    },
    'aloha_static_pingpong': {
        'hf_repo': 'lerobot/aloha_static_pingpong_test',
        'type': 'lerobot',
        'description': 'ALOHA static pingpong (6K, 14-dim bimanual)',
    },
    'aloha_static_tape': {
        'hf_repo': 'lerobot/aloha_static_tape',
        'type': 'lerobot',
        'description': 'ALOHA static tape (35K, 14-dim bimanual)',
    },
    'aloha_static_pro_pencil': {
        'hf_repo': 'lerobot/aloha_static_pro_pencil',
        'type': 'lerobot',
        'description': 'ALOHA static pro pencil (8.75K, 14-dim bimanual)',
    },
    'aloha_static_candy': {
        'hf_repo': 'lerobot/aloha_static_candy',
        'type': 'lerobot',
        'description': 'ALOHA static candy (35K, 14-dim bimanual)',
    },
    'aloha_static_fork': {
        'hf_repo': 'lerobot/aloha_static_fork_pick_up',
        'type': 'lerobot',
        'description': 'ALOHA static fork pick up (60K, 14-dim bimanual)',
    },
    'aloha_static_velcro': {
        'hf_repo': 'lerobot/aloha_static_thread_velcro',
        'type': 'lerobot',
        'description': 'ALOHA static thread velcro (34.3K, 14-dim bimanual)',
    },
    'aloha_static_battery': {
        'hf_repo': 'lerobot/aloha_static_battery',
        'type': 'lerobot',
        'description': 'ALOHA static battery (29.4K, 14-dim bimanual)',
    },
    'aloha_static_screw': {
        'hf_repo': 'lerobot/aloha_static_screw_driver',
        'type': 'lerobot',
        'description': 'ALOHA static screw driver (20K, 14-dim bimanual)',
    },
    'aloha_static_towel': {
        'hf_repo': 'lerobot/aloha_static_towel',
        'type': 'lerobot',
        'description': 'ALOHA static towel (25K, 14-dim bimanual)',
    },
    'aloha_static_ziploc': {
        'hf_repo': 'lerobot/aloha_static_ziploc_slide',
        'type': 'lerobot',
        'description': 'ALOHA static ziploc (16.8K, 14-dim bimanual)',
    },
    # ── ALOHA Mobile variants (LeRobot format, 16-dim bimanual+base) ──
    'aloha_mobile_cabinet': {
        'hf_repo': 'lerobot/aloha_mobile_cabinet',
        'type': 'lerobot',
        'description': 'ALOHA mobile cabinet (128K, 16-dim bimanual+base)',
    },
    'aloha_mobile_chair': {
        'hf_repo': 'lerobot/aloha_mobile_chair',
        'type': 'lerobot',
        'description': 'ALOHA mobile chair (110K, 16-dim bimanual+base)',
    },
    'aloha_mobile_wash_pan': {
        'hf_repo': 'lerobot/aloha_mobile_wash_pan',
        'type': 'lerobot',
        'description': 'ALOHA mobile wash pan (55K, 16-dim bimanual+base)',
    },
    'aloha_mobile_wipe_wine': {
        'hf_repo': 'lerobot/aloha_mobile_wipe_wine',
        'type': 'lerobot',
        'description': 'ALOHA mobile wipe wine (65K, 16-dim bimanual+base)',
    },
    'aloha_mobile_elevator': {
        'hf_repo': 'lerobot/aloha_mobile_elevator',
        'type': 'lerobot',
        'description': 'ALOHA mobile elevator (45K, 16-dim bimanual+base)',
    },
    'aloha_mobile_shrimp': {
        'hf_repo': 'lerobot/aloha_mobile_shrimp',
        'type': 'lerobot',
        'description': 'ALOHA mobile shrimp (67.5K, 16-dim bimanual+base)',
    },
    # ── Additional OXE datasets (LeRobot format) ──
    'berkeley_rpt': {
        'hf_repo': 'lerobot/berkeley_rpt',
        'type': 'lerobot',
        'description': 'Berkeley RPT (393K, 8-dim joint delta)',
    },
    'toto': {
        'hf_repo': 'lerobot/toto',
        'type': 'lerobot',
        'description': 'TOTO (326K, 8-dim delta EEF)',
    },
    'stanford_robocook': {
        'hf_repo': 'lerobot/stanford_robocook',
        'type': 'lerobot',
        'description': 'Stanford RoboCook (113K, 7-dim EEF vel)',
    },
    'berkeley_mvp': {
        'hf_repo': 'lerobot/berkeley_mvp',
        'type': 'lerobot',
        'description': 'Berkeley MVP (45.3K, 7-dim delta EEF)',
    },
    'kaist_nonprehensile': {
        'hf_repo': 'lerobot/kaist_nonprehensile',
        'type': 'lerobot',
        'description': 'KAIST nonprehensile (32.4K, 4-dim EEF vel)',
    },
    'ucsd_pick_place': {
        'hf_repo': 'lerobot/ucsd_pick_and_place_dataset',
        'type': 'lerobot',
        'description': 'UCSD pick and place (67.8K, 8-dim delta EEF)',
    },
    'ucsd_kitchen': {
        'hf_repo': 'lerobot/ucsd_kitchen_dataset',
        'type': 'lerobot',
        'description': 'UCSD kitchen (3.97K, 8-dim delta EEF)',
    },
    'asu_table_top': {
        'hf_repo': 'lerobot/asu_table_top',
        'type': 'lerobot',
        'description': 'ASU table top (26.1K, 9-dim joint vel)',
    },
    'utokyo_pr2_fridge': {
        'hf_repo': 'lerobot/utokyo_pr2_opening_fridge',
        'type': 'lerobot',
        'description': 'UTokyo PR2 fridge (11.5K, 8-dim delta EEF)',
    },
    'utokyo_pr2_tabletop': {
        'hf_repo': 'lerobot/utokyo_pr2_tabletop_manipulation',
        'type': 'lerobot',
        'description': 'UTokyo PR2 tabletop (32.7K, 8-dim delta EEF)',
    },
    'utokyo_xarm_bimanual': {
        'hf_repo': 'lerobot/utokyo_xarm_bimanual',
        'type': 'lerobot',
        'description': 'UTokyo xArm bimanual (1.51K, 14-dim bimanual)',
    },
    'tokyo_u_lsmo': {
        'hf_repo': 'lerobot/tokyo_u_lsmo',
        'type': 'lerobot',
        'description': 'Tokyo U LSMO (11.9K, 8-dim delta EEF)',
    },
    'dlr_sara_grid': {
        'hf_repo': 'lerobot/dlr_sara_grid_clamp',
        'type': 'lerobot',
        'description': 'DLR SARA grid clamp (7.62K, 8-dim delta EEF)',
    },
    'dlr_sara_pour': {
        'hf_repo': 'lerobot/dlr_sara_pour',
        'type': 'lerobot',
        'description': 'DLR SARA pour (13K, 8-dim delta EEF)',
    },
    'dlr_edan': {
        'hf_repo': 'lerobot/dlr_edan_shared_control',
        'type': 'lerobot',
        'description': 'DLR EDAN shared control (8.93K, 8-dim delta EEF)',
    },
    'nyu_rot': {
        'hf_repo': 'lerobot/nyu_rot_dataset',
        'type': 'lerobot',
        'description': 'NYU ROT (440, 8-dim delta EEF)',
    },
    'usc_cloth_sim': {
        'hf_repo': 'lerobot/usc_cloth_sim',
        'type': 'lerobot',
        'description': 'USC cloth sim (100K, 4-dim delta EEF)',
    },
    'cmu_franka_exploration': {
        'hf_repo': 'lerobot/cmu_franka_exploration_dataset',
        'type': 'lerobot',
        'description': 'CMU Franka exploration (1.99K, 8-dim delta EEF)',
    },
    'imperialcollege_sawyer': {
        'hf_repo': 'lerobot/imperialcollege_sawyer_wrist_cam',
        'type': 'lerobot',
        'description': 'Imperial College Sawyer (7.15K, 7-dim EEF vel)',
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
    'utaustin_mutex': ['image', 'wrist_image'],
    'cmu_stretch': ['image'],
    'nyu_franka': ['image'],
    'stanford_hydra': ['image'],
    'dexwild': ['thumb_cam', 'pinky_cam'],
    # Open X-Embodiment additions (auto-detected by _detect_image_keys if empty)
    'bridgev2': ['image'],
    'kuka': ['image'],
    'berkeley_fanuc': ['image', 'wrist_image'],
    'cmu_play_fusion': ['image'],
    'jaco_play': ['image'],
    'austin_buds': ['image'],
    'austin_sailor': ['image'],
    'austin_sirius': ['image'],
    'columbia_pusht': ['image'],
    'nyu_door': ['image'],
    # ALOHA Static variants (top camera)
    'aloha_static_cups_open': ['top'],
    'aloha_static_vinh_cup': ['top'],
    'aloha_static_vinh_cup_left': ['top'],
    'aloha_static_coffee': ['top'],
    'aloha_static_pingpong': ['top'],
    'aloha_static_tape': ['top'],
    'aloha_static_pro_pencil': ['top'],
    'aloha_static_candy': ['top'],
    'aloha_static_fork': ['top'],
    'aloha_static_velcro': ['top'],
    'aloha_static_battery': ['top'],
    'aloha_static_screw': ['top'],
    'aloha_static_towel': ['top'],
    'aloha_static_ziploc': ['top'],
    # ALOHA Mobile variants
    'aloha_mobile_cabinet': ['image'],
    'aloha_mobile_chair': ['image'],
    'aloha_mobile_wash_pan': ['image'],
    'aloha_mobile_wipe_wine': ['image'],
    'aloha_mobile_elevator': ['image'],
    'aloha_mobile_shrimp': ['image'],
    # Additional OXE datasets
    'berkeley_rpt': ['image'],
    'toto': ['image'],
    'stanford_robocook': ['image'],
    'berkeley_mvp': ['image'],
    'kaist_nonprehensile': ['image'],
    'ucsd_pick_place': ['image'],
    'ucsd_kitchen': ['image'],
    'asu_table_top': ['image'],
    'utokyo_pr2_fridge': ['image'],
    'utokyo_pr2_tabletop': ['image'],
    'utokyo_xarm_bimanual': ['image'],
    'tokyo_u_lsmo': ['image'],
    'dlr_sara_grid': ['image'],
    'dlr_sara_pour': ['image'],
    'dlr_edan': ['image'],
    'nyu_rot': ['image'],
    'usc_cloth_sim': ['image'],
    'cmu_franka_exploration': ['image'],
    'imperialcollege_sawyer': ['image'],
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
# Dexora — multi-task LeRobot repo, each task is a sub-dataset
# =============================================================================

DEXORA_TASKS = [
    'pick_yellow_egg', 'close_laptop_bimanual', 'stack_colored_square_blocks',
    'fold_towel_bimanual', 'hammer_nail', 'open_box_lid_bimanual',
    'place_cylinder_on_base', 'sweep_garbage_into_dustpan',
    'turn_rubiks_cube_bimanual', 'cut_green_onion',
]


def prepare_dexora(
    ds_info: dict,
    output_dir: Path,
    data_root: Path,
    image_size: int = 448,
    max_episodes: int = None,
    **kwargs,
):
    """Download + convert Dexora tasks via LeRobot sub-dataset loading.

    Dexora/Dexora_Real-World_Dataset is a multi-task repo where each task
    is a proper LeRobot v2 dataset at dexora/{task_name}/.  We download
    a subset of tasks and merge episodes into a single output directory.
    """
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = ds_info['hf_repo']

    embodiment_id = DATASET_EMBODIMENT.get('dexora', 6)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('dexora', 39)
    field_names = DATASET_FIELD_FORMATS.get('dexora')
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim, dataset_name='dexora')

    all_actions = []
    ep_metadata_list = []
    ep_counter = 0
    tasks_to_try = DEXORA_TASKS

    for task_name in tasks_to_try:
        if max_episodes and ep_counter >= max_episodes:
            break

        logger.info(f"  Dexora task: {task_name}")
        task_prefix = f"dexora/{task_name}/"

        try:
            task_dir = data_root / 'dexora' / task_name
            if not (task_dir / 'meta' / 'info.json').exists():
                logger.info(f"    Downloading {task_name}...")
                snapshot_download(
                    repo_id=hf_repo, repo_type='dataset',
                    local_dir=str(data_root / 'dexora_repo'),
                    allow_patterns=[f"{task_prefix}**"],
                )
                import shutil
                src = data_root / 'dexora_repo' / 'dexora' / task_name
                if src.exists():
                    task_dir.mkdir(parents=True, exist_ok=True)
                    if task_dir != src:
                        shutil.copytree(str(src), str(task_dir), dirs_exist_ok=True)

            if not (task_dir / 'meta' / 'info.json').exists():
                logger.warning(f"    Skipping {task_name}: no meta/info.json")
                continue

            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            lerobot_ds = LeRobotDataset(
                hf_repo, video_backend='pyav',
                root=str(task_dir),
            )
        except Exception as e:
            logger.warning(f"    Failed to load {task_name}: {e}")
            try:
                from datasets import load_dataset
                parquet_dir = task_dir / 'data'
                parquet_files = sorted(parquet_dir.rglob('*.parquet')) if parquet_dir.exists() else []
                if not parquet_files:
                    continue
                hf_ds = load_dataset('parquet', data_files=[str(p) for p in parquet_files], split='train')
                lerobot_ds = None
            except Exception as e2:
                logger.warning(f"    Also failed parquet fallback: {e2}")
                continue

        try:
            if lerobot_ds is not None:
                hf = lerobot_ds.hf_dataset
            else:
                hf = hf_ds

            total_frames = len(hf)
            logger.info(f"    {task_name}: {total_frames} frames")

            episodes = defaultdict(list)
            ep_col = hf['episode_index']
            for i in range(total_frames):
                episodes[int(ep_col[i])].append(i)

            ep_ids = sorted(episodes.keys())
            remaining = (max_episodes - ep_counter) if max_episodes else len(ep_ids)
            ep_ids = ep_ids[:remaining]

            for ep_id in tqdm(ep_ids, desc=f"    {task_name}", leave=False):
                frame_indices = sorted(episodes[ep_id])
                ep_len = len(frame_indices)

                ep_filename = f"ep_{ep_counter:06d}.hdf5"
                ep_path = output_dir / ep_filename

                with h5py.File(str(ep_path), 'w') as hf_out:
                    actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                    for t, fi in enumerate(frame_indices):
                        act = hf[fi]['action']
                        if act is not None:
                            a = np.asarray(act).flatten().astype(np.float32)
                            actions_native[t, :min(len(a), native_dim)] = a[:native_dim]

                    if field_names:
                        actions_unified, _ = assemble_state_vec_batch(actions_native, field_names)
                    else:
                        actions_unified = map_to_unified(actions_native, embodiment_id)
                    hf_out.create_dataset('actions', data=actions_unified)
                    all_actions.append(actions_unified)
                    hf_out.create_dataset('action_mask', data=mask_info.mask)

                    lang = task_name.replace('_', ' ')
                    hf_out.create_dataset('language', data=lang)

                    # Images: try video decode or per-frame
                    n_views_written = 0
                    if lerobot_ds is not None:
                        sample0 = lerobot_ds[0]
                        img_keys = [k for k in sample0.keys() if 'image' in k.lower()]
                        for v_idx, img_key in enumerate(img_keys[:4]):
                            images = []
                            for fi in frame_indices:
                                try:
                                    s = lerobot_ds[fi]
                                    iv = s.get(img_key)
                                    if iv is not None:
                                        img = _lerobot_to_numpy_image(iv)
                                        if img is not None:
                                            images.append(img)
                                            continue
                                except Exception:
                                    pass
                                if images:
                                    images.append(np.zeros_like(images[0]))
                            if images:
                                images = [im for im in images if im is not None and isinstance(im, np.ndarray)]
                                if images:
                                    frames = np.stack(images).astype(np.uint8)
                                    hf_out.create_dataset(
                                        f'images/view_{n_views_written}', data=frames,
                                        chunks=(1,) + frames.shape[1:],
                                        compression='gzip', compression_opts=1,
                                    )
                                    n_views_written += 1

                    hf_out.attrs['dataset_name'] = 'dexora'
                    hf_out.attrs['embodiment_id'] = embodiment_id
                    hf_out.attrs['episode_length'] = ep_len
                    hf_out.attrs['n_views'] = n_views_written
                    hf_out.attrs['action_dim'] = native_dim
                    hf_out.attrs['image_size'] = 0

                ep_metadata_list.append({'filename': ep_filename, 'length': ep_len})
                ep_counter += 1

                if max_episodes and ep_counter >= max_episodes:
                    break

        except Exception as e:
            logger.warning(f"    Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not ep_metadata_list:
        logger.error("  No Dexora episodes converted")
        return False

    _write_metadata(output_dir, 'dexora', embodiment_id, False,
                    ['top', 'wrist_left', 'wrist_right', 'front'],
                    all_actions, ep_metadata_list)
    logger.info(f"  ✅ dexora: {len(ep_metadata_list)} episodes from {len(DEXORA_TASKS)} tasks → {output_dir}")
    return True


# =============================================================================
# HF Generic — standard HuggingFace datasets (parquet/video, not LeRobot)
# =============================================================================

def prepare_hf_generic(
    ds_name: str,
    ds_info: dict,
    output_dir: Path,
    data_root: Path,
    image_size: int = 448,
    max_episodes: int = None,
    **kwargs,
):
    """Download + convert a generic HuggingFace dataset (parquet with video).

    For datasets like jxie/bridge_data_v2 where each row is one episode
    with a 'video' column (video file) + 'text' column (language instruction).
    Requires FFmpeg for video decoding.
    """
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = ds_info['hf_repo']

    logger.info(f"  Loading HF generic dataset: {hf_repo}")
    try:
        hf_ds = load_dataset(hf_repo, split='train')
        total = len(hf_ds)
        logger.info(f"  Loaded {total} episodes (video-per-row format)")
    except Exception as e:
        logger.error(f"  Failed to load {hf_repo}: {e}")
        import traceback
        traceback.print_exc()
        return False

    embodiment_id = DATASET_EMBODIMENT.get(ds_name, 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name, 7)
    field_names = DATASET_FIELD_FORMATS.get(ds_name)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim, dataset_name=ds_name)

    n_episodes = min(max_episodes, total) if max_episodes else total
    all_actions = []
    ep_metadata_list = []

    try:
        import av
    except ImportError:
        logger.error("  pyav not installed — required for video decoding. pip install av")
        return False

    cols = hf_ds.column_names
    has_video = 'video' in cols
    has_text = 'text' in cols

    if not has_video:
        logger.warning("  No 'video' column — falling back to _convert_hf_dataset_to_episodes")
        return _convert_hf_dataset_to_episodes(
            hf_ds, ds_name, output_dir, image_size, max_episodes,
        )

    import io

    # Get non-video columns for metadata access without triggering video decode
    non_video_cols = [c for c in cols if c != 'video']
    hf_ds_meta = hf_ds.select_columns(non_video_cols) if non_video_cols else None

    # Access raw video bytes from the underlying Arrow table to avoid torchcodec
    # HF datasets wraps parquet in Arrow; video column stores bytes directly
    logger.info(f"  Columns: {cols}")
    if has_video:
        # Peek at first row to understand format
        try:
            raw_table = hf_ds.data  # pyarrow Table
            video_col = raw_table.column('video')
            first_chunk = video_col.chunk(0)
            sample_val = first_chunk[0].as_py()
            logger.info(f"  Video column raw type: {type(sample_val)}, "
                        f"keys={list(sample_val.keys()) if isinstance(sample_val, dict) else 'N/A'}")
        except Exception as e:
            logger.info(f"  Could not peek video column: {e}")

    for ep_num in tqdm(range(n_episodes), desc=f"  {ds_name}"):
        try:
            row = hf_ds_meta[ep_num] if hf_ds_meta else {}

            # Read video bytes from raw Arrow table (bypasses torchcodec)
            video_source = None
            try:
                raw_table = hf_ds.data
                video_col = raw_table.column('video')
                # Find which chunk contains this index
                offset = 0
                for chunk in video_col.chunks:
                    if ep_num < offset + len(chunk):
                        raw_val = chunk[ep_num - offset].as_py()
                        break
                    offset += len(chunk)
                else:
                    raw_val = None

                if isinstance(raw_val, dict):
                    if raw_val.get('bytes'):
                        video_source = io.BytesIO(raw_val['bytes'])
                    elif raw_val.get('path'):
                        video_source = raw_val['path']
                elif isinstance(raw_val, bytes):
                    video_source = io.BytesIO(raw_val)
                elif isinstance(raw_val, str):
                    video_source = raw_val
            except Exception as e:
                logger.warning(f"  Ep {ep_num}: raw Arrow read failed: {e}")

            if video_source is None:
                if ep_num < 3:
                    logger.warning(f"  Ep {ep_num}: no video source found")
                continue

            container = av.open(video_source)
            stream = container.streams.video[0]
            frames = []
            for frame in container.decode(stream):
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
            container.close()

            ep_len = len(frames)
            if ep_len == 0:
                continue

            ep_filename = f"ep_{ep_num:06d}.hdf5"
            ep_path = output_dir / ep_filename

            with h5py.File(str(ep_path), 'w') as hf_out:
                img_arr = np.stack(frames).astype(np.uint8)
                hf_out.create_dataset(
                    'images/view_0', data=img_arr,
                    chunks=(1,) + img_arr.shape[1:],
                    compression='gzip', compression_opts=1,
                )

                actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                if field_names:
                    actions_unified, _ = assemble_state_vec_batch(actions_native, field_names)
                else:
                    actions_unified = map_to_unified(actions_native, embodiment_id)
                hf_out.create_dataset('actions', data=actions_unified)
                all_actions.append(actions_unified)
                hf_out.create_dataset('action_mask', data=mask_info.mask)

                lang = row.get('text', '') if has_text else ''
                hf_out.create_dataset('language', data=lang or '')

                hf_out.attrs['dataset_name'] = ds_name
                hf_out.attrs['embodiment_id'] = embodiment_id
                hf_out.attrs['episode_length'] = ep_len
                hf_out.attrs['n_views'] = 1
                hf_out.attrs['action_dim'] = native_dim
                hf_out.attrs['image_size'] = 0

            ep_metadata_list.append({'filename': ep_filename, 'length': ep_len})

        except Exception as e:
            logger.warning(f"  Error processing episode {ep_num}: {e}")
            continue

    if not ep_metadata_list:
        logger.error(f"  No {ds_name} episodes converted")
        return False

    view_names = DATASET_VIEW_NAMES.get(ds_name, ['view_0'])
    _write_metadata(output_dir, ds_name, embodiment_id, False,
                    view_names, all_actions, ep_metadata_list)
    logger.info(f"  ✅ {ds_name}: {len(ep_metadata_list)} episodes → {output_dir}")
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
    'dexwild': 'dexwild',
    'dexora': 'dexora',
    'hf_generic': 'hf_generic',
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
        elif ds_type == 'dexwild':
            ok = prepare_dexwild(
                ds_info, output_dir, data_root,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
            )
        elif ds_type == 'dexora':
            ok = prepare_dexora(
                ds_info, output_dir, data_root,
                image_size=args.image_size,
                max_episodes=args.max_episodes,
            )
        elif ds_type == 'hf_generic':
            ok = prepare_hf_generic(
                ds_name, ds_info, output_dir, data_root,
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

