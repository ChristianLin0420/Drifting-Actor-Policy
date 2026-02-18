#!/usr/bin/env python3
"""
Convert raw datasets to Episode-Centric HDF5 format.

Phase 2 of the data pipeline:
  Phase 1: download_datasets.py → raw data (Arrow, PerAct, DexGraspNet)
  Phase 2: convert_to_episodes.py → standardized HDF5 per-episode

Usage:
    python scripts/convert_to_episodes.py --dataset aloha --image-size 448
    python scripts/convert_to_episodes.py --dataset dexgraspnet --image-size 448
    python scripts/convert_to_episodes.py --all --image-size 448

Output:
    data/episodes/{dataset_name}/
    ├── metadata.json
    ├── ep_000000.hdf5  (or scene_XXXX.hdf5 for dexgraspnet)
    └── ...
"""

import sys
import json
import argparse
import logging
import h5py
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.data.action_mapping import (
    DATASET_EMBODIMENT, DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM,
    LEROBOT_DATASETS, UNIFIED_ACTION_DIM,
    map_to_unified, get_action_mask,
)


# ─── Dataset-specific view name registries ───
DATASET_VIEW_NAMES = {
    'rlbench': ['front_rgb', 'wrist_rgb'],
    'aloha': ['top'],
    'bc_z': ['image', 'hand_image'],
    'taco_play': ['image'],
    'utaustin_mutex': [],  # no images
    'cmu_stretch': ['image'],
    'nyu_franka': ['image'],
    'stanford_hydra': ['image'],
    'dexgraspnet': [f'view_{i}' for i in range(8)],
    'dexwild': ['zed_obs', 'left_thumb_cam', 'right_thumb_cam'],
}
# behavior1k defaults
for i in range(50):
    DATASET_VIEW_NAMES[f'behavior1k_t{i:04d}'] = ['head', 'left_wrist', 'right_wrist']


def convert_lerobot_dataset(
    ds_name: str,
    data_root: str = './data',
    output_root: str = './data/episodes',
    image_size: int = 448,
    max_episodes: int = None,
):
    """Convert a LeRobot-format dataset (Arrow) to episode HDF5."""
    from datasets import load_from_disk

    arrow_path = Path(data_root) / ds_name / 'arrow_data'
    output_dir = Path(output_root) / ds_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not arrow_path.exists():
        logger.error(f"Arrow data not found: {arrow_path}")
        return False

    logger.info(f"Converting {ds_name} from {arrow_path}")

    # Load Arrow dataset
    ds = load_from_disk(str(arrow_path))
    logger.info(f"  Loaded {len(ds)} samples")

    # Detect image keys
    all_cols = ds.column_names
    image_keys = _detect_image_keys(all_cols)
    logger.info(f"  Image keys: {image_keys}")

    # Detect language key
    language_key = None
    for k in ['task', 'language_instruction', 'language', 'text']:
        if k in all_cols:
            language_key = k
            break

    # Action info
    embodiment_id = DATASET_EMBODIMENT.get(ds_name, 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name, 7)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    # Group by episode
    episodes = defaultdict(list)
    for i in range(len(ds)):
        ep_idx = ds[i].get('episode_index', 0)
        episodes[ep_idx].append(i)

    # Sort episodes
    ep_ids = sorted(episodes.keys())
    if max_episodes:
        ep_ids = ep_ids[:max_episodes]

    logger.info(f"  {len(ep_ids)} episodes to convert")

    # Action statistics accumulator
    all_actions = []
    ep_metadata_list = []

    for ep_num, ep_id in enumerate(tqdm(ep_ids, desc=f"Converting {ds_name}")):
        frame_indices = sorted(episodes[ep_id])
        ep_len = len(frame_indices)

        ep_filename = f"ep_{ep_num:06d}.hdf5"
        ep_path = output_dir / ep_filename

        with h5py.File(str(ep_path), 'w') as hf:
            # ── Images ──
            n_views = len(image_keys) if image_keys else 0
            if n_views > 0:
                # Read first image to get dimensions
                first_img = _extract_image(ds[frame_indices[0]], image_keys[0], image_size)
                if first_img is not None:
                    H, W = first_img.shape[:2]
                else:
                    H, W = image_size, image_size

                for v_idx, img_key in enumerate(image_keys):
                    view_data = np.zeros((ep_len, H, W, 3), dtype=np.uint8)
                    for t, fi in enumerate(frame_indices):
                        img = _extract_image(ds[fi], img_key, image_size)
                        if img is not None:
                            view_data[t] = img
                    hf.create_dataset(
                        f'images/view_{v_idx}', data=view_data,
                        chunks=(1, H, W, 3), compression='gzip', compression_opts=1,
                    )

            # ── Actions (map to 128-dim) ──
            actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
            for t, fi in enumerate(frame_indices):
                row = ds[fi]
                act = row.get('action')
                if act is not None:
                    act = np.array(act, dtype=np.float32)
                    actions_native[t, :min(len(act), native_dim)] = act[:native_dim]

            # Map to unified 128-dim
            actions_unified = map_to_unified(actions_native, embodiment_id)
            hf.create_dataset('actions', data=actions_unified)
            all_actions.append(actions_unified)

            # ── Action mask ──
            hf.create_dataset('action_mask', data=mask_info.mask)

            # ── Proprioception ──
            proprio_keys = [k for k in all_cols if 'state' in k.lower() or 'proprio' in k.lower() or k == 'observation.state']
            if proprio_keys:
                proprio_data = np.zeros((ep_len, UNIFIED_ACTION_DIM), dtype=np.float32)
                for t, fi in enumerate(frame_indices):
                    row = ds[fi]
                    for pk in proprio_keys:
                        val = row.get(pk)
                        if val is not None:
                            val = np.atleast_1d(np.array(val, dtype=np.float32))
                            # Map proprio to unified space (same as actions)
                            p_unified = map_to_unified(
                                val[:native_dim].reshape(1, -1), embodiment_id
                            )[0]
                            proprio_data[t] = p_unified
                            break
                hf.create_dataset('proprio', data=proprio_data)

            # ── Language ──
            lang = ""
            if language_key:
                first_row = ds[frame_indices[0]]
                lang_val = first_row.get(language_key, "")
                if lang_val:
                    lang = str(lang_val)
            hf.create_dataset('language', data=lang)

            # ── Attributes ──
            hf.attrs['dataset_name'] = ds_name
            hf.attrs['embodiment_id'] = embodiment_id
            hf.attrs['episode_length'] = ep_len
            hf.attrs['n_views'] = n_views
            hf.attrs['action_dim'] = native_dim
            hf.attrs['image_size'] = image_size

        ep_metadata_list.append({
            'filename': ep_filename,
            'length': ep_len,
            'n_grasps': 0,
        })

    # Compute action statistics
    if all_actions:
        all_actions_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_actions_cat.mean(axis=0).tolist()
        action_std = all_actions_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    # Write metadata
    view_names = DATASET_VIEW_NAMES.get(ds_name, [f'view_{i}' for i in range(len(image_keys))])
    metadata = {
        'dataset_name': ds_name,
        'embodiment_id': embodiment_id,
        'is_static': False,
        'view_names': view_names[:len(image_keys)],
        'total_samples': sum(ep['length'] for ep in ep_metadata_list),
        'total_episodes': len(ep_metadata_list),
        'action_stats': {
            'mean': action_mean,
            'std': action_std,
        },
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"  ✅ {ds_name}: {len(ep_metadata_list)} episodes, "
        f"{sum(ep['length'] for ep in ep_metadata_list)} frames → {output_dir}"
    )
    return True


def convert_rlbench_dataset(
    data_root: str = './data',
    output_root: str = './data/episodes',
    image_size: int = 448,
    max_episodes: int = None,
):
    """Convert RLBench PerAct format to episode HDF5."""
    import pickle

    rlbench_dir = Path(data_root) / 'rlbench'
    output_dir = Path(output_root) / 'rlbench'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rlbench_dir.exists():
        logger.error(f"RLBench not found: {rlbench_dir}")
        return False

    # Find episode directories
    ep_dirs = sorted(rlbench_dir.rglob('episode*'))
    ep_dirs = [d for d in ep_dirs if d.is_dir() and (d / 'low_dim_obs.pkl').exists()]

    if not ep_dirs:
        logger.error("No RLBench episodes found")
        return False

    if max_episodes:
        ep_dirs = ep_dirs[:max_episodes]

    logger.info(f"Converting rlbench: {len(ep_dirs)} episodes")

    embodiment_id = DATASET_EMBODIMENT.get('rlbench', 0)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('rlbench', 8)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    cameras = ['front_rgb', 'wrist_rgb']
    all_actions = []
    ep_metadata_list = []

    for ep_num, ep_dir in enumerate(tqdm(ep_dirs, desc="Converting rlbench")):
        try:
            # Load observations
            from drifting_vla.data.rlbench_dataset import _load_pickle
            obs_list = _load_pickle(ep_dir / 'low_dim_obs.pkl')
            ep_len = len(obs_list)
            if ep_len == 0:
                continue

            ep_filename = f"ep_{ep_num:06d}.hdf5"
            ep_path = output_dir / ep_filename

            with h5py.File(str(ep_path), 'w') as hf:
                # ── Images ──
                for v_idx, cam in enumerate(cameras):
                    cam_dir = ep_dir / cam
                    if not cam_dir.exists():
                        continue
                    img_files = sorted(cam_dir.glob('*.png'))
                    if not img_files:
                        continue

                    # Read first to get size
                    first = cv2.imread(str(img_files[0]))
                    if first is None:
                        continue
                    first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
                    first = cv2.resize(first, (image_size, image_size))

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

                # ── Actions ──
                actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                for t, obs in enumerate(obs_list):
                    try:
                        action = np.concatenate([
                            obs.gripper_pose,  # [7]: xyz + quat
                            [obs.gripper_open],  # [1]: gripper
                        ]).astype(np.float32)
                        actions_native[t, :min(len(action), native_dim)] = action[:native_dim]
                    except (AttributeError, TypeError):
                        pass

                actions_unified = map_to_unified(actions_native, embodiment_id)
                hf.create_dataset('actions', data=actions_unified)
                all_actions.append(actions_unified)

                hf.create_dataset('action_mask', data=mask_info.mask)

                # ── Language ──
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

                # Fallback: task name from path
                if not lang:
                    parts = ep_dir.parts
                    for p in parts:
                        if p not in ('train', 'val', 'test', 'all_variations', 'episodes') and not p.startswith('episode'):
                            lang = p.replace('_', ' ')

                hf.create_dataset('language', data=lang)

                hf.attrs['dataset_name'] = 'rlbench'
                hf.attrs['embodiment_id'] = embodiment_id
                hf.attrs['episode_length'] = ep_len
                hf.attrs['n_views'] = len(cameras)
                hf.attrs['action_dim'] = native_dim
                hf.attrs['image_size'] = image_size

            ep_metadata_list.append({
                'filename': ep_filename,
                'length': ep_len,
            })
        except Exception as e:
            logger.warning(f"  Error converting {ep_dir.name}: {e}")
            continue

    # Stats + metadata
    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_cat.mean(axis=0).tolist()
        action_std = all_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    metadata = {
        'dataset_name': 'rlbench',
        'embodiment_id': embodiment_id,
        'is_static': False,
        'view_names': cameras,
        'total_samples': sum(ep['length'] for ep in ep_metadata_list),
        'total_episodes': len(ep_metadata_list),
        'action_stats': {'mean': action_mean, 'std': action_std},
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✅ rlbench: {len(ep_metadata_list)} episodes → {output_dir}")
    return True


def convert_dexgraspnet_dataset(
    data_root: str = './data',
    output_root: str = './data/episodes',
    image_size: int = 448,
    max_scenes: int = None,
):
    """Convert DexGraspNet scenes to scene-based HDF5."""
    from scipy.spatial.transform import Rotation

    dex_dir = Path(data_root) / 'dexgraspnet'
    output_dir = Path(output_root) / 'dexgraspnet'
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes_dir = dex_dir / 'scenes'
    rendered_dir = dex_dir / 'rendered'

    if not scenes_dir.exists():
        logger.error(f"DexGraspNet scenes not found: {scenes_dir}")
        return False

    scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and d.name.startswith('scene_')])
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    logger.info(f"Converting dexgraspnet: {len(scene_dirs)} scenes")

    embodiment_id = DATASET_EMBODIMENT.get('dexgraspnet', 3)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('dexgraspnet', 23)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    all_actions = []
    ep_metadata_list = []

    for scene_dir in tqdm(scene_dirs, desc="Converting dexgraspnet"):
        scene_name = scene_dir.name

        # Find grasp files (NPZ)
        npz_files = sorted((scene_dir / 'realsense').glob('*.npz')) if (scene_dir / 'realsense').exists() else []
        if not npz_files:
            continue

        # Load rendered images
        images_list = []
        render_scene_dir = rendered_dir / scene_name if rendered_dir.exists() else None
        if render_scene_dir and render_scene_dir.exists():
            view_files = sorted(render_scene_dir.glob('view_*.png'))
            for vf in view_files[:8]:
                try:
                    img = cv2.imread(str(vf))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (image_size, image_size))
                        images_list.append(img)
                except Exception:
                    pass

        if not images_list:
            # Try depth fallback
            continue  # Skip scenes without rendered images

        n_views = len(images_list)
        images_arr = np.stack(images_list, axis=0).astype(np.uint8)  # [V, H, W, 3]

        # Collect all grasps from all viewpoints
        all_grasps = []
        object_names = set()
        for npz_file in npz_files:
            try:
                data = np.load(str(npz_file), allow_pickle=True)
                translations = data.get('translation')
                rotations = data.get('rotation')
                if translations is None or rotations is None:
                    continue

                K = translations.shape[0]
                for k in range(K):
                    # Build 23-dim action: wrist_xyz(3) + quat(4) + fingers(16)
                    xyz = translations[k].astype(np.float32)  # [3]
                    rot_mat = rotations[k].astype(np.float32)  # [3, 3]
                    try:
                        quat = Rotation.from_matrix(rot_mat).as_quat().astype(np.float32)  # [4] xyzw
                    except Exception:
                        quat = np.array([0, 0, 0, 1], dtype=np.float32)

                    fingers = np.zeros(16, dtype=np.float32)
                    for j in range(16):
                        jkey = f'j{j}'
                        if jkey in data:
                            jvals = data[jkey]
                            if k < len(jvals):
                                fingers[j] = float(jvals[k])

                    action_native = np.concatenate([xyz, quat, fingers])  # [23]
                    all_grasps.append(action_native)

                # Parse object names from XML
                xml_name = npz_file.stem + '.xml'
                xml_path = npz_file.parent / 'annotations' / xml_name
                if xml_path.exists():
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(str(xml_path))
                        for obj in tree.findall('.//object'):
                            name = obj.get('name', '')
                            if name:
                                object_names.add(name)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"  Error reading {npz_file.name}: {e}")
                continue

        if not all_grasps:
            continue

        grasps_native = np.stack(all_grasps, axis=0)  # [K, 23]
        grasps_unified = map_to_unified(grasps_native, embodiment_id)  # [K, 128]
        all_actions.append(grasps_unified)

        # Write scene HDF5
        scene_filename = f"{scene_name}.hdf5"
        scene_path = output_dir / scene_filename

        with h5py.File(str(scene_path), 'w') as hf:
            hf.create_dataset('images', data=images_arr, compression='gzip', compression_opts=1)
            hf.create_dataset('grasps', data=grasps_unified)
            hf.create_dataset('action_mask', data=mask_info.mask)

            obj_names_list = sorted(object_names)
            if obj_names_list:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset('object_names', data=obj_names_list, dtype=dt)
                hf.create_dataset('language', data=f"grasp the {obj_names_list[0]}")
            else:
                hf.create_dataset('language', data=f"grasp object in {scene_name}")

            hf.attrs['dataset_name'] = 'dexgraspnet'
            hf.attrs['embodiment_id'] = embodiment_id
            hf.attrs['n_grasps'] = len(all_grasps)
            hf.attrs['n_views'] = n_views
            hf.attrs['image_size'] = image_size

        ep_metadata_list.append({
            'filename': scene_filename,
            'length': 1,
            'n_grasps': len(all_grasps),
        })

    # Stats
    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_cat.mean(axis=0).tolist()
        action_std = all_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    total_grasps = sum(ep.get('n_grasps', 0) for ep in ep_metadata_list)
    metadata = {
        'dataset_name': 'dexgraspnet',
        'embodiment_id': embodiment_id,
        'is_static': True,
        'view_names': [f'view_{i}' for i in range(8)],
        'total_samples': total_grasps,
        'total_episodes': len(ep_metadata_list),
        'action_stats': {'mean': action_mean, 'std': action_std},
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✅ dexgraspnet: {len(ep_metadata_list)} scenes, {total_grasps} grasps → {output_dir}")
    return True


def convert_dexwild_dataset(
    data_root: str = './data',
    output_root: str = './data/episodes',
    image_size: int = 448,
    max_episodes: int = None,
):
    """Convert DexWild dataset (HDF5 format) to episode HDF5.

    DexWild HDF5 structure (per task file, e.g. robot_pour_data.hdf5):
      ep_XXXX/
        right_arm_eef/right_arm_eef     [T, 8] float64  — wrist EEF (xyz+quat+grip)
        right_leapv2/right_leapv2       [T, 18] float64 — LEAP hand v2 joint positions
        right_thumb_cam/timestamp.jpg   [H, W, 3] uint8  — thumb camera frames
        right_pinky_cam/timestamp.jpg   [H, W, 3] uint8  — pinky camera frames
        timesteps/timesteps             scalar string     — timestamps

    Maps to EMBODIMENT_DEXHAND:
        wrist EEF[0:6] (xyz + euler/quat) → Region A [0:6]
        LEAP hand joints[0:16] → Region D [32:48]
    """
    dex_dir = Path(data_root) / 'dexwild'
    output_dir = Path(output_root) / 'dexwild'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dex_dir.exists():
        logger.error(f"DexWild data not found: {dex_dir}")
        return False

    # Find all source HDF5 files
    src_hdf5_files = sorted(dex_dir.glob('**/*.hdf5'))
    if not src_hdf5_files:
        logger.error(f"No HDF5 files found in {dex_dir}")
        return False

    logger.info(f"Converting dexwild from {len(src_hdf5_files)} HDF5 source files")

    embodiment_id = DATASET_EMBODIMENT.get('dexwild', 3)
    native_dim = DATASET_NATIVE_ACTION_DIM.get('dexwild', 22)
    mask_info = get_action_mask(embodiment_id, native_dim=native_dim)

    all_actions = []
    ep_metadata_list = []
    ep_counter = 0

    for src_path in src_hdf5_files:
        task_name = src_path.stem.replace('_data', '').replace('robot_', '').replace('human_', '')
        logger.info(f"  Reading {src_path.name} (task={task_name})...")

        try:
            with h5py.File(str(src_path), 'r') as src:
                ep_keys = sorted([k for k in src.keys() if k.startswith('ep_')])
                if max_episodes:
                    ep_keys = ep_keys[:max_episodes]

                for ep_key in tqdm(ep_keys, desc=f"  {src_path.stem}"):
                    try:
                        ep = src[ep_key]

                        # ── Find hand joint data (LEAP v2 preferred) ──
                        # DexWild format: col 0 = timestamp, cols 1: = data
                        finger_joints = None
                        for hand_key in ['right_leapv2', 'left_leapv2', 'right_leapv1', 'left_leapv1']:
                            if hand_key in ep and hand_key in ep[hand_key]:
                                raw = ep[hand_key][hand_key][:]  # [T, 18] (col0=ts)
                                finger_joints = raw[:, 1:].astype(np.float32)  # skip timestamp
                                break

                        if finger_joints is None:
                            continue

                        ep_len = finger_joints.shape[0]
                        n_finger = min(finger_joints.shape[1], 16)

                        # ── Find wrist EEF (col 0 = timestamp, cols 1:8 = xyz+quat) ──
                        wrist_eef = np.zeros((ep_len, 7), dtype=np.float32)
                        for eef_key in ['right_arm_eef', 'left_arm_eef']:
                            if eef_key in ep and eef_key in ep[eef_key]:
                                raw = ep[eef_key][eef_key][:].astype(np.float32)  # [T, 8]
                                # col 0 = timestamp, cols 1:8 = xyz(3)+quat(4)
                                T_w = min(raw.shape[0], ep_len)
                                wrist_eef[:T_w, :7] = raw[:T_w, 1:8]
                                break

                        # ── Build native 23-dim action: wrist(7) + fingers(16) ──
                        actions_native = np.zeros((ep_len, native_dim), dtype=np.float32)
                        actions_native[:, :7] = wrist_eef
                        actions_native[:, 7:7 + n_finger] = finger_joints[:, :n_finger]

                        actions_unified = map_to_unified(actions_native, embodiment_id)
                        all_actions.append(actions_unified)

                        # ── Extract camera images from HDF5 groups ──
                        ep_filename = f"ep_{ep_counter:06d}.hdf5"
                        ep_path = output_dir / ep_filename

                        with h5py.File(str(ep_path), 'w') as hf:
                            n_views_written = 0
                            cam_names_found = []
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
                                    img = cam_group[img_keys[t]][:]  # [H, W, 3] uint8
                                    if img.shape[0] != image_size or img.shape[1] != image_size:
                                        img = cv2.resize(img, (image_size, image_size))
                                    view_data[t] = img

                                # Repeat last frame for padding
                                if 0 < n_frames < ep_len:
                                    view_data[n_frames:] = view_data[n_frames - 1]

                                hf.create_dataset(
                                    f'images/view_{n_views_written}', data=view_data,
                                    chunks=(1, image_size, image_size, 3),
                                    compression='gzip', compression_opts=1,
                                )
                                n_views_written += 1
                                cam_names_found.append(cam_key)

                            if n_views_written == 0:
                                ep_path.unlink(missing_ok=True)
                                continue

                            hf.create_dataset('actions', data=actions_unified)
                            hf.create_dataset('action_mask', data=mask_info.mask)
                            hf.create_dataset('language', data=f"pour the liquid")

                            hf.attrs['dataset_name'] = 'dexwild'
                            hf.attrs['embodiment_id'] = embodiment_id
                            hf.attrs['episode_length'] = ep_len
                            hf.attrs['n_views'] = n_views_written
                            hf.attrs['action_dim'] = native_dim
                            hf.attrs['image_size'] = image_size

                        ep_metadata_list.append({'filename': ep_filename, 'length': ep_len})
                        ep_counter += 1

                    except Exception as e:
                        logger.debug(f"    Error in {ep_key}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"  Error reading {src_path.name}: {e}")
            continue

    if not ep_metadata_list:
        logger.error("No DexWild episodes converted")
        return False

    # Stats + metadata
    if all_actions:
        all_cat = np.concatenate(all_actions, axis=0)
        action_mean = all_cat.mean(axis=0).tolist()
        action_std = all_cat.std(axis=0).tolist()
    else:
        action_mean = [0.0] * UNIFIED_ACTION_DIM
        action_std = [1.0] * UNIFIED_ACTION_DIM

    metadata = {
        'dataset_name': 'dexwild',
        'embodiment_id': embodiment_id,
        'is_static': False,
        'view_names': ['thumb_cam', 'pinky_cam'],
        'total_samples': sum(ep['length'] for ep in ep_metadata_list),
        'total_episodes': len(ep_metadata_list),
        'action_stats': {'mean': action_mean, 'std': action_std},
        'episodes': ep_metadata_list,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✅ dexwild: {len(ep_metadata_list)} episodes → {output_dir}")
    return True


# ─── Helpers ───

def _detect_image_keys(column_names: list) -> list:
    """Detect RGB image columns from Arrow dataset."""
    img_keys = []
    depth_seg_patterns = [
        'images.depth', 'images.seg_', 'images.segmentation',
        'images.mask', 'images.normal',
    ]
    for col in sorted(column_names):
        cl = col.lower()
        if 'image' not in cl:
            continue
        if any(pat in cl for pat in depth_seg_patterns):
            continue
        img_keys.append(col)
    return img_keys


def _extract_image(row: dict, key: str, target_size: int) -> np.ndarray:
    """Extract and resize a single image from a dataset row.

    Returns: [H, W, 3] uint8 or None.
    """
    val = row.get(key)
    if val is None:
        return None

    try:
        # PIL Image
        if hasattr(val, 'convert'):
            img = np.array(val.convert('RGB'))
        # Torch tensor
        elif hasattr(val, 'numpy'):
            img = val.numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)  # CHW → HWC
            if img.dtype in (np.float32, np.float64):
                img = (img * 255).clip(0, 255).astype(np.uint8)
        # Dict with 'bytes' (Arrow image format)
        elif isinstance(val, dict) and 'bytes' in val:
            import io
            from PIL import Image
            img = np.array(Image.open(io.BytesIO(val['bytes'])).convert('RGB'))
        # Numpy array
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


# ─── Main ───

ALL_DATASETS = ['rlbench'] + sorted(LEROBOT_DATASETS) + ['dexgraspnet', 'dexwild']


# Approximate episode counts per dataset (for --data-fraction computation)
APPROX_EPISODES = {
    'rlbench': 18_000, 'dexgraspnet': 100, 'aloha': 400, 'droid': 90_000,
    'bc_z': 4_000, 'taco_play': 12_000, 'utaustin_mutex': 1_500,
    'cmu_stretch': 130, 'nyu_franka': 450, 'stanford_hydra': 600,
    'dexwild': 9_500,
}


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to Episode HDF5')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--all', action='store_true', help='Convert all datasets')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-root', type=str, default='./data/episodes')
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--data-fraction', type=float, default=None,
                        help='Fraction of episodes to convert (0.0-1.0). '
                             'E.g., 0.1 = convert ~10%% of episodes per dataset. '
                             'Overrides --max-episodes.')
    args = parser.parse_args()

    if args.all:
        datasets = ALL_DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    data_fraction = args.data_fraction

    if data_fraction is not None and 0 < data_fraction < 1.0:
        logger.info(f"Data fraction mode: converting {data_fraction:.0%} of each dataset")

    results = {}
    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting: {ds_name}")
        logger.info(f"{'='*60}")

        # Compute max_episodes from data_fraction
        max_episodes = args.max_episodes
        if data_fraction is not None and 0 < data_fraction < 1.0:
            approx_eps = APPROX_EPISODES.get(ds_name.split('_t')[0] if 'behavior1k' in ds_name else ds_name, 500)
            max_episodes = max(1, int(approx_eps * data_fraction))
            logger.info(f"  Fraction {data_fraction:.0%} of ~{approx_eps} episodes ≈ {max_episodes} episodes")

        try:
            if ds_name == 'rlbench':
                ok = convert_rlbench_dataset(
                    args.data_root, args.output_root, args.image_size, max_episodes
                )
            elif ds_name == 'dexgraspnet':
                ok = convert_dexgraspnet_dataset(
                    args.data_root, args.output_root, args.image_size, max_episodes
                )
            elif ds_name == 'dexwild':
                ok = convert_dexwild_dataset(
                    args.data_root, args.output_root, args.image_size, max_episodes
                )
            elif ds_name in LEROBOT_DATASETS:
                ok = convert_lerobot_dataset(
                    ds_name, args.data_root, args.output_root, args.image_size, max_episodes
                )
            else:
                logger.warning(f"Unknown dataset: {ds_name}")
                ok = False
        except Exception as e:
            logger.error(f"Failed to convert {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            ok = False

        results[ds_name] = ok

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Conversion Summary:")
    for name, ok in results.items():
        status = '✅ PASS' if ok else '❌ FAIL'
        logger.info(f"  {status}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        logger.error(f"\n{len(failed)} dataset(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        logger.info(f"\nAll {len(results)} dataset(s) converted successfully!")


if __name__ == '__main__':
    main()



