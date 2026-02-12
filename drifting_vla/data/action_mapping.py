"""
Unified Action Space Mapping
=============================

Maps embodiment-specific actions to/from a fixed 128-dim unified vector.

Layout:
  [0-6]    Arm 1 joints or EE pose (xyz + quat)
  [7]      Gripper 1
  [8-14]   Arm 2 joints or second arm
  [15]     Gripper 2
  [16-31]  Joint angles (dexterous hand fingers)
  [32-127] Padding (zeros)

Embodiments:
  0 = Parallel gripper:  dims 0-7 active (up to 8 dims)
  1 = Bimanual:          dims 0-15 active (up to 16 dims)
  2 = Dexterous hand:    dims 0-6 + 16-31 active (up to 23 dims)
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


UNIFIED_ACTION_DIM = 128

# Embodiment IDs
EMBODIMENT_GRIPPER = 0
EMBODIMENT_BIMANUAL = 1
EMBODIMENT_DEXHAND = 2

# Embodiment names
EMBODIMENT_NAMES = {
    0: 'gripper',
    1: 'bimanual',
    2: 'dex_hand',
}

# ─────────────────────────────────────────────────────────────────────
# Dataset Registry (all verified working with lerobot v0.4+)
# ─────────────────────────────────────────────────────────────────────

# Dataset → Embodiment mapping
DATASET_EMBODIMENT = {
    # Gripper (single arm, 7-8 dim)
    'rlbench': EMBODIMENT_GRIPPER,
    'droid': EMBODIMENT_GRIPPER,
    'bc_z': EMBODIMENT_GRIPPER,
    'taco_play': EMBODIMENT_GRIPPER,
    'utaustin_mutex': EMBODIMENT_GRIPPER,
    'cmu_stretch': EMBODIMENT_GRIPPER,
    'stanford_hydra': EMBODIMENT_GRIPPER,
    # Bimanual (14-15 dim, two arms)
    'aloha': EMBODIMENT_BIMANUAL,
    'nyu_franka': EMBODIMENT_BIMANUAL,
    **{f'behavior1k_t{i:04d}': EMBODIMENT_BIMANUAL for i in range(50)},
    # Dexterous hand (23 dim, wrist + fingers)
    'dexgraspnet': EMBODIMENT_DEXHAND,
}

# HuggingFace repos (all verified downloadable)
DATASET_HF_REPOS = {
    'rlbench': 'hqfang/rlbench-18-tasks',
    'dexgraspnet': 'lhrlhr/DexGraspNet2.0',
    'aloha': 'lerobot/aloha_sim_transfer_cube_human_image',
    'droid': 'lerobot/droid_1.0.1',
    'bc_z': 'lerobot/berkeley_autolab_ur5',
    'taco_play': 'lerobot/taco_play',
    'utaustin_mutex': 'lerobot/utaustin_mutex',
    'cmu_stretch': 'lerobot/cmu_stretch',
    'nyu_franka': 'lerobot/nyu_franka_play_dataset',
    'stanford_hydra': 'lerobot/stanford_hydra_dataset',
    **{f'behavior1k_t{i:04d}': f'lerobot/behavior1k-task{i:04d}' for i in range(50)},
}

# Datasets loaded via lerobot package
LEROBOT_DATASETS = {
    'aloha', 'droid', 'bc_z',
    'taco_play', 'utaustin_mutex', 'cmu_stretch',
    'nyu_franka', 'stanford_hydra',
    *{f'behavior1k_t{i:04d}' for i in range(50)},
}

# Native action dimensions per dataset
DATASET_NATIVE_ACTION_DIM = {
    'rlbench': 8,           # xyz(3) + quat(4) + gripper(1)
    'dexgraspnet': 23,      # wrist(7) + joints(16)
    'aloha': 14,            # left(7) + right(7) joint positions
    'droid': 7,             # delta EE(6) + gripper(1)
    'bc_z': 7,              # delta EE(6) + gripper(1)
    'taco_play': 7,         # delta EE(6) + gripper(1)
    'utaustin_mutex': 7,    # delta EE(6) + gripper(1)
    'cmu_stretch': 8,       # delta EE(6) + gripper(1) + base(1)
    'nyu_franka': 15,       # bimanual franka joints
    'stanford_hydra': 7,    # delta EE(6) + gripper(1)
    **{f'behavior1k_t{i:04d}': 23 for i in range(50)},   # bimanual robot in OmniGibson
}

# ─────────────────────────────────────────────────────────────────────
# Dataset Size Reference (for memory planning)
# ─────────────────────────────────────────────────────────────────────
# Dataset                   Samples       Act  Embody      Disk(est)   VLM feat
# ───────────────────────────────────────────────────────────────────────────────
# rlbench (18 tasks)        ~270,000       8   gripper     ~50 GB      ~1.1 GB
# dexgraspnet 2.0           ~500,000      23   dex hand    ~100 GB     ~2.0 GB
# aloha (sim cube)            20,000      14   bimanual    ~8 GB       ~80 MB
# droid 1.0.1            ~25,500,000       7   gripper     ~2 TB       ~102 GB
# bc_z (autolab)              97,939       7   gripper     ~30 GB      ~400 MB
# taco_play                  237,798       7   gripper     ~30 GB      ~950 MB
# utaustin_mutex             361,883       7   gripper     ~40 GB      ~1.4 GB
# cmu_stretch                 25,016       8   gripper     ~5 GB       ~100 MB
# nyu_franka                  44,875      15   bimanual    ~12 GB      ~180 MB
# stanford_hydra             358,234       7   gripper     ~40 GB      ~1.4 GB
# behavior1k (×50 tasks)  ~5,000,000+     23   bimanual   ~500 GB     ~20 GB
# ───────────────────────────────────────────────────────────────────────────────
# TOTAL                  ~32,400,000+                      ~2.8 TB     ~130 GB
# ───────────────────────────────────────────────────────────────────────────────


@dataclass
class ActionMaskInfo:
    """Action mask with metadata."""
    mask: np.ndarray           # [128] bool
    active_dims: int           # Number of active dimensions
    native_dim: int            # Native action dimensionality
    quat_dims: Optional[list]  # Which dims are quaternion (for normalization)


def get_action_mask(embodiment_id: int) -> ActionMaskInfo:
    """
    Get the action mask for an embodiment type.

    Args:
        embodiment_id: 0=gripper, 1=bimanual, 2=dex_hand

    Returns:
        ActionMaskInfo with mask and metadata.
    """
    mask = np.zeros(UNIFIED_ACTION_DIM, dtype=bool)

    if embodiment_id == EMBODIMENT_GRIPPER:
        mask[:8] = True
        return ActionMaskInfo(
            mask=mask, active_dims=8, native_dim=8,
            quat_dims=[3, 4, 5, 6],
        )

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        mask[:16] = True
        return ActionMaskInfo(
            mask=mask, active_dims=16, native_dim=16,
            quat_dims=None,
        )

    elif embodiment_id == EMBODIMENT_DEXHAND:
        mask[:7] = True
        mask[16:32] = True
        return ActionMaskInfo(
            mask=mask, active_dims=23, native_dim=23,
            quat_dims=[3, 4, 5, 6],
        )

    else:
        raise ValueError(f"Unknown embodiment_id: {embodiment_id}")


def map_to_unified(
    action: np.ndarray,
    embodiment_id: int,
) -> np.ndarray:
    """
    Map native action to 128-dim unified space.

    Args:
        action: [D_native] or [T, D_native] native action
        embodiment_id: 0=gripper, 1=bimanual, 2=dex_hand

    Returns:
        unified: [128] or [T, 128] zero-padded action
    """
    is_sequence = action.ndim == 2
    if not is_sequence:
        action = action[np.newaxis]

    T = action.shape[0]
    unified = np.zeros((T, UNIFIED_ACTION_DIM), dtype=np.float32)
    D_native = action.shape[1]

    if embodiment_id == EMBODIMENT_GRIPPER:
        n = min(D_native, 8)
        unified[:, :n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        n = min(D_native, 16)
        unified[:, :n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_DEXHAND:
        n_wrist = min(D_native, 7)
        unified[:, :n_wrist] = action[:, :n_wrist]
        if D_native > 7:
            n_joints = min(D_native - 7, 16)
            unified[:, 16:16+n_joints] = action[:, 7:7+n_joints]

    if not is_sequence:
        return unified[0]
    return unified


def extract_from_unified(
    unified: np.ndarray,
    embodiment_id: int,
) -> np.ndarray:
    """Extract native action from 128-dim unified space."""
    is_sequence = unified.ndim == 2
    if not is_sequence:
        unified = unified[np.newaxis]

    if embodiment_id == EMBODIMENT_GRIPPER:
        action = unified[:, :8]
    elif embodiment_id == EMBODIMENT_BIMANUAL:
        action = unified[:, :16]
    elif embodiment_id == EMBODIMENT_DEXHAND:
        wrist = unified[:, :7]
        joints = unified[:, 16:32]
        action = np.concatenate([wrist, joints], axis=-1)

    if not is_sequence:
        return action[0]
    return action


def normalize_quaternion_in_unified(
    action: np.ndarray,
    embodiment_id: int,
) -> np.ndarray:
    """Normalize quaternion dimensions in unified action."""
    info = get_action_mask(embodiment_id)
    if info.quat_dims is None:
        return action

    action = action.copy()
    if action.ndim == 1:
        quat = action[info.quat_dims]
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            action[info.quat_dims] = quat / norm
        else:
            action[info.quat_dims] = [0, 0, 0, 1]
    else:
        for t in range(action.shape[0]):
            quat = action[t, info.quat_dims]
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                action[t, info.quat_dims] = quat / norm
            else:
                action[t, info.quat_dims] = [0, 0, 0, 1]

    return action


# Pre-computed masks as torch tensors (for loss computation)
_MASK_CACHE = {}

def get_action_mask_tensor(
    embodiment_id: int,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Get action mask as a torch tensor (cached)."""
    key = (embodiment_id, device)
    if key not in _MASK_CACHE:
        info = get_action_mask(embodiment_id)
        _MASK_CACHE[key] = torch.from_numpy(info.mask).to(device)
    return _MASK_CACHE[key]

