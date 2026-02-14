"""
Unified Action Space Mapping (128-dim)
========================================

Non-overlapping regions for different action representations:

  Region A: Single-arm EEF pose          [0:3]=xyz, [3:7]=quat, [7]=grip   (8 dims)
  Region B: Single-arm joint positions   [8:15]=j0-j6, [15]=grip            (8 dims)
  Region C: Bimanual joints              [16:23]=left_j0-j6, [23]=left_grip,
                                         [24:31]=right_j0-j6, [31]=right_grip (16 dims)
  Region D: Dexterous hand fingers       [32:48]=16 finger joints            (16 dims)
  Region E: Extra / base                 [48:56]=base vel, etc.              (8 dims)
  Padding:                               [56:127]=zeros                      (72 dims)

Embodiment → Region mapping:
  Single-arm EEF     (rlbench, utaustin_mutex, stanford_hydra, cmu_stretch) → Region A
  Single-arm Joint   (droid, bc_z, taco_play)                                → Region B
  Bimanual           (aloha, nyu_franka)                                     → Region C
  Bimanual + Mobile  (behavior1k R1Pro)                                      → Region C (16 arm) + Region E (7 base/torso)
  Dexterous hand     (dexgraspnet)                                           → Region A (wrist) + D (fingers)
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


UNIFIED_ACTION_DIM = 128

# ─── Region definitions (non-overlapping) ───
REGION_EEF_START = 0        # Single-arm EEF: [0:8]
REGION_EEF_END = 8
REGION_JOINT_START = 8      # Single-arm joints: [8:16]
REGION_JOINT_END = 16
REGION_BIMANUAL_START = 16  # Bimanual: [16:32]
REGION_BIMANUAL_END = 32
REGION_HAND_START = 32      # Dex hand fingers: [32:48]
REGION_HAND_END = 48
REGION_EXTRA_START = 48     # Extra (base, etc): [48:56]
REGION_EXTRA_END = 56

# Embodiment IDs
EMBODIMENT_GRIPPER_EEF = 0       # Single arm, absolute EEF pose control
EMBODIMENT_GRIPPER_JOINT = 1     # Single arm, joint position control
EMBODIMENT_BIMANUAL = 2          # Two arms, joint control (≤16 dims)
EMBODIMENT_DEXHAND = 3           # Dexterous hand (wrist EEF + finger joints)
EMBODIMENT_GRIPPER_DELTA_EEF = 4 # Single arm, delta EEF control
EMBODIMENT_BIMANUAL_MOBILE = 5   # Bimanual + base/torso (>16 dims, e.g. R1Pro 23-dim)

# Legacy alias
EMBODIMENT_GRIPPER = 0

EMBODIMENT_NAMES = {
    0: 'gripper_eef',
    1: 'gripper_joint',
    2: 'bimanual',
    3: 'dex_hand',
    4: 'gripper_delta_eef',
    5: 'bimanual_mobile',
}

# ─── Dataset Registry ───

# Dataset → Embodiment (determines which REGION is used)
DATASET_EMBODIMENT = {
    # Single-arm Absolute EEF → Region A [0:8]
    'rlbench': EMBODIMENT_GRIPPER_EEF,
    # Single-arm Delta EEF → Region E [48:56]
    'utaustin_mutex': EMBODIMENT_GRIPPER_DELTA_EEF,
    'stanford_hydra': EMBODIMENT_GRIPPER_DELTA_EEF,
    'cmu_stretch': EMBODIMENT_GRIPPER_DELTA_EEF,
    # Single-arm Joint → Region B [8:16]
    'droid': EMBODIMENT_GRIPPER_JOINT,
    'bc_z': EMBODIMENT_GRIPPER_JOINT,
    'taco_play': EMBODIMENT_GRIPPER_JOINT,
    # Bimanual → Region C [16:32]
    'aloha': EMBODIMENT_BIMANUAL,
    'nyu_franka': EMBODIMENT_BIMANUAL,
    # Bimanual + Mobile (R1Pro: 16 arm + 7 base/torso) → Region C + Region E
    **{f'behavior1k_t{i:04d}': EMBODIMENT_BIMANUAL_MOBILE for i in range(50)},
    # Dexterous hand → Region A (wrist) + Region D (fingers)
    'dexgraspnet': EMBODIMENT_DEXHAND,
}

# HuggingFace repos
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

LEROBOT_DATASETS = {
    'aloha', 'droid', 'bc_z',
    'taco_play', 'utaustin_mutex', 'cmu_stretch',
    'nyu_franka', 'stanford_hydra',
    *{f'behavior1k_t{i:04d}' for i in range(50)},
}

DATASET_NATIVE_ACTION_DIM = {
    'rlbench': 8,           # EEF: xyz(3) + quat(4) + gripper(1)
    'dexgraspnet': 23,      # EEF wrist(7) + finger joints(16)
    'aloha': 14,            # Bimanual: left(7) + right(7) joint positions
    'droid': 7,             # Joint: j0-j6 + gripper
    'bc_z': 7,              # Joint: j0-j6 + gripper
    'taco_play': 7,         # Joint: j0-j6 + gripper
    'utaustin_mutex': 7,    # EEF: delta_xyz(3) + delta_rot(3) + gripper
    'cmu_stretch': 8,       # EEF: delta_xyz(3) + delta_rot(3) + gripper + base
    'nyu_franka': 15,       # Bimanual: left(7) + right(7) + extra
    'stanford_hydra': 7,    # EEF: delta_xyz(3) + delta_rot(3) + gripper
    **{f'behavior1k_t{i:04d}': 23 for i in range(50)},
}

DATASET_ACTION_FORMAT = {
    'rlbench': 'absolute_ee',
    'dexgraspnet': 'absolute_ee',
    'aloha': 'absolute_joints',
    'droid': 'joint_position',
    'bc_z': 'joint_position',
    'taco_play': 'joint_position',
    'utaustin_mutex': 'delta_ee',
    'cmu_stretch': 'delta_ee',
    'nyu_franka': 'absolute_joints',
    'stanford_hydra': 'delta_ee',
    **{f'behavior1k_t{i:04d}': 'absolute_joints' for i in range(50)},
}


# ─── Action mask and mapping ───

@dataclass
class ActionMaskInfo:
    """Action mask with metadata."""
    mask: np.ndarray
    active_dims: int
    native_dim: int
    quat_dims: Optional[list]


def get_action_mask(embodiment_id: int, native_dim: Optional[int] = None) -> ActionMaskInfo:
    """Get the action mask for an embodiment type.
    
    Args:
        embodiment_id: Which embodiment (determines which region).
        native_dim: Actual number of native action dims for this dataset.
            If provided, the mask is tightened to only cover exactly
            native_dim positions within the region (no over-counting).
            Example: aloha has native_dim=14 in bimanual region [16:32],
            so only [16:30] is marked active (not the full [16:32]).
    """
    mask = np.zeros(UNIFIED_ACTION_DIM, dtype=bool)

    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        # Region A: [0:8]
        n = min(native_dim, 8) if native_dim else 8
        mask[REGION_EEF_START:REGION_EEF_START + n] = True
        quat_dims = [3, 4, 5, 6] if n >= 7 else None
        return ActionMaskInfo(mask=mask, active_dims=n, native_dim=n, quat_dims=quat_dims)

    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        # Region B: [8:16]
        n = min(native_dim, 8) if native_dim else 8
        mask[REGION_JOINT_START:REGION_JOINT_START + n] = True
        return ActionMaskInfo(mask=mask, active_dims=n, native_dim=n, quat_dims=None)

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        # Region C: [16:32]
        n = min(native_dim, 16) if native_dim else 16
        mask[REGION_BIMANUAL_START:REGION_BIMANUAL_START + n] = True
        return ActionMaskInfo(mask=mask, active_dims=n, native_dim=n, quat_dims=None)

    elif embodiment_id == EMBODIMENT_DEXHAND:
        # Region A [0:7] (wrist EEF) + Region D [32:48] (fingers)
        n_wrist = 7
        n_fingers = min(native_dim - 7, 16) if native_dim and native_dim > 7 else 16
        mask[REGION_EEF_START:REGION_EEF_START + n_wrist] = True
        mask[REGION_HAND_START:REGION_HAND_START + n_fingers] = True
        total = n_wrist + n_fingers
        return ActionMaskInfo(mask=mask, active_dims=total, native_dim=total, quat_dims=[3, 4, 5, 6])

    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        # Region E [48:56] — delta EEF
        n = min(native_dim, 8) if native_dim else 8
        mask[REGION_EXTRA_START:REGION_EXTRA_START + n] = True
        return ActionMaskInfo(mask=mask, active_dims=n, native_dim=n, quat_dims=None)

    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        # Region C [16:32] (bimanual arms) + Region E [48:55] (base/torso)
        n_arms = 16
        n_extra = min(native_dim - 16, 8) if native_dim and native_dim > 16 else 7
        mask[REGION_BIMANUAL_START:REGION_BIMANUAL_START + n_arms] = True
        mask[REGION_EXTRA_START:REGION_EXTRA_START + n_extra] = True
        total = n_arms + n_extra
        return ActionMaskInfo(mask=mask, active_dims=total, native_dim=total, quat_dims=None)

    else:
        raise ValueError(f"Unknown embodiment_id: {embodiment_id}")


def map_to_unified(action: np.ndarray, embodiment_id: int) -> np.ndarray:
    """
    Map native action to 128-dim unified space (non-overlapping regions).

    Region A [0:8]:   Single-arm EEF (xyz+quat+grip)
    Region B [8:16]:  Single-arm joints (j0-j6+grip)
    Region C [16:32]: Bimanual joints (left 8 + right 8)
    Region D [32:48]: Dex hand fingers (16 joints)
    """
    is_sequence = action.ndim == 2
    if not is_sequence:
        action = action[np.newaxis]

    T = action.shape[0]
    D_native = action.shape[1]
    unified = np.zeros((T, UNIFIED_ACTION_DIM), dtype=np.float32)

    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        # EEF datasets → Region A [0:8]
        n = min(D_native, 8)
        unified[:, REGION_EEF_START:REGION_EEF_START + n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        # Joint datasets → Region B [8:16]
        n = min(D_native, 8)
        unified[:, REGION_JOINT_START:REGION_JOINT_START + n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        # Bimanual datasets → Region C [16:32]
        n = min(D_native, 16)
        unified[:, REGION_BIMANUAL_START:REGION_BIMANUAL_START + n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_DEXHAND:
        # Dex hand → Region A [0:7] (wrist) + Region D [32:48] (fingers)
        n_wrist = min(D_native, 7)
        unified[:, REGION_EEF_START:REGION_EEF_START + n_wrist] = action[:, :n_wrist]
        if D_native > 7:
            n_fingers = min(D_native - 7, 16)
            unified[:, REGION_HAND_START:REGION_HAND_START + n_fingers] = action[:, 7:7 + n_fingers]

    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        # Delta EEF → Region E [48:56]
        n = min(D_native, 8)
        unified[:, REGION_EXTRA_START:REGION_EXTRA_START + n] = action[:, :n]

    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        # Bimanual mobile → Region C [16:32] (arms) + Region E [48:55] (base/torso)
        # First 16 dims = bimanual arms
        n_arms = min(D_native, 16)
        unified[:, REGION_BIMANUAL_START:REGION_BIMANUAL_START + n_arms] = action[:, :n_arms]
        # Remaining dims (17-23) = base/torso/head → Region E
        if D_native > 16:
            n_extra = min(D_native - 16, 7)
            unified[:, REGION_EXTRA_START:REGION_EXTRA_START + n_extra] = action[:, 16:16 + n_extra]

    if not is_sequence:
        return unified[0]
    return unified


def extract_from_unified(unified: np.ndarray, embodiment_id: int) -> np.ndarray:
    """Extract native action from 128-dim unified space."""
    is_sequence = unified.ndim == 2
    if not is_sequence:
        unified = unified[np.newaxis]

    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        action = unified[:, REGION_EEF_START:REGION_EEF_END]
    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        action = unified[:, REGION_JOINT_START:REGION_JOINT_END]
    elif embodiment_id == EMBODIMENT_BIMANUAL:
        action = unified[:, REGION_BIMANUAL_START:REGION_BIMANUAL_END]
    elif embodiment_id == EMBODIMENT_DEXHAND:
        wrist = unified[:, REGION_EEF_START:REGION_EEF_START + 7]
        fingers = unified[:, REGION_HAND_START:REGION_HAND_END]
        action = np.concatenate([wrist, fingers], axis=-1)
    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        action = unified[:, REGION_EXTRA_START:REGION_EXTRA_END]
    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        arms = unified[:, REGION_BIMANUAL_START:REGION_BIMANUAL_END]  # [T, 16]
        extra = unified[:, REGION_EXTRA_START:REGION_EXTRA_START + 7]  # [T, 7]
        action = np.concatenate([arms, extra], axis=-1)  # [T, 23]
    else:
        raise ValueError(f"Unknown embodiment_id: {embodiment_id}")

    if not is_sequence:
        return action[0]
    return action


def normalize_quaternion_in_unified(action: np.ndarray, embodiment_id: int) -> np.ndarray:
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


# Pre-computed masks as torch tensors
_MASK_CACHE = {}

def get_action_mask_tensor(embodiment_id: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Get action mask as a torch tensor (cached)."""
    key = (embodiment_id, device)
    if key not in _MASK_CACHE:
        info = get_action_mask(embodiment_id)
        _MASK_CACHE[key] = torch.from_numpy(info.mask).to(device)
    return _MASK_CACHE[key]
