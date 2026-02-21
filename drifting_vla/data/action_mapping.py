"""
Unified Action Space Mapping — RDT-1B Compatible (128-dim)
============================================================

Semantic named-field mapping where each physical quantity gets a named slot.
A single robot can fill MULTIPLE regions simultaneously (e.g., joint AND EEF).

Layout (RDT-1B compatible + Dex Hand extension):
  [0, 10):   right arm joint positions
  [10, 15):  right gripper joint positions
  [15, 25):  right arm joint velocities
  [25, 30):  right gripper joint velocities
  [30, 33):  right EEF positions (xyz)
  [33, 39):  right EEF 6D rotation (continuous, NOT quaternion)
  [39, 42):  right EEF velocities
  [42, 45):  right EEF angular velocities
  [45, 50):  head/torso/spine joints + reserved  (RDT-1B: reserved)
  [50, 60):  left arm joint positions
  [60, 65):  left gripper joint positions
  [65, 75):  left arm joint velocities
  [75, 80):  left gripper joint velocities
  [80, 83):  left EEF positions (xyz)
  [83, 89):  left EEF 6D rotation
  [89, 92):  left EEF velocities
  [92, 95):  left EEF angular velocities
  [95, 100): reserved (RDT-1B compatible)
  [100, 103): base velocities (x, y, angular)
  [103, 115): right dexterous finger joints (12 DOF max)
  [115, 127): left dexterous finger joints (12 DOF max)
  [127, 128): reserved

References:
  - RDT-1B: configs/state_vec.py
  - 6D rotation: Zhou et al., CVPR 2019 "On the Continuity of Rotation Representations"
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


UNIFIED_ACTION_DIM = 128

# =============================================================================
# Semantic State Vector Index Mapping (RDT-1B compatible + Dex Hand)
# =============================================================================

STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{f'arm_joint_{i}_pos': i for i in range(10)},
    **{f'right_arm_joint_{i}_pos': i for i in range(10)},
    # [10, 15): right gripper joint positions
    **{f'gripper_joint_{i}_pos': i + 10 for i in range(5)},
    **{f'right_gripper_joint_{i}_pos': i + 10 for i in range(5)},
    'gripper_open': 10,
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{f'arm_joint_{i}_vel': i + 15 for i in range(10)},
    **{f'right_arm_joint_{i}_vel': i + 15 for i in range(10)},
    # [25, 30): right gripper joint velocities
    **{f'gripper_joint_{i}_vel': i + 25 for i in range(5)},
    **{f'right_gripper_joint_{i}_vel': i + 25 for i in range(5)},
    'gripper_open_vel': 25,
    # [30, 33): right EEF positions
    'eef_pos_x': 30, 'right_eef_pos_x': 30,
    'eef_pos_y': 31, 'right_eef_pos_y': 31,
    'eef_pos_z': 32, 'right_eef_pos_z': 32,
    # [33, 39): right EEF 6D rotation (continuous)
    **{f'eef_angle_{i}': 33 + i for i in range(6)},
    **{f'right_eef_angle_{i}': 33 + i for i in range(6)},
    # [39, 42): right EEF velocities
    'eef_vel_x': 39, 'eef_vel_y': 40, 'eef_vel_z': 41,
    # [42, 45): right EEF angular velocities
    'eef_angular_vel_roll': 42, 'eef_angular_vel_pitch': 43, 'eef_angular_vel_yaw': 44,
    # [45, 50): head/torso/spine joints (reserved in RDT-1B, extended for Dexora-class robots)
    'head_joint_0': 45, 'head_joint_1': 46, 'spine_joint': 47,
    # [50, 60): left arm joint positions
    **{f'left_arm_joint_{i}_pos': i + 50 for i in range(10)},
    # [60, 65): left gripper joint positions
    **{f'left_gripper_joint_{i}_pos': i + 60 for i in range(5)},
    'left_gripper_open': 60,
    # [65, 75): left arm joint velocities
    **{f'left_arm_joint_{i}_vel': i + 65 for i in range(10)},
    # [75, 80): left gripper joint velocities
    **{f'left_gripper_joint_{i}_vel': i + 75 for i in range(5)},
    # [80, 83): left EEF positions
    'left_eef_pos_x': 80, 'left_eef_pos_y': 81, 'left_eef_pos_z': 82,
    # [83, 89): left EEF 6D rotation
    **{f'left_eef_angle_{i}': 83 + i for i in range(6)},
    # [89, 92): left EEF velocities
    'left_eef_vel_x': 89, 'left_eef_vel_y': 90, 'left_eef_vel_z': 91,
    # [92, 95): left EEF angular velocities
    'left_eef_angular_vel_roll': 92, 'left_eef_angular_vel_pitch': 93, 'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved (RDT-1B compatible)
    # [100, 103): base velocities
    'base_vel_x': 100, 'base_vel_y': 101, 'base_angular_vel': 102,
    # [103, 115): right dexterous finger joints (12 DOF max)
    # [115, 127): left dexterous finger joints (12 DOF max)
    # Single-hand datasets (DexWild 16-DOF) use dex_finger_joint_{0-15} → [103:119]
    # Bimanual dex (Dexora 12+12) uses right {0-11} → [103:115], left {12-23} → [115:127]
    **{f'dex_finger_joint_{i}_pos': 103 + i for i in range(24)},
    # [127, 128): reserved
}


# =============================================================================
# 6D Rotation Conversion
# =============================================================================

def quaternion_to_6d_rotation(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 6D continuous rotation.

    Reference: Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.

    Returns: [6] — first two columns of 3x3 rotation matrix, flattened.
    """
    from scipy.spatial.transform import Rotation
    if np.linalg.norm(quat) < 1e-8:
        return np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # identity
    R = Rotation.from_quat(quat).as_matrix()  # [3, 3]
    return R[:, :2].T.flatten().astype(np.float32)  # [6]


def rotation_6d_to_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation back to quaternion [x,y,z,w].

    Reconstructs the third column via cross product, then converts.
    """
    from scipy.spatial.transform import Rotation
    r6 = rot6d.reshape(2, 3)
    # Gram-Schmidt orthogonalize
    a1 = r6[0] / (np.linalg.norm(r6[0]) + 1e-8)
    a2 = r6[1] - np.dot(a1, r6[1]) * a1
    a2 = a2 / (np.linalg.norm(a2) + 1e-8)
    a3 = np.cross(a1, a2)
    R = np.stack([a1, a2, a3], axis=1)  # [3, 3]
    return Rotation.from_matrix(R).as_quat().astype(np.float32)


# =============================================================================
# Per-Dataset Field Format Strings (maps native action dims to named fields)
# =============================================================================

DATASET_FIELD_FORMATS = {
    'rlbench': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2', 'eef_angle_3',
        'gripper_open',
    ],
    'droid': [
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'gripper_open',
    ],
    'bc_z': [
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'gripper_open',
    ],
    'taco_play': [
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'gripper_open',
    ],
    'utaustin_mutex': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2',
        'gripper_open',
    ],
    'stanford_hydra': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2',
        'gripper_open',
    ],
    'cmu_stretch': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2',
        'gripper_open', 'base_vel_x',
    ],
    'aloha': [
        'left_arm_joint_0_pos', 'left_arm_joint_1_pos', 'left_arm_joint_2_pos',
        'left_arm_joint_3_pos', 'left_arm_joint_4_pos', 'left_arm_joint_5_pos',
        'left_gripper_open',
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'gripper_open',
    ],
    'nyu_franka': [
        'left_arm_joint_0_pos', 'left_arm_joint_1_pos', 'left_arm_joint_2_pos',
        'left_arm_joint_3_pos', 'left_arm_joint_4_pos', 'left_arm_joint_5_pos',
        'left_gripper_open',
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'gripper_open',
        'base_vel_x',
    ],
    'dexgraspnet': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2', 'eef_angle_3',
        *[f'dex_finger_joint_{i}_pos' for i in range(16)],
    ],
    'dexwild': [
        'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
        'eef_angle_0', 'eef_angle_1', 'eef_angle_2', 'eef_angle_3',
        *[f'dex_finger_joint_{i}_pos' for i in range(16)],
    ],
    'dexora': [
        # Left arm joints [50:56]
        *[f'left_arm_joint_{i}_pos' for i in range(6)],
        # Right arm joints [0:6]
        *[f'arm_joint_{i}_pos' for i in range(6)],
        # Left hand 12 finger joints [115:127]
        *[f'dex_finger_joint_{12+i}_pos' for i in range(12)],
        # Right hand 12 finger joints [103:115]
        *[f'dex_finger_joint_{i}_pos' for i in range(12)],
        # Head + spine [45:48]
        'head_joint_0', 'head_joint_1', 'spine_joint',
    ],
}

# Behavior 1K: bimanual (8+8 arms) + base/torso (7)
for _i in range(50):
    DATASET_FIELD_FORMATS[f'behavior1k_t{_i:04d}'] = [
        'left_arm_joint_0_pos', 'left_arm_joint_1_pos', 'left_arm_joint_2_pos',
        'left_arm_joint_3_pos', 'left_arm_joint_4_pos', 'left_arm_joint_5_pos',
        'left_arm_joint_6_pos', 'left_gripper_open',
        'arm_joint_0_pos', 'arm_joint_1_pos', 'arm_joint_2_pos',
        'arm_joint_3_pos', 'arm_joint_4_pos', 'arm_joint_5_pos',
        'arm_joint_6_pos', 'gripper_open',
        'base_vel_x', 'base_vel_y', 'base_angular_vel',
        # remaining 4 dims: head/torso
        'eef_vel_x', 'eef_vel_y', 'eef_vel_z', 'eef_angular_vel_roll',
    ]


# =============================================================================
# assemble_state_vec — RDT-1B style named-field mapping
# =============================================================================

def assemble_state_vec(
    values: np.ndarray,
    field_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map native action vector to 128-dim via named fields (RDT-1B style).

    Args:
        values: [D] native action vector
        field_names: list of D field names from STATE_VEC_IDX_MAPPING

    Returns:
        (state_vec [128], mask_vec [128])
    """
    state_vec = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)
    mask_vec = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)

    for i, name in enumerate(field_names):
        if i >= len(values):
            break
        if name in STATE_VEC_IDX_MAPPING:
            idx = STATE_VEC_IDX_MAPPING[name]
            state_vec[idx] = values[i]
            mask_vec[idx] = 1.0

    return state_vec, mask_vec


def assemble_state_vec_batch(
    actions: np.ndarray,
    field_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch version: [T, D_native] → [T, 128] + [128] mask."""
    T = actions.shape[0]
    unified = np.zeros((T, UNIFIED_ACTION_DIM), dtype=np.float32)
    mask = np.zeros(UNIFIED_ACTION_DIM, dtype=np.float32)

    for i, name in enumerate(field_names):
        if name in STATE_VEC_IDX_MAPPING:
            idx = STATE_VEC_IDX_MAPPING[name]
            if i < actions.shape[1]:
                unified[:, idx] = actions[:, i]
            mask[idx] = 1.0

    return unified, mask


# =============================================================================
# Embodiment IDs (kept for backward compat)
# =============================================================================

EMBODIMENT_GRIPPER_EEF = 0
EMBODIMENT_GRIPPER_JOINT = 1
EMBODIMENT_BIMANUAL = 2
EMBODIMENT_DEXHAND = 3
EMBODIMENT_GRIPPER_DELTA_EEF = 4
EMBODIMENT_BIMANUAL_MOBILE = 5
EMBODIMENT_BIMANUAL_DEX = 6       # Bimanual + dexterous hands (Dexora-class)
EMBODIMENT_GRIPPER = 0  # legacy alias

EMBODIMENT_NAMES = {
    0: 'gripper_eef',
    1: 'gripper_joint',
    2: 'bimanual',
    3: 'dex_hand',
    4: 'gripper_delta_eef',
    5: 'bimanual_mobile',
    6: 'bimanual_dex',
}

# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_EMBODIMENT = {
    'rlbench': EMBODIMENT_GRIPPER_EEF,
    'utaustin_mutex': EMBODIMENT_GRIPPER_DELTA_EEF,
    'stanford_hydra': EMBODIMENT_GRIPPER_DELTA_EEF,
    'cmu_stretch': EMBODIMENT_GRIPPER_DELTA_EEF,
    'droid': EMBODIMENT_GRIPPER_JOINT,
    'bc_z': EMBODIMENT_GRIPPER_JOINT,
    'taco_play': EMBODIMENT_GRIPPER_JOINT,
    'aloha': EMBODIMENT_BIMANUAL,
    'nyu_franka': EMBODIMENT_BIMANUAL,
    **{f'behavior1k_t{i:04d}': EMBODIMENT_BIMANUAL_MOBILE for i in range(50)},
    'dexgraspnet': EMBODIMENT_DEXHAND,
    'dexwild': EMBODIMENT_DEXHAND,
    'dexora': EMBODIMENT_BIMANUAL_DEX,
}

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
    'dexwild': 'boardd/dexwild-dataset',
    'dexora': 'Dexora/Dexora_Real-World_Dataset',
}

LEROBOT_DATASETS = {
    'aloha', 'droid', 'bc_z',
    'taco_play', 'utaustin_mutex', 'cmu_stretch',
    'nyu_franka', 'stanford_hydra',
    *{f'behavior1k_t{i:04d}' for i in range(50)},
    'dexora',
}

DATASET_NATIVE_ACTION_DIM = {
    'rlbench': 8,
    'dexgraspnet': 23,
    'aloha': 14,
    'droid': 7,
    'bc_z': 7,
    'taco_play': 7,
    'utaustin_mutex': 7,
    'cmu_stretch': 8,
    'nyu_franka': 15,
    'stanford_hydra': 7,
    **{f'behavior1k_t{i:04d}': 23 for i in range(50)},
    'dexwild': 23,
    'dexora': 39,
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
    'dexwild': 'absolute_ee',
    'dexora': 'absolute_joints',
}


# =============================================================================
# Action mask (derived from field format)
# =============================================================================

@dataclass
class ActionMaskInfo:
    """Action mask with metadata."""
    mask: np.ndarray
    active_dims: int
    native_dim: int
    quat_dims: Optional[list]


def get_action_mask(embodiment_id: int, native_dim: Optional[int] = None,
                    dataset_name: Optional[str] = None) -> ActionMaskInfo:
    """Get the action mask for an embodiment/dataset.

    If dataset_name is provided AND has a DATASET_FIELD_FORMATS entry,
    the mask is computed from the named fields (precise).
    Otherwise falls back to embodiment-based region logic (legacy).
    """
    # ── New path: field-format-based mask ──
    if dataset_name and dataset_name in DATASET_FIELD_FORMATS:
        fields = DATASET_FIELD_FORMATS[dataset_name]
        mask = np.zeros(UNIFIED_ACTION_DIM, dtype=bool)
        for name in fields:
            if name in STATE_VEC_IDX_MAPPING:
                mask[STATE_VEC_IDX_MAPPING[name]] = True
        active = int(mask.sum())
        # Detect quaternion dims (legacy compat — [33:37] are rotation angles)
        quat_dims = None
        if any(f'eef_angle_{i}' in fields for i in range(4)):
            quat_dims = [33, 34, 35, 36]
        return ActionMaskInfo(mask=mask, active_dims=active,
                              native_dim=native_dim or active, quat_dims=quat_dims)

    # ── Legacy path: embodiment region-based ──
    mask = np.zeros(UNIFIED_ACTION_DIM, dtype=bool)
    nd = native_dim or 8

    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        n = min(nd, 8)
        mask[30:30 + min(n, 3)] = True  # EEF pos
        if n > 3:
            mask[33:33 + min(n - 3, 4)] = True  # rotation
        if n > 7:
            mask[10] = True  # gripper_open
        quat_dims = [33, 34, 35, 36] if n >= 7 else None
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd, quat_dims=quat_dims)

    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        n = min(nd, 7)
        mask[0:n] = True  # arm joints
        if nd > 6:
            mask[10] = True  # gripper_open
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd, quat_dims=None)

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        n = min(nd, 14)
        n_left = min(n, 7)
        mask[50:50 + min(n_left, 6)] = True
        if n_left > 6:
            mask[60] = True
        if n > 7:
            n_right = min(n - 7, 7)
            mask[0:min(n_right, 6)] = True
            if n_right > 6:
                mask[10] = True
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd, quat_dims=None)

    elif embodiment_id == EMBODIMENT_DEXHAND:
        mask[30:33] = True  # EEF pos
        mask[33:37] = True  # rotation
        n_fingers = min(nd - 7, 16) if nd > 7 else 16
        mask[103:103 + n_fingers] = True  # dex fingers
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd,
                              quat_dims=[33, 34, 35, 36])

    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        n = min(nd, 8)
        mask[30:30 + min(n, 3)] = True
        if n > 3:
            mask[33:33 + min(n - 3, 3)] = True
        if n > 6:
            mask[10] = True
        if n > 7:
            mask[100] = True
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd, quat_dims=None)

    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        # Arms: left [50:58] + right [0:8]
        mask[50:58] = True
        mask[0:8] = True
        # Base
        mask[100:103] = True
        # Extra
        if nd > 19:
            mask[39:39 + min(nd - 19, 4)] = True
        return ActionMaskInfo(mask=mask, active_dims=int(mask.sum()), native_dim=nd, quat_dims=None)

    else:
        raise ValueError(f"Unknown embodiment_id: {embodiment_id}")


# =============================================================================
# map_to_unified / extract_from_unified — uses field formats when available
# =============================================================================

def map_to_unified(action: np.ndarray, embodiment_id: int,
                   dataset_name: Optional[str] = None) -> np.ndarray:
    """Map native action to 128-dim unified space.

    If dataset_name has a DATASET_FIELD_FORMATS entry, uses named-field mapping.
    Otherwise falls back to simple region slicing (legacy).
    """
    is_sequence = action.ndim == 2
    if not is_sequence:
        action = action[np.newaxis]

    # ── New path: named-field mapping ──
    if dataset_name and dataset_name in DATASET_FIELD_FORMATS:
        fields = DATASET_FIELD_FORMATS[dataset_name]
        unified, _ = assemble_state_vec_batch(action, fields)
        return unified if is_sequence else unified[0]

    # ── Legacy path: region slicing ──
    T = action.shape[0]
    D_native = action.shape[1]
    unified = np.zeros((T, UNIFIED_ACTION_DIM), dtype=np.float32)

    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        n = min(D_native, 3)
        unified[:, 30:30 + n] = action[:, :n]
        if D_native > 3:
            n_rot = min(D_native - 3, 4)
            unified[:, 33:33 + n_rot] = action[:, 3:3 + n_rot]
        if D_native > 7:
            unified[:, 10] = action[:, 7]

    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        n = min(D_native, 6)
        unified[:, 0:n] = action[:, :n]
        if D_native > 6:
            unified[:, 10] = action[:, 6]

    elif embodiment_id == EMBODIMENT_BIMANUAL:
        n = min(D_native, 14)
        n_left = min(n, 7)
        unified[:, 50:50 + min(n_left, 6)] = action[:, :min(n_left, 6)]
        if n_left > 6:
            unified[:, 60] = action[:, 6]
        if n > 7:
            n_right = min(n - 7, 7)
            unified[:, 0:min(n_right, 6)] = action[:, 7:7 + min(n_right, 6)]
            if n_right > 6:
                unified[:, 10] = action[:, 13]

    elif embodiment_id == EMBODIMENT_DEXHAND:
        n_wrist = min(D_native, 7)
        unified[:, 30:30 + min(n_wrist, 3)] = action[:, :min(n_wrist, 3)]
        if n_wrist > 3:
            unified[:, 33:33 + min(n_wrist - 3, 4)] = action[:, 3:n_wrist]
        if D_native > 7:
            n_fingers = min(D_native - 7, 16)
            unified[:, 103:103 + n_fingers] = action[:, 7:7 + n_fingers]

    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        n = min(D_native, 3)
        unified[:, 30:30 + n] = action[:, :n]
        if D_native > 3:
            n_rot = min(D_native - 3, 3)
            unified[:, 33:33 + n_rot] = action[:, 3:3 + n_rot]
        if D_native > 6:
            unified[:, 10] = action[:, 6]
        if D_native > 7:
            unified[:, 100] = action[:, 7]

    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        # Left arm [50:58]
        n_left = min(D_native, 8)
        unified[:, 50:50 + n_left] = action[:, :n_left]
        # Right arm [0:8]
        if D_native > 8:
            n_right = min(D_native - 8, 8)
            unified[:, 0:n_right] = action[:, 8:8 + n_right]
        # Base [100:103]
        if D_native > 16:
            n_base = min(D_native - 16, 3)
            unified[:, 100:100 + n_base] = action[:, 16:16 + n_base]
        # Extra
        if D_native > 19:
            n_extra = min(D_native - 19, 4)
            unified[:, 39:39 + n_extra] = action[:, 19:19 + n_extra]

    if not is_sequence:
        return unified[0]
    return unified


def extract_from_unified(unified: np.ndarray, embodiment_id: int,
                         dataset_name: Optional[str] = None) -> np.ndarray:
    """Extract native action from 128-dim unified space."""
    is_sequence = unified.ndim == 2
    if not is_sequence:
        unified = unified[np.newaxis]

    if dataset_name and dataset_name in DATASET_FIELD_FORMATS:
        fields = DATASET_FIELD_FORMATS[dataset_name]
        T = unified.shape[0]
        native = np.zeros((T, len(fields)), dtype=np.float32)
        for i, name in enumerate(fields):
            if name in STATE_VEC_IDX_MAPPING:
                idx = STATE_VEC_IDX_MAPPING[name]
                native[:, i] = unified[:, idx]
        return native if is_sequence else native[0]

    # Legacy extraction
    if embodiment_id == EMBODIMENT_GRIPPER_EEF:
        pos = unified[:, 30:33]
        rot = unified[:, 33:37]
        grip = unified[:, 10:11]
        action = np.concatenate([pos, rot, grip], axis=-1)
    elif embodiment_id == EMBODIMENT_GRIPPER_JOINT:
        joints = unified[:, 0:6]
        grip = unified[:, 10:11]
        action = np.concatenate([joints, grip], axis=-1)
    elif embodiment_id == EMBODIMENT_BIMANUAL:
        left = np.concatenate([unified[:, 50:56], unified[:, 60:61]], axis=-1)
        right = np.concatenate([unified[:, 0:6], unified[:, 10:11]], axis=-1)
        action = np.concatenate([left, right], axis=-1)
    elif embodiment_id == EMBODIMENT_DEXHAND:
        wrist = np.concatenate([unified[:, 30:33], unified[:, 33:37]], axis=-1)
        fingers = unified[:, 103:119]
        action = np.concatenate([wrist, fingers], axis=-1)
    elif embodiment_id == EMBODIMENT_GRIPPER_DELTA_EEF:
        pos = unified[:, 30:33]
        rot = unified[:, 33:36]
        grip = unified[:, 10:11]
        action = np.concatenate([pos, rot, grip], axis=-1)
    elif embodiment_id == EMBODIMENT_BIMANUAL_MOBILE:
        left = unified[:, 50:58]
        right = unified[:, 0:8]
        base = unified[:, 100:103]
        extra = unified[:, 39:43]
        action = np.concatenate([left, right, base, extra], axis=-1)
    else:
        raise ValueError(f"Unknown embodiment_id: {embodiment_id}")

    if not is_sequence:
        return action[0]
    return action


def normalize_quaternion_in_unified(action: np.ndarray, embodiment_id: int) -> np.ndarray:
    """Normalize rotation dimensions in unified action (legacy compat)."""
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
        for t in range(action.shape[0]):
            quat = action[t, info.quat_dims]
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                action[t, info.quat_dims] = quat / norm
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
