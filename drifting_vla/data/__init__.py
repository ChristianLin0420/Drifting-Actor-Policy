"""
Drifting-VLA Data Module
=============================

Multi-dataset loading with unified 128-dim action space:
    - UnifiedDataset: Multi-source loader with per-dataset normalization
    - RLBenchDataset: PerAct format gripper demonstrations
    - DexGraspNetDataset: Dexterous hand grasping data
    - action_mapping: Embodiment-specific action ↔ 128-dim mapping
    - sample_queue: Positive/negative sample queues for drifting loss
"""

from drifting_vla.data.base_dataset import BaseVLADataset
from drifting_vla.data.rlbench_dataset import RLBenchDataset
from drifting_vla.data.dexgraspnet_dataset import DexGraspNetDataset
from drifting_vla.data.lerobot_dataset import LeRobotDataset
from drifting_vla.data.unified_dataset import UnifiedDataset, collate_unified
from drifting_vla.data.sample_queue import SampleQueue, GlobalSampleQueue, NegativeSampleQueue
from drifting_vla.data.action_mapping import (
    UNIFIED_ACTION_DIM, DATASET_EMBODIMENT, LEROBOT_DATASETS,
    DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM,
    map_to_unified, extract_from_unified, get_action_mask,
)

__all__ = [
    "BaseVLADataset",
    "RLBenchDataset",
    "DexGraspNetDataset",
    "LeRobotDataset",
    "UnifiedDataset",
    "collate_unified",
    "SampleQueue",
    "GlobalSampleQueue",
    "NegativeSampleQueue",
    "UNIFIED_ACTION_DIM",
    "DATASET_EMBODIMENT",
    "LEROBOT_DATASETS",
    "DATASET_HF_REPOS",
    "DATASET_NATIVE_ACTION_DIM",
    "map_to_unified",
    "extract_from_unified",
    "get_action_mask",
]
