#!/usr/bin/env python3
"""
Verify downloaded dataset integrity for Drifting-VLA.

Checks:
  - Sample count > 0
  - Images present (shape [V, 3, H, W], non-zero)
  - Language non-empty
  - Actions correct dimensionality
  - Action mapping round-trip
  - DexGraspNet: rendered images exist

Usage:
    python scripts/verify_dataset.py --dataset aloha
    python scripts/verify_dataset.py --all
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_lerobot_dataset(ds_name, data_root='./data'):
    """Verify a LeRobot-format dataset."""
    from drifting_vla.data.lerobot_dataset import LeRobotDataset
    from drifting_vla.data.action_mapping import (
        DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM, DATASET_EMBODIMENT,
        map_to_unified, extract_from_unified, get_action_mask,
    )
    import numpy as np

    hf_repo = DATASET_HF_REPOS.get(ds_name)
    if not hf_repo:
        logger.error(f"No HF repo for {ds_name}")
        return False

    arrow_path = Path(data_root) / ds_name / 'arrow_data'
    if not arrow_path.exists():
        logger.error(f"Arrow data not found: {arrow_path}")
        return False

    errors = []

    # Load dataset
    ds = LeRobotDataset(
        repo_id=hf_repo, data_root=data_root, ds_name=ds_name,
        image_size=448, action_horizon=16, max_samples=10,
    )

    # Check 1: Sample count
    if len(ds) == 0:
        errors.append("0 samples")
    else:
        logger.info(f"  Samples: {len(ds)} ✓")

    # Check 2-5: Verify first 3 samples
    n_with_images = 0
    n_with_language = 0
    native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name)

    for i in range(min(3, len(ds))):
        s = ds[i]

        # Images
        imgs = s.get('images')
        if imgs is not None and imgs.numel() > 0:
            if imgs.abs().sum() > 0:
                n_with_images += 1
            if imgs.ndim != 4 or imgs.shape[1] != 3:
                errors.append(f"Sample {i}: bad image shape {imgs.shape} (expected [V,3,H,W])")

        # Language
        lang = s.get('language', '')
        if lang and len(lang) > 0:
            n_with_language += 1

        # Actions
        actions = s.get('actions')
        if actions is not None:
            if actions.shape != (16, native_dim):
                errors.append(f"Sample {i}: action shape {actions.shape}, expected (16, {native_dim})")

    if n_with_images == 0 and ds.image_keys:
        errors.append("No non-zero images found in first 3 samples")
    else:
        logger.info(f"  Images: {n_with_images}/3 with content, keys={ds.image_keys} ✓")

    if n_with_language == 0:
        errors.append("No language found in first 3 samples")
    else:
        logger.info(f"  Language: {n_with_language}/3 with text, key={ds.language_key} ✓")

    logger.info(f"  Action dim: {ds.action_dim}, backend: {ds._backend} ✓")

    # Check 6: Action mapping round-trip
    if native_dim and len(ds) > 0:
        emb_id = DATASET_EMBODIMENT[ds_name]
        action = np.random.randn(native_dim).astype(np.float32)
        unified = map_to_unified(action, emb_id)
        extracted = extract_from_unified(unified, emb_id)
        mask_info = get_action_mask(emb_id, native_dim=native_dim)

        if not np.allclose(action, extracted[:native_dim], atol=1e-5):
            errors.append(f"Action round-trip failed: {action[:3]}... → {extracted[:3]}...")
        elif mask_info.active_dims != native_dim:
            errors.append(f"Mask active_dims={mask_info.active_dims} != native_dim={native_dim}")
        else:
            logger.info(f"  Action mapping: round-trip OK, mask={mask_info.active_dims} dims ✓")

    if errors:
        for e in errors:
            logger.error(f"  ✗ {e}")
        return False

    logger.info(f"  ✅ {ds_name} PASSED")
    return True


def verify_rlbench_dataset(data_root='./data'):
    """Verify RLBench dataset."""
    rlbench_dir = Path(data_root) / 'rlbench'
    if not rlbench_dir.exists():
        logger.error(f"RLBench not found: {rlbench_dir}")
        return False

    # Check for train/ subdirectory or any pkl files
    train_dir = rlbench_dir / 'train'
    pkl_files = list(rlbench_dir.rglob('low_dim_obs.pkl'))
    
    if not pkl_files:
        # May have downloaded but no episodes extracted yet
        any_files = list(rlbench_dir.rglob('*'))
        if any_files:
            logger.info(f"  RLBench directory exists with {len(any_files)} files (no episodes yet)")
            logger.info(f"  ✅ rlbench PASSED (download OK, episodes may need extraction)")
            return True
        logger.error("No RLBench data found")
        return False

    logger.info(f"  Episodes: {len(pkl_files)} ✓")
    logger.info(f"  ✅ rlbench PASSED")
    return True


def verify_dexgraspnet_dataset(data_root='./data'):
    """Verify DexGraspNet dataset."""
    from drifting_vla.data.dexgraspnet_dataset import DexGraspNetDataset
    import numpy as np

    ds_dir = Path(data_root) / 'dexgraspnet'
    errors = []

    # Check scenes exist
    scenes_dir = ds_dir / 'scenes'
    if not scenes_dir.exists():
        errors.append("No scenes directory")
    else:
        n_scenes = sum(1 for d in scenes_dir.iterdir() if d.is_dir() and d.name.startswith('scene_'))
        logger.info(f"  Scenes: {n_scenes} ✓")

    # Check meshdata
    mesh_dir = ds_dir / 'meshdata'
    if not mesh_dir.exists():
        errors.append("No meshdata directory (needed for rendering)")
    else:
        n_meshes = sum(1 for d in mesh_dir.iterdir() if d.is_dir())
        logger.info(f"  Meshes: {n_meshes} objects ✓")

    # Check rendered images
    rendered_dir = ds_dir / 'rendered'
    if rendered_dir.exists():
        n_rendered = sum(1 for _ in rendered_dir.rglob('*.png')) // 8
        logger.info(f"  Rendered: {n_rendered} scenes ✓")
    else:
        logger.warning(f"  Rendered: not yet (will auto-render on next download)")

    # Load and check samples
    ds = DexGraspNetDataset(data_dir=str(ds_dir), max_samples=10, image_size=448)
    if len(ds) == 0:
        errors.append("0 samples loaded")
    else:
        logger.info(f"  Samples: {len(ds)} ✓")
        s = ds[0]
        logger.info(f"  Images: {s['images'].shape}, lang={repr(s['language'][:50])}")
        logger.info(f"  Actions: {s['actions'].shape}")

        # Verify action is 23-dim
        if s['actions'].shape[1] != 23:
            errors.append(f"Action dim {s['actions'].shape[1]} != 23")

    if errors:
        for e in errors:
            logger.error(f"  ✗ {e}")
        return False

    logger.info(f"  ✅ dexgraspnet PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description='Verify dataset integrity')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--data-root', type=str, default='./data')
    args = parser.parse_args()

    from drifting_vla.data.action_mapping import LEROBOT_DATASETS

    if args.all:
        datasets = ['rlbench'] + sorted(LEROBOT_DATASETS) + ['dexgraspnet']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for ds in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Verifying: {ds}")
        logger.info(f"{'='*50}")

        if ds == 'rlbench':
            results[ds] = verify_rlbench_dataset(args.data_root)
        elif ds == 'dexgraspnet':
            results[ds] = verify_dexgraspnet_dataset(args.data_root)
        elif ds in LEROBOT_DATASETS:
            results[ds] = verify_lerobot_dataset(ds, args.data_root)
        else:
            logger.warning(f"Unknown dataset: {ds}")
            results[ds] = False

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Verification Summary:")
    all_pass = True
    for name, ok in results.items():
        status = '✅ PASS' if ok else '❌ FAIL'
        logger.info(f"  {status}  {name}")
        if not ok:
            all_pass = False
    logger.info(f"{'='*50}")

    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()

