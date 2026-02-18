#!/usr/bin/env python3
"""
Validate Episode HDF5 datasets for training readiness.

4-Tier validation:
  Tier 1: Schema — check 10 samples: shapes, dtypes, non-NaN
  Tier 2: Full-scan — iterate ALL samples for errors
  Tier 3: DataLoader — stress test with num_workers + batching
  Tier 4: Integration — UnifiedDataset with ALL datasets

Usage:
    python scripts/validate_training.py --dataset aloha --tier 1
    python scripts/validate_training.py --dataset aloha --tier 3
    python scripts/validate_training.py --all --tier 4
"""

import sys
import argparse
import logging
import time
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_tier1_schema(dataset_name: str, episodes_root: str = './data/episodes') -> bool:
    """Tier 1: Check first 10 samples for correct shapes, dtypes, non-NaN."""
    from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset

    ep_dir = Path(episodes_root) / dataset_name
    if not ep_dir.exists():
        logger.error(f"Episode dir not found: {ep_dir}")
        return False

    ds = EpisodeHDF5Dataset(str(ep_dir), max_samples=10)
    errors = []

    if len(ds) == 0:
        errors.append("0 samples")
    else:
        logger.info(f"  Samples: {len(ds)}")

    for i in range(min(10, len(ds))):
        try:
            s = ds[i]
        except Exception as e:
            errors.append(f"Sample {i}: load error: {e}")
            continue

        # Images
        imgs = s.get('images')
        if imgs is not None:
            if imgs.ndim != 4 or imgs.shape[1] != 3:
                errors.append(f"Sample {i}: image shape {imgs.shape} (expected [N, 3, H, W])")
            if torch.isnan(imgs).any():
                errors.append(f"Sample {i}: NaN in images")
            if imgs.abs().sum() == 0 and s['num_views'] > 0:
                errors.append(f"Sample {i}: all-zero images (but {s['num_views']} views expected)")

        # Actions
        acts = s.get('actions')
        if acts is not None:
            if acts.shape != (16, 128):
                errors.append(f"Sample {i}: action shape {acts.shape} (expected [16, 128])")
            if torch.isnan(acts).any():
                errors.append(f"Sample {i}: NaN in actions")

        # Action mask
        mask = s.get('action_mask')
        if mask is not None:
            if mask.shape != (128,):
                errors.append(f"Sample {i}: mask shape {mask.shape}")
            n_active = int(mask.sum())
            if n_active == 0:
                errors.append(f"Sample {i}: action mask all zeros")

        # Language
        lang = s.get('language', '')
        if not lang and i == 0:
            logger.warning(f"  Sample {i}: no language (may be OK for some datasets)")

        # Metadata
        n_views = s.get('num_views', 0)
        n_frames = s.get('num_frames', 0)
        if n_views == 0:
            logger.warning(f"  Sample {i}: num_views=0")

    ds.close()

    if errors:
        for e in errors:
            logger.error(f"  ✗ {e}")
        return False

    logger.info(f"  ✅ Tier 1 PASSED: {dataset_name}")
    return True


def validate_tier2_fullscan(dataset_name: str, episodes_root: str = './data/episodes') -> bool:
    """Tier 2: Iterate ALL samples, check for errors."""
    from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset

    ep_dir = Path(episodes_root) / dataset_name
    ds = EpisodeHDF5Dataset(str(ep_dir))

    n_errors = 0
    n_total = len(ds)

    for i in tqdm(range(n_total), desc=f"Full-scan {dataset_name}"):
        try:
            s = ds[i]
            if torch.isnan(s['actions']).any():
                n_errors += 1
                if n_errors <= 5:
                    logger.warning(f"  NaN in actions at sample {i}")
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                logger.warning(f"  Error at sample {i}: {e}")

    ds.close()

    if n_errors > 0:
        logger.error(f"  ✗ Tier 2: {n_errors}/{n_total} errors")
        return False

    logger.info(f"  ✅ Tier 2 PASSED: {dataset_name} ({n_total} samples, 0 errors)")
    return True


def validate_tier3_dataloader(
    dataset_name: str,
    episodes_root: str = './data/episodes',
    num_workers: int = 4,
    batch_size: int = 16,
    n_batches: int = 100,
) -> bool:
    """Tier 3: DataLoader stress test with multiprocessing."""
    from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset
    from torch.utils.data import DataLoader

    ep_dir = Path(episodes_root) / dataset_name
    ds = EpisodeHDF5Dataset(str(ep_dir))

    def collate(batch):
        """Simple collate for testing."""
        max_imgs = max(s['images'].shape[0] for s in batch)
        C, H, W = batch[0]['images'].shape[1:]
        padded = []
        for s in batch:
            img = s['images']
            if img.shape[0] < max_imgs:
                pad = torch.zeros(max_imgs - img.shape[0], C, H, W)
                img = torch.cat([img, pad], dim=0)
            padded.append(img)
        return {
            'images': torch.stack(padded),
            'actions': torch.stack([s['actions'] for s in batch]),
            'action_mask': torch.stack([s['action_mask'] for s in batch]),
        }

    loader = DataLoader(
        ds, batch_size=min(batch_size, len(ds)),
        shuffle=True, num_workers=num_workers,
        collate_fn=collate, pin_memory=False,
    )

    n_errors = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= n_batches:
            break
        try:
            assert batch['images'].ndim == 5, f"images ndim={batch['images'].ndim}"
            assert batch['actions'].shape[-1] == 128, f"action dim={batch['actions'].shape[-1]}"
            assert not torch.isnan(batch['actions']).any(), "NaN in actions"
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                logger.warning(f"  Batch {batch_idx}: {e}")

    elapsed = time.time() - start
    actual_batches = min(n_batches, len(loader))
    samples_per_sec = (actual_batches * batch_size) / max(elapsed, 0.001)

    ds.close()

    if n_errors > 0:
        logger.error(f"  ✗ Tier 3: {n_errors} errors in {actual_batches} batches")
        return False

    logger.info(
        f"  ✅ Tier 3 PASSED: {dataset_name} "
        f"({actual_batches} batches, {samples_per_sec:.0f} samples/s, "
        f"workers={num_workers})"
    )
    return True


def validate_tier4_integration(
    episodes_root: str = './data/episodes',
    n_batches: int = 200,
) -> bool:
    """Tier 4: Full integration test with UnifiedDataset + all datasets."""
    from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset

    ep_root = Path(episodes_root)
    all_ds_dirs = [d for d in ep_root.iterdir() if d.is_dir() and (d / 'metadata.json').exists()]

    if not all_ds_dirs:
        logger.error(f"No episode datasets found in {ep_root}")
        return False

    logger.info(f"Integration test with {len(all_ds_dirs)} datasets")

    # Load all datasets
    datasets = {}
    for d in all_ds_dirs:
        try:
            ds = EpisodeHDF5Dataset(str(d), max_samples=500)
            if len(ds) > 0:
                datasets[d.name] = ds
                logger.info(f"  Loaded {d.name}: {len(ds)} samples")
        except Exception as e:
            logger.warning(f"  Failed to load {d.name}: {e}")

    if not datasets:
        logger.error("No datasets loaded")
        return False

    # Simple round-robin test
    n_errors = 0
    sample_count = 0

    for batch_idx in range(n_batches):
        batch = []
        for ds_name, ds in datasets.items():
            idx = batch_idx % len(ds)
            try:
                s = ds[idx]
                batch.append(s)
                sample_count += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning(f"  Error from {ds_name}[{idx}]: {e}")

        # Check batch compatibility
        if batch:
            shapes = set(s['actions'].shape for s in batch)
            if len(shapes) > 1:
                n_errors += 1
                logger.warning(f"  Batch {batch_idx}: mixed action shapes: {shapes}")

    for ds in datasets.values():
        ds.close()

    if n_errors > 0:
        logger.error(f"  ✗ Tier 4: {n_errors} errors across {sample_count} samples")
        return False

    logger.info(f"  ✅ Tier 4 PASSED: {len(datasets)} datasets, {sample_count} samples, 0 errors")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate training readiness')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--tier', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--episodes-root', type=str, default='./data/episodes')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    if args.tier == 4:
        # Integration test — uses all datasets
        ok = validate_tier4_integration(args.episodes_root)
        sys.exit(0 if ok else 1)

    # Per-dataset validation
    if args.all:
        ep_root = Path(args.episodes_root)
        datasets = [d.name for d in ep_root.iterdir() if d.is_dir() and (d / 'metadata.json').exists()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    results = {}
    for ds_name in sorted(datasets):
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating: {ds_name} (Tier {args.tier})")
        logger.info(f"{'='*50}")

        if args.tier == 1:
            results[ds_name] = validate_tier1_schema(ds_name, args.episodes_root)
        elif args.tier == 2:
            results[ds_name] = validate_tier2_fullscan(ds_name, args.episodes_root)
        elif args.tier == 3:
            results[ds_name] = validate_tier3_dataloader(
                ds_name, args.episodes_root, args.num_workers, args.batch_size
            )

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Validation Summary (Tier {args.tier}):")
    for name, ok in results.items():
        status = '✅ PASS' if ok else '❌ FAIL'
        logger.info(f"  {status}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()













