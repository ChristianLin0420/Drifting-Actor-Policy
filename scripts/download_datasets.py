#!/usr/bin/env python3
"""
Download real data for Drifting-VLA pretraining pipeline.

Downloads N real samples per dataset from HuggingFace for testing/debugging.

Usage:
    python scripts/download_datasets.py --test                  # 50 samples each
    python scripts/download_datasets.py --dataset aloha --test  # specific dataset
    python scripts/download_datasets.py --all                   # full datasets
"""

import argparse
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = Path('./data')
TEST_SAMPLES = 50


# ──────────────────────────────────────────────────────────────────────
# All verified datasets
# ──────────────────────────────────────────────────────────────────────

DATASETS = {
    'rlbench': {
        'hf_repo': 'hqfang/rlbench-18-tasks',
        'type': 'rlbench',
        'description': 'RLBench 18-task sim (270K, 8-dim gripper)',
    },
    'dexgraspnet': {
        'hf_repo': 'lhrlhr/DexGraspNet2.0',
        'type': 'hf_generic',
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
    **{f'behavior1k_t{i:04d}': {
        'hf_repo': f'lerobot/behavior1k-task{i:04d}',
        'type': 'lerobot',
        'description': f'Behavior 1K task {i:04d} OmniGibson sim (23-dim bimanual)',
    } for i in range(50)},
}


# ──────────────────────────────────────────────────────────────────────
# Download functions
# ──────────────────────────────────────────────────────────────────────

def download_lerobot_dataset(name, info, dest, n_samples=None):
    """
    Download a LeRobot-format dataset — only the samples needed, not the full dataset.

    Strategy:
      1. Try HuggingFace `datasets` library with split slicing (downloads only needed parquet chunks)
      2. Fallback to lerobot API if datasets library fails (downloads full cache)
    """
    hf_repo = info['hf_repo']
    save_path = dest / name / 'arrow_data'

    if save_path.exists():
        logger.info(f"  ✓ {name} already downloaded at {save_path}")
        return True

    logger.info(f"  Downloading {hf_repo} ({n_samples or 'all'} samples) ...")

    # --- Strategy 1: HuggingFace datasets library (partial download) ---
    try:
        from datasets import load_dataset

        if n_samples:
            # Only download the first n_samples from the 'train' split
            # This downloads only the parquet chunks needed, NOT the full dataset
            hf_ds = load_dataset(hf_repo, split=f'train[:{n_samples}]')
        else:
            hf_ds = load_dataset(hf_repo, split='train')

        save_path.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(save_path))

        logger.info(f"  ✓ {name}: {len(hf_ds)} samples saved (HF datasets, partial download)")
        if len(hf_ds) > 0:
            cols = hf_ds.column_names[:8]
            logger.info(f"    Columns: {cols}")
        return True

    except Exception as e1:
        logger.warning(f"  HF datasets failed: {e1}")

    # --- Strategy 2: lerobot API (downloads full cache, extracts n_samples) ---
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as LDS
        from datasets import Dataset as HFDataset

        logger.info(f"  Falling back to lerobot API (may download full dataset to cache)...")
        ds = LDS(hf_repo)
        total = len(ds)

        if n_samples and n_samples < total:
            indices = list(range(n_samples))
        else:
            indices = list(range(total))
            n_samples = total

        records = []
        for i in indices:
            try:
                row = ds[i]
                record = {}
                for k, v in row.items():
                    if hasattr(v, 'numpy'):
                        record[k] = v.numpy().tolist()
                    else:
                        record[k] = v
                records.append(record)
            except Exception:
                continue  # Skip failed samples (video decode errors)

        if not records:
            logger.error(f"  ✗ No valid samples from {hf_repo}")
            return False

        hf_ds = HFDataset.from_list(records)
        save_path.mkdir(parents=True, exist_ok=True)
        hf_ds.save_to_disk(str(save_path))

        logger.info(f"  ✓ {name}: {len(records)} samples saved (lerobot API)")

        # Clean lerobot cache to save disk
        import shutil
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'lerobot'
        if cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / 1e9
            if cache_size > 1.0:  # Only clean if > 1 GB
                logger.info(f"  Cleaning lerobot cache ({cache_size:.1f} GB)...")
                shutil.rmtree(str(cache_dir), ignore_errors=True)

        return True

    except Exception as e2:
        logger.error(f"  ✗ Failed to download {hf_repo}: {e2}")
        import traceback
        traceback.print_exc()
        return False


def download_rlbench_dataset(name, info, dest, n_samples=None):
    """Download RLBench data (PerAct format)."""
    from huggingface_hub import snapshot_download

    rlbench_dir = dest / 'rlbench'
    train_dir = rlbench_dir / 'train'
    if train_dir.exists() and any(train_dir.rglob('*.pkl')):
        n_eps = sum(1 for _ in train_dir.rglob('low_dim_obs.pkl'))
        logger.info(f"  ✓ RLBench already exists: {n_eps} episodes")
        return True

    tasks = ['close_jar']
    for task in tasks:
        logger.info(f"  Downloading RLBench/{task} ...")
        try:
            patterns = [f'train/{task}/all_variations/episodes/episode0/**',
                        f'train/{task}/all_variations/episodes/episode1/**'] if n_samples else [f'train/{task}/**']
            snapshot_download(
                repo_id=info['hf_repo'], repo_type='dataset',
                local_dir=str(rlbench_dir),
                allow_patterns=patterns, ignore_patterns=['*.avi', '*.mp4'],
            )
            logger.info(f"  ✓ RLBench/{task} downloaded")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            return False
    return True


def download_hf_generic(name, info, dest, n_samples=None):
    """Download a generic HuggingFace dataset."""
    from huggingface_hub import snapshot_download

    ds_dir = dest / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {info['hf_repo']} ...")
    try:
        snapshot_download(
            repo_id=info['hf_repo'], repo_type='dataset',
            local_dir=str(ds_dir),
            allow_patterns=['*.parquet', '*.json', '*.h5', '*.hdf5', '*.npz', 'README*'],
            ignore_patterns=['*.ply', '*.obj', '*.stl', '*.png', '*.jpg'],
        )
        logger.info(f"  ✓ {name} downloaded")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


DOWNLOAD_FN = {
    'lerobot': download_lerobot_dataset,
    'rlbench': download_rlbench_dataset,
    'hf_generic': download_hf_generic,
}


def main():
    parser = argparse.ArgumentParser(description='Download datasets for Drifting-VLA')
    parser.add_argument('--test', action='store_true', help=f'Download {TEST_SAMPLES} real samples per dataset')
    parser.add_argument('--all', action='store_true', help='Download full datasets')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--n-samples', type=int, default=None)
    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = Path(args.data_root)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples or (TEST_SAMPLES if args.test else None)
    datasets_to_dl = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    logger.info(f"{'='*60}")
    logger.info(f"Drifting-VLA Dataset Download")
    logger.info(f"  Mode: {'TEST (' + str(n_samples) + ' samples)' if n_samples else 'FULL'}")
    logger.info(f"  Datasets: {list(datasets_to_dl.keys())}")
    logger.info(f"{'='*60}")

    results = {}
    for ds_name, ds_info in datasets_to_dl.items():
        logger.info(f"\n--- {ds_name} ({ds_info['description']}) ---")
        fn = DOWNLOAD_FN.get(ds_info['type'], download_hf_generic)
        results[ds_name] = fn(ds_name, ds_info, DATA_ROOT, n_samples)

    logger.info(f"\n{'='*60}")
    logger.info("Summary:")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()

