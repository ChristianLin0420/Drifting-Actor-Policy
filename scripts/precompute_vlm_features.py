#!/usr/bin/env python3
"""
Pre-compute VLM features for all datasets.

This script runs the VLM (Qwen3-VL-2B or PaliGemma2-3B) once on each sample
and saves the hidden states to HDF5 for fast training.

Each sample's images + language instruction are fed through the VLM together
to produce joint vision-language features. This ensures the features encode
both the visual scene AND the task instruction — they are always aligned.

Usage:
    # Pre-compute for a specific dataset
    python scripts/precompute_vlm_features.py --dataset rlbench --vlm qwen3vl

    # Pre-compute for all datasets
    python scripts/precompute_vlm_features.py --all --vlm qwen3vl

    # Limit samples (for testing on A40)
    python scripts/precompute_vlm_features.py --dataset rlbench --vlm qwen3vl --max-samples 100

Time estimate:
    - Qwen3-VL-2B: ~40ms per sample per GPU
    - 1M samples on 8 GPUs: ~83 minutes
    - Storage: ~4KB per sample → 1M samples ≈ 4 GB
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import torch
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


VLM_HIDDEN_DIMS = {
    'qwen3vl': 1536,
    'paligemma2': 2048,
}


def precompute_vlm_features(
    dataset_name: str,
    dataset,
    output_dir: Path,
    vlm_model_key: str = 'qwen3vl',
    batch_size: int = 8,
    device: str = 'cuda',
    max_samples: int = None,
):
    """
    Pre-compute VLM features using the actual VLM model.
    
    For each sample, feeds images + language through the VLM to produce
    aligned vision-language features.
    """
    from drifting_vla.models.vlm_backbone import VLMBackbone, VLMConfig

    hidden_dim = VLM_HIDDEN_DIMS[vlm_model_key]
    output_path = output_dir / f'{dataset_name}_vlm_features.h5'

    n_samples = len(dataset)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    logger.info(f"Pre-computing VLM features for {dataset_name}: {n_samples} samples")
    logger.info(f"  VLM: {vlm_model_key}, device: {device}")

    # Initialize VLM
    vlm_config = VLMConfig(model_key=vlm_model_key, dit_hidden_dim=768)
    vlm = VLMBackbone(vlm_config)
    vlm.load_vlm(torch.device(device))
    vlm.eval()

    n_skipped = 0

    with h5py.File(str(output_path), 'w') as f:
        f.attrs['vlm_model'] = vlm_model_key
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['n_samples'] = n_samples

        for i in tqdm(range(n_samples), desc=f'VLM features for {dataset_name}'):
            try:
                sample = dataset[i]
                images = sample['images']      # [V, 3, H, W] or [3, H, W]
                language = sample.get('language', '')

                if isinstance(images, np.ndarray):
                    images = torch.from_numpy(images).float()

                # Ensure [V, 3, H, W]
                if images.dim() == 3:
                    images = images.unsqueeze(0)

                images = images.to(device)

                # Warn if language is empty (data alignment issue)
                if not language:
                    n_skipped += 1
                    if n_skipped <= 5:
                        logger.warning(f"  Sample {i}: empty language instruction")

                # Extract features (images + language jointly through VLM)
                with torch.no_grad():
                    hidden, pooled = vlm.extract_features(images, language)

                # Save
                grp = f.create_group(f'sample_{i}')
                grp.create_dataset('hidden', data=hidden.cpu().half().numpy(), compression='gzip', compression_opts=4)
                grp.create_dataset('pooled', data=pooled.cpu().half().numpy())
                grp.attrs['sequence_length'] = hidden.shape[0]
                grp.attrs['language'] = language  # Store language for verification

            except Exception as e:
                logger.warning(f"  Failed on sample {i}: {e}")
                # Skip — do NOT generate fake features
                continue

    if n_skipped > 0:
        logger.warning(f"  {n_skipped}/{n_samples} samples had empty language instructions")

    file_size = output_path.stat().st_size / 1e6
    logger.info(f"✓ Saved VLM features to {output_path} ({file_size:.1f} MB)")


def load_dataset_for_precompute(dataset_name: str, data_root: str):
    """Load a dataset for feature pre-computation."""
    from drifting_vla.data.action_mapping import LEROBOT_DATASETS, DATASET_HF_REPOS

    data_dir = Path(data_root) / dataset_name

    if dataset_name == 'rlbench':
        from drifting_vla.data.rlbench_dataset import RLBenchDataset
        return RLBenchDataset(
            data_dir=str(data_dir),
            split='train',
            image_size=448,
            cameras=['front', 'wrist'],
            action_horizon=1,
        )

    elif dataset_name == 'dexgraspnet':
        from drifting_vla.data.dexgraspnet_dataset import DexGraspNetDataset
        return DexGraspNetDataset(
            data_dir=str(data_dir),
            image_size=448,
            action_horizon=1,
        )

    elif dataset_name in LEROBOT_DATASETS:
        from drifting_vla.data.lerobot_dataset import LeRobotDataset
        hf_repo = DATASET_HF_REPOS.get(dataset_name)
        if hf_repo is None:
            raise ValueError(f"No HF repo configured for {dataset_name}")
        return LeRobotDataset(
            repo_id=hf_repo,
            image_size=448,
            action_horizon=1,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: rlbench, dexgraspnet, {', '.join(LEROBOT_DATASETS)}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute VLM features for training')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--all', action='store_true', help='Process all datasets')
    parser.add_argument('--vlm', type=str, default='qwen3vl', choices=['qwen3vl', 'paligemma2'])
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./data/vlm_features')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-samples', type=int, default=None)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        from drifting_vla.data.action_mapping import LEROBOT_DATASETS
        datasets_to_process = ['rlbench', 'dexgraspnet'] + sorted(LEROBOT_DATASETS)
    elif args.dataset:
        datasets_to_process = [args.dataset]
    else:
        parser.print_help()
        return

    for ds_name in datasets_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ds_name}")
        logger.info(f"{'='*60}")

        try:
            dataset = load_dataset_for_precompute(ds_name, args.data_root)

            precompute_vlm_features(
                ds_name, dataset, output_dir,
                vlm_model_key=args.vlm,
                batch_size=args.batch_size,
                device=args.device,
                max_samples=args.max_samples,
            )

        except Exception as e:
            logger.error(f"Failed to process {ds_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()



Pre-compute VLM features for all datasets.

This script runs the VLM (Qwen3-VL-2B or PaliGemma2-3B) once on each sample
and saves the hidden states to HDF5 for fast training.

Each sample's images + language instruction are fed through the VLM together
to produce joint vision-language features. This ensures the features encode
both the visual scene AND the task instruction — they are always aligned.

Usage:
    # Pre-compute for a specific dataset
    python scripts/precompute_vlm_features.py --dataset rlbench --vlm qwen3vl

    # Pre-compute for all datasets
    python scripts/precompute_vlm_features.py --all --vlm qwen3vl

    # Limit samples (for testing on A40)
    python scripts/precompute_vlm_features.py --dataset rlbench --vlm qwen3vl --max-samples 100

Time estimate:
    - Qwen3-VL-2B: ~40ms per sample per GPU
    - 1M samples on 8 GPUs: ~83 minutes
    - Storage: ~4KB per sample → 1M samples ≈ 4 GB
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import torch
import logging
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


VLM_HIDDEN_DIMS = {
    'qwen3vl': 1536,
    'paligemma2': 2048,
}


