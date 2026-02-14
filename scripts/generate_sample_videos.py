#!/usr/bin/env python3
"""
Generate sample visualizations from downloaded datasets.

Creates per-dataset sample images and action plots for visual inspection.

Usage:
    python scripts/generate_sample_videos.py --all
    python scripts/generate_sample_videos.py --dataset aloha
    python scripts/generate_sample_videos.py --dataset bc_z --n-frames 20
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_dataset(ds_name, data_root='./data', output_dir='./sample_videos', n_frames=10):
    """Generate sample visualization for a dataset."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return False

    from drifting_vla.data.lerobot_dataset import LeRobotDataset
    from drifting_vla.data.action_mapping import LEROBOT_DATASETS, DATASET_HF_REPOS

    out_dir = Path(output_dir) / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if ds_name not in LEROBOT_DATASETS:
        logger.warning(f"  {ds_name} not a LeRobot dataset, skipping visualization")
        return False

    hf_repo = DATASET_HF_REPOS.get(ds_name)
    if not hf_repo:
        logger.warning(f"  No HF repo for {ds_name}")
        return False

    arrow_path = Path(data_root) / ds_name / 'arrow_data'
    if not arrow_path.exists():
        logger.warning(f"  Arrow data not found at {arrow_path}. Run download first.")
        return False

    logger.info(f"Loading {ds_name}...")
    ds = LeRobotDataset(
        repo_id=hf_repo,
        image_size=448,
        action_horizon=16,
        max_samples=n_frames,
        data_root=data_root,
        ds_name=ds_name,
    )

    if len(ds) == 0:
        logger.warning(f"  {ds_name}: 0 samples, skipping")
        return False

    logger.info(f"  {ds_name}: {len(ds)} samples, backend={ds._backend}")
    logger.info(f"  image_keys: {ds.image_keys}")
    logger.info(f"  language_key: {ds.language_key}")
    logger.info(f"  action_dim: {ds.action_dim}")

    # --- 1. Sample grid: images + language ---
    n_show = min(n_frames, len(ds))
    n_views = max(len(ds.image_keys), 1)

    fig, axes = plt.subplots(n_show, n_views + 1, figsize=(4 * (n_views + 1), 3 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    if n_views + 1 == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'{ds_name} — {len(ds)} samples, action_dim={ds.action_dim}', fontsize=14, fontweight='bold')

    all_actions = []
    for i in range(n_show):
        sample = ds[i]
        images = sample['images']  # [T*V, 3, H, W]
        actions = sample['actions']  # [T, D]
        language = sample.get('language', '')
        proprio = sample.get('proprio', None)

        all_actions.append(actions.numpy())

        # Show images (first frame only, all views)
        for v in range(n_views):
            ax = axes[i, v]
            if v < images.shape[0]:
                img = images[v].numpy()  # [3, H, W]
                if img.max() <= 1.0:
                    img = img * 255
                img = img.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)  # [H, W, 3]
                ax.imshow(img)
                if i == 0 and v < len(ds.image_keys):
                    ax.set_title(ds.image_keys[v].split('.')[-1], fontsize=8)
            else:
                ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
                ax.set_title('no image', fontsize=8)
            ax.axis('off')

        # Action bar chart
        ax_act = axes[i, n_views]
        action_vals = actions[0].numpy()  # First timestep
        ax_act.barh(range(len(action_vals)), action_vals, height=0.7, color='steelblue')
        ax_act.set_xlim(-3, 3)
        ax_act.set_ylabel(f'sample {i}', fontsize=7)
        if language:
            ax_act.set_title(language[:60], fontsize=7, wrap=True)
        else:
            ax_act.set_title('(no language)', fontsize=7, color='gray')
        ax_act.tick_params(labelsize=6)

    plt.tight_layout()
    grid_path = out_dir / 'sample_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved {grid_path}")

    # --- 2. Action distribution plot ---
    if all_actions:
        all_actions = np.stack(all_actions)[:, 0, :]  # [N, D] first timestep
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Action heatmap
        ax = axes[0]
        im = ax.imshow(all_actions.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Action dim')
        ax.set_title(f'{ds_name} — Action heatmap')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Action histograms (first 8 dims)
        ax = axes[1]
        n_dims = min(8, all_actions.shape[1])
        for d in range(n_dims):
            ax.hist(all_actions[:, d], bins=20, alpha=0.5, label=f'dim {d}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{ds_name} — Action distributions (first {n_dims} dims)')
        ax.legend(fontsize=6)

        plt.tight_layout()
        dist_path = out_dir / 'action_distribution.png'
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved {dist_path}")

    # --- 3. Summary text ---
    summary_path = out_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"HF repo: {hf_repo}\n")
        f.write(f"Backend: {ds._backend}\n")
        f.write(f"Samples: {len(ds)}\n")
        f.write(f"Action dim: {ds.action_dim}\n")
        f.write(f"Image keys: {ds.image_keys}\n")
        f.write(f"Language key: {ds.language_key}\n")
        if all_actions is not None and len(all_actions) > 0:
            f.write(f"Action mean: {all_actions.mean(0)[:8]}\n")
            f.write(f"Action std:  {all_actions.std(0)[:8]}\n")
        # Show first 3 language samples
        for j in range(min(3, len(ds))):
            s = ds[j]
            f.write(f"Sample {j} language: {repr(s.get('language', ''))}\n")
    logger.info(f"  ✓ Saved {summary_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate sample visualizations')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--all', action='store_true', help='Visualize all downloaded datasets')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./sample_videos')
    parser.add_argument('--n-frames', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()

    from drifting_vla.data.action_mapping import LEROBOT_DATASETS

    if args.all:
        datasets = sorted(LEROBOT_DATASETS)
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return

    results = {}
    for ds_name in datasets:
        arrow_path = Path(args.data_root) / ds_name / 'arrow_data'
        if not arrow_path.exists():
            continue
        logger.info(f"\n{'='*50}")
        logger.info(f"Visualizing: {ds_name}")
        logger.info(f"{'='*50}")
        results[ds_name] = visualize_dataset(
            ds_name, args.data_root, args.output_dir, args.n_frames
        )

    logger.info(f"\n{'='*50}")
    logger.info("Summary:")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    logger.info(f"Output: {args.output_dir}/")
    logger.info(f"{'='*50}")


if __name__ == '__main__':
    main()






