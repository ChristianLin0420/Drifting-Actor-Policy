#!/usr/bin/env python3
"""
Download RLBench dataset from Hugging Face.

This script downloads pre-generated RLBench demonstrations from:
https://huggingface.co/datasets/hqfang/rlbench-18-tasks

The dataset contains 18 tasks with:
- Train: 100 episodes per task
- Val: 25 episodes per task
- Test: 25 episodes per task

Usage:
    python scripts/download_rlbench_data.py --output-dir ./data/rlbench
    python scripts/download_rlbench_data.py --output-dir ./data/rlbench --tasks reach_target,pick_up_cup
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dataset repository on Hugging Face
REPO_ID = "hqfang/rlbench-18-tasks"

# Available tasks in the dataset (18 tasks from PerAct)
AVAILABLE_TASKS = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]


def download_task(
    task: str,
    output_dir: Path,
    splits: list[str] = ["train", "val", "test"],
) -> None:
    """
    Download a single task from the dataset.
    
    Args:
        task: Task name to download
        output_dir: Directory to save the data
        splits: Which splits to download (train, val, test)
    """
    logger.info(f"Downloading task: {task}")
    
    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # The zip file name pattern (files are under data/ prefix in the repo)
        zip_filename = f"data/{split}/{task}.zip"
        
        try:
            # Download from Hugging Face
            logger.info(f"  Downloading {split}/{task}.zip...")
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_filename,
                repo_type="dataset",
                local_dir=output_dir,
            )
            
            # Extract the zip file (downloaded to output_dir/data/split/task.zip)
            zip_path = output_dir / zip_filename
            if zip_path.exists():
                logger.info(f"  Extracting {split}/{task}.zip...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(split_dir)
                
                # Remove the zip file to save space
                zip_path.unlink()
                logger.info(f"  Completed {split}/{task}")
            
        except Exception as e:
            logger.warning(f"  Failed to download {split}/{task}: {e}")


def list_available_tasks() -> list[str]:
    """
    List all available tasks in the dataset.
    
    Returns:
        List of task names
    """
    try:
        files = list_repo_files(REPO_ID, repo_type="dataset")
        tasks = set()
        for f in files:
            if f.endswith('.zip') and '/' in f:
                task_name = f.split('/')[-1].replace('.zip', '')
                tasks.add(task_name)
        return sorted(tasks)
    except Exception:
        return AVAILABLE_TASKS


def main():
    parser = argparse.ArgumentParser(
        description="Download RLBench dataset from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/rlbench",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to download (default: all)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to download"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_tasks:
        logger.info("Available tasks:")
        for task in AVAILABLE_TASKS:
            print(f"  - {task}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [s.strip() for s in args.splits.split(",")]
    
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        # Validate tasks
        for task in tasks:
            if task not in AVAILABLE_TASKS:
                logger.warning(f"Task '{task}' not in available tasks, skipping")
        tasks = [t for t in tasks if t in AVAILABLE_TASKS]
    else:
        tasks = AVAILABLE_TASKS
    
    logger.info(f"Downloading {len(tasks)} tasks to {output_dir}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Tasks: {tasks}")
    
    for task in tasks:
        download_task(task, output_dir, splits)
    
    logger.info("Download complete!")
    logger.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()

