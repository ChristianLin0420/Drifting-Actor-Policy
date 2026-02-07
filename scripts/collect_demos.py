#!/usr/bin/env python3
"""
RLBench Demo Collection Script
==============================

Collect expert demonstrations from RLBench tasks.

Usage:
    python scripts/collect_demos.py --task reach_target --num-episodes 100
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.data import RLBenchCollector, CollectorConfig

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collect RLBench demonstrations')
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='reach_target',
        help='Task name to collect (or comma-separated list)'
    )
    parser.add_argument(
        '--num-episodes', '-n',
        type=int,
        default=100,
        help='Number of episodes per task'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./data/rlbench',
        help='Output directory'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image resolution'
    )
    parser.add_argument(
        '--cameras',
        type=str,
        default='front,wrist',
        help='Camera names (comma-separated)'
    )
    parser.add_argument(
        '--save-depth',
        action='store_true',
        help='Save depth images'
    )
    
    return parser.parse_args()


def main():
    """Main collection function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Parse arguments
    tasks = [t.strip() for t in args.task.split(',')]
    cameras = [c.strip() for c in args.cameras.split(',')]
    
    logger.info(f"Collecting demonstrations for tasks: {tasks}")
    logger.info(f"Episodes per task: {args.num_episodes}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create collector
    config = CollectorConfig(
        tasks=tasks,
        num_episodes=args.num_episodes,
        image_size=args.image_size,
        cameras=cameras,
        save_depth=args.save_depth,
        output_dir=args.output_dir,
    )
    
    collector = RLBenchCollector(config)
    
    try:
        # Collect demonstrations
        results = collector.collect()
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info("COLLECTION RESULTS")
        logger.info(f"{'='*50}")
        
        total = 0
        for task, count in results.items():
            logger.info(f"  {task}: {count} episodes")
            total += count
        
        logger.info(f"\nTotal: {total} episodes")
        logger.info(f"Output: {args.output_dir}")
        
    finally:
        collector.close()
    
    logger.info("Collection complete!")


if __name__ == '__main__':
    main()


