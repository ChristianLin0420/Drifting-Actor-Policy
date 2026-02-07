#!/usr/bin/env python3
"""
Drifting-VLA Evaluation Script
==============================

Evaluate trained models on RLBench tasks.

Usage:
    python scripts/evaluate.py --checkpoint /path/to/model.pt --tasks reach_target,pick_up_cup
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.inference import DriftingVLAPolicy, PolicyConfig
from drifting_vla.inference.rollout import evaluate_policy, save_rollout_video, RolloutConfig
from drifting_vla.envs import RLBenchEnvironment, EnvConfig
from drifting_vla.logging import WandBLogger, LoggerConfig

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Drifting-VLA model')
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tasks', '-t',
        type=str,
        default='reach_target',
        help='Comma-separated list of tasks to evaluate'
    )
    parser.add_argument(
        '--num-episodes', '-n',
        type=int,
        default=10,
        help='Number of episodes per task'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./eval_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--cfg-scale',
        type=float,
        default=2.0,
        help='Classifier-free guidance scale'
    )
    parser.add_argument(
        '--save-videos',
        action='store_true',
        help='Save rollout videos'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='WandB project for logging results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(',')]
    logger.info(f"Evaluating on tasks: {tasks}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    logger.info(f"Loading model from {args.checkpoint}")
    policy_config = PolicyConfig(
        cfg_scale=args.cfg_scale,
        device=args.device,
        use_ema=True,
        temporal_ensemble=True,
    )
    policy = DriftingVLAPolicy.from_checkpoint(args.checkpoint, policy_config)
    
    # Setup environment
    env_config = EnvConfig(
        task=tasks,
        image_size=224,
        cameras=['front'],
        max_episode_length=args.max_steps,
        headless=True,
    )
    env = RLBenchEnvironment(env_config)
    
    # Setup rollout config
    rollout_config = RolloutConfig(
        max_steps=args.max_steps,
        save_video=args.save_videos,
        video_fps=10,
        verbose=True,
    )
    
    # Setup WandB logging
    wandb_logger = None
    if args.wandb_project:
        wandb_logger = WandBLogger(
            LoggerConfig(
                project=args.wandb_project,
                tags=['evaluation'],
            )
        )
        wandb_logger.init()
    
    try:
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluate_policy(
            env=env,
            policy=policy,
            tasks=tasks,
            num_episodes=args.num_episodes,
            config=rollout_config,
        )
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Overall Success Rate: {results['success_rate']:.2%}")
        logger.info(f"Average Episode Length: {results['avg_episode_length']:.1f}")
        logger.info(f"\nPer-Task Success Rates:")
        for task, rate in results['per_task_success'].items():
            logger.info(f"  {task}: {rate:.2%}")
        
        # Save results to JSON
        results_json = {
            'success_rate': float(results['success_rate']),
            'avg_episode_length': float(results['avg_episode_length']),
            'per_task_success': {k: float(v) for k, v in results['per_task_success'].items()},
            'checkpoint': args.checkpoint,
            'cfg_scale': args.cfg_scale,
            'num_episodes': args.num_episodes,
        }
        
        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"\nSaved results to {results_path}")
        
        # Save videos
        if args.save_videos:
            videos_dir = output_dir / 'videos'
            videos_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(results['results']):
                if result.frames:
                    video_path = videos_dir / f'rollout_{i}_{result.info["task"]}.mp4'
                    save_rollout_video(result, str(video_path))
        
        # Log to WandB
        if wandb_logger:
            wandb_logger.log({
                'eval/success_rate': results['success_rate'],
                'eval/avg_episode_length': results['avg_episode_length'],
            })
            
            for task, rate in results['per_task_success'].items():
                wandb_logger.log({f'eval/success_{task}': rate})
            
            wandb_logger.finish()
        
    finally:
        env.close()
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()


