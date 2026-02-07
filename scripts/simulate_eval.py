#!/usr/bin/env python3
"""
Simulation-Based Evaluation Script
===================================

Evaluate a trained Drifting-VLA model in RLBench simulation.
Records rollout videos and computes success rates.

Usage:
    python scripts/simulate_eval.py checkpoint=checkpoints/checkpoint_step_10000.pt
    python scripts/simulate_eval.py checkpoint=checkpoints/best.pt tasks='[close_jar,open_drawer]'
    
Requirements:
    - RLBench and CoppeliaSim must be installed
    - Xvfb must be running for headless mode
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def setup_display():
    """Setup virtual display for headless rendering."""
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':99'
        logger.info("Set DISPLAY=:99 for Xvfb")


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    from drifting_vla.models import DriftingVLA, DriftingVLAConfig
    from drifting_vla.models.vision_encoder import VisionEncoderConfig
    from drifting_vla.models.language_encoder import LanguageEncoderConfig
    from drifting_vla.models.fusion import FusionConfig
    from drifting_vla.models.dit import DiTConfig
    from drifting_vla.models.action_decoder import ActionDecoderConfig
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Build config from checkpoint
    if isinstance(config, dict):
        model_cfg = config.get('model', config)
        
        vision_cfg = VisionEncoderConfig(
            model_name=model_cfg.get('vision', {}).get('model_name', 'dinov2_vitb14'),
            image_size=model_cfg.get('vision', {}).get('image_size', 224),
            hidden_dim=model_cfg.get('vision', {}).get('hidden_dim', 768),
            freeze=model_cfg.get('vision', {}).get('freeze', True),
        )
        
        language_cfg = LanguageEncoderConfig(
            model_name=model_cfg.get('language', {}).get('model_name', 'openai/clip-vit-large-patch14'),
            hidden_dim=model_cfg.get('language', {}).get('hidden_dim', 768),
            max_length=model_cfg.get('language', {}).get('max_length', 77),
            freeze=model_cfg.get('language', {}).get('freeze', True),
        )
        
        fusion_cfg = FusionConfig(
            hidden_dim=model_cfg.get('fusion', {}).get('hidden_dim', 768),
            num_heads=model_cfg.get('fusion', {}).get('num_heads', 12),
            num_layers=model_cfg.get('fusion', {}).get('num_layers', 2),
        )
        
        transformer_cfg = DiTConfig(
            hidden_dim=model_cfg.get('transformer', {}).get('hidden_dim', 768),
            num_heads=model_cfg.get('transformer', {}).get('num_heads', 12),
            num_layers=model_cfg.get('transformer', {}).get('num_layers', 12),
            use_flash_attn=False,  # Disable for evaluation stability
        )
        
        action_decoder_cfg = ActionDecoderConfig(
            hidden_dim=model_cfg.get('action_decoder', {}).get('hidden_dim', 768),
            action_horizon=model_cfg.get('action_decoder', {}).get('action_horizon', 16),
            position_dim=model_cfg.get('action_decoder', {}).get('position_dim', 3),
            rotation_dim=model_cfg.get('action_decoder', {}).get('rotation_dim', 4),
            gripper_dim=model_cfg.get('action_decoder', {}).get('gripper_dim', 1),
        )
        
        model_config = DriftingVLAConfig(
            vision=vision_cfg,
            language=language_cfg,
            fusion=fusion_cfg,
            transformer=transformer_cfg,
            action_decoder=action_decoder_cfg,
            hidden_dim=model_cfg.get('hidden_dim', 768),
            action_horizon=model_cfg.get('action_horizon', 16),
            noise_dim=model_cfg.get('noise_dim', 64),
        )
    else:
        model_config = config
    
    model = DriftingVLA(model_config)
    
    # Load weights (prefer EMA)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict']['ema_model'])
        logger.info("Loaded EMA weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw checkpoint weights")
    
    model = model.to(device)
    model.eval()
    
    return model


def run_simulation_eval(
    model,
    tasks: list[str],
    num_episodes: int,
    max_steps: int,
    record_video: bool,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """
    Run simulation evaluation.
    
    Args:
        model: Trained DriftingVLA model.
        tasks: List of task names to evaluate.
        num_episodes: Episodes per task.
        max_steps: Max steps per episode.
        record_video: Whether to record videos.
        output_dir: Directory for saving results.
        device: Torch device.
    
    Returns:
        Dict with evaluation metrics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    from drifting_vla.envs import RLBenchEnvironment
    from drifting_vla.envs.base_env import EnvConfig
    from drifting_vla.inference.policy import DriftingVLAPolicy, PolicyConfig
    from drifting_vla.inference.rollout import rollout_episode, RolloutConfig, save_rollout_video
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / 'videos'
    video_dir.mkdir(exist_ok=True)
    
    # Create policy
    policy_config = PolicyConfig(
        cfg_scale=2.0,
        action_horizon=16,
        use_ema=False,  # Already using EMA weights
        device=str(device),
        temporal_ensemble=True,
    )
    
    policy = DriftingVLAPolicy(model=model, config=policy_config)
    
    rollout_config = RolloutConfig(
        max_steps=max_steps,
        save_video=record_video,
        video_fps=10,
        verbose=True,
    )
    
    all_results = []
    per_task_success = {}
    
    for task in tasks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating task: {task}")
        logger.info(f"{'='*60}")
        
        # Create environment for this task
        env_config = EnvConfig(
            task=task,
            image_size=224,
            cameras=['front'],
            max_episode_length=max_steps,
            headless=True,
        )
        
        try:
            env = RLBenchEnvironment(env_config)
        except Exception as e:
            logger.error(f"Failed to create environment for {task}: {e}")
            per_task_success[task] = 0.0
            continue
        
        task_successes = []
        
        for ep in range(num_episodes):
            logger.info(f"\n--- Episode {ep + 1}/{num_episodes} ---")
            
            try:
                result = rollout_episode(
                    env=env,
                    policy=policy,
                    config=rollout_config,
                    task=task,
                    seed=ep,
                )
                
                all_results.append(result)
                task_successes.append(float(result.success))
                
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                logger.info(f"{status} - Length: {result.episode_length}")
                
                # Save video
                if record_video and result.frames:
                    video_path = video_dir / f"{task}_ep{ep:02d}_{'success' if result.success else 'fail'}.mp4"
                    save_rollout_video(result, str(video_path), fps=10)
                
            except Exception as e:
                logger.warning(f"Rollout failed: {e}")
                task_successes.append(0.0)
        
        env.close()
        
        task_success_rate = np.mean(task_successes) if task_successes else 0.0
        per_task_success[task] = task_success_rate
        logger.info(f"\nTask {task} success rate: {task_success_rate:.2%}")
    
    # Compute overall metrics
    overall_success = np.mean([r.success for r in all_results]) if all_results else 0.0
    avg_length = np.mean([r.episode_length for r in all_results]) if all_results else 0.0
    
    metrics = {
        'overall_success_rate': overall_success,
        'avg_episode_length': avg_length,
        'num_episodes': len(all_results),
        'per_task_success': per_task_success,
    }
    
    return metrics


def generate_report(metrics: dict, output_dir: Path):
    """Generate evaluation report."""
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DRIFTING-VLA SIMULATION EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Success Rate: {metrics['overall_success_rate']:.2%}\n")
        f.write(f"Avg Episode Length: {metrics['avg_episode_length']:.1f}\n")
        f.write(f"Total Episodes: {metrics['num_episodes']}\n\n")
        
        f.write("PER-TASK SUCCESS RATES\n")
        f.write("-"*40 + "\n")
        for task, rate in metrics['per_task_success'].items():
            f.write(f"  {task}: {rate:.2%}\n")
    
    logger.info(f"Report saved to {report_path}")


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    setup_display()
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # Get checkpoint path
    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path is None:
        # Look for latest checkpoint
        checkpoint_dir = Path(cfg.training.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
        if not checkpoints:
            logger.error("No checkpoint found. Specify with checkpoint=PATH")
            return
        checkpoint_path = str(max(checkpoints, key=lambda p: int(p.stem.split('_')[-1])))
        logger.info(f"Using latest checkpoint: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get evaluation settings
    sim_cfg = cfg.training.get('simulation_eval', {})
    tasks = sim_cfg.get('tasks', None)
    if tasks is None:
        tasks = list(cfg.data.tasks)
    
    num_episodes = sim_cfg.get('num_episodes', 5)
    max_steps = sim_cfg.get('max_steps', 200)
    record_video = sim_cfg.get('record_video', True)
    
    # Output directory
    output_dir = Path(cfg.training.checkpoint_dir) / 'sim_eval' / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info(f"\n{'='*60}")
    logger.info("SIMULATION EVALUATION")
    logger.info(f"{'='*60}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Episodes per task: {num_episodes}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Record video: {record_video}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60 + "\n")
    
    # Load model
    logger.info("Loading model...")
    model = build_model_from_checkpoint(checkpoint_path, device)
    
    # Run evaluation
    logger.info("\nStarting simulation evaluation...")
    metrics = run_simulation_eval(
        model=model,
        tasks=tasks,
        num_episodes=num_episodes,
        max_steps=max_steps,
        record_video=record_video,
        output_dir=output_dir,
        device=device,
    )
    
    # Generate report
    generate_report(metrics, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Overall Success Rate: {metrics['overall_success_rate']:.2%}")
    logger.info(f"Average Episode Length: {metrics['avg_episode_length']:.1f}")
    logger.info(f"Results saved to: {output_dir}")
    
    # Log to WandB if available
    if cfg.training.wandb.project:
        try:
            import wandb
            
            wandb.init(
                project=cfg.training.wandb.project,
                entity=cfg.training.wandb.get('entity'),
                name=f"sim_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['simulation_eval'],
            )
            
            # Log metrics
            wandb.log({
                'sim/success_rate': metrics['overall_success_rate'],
                'sim/avg_episode_length': metrics['avg_episode_length'],
                **{f'sim/success_{k}': v for k, v in metrics['per_task_success'].items()}
            })
            
            # Log videos
            video_dir = output_dir / 'videos'
            for video_file in video_dir.glob('*.mp4'):
                wandb.log({
                    f"sim/video/{video_file.stem}": wandb.Video(str(video_file))
                })
            
            wandb.finish()
            
        except Exception as e:
            logger.warning(f"Failed to log to WandB: {e}")


if __name__ == '__main__':
    main()

