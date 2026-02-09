#!/usr/bin/env python3
"""
Drifting-VLA Training Script
============================

Main entry point for training Drifting-VLA models.

Usage:
    python scripts/train.py
    python scripts/train.py model=dit_l2 training=large_scale
    
    # With overrides:
    python scripts/train.py training.learning_rate=1e-4
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.models import DriftingVLA, DriftingVLAConfig
from drifting_vla.training import DriftingVLATrainer
from drifting_vla.training.trainer import TrainerConfig, VisualizationConfig, SimulationEvalConfig
from drifting_vla.data import RLBenchDataset, VLATransforms, SampleQueue
from drifting_vla.logging import WandBLogger, LoggerConfig, generate_run_name
from drifting_vla.utils.distributed import setup_distributed, cleanup_distributed

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Configure logging based on rank."""
    rank = int(os.environ.get('RANK', 0))
    level = logging.INFO if rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format=f'[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def build_model(cfg: DictConfig) -> DriftingVLA:
    """Build model from config."""
    from drifting_vla.models.vision_encoder import VisionEncoderConfig
    from drifting_vla.models.language_encoder import LanguageEncoderConfig
    from drifting_vla.models.fusion import FusionConfig
    from drifting_vla.models.dit import DiTConfig
    from drifting_vla.models.action_decoder import ActionDecoderConfig
    
    # Build sub-configs from Hydra config
    vision_cfg = VisionEncoderConfig(
        model_name=cfg.model.vision.model_name,
        image_size=cfg.model.vision.image_size,
        hidden_dim=cfg.model.vision.hidden_dim,
        freeze=cfg.model.vision.freeze,
    )
    
    language_cfg = LanguageEncoderConfig(
        model_name=cfg.model.language.model_name,
        hidden_dim=cfg.model.language.hidden_dim,
        max_length=cfg.model.language.max_length,
        freeze=cfg.model.language.freeze,
    )
    
    fusion_cfg = FusionConfig(
        hidden_dim=cfg.model.fusion.hidden_dim,
        num_heads=cfg.model.fusion.num_heads,
        num_layers=cfg.model.fusion.num_layers,
        dropout=cfg.model.fusion.dropout,
    )
    
    transformer_cfg = DiTConfig(
        hidden_dim=cfg.model.transformer.hidden_dim,
        num_heads=cfg.model.transformer.num_heads,
        num_layers=cfg.model.transformer.num_layers,
        mlp_ratio=cfg.model.transformer.mlp_ratio,
        dropout=cfg.model.transformer.dropout,
        use_flash_attn=cfg.model.transformer.use_flash_attn,
        conditioning_dim=cfg.model.transformer.conditioning_dim,
    )
    
    action_decoder_cfg = ActionDecoderConfig(
        hidden_dim=cfg.model.action_decoder.hidden_dim,
        action_horizon=cfg.model.action_decoder.action_horizon,
        position_dim=cfg.model.action_decoder.position_dim,
        rotation_dim=cfg.model.action_decoder.rotation_dim,
        gripper_dim=cfg.model.action_decoder.gripper_dim,
        num_layers=cfg.model.action_decoder.num_layers,
        dropout=cfg.model.action_decoder.dropout,
    )
    
    model_cfg = DriftingVLAConfig(
        vision=vision_cfg,
        language=language_cfg,
        fusion=fusion_cfg,
        transformer=transformer_cfg,
        action_decoder=action_decoder_cfg,
        hidden_dim=cfg.model.hidden_dim,
        action_horizon=cfg.model.action_horizon,
        noise_dim=cfg.model.noise_dim,
        cfg_scale_range=tuple(cfg.model.cfg_scale_range),
        cfg_dropout=cfg.model.cfg_dropout,
    )
    
    model = DriftingVLA(model_cfg)
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")
    
    return model


def build_dataloaders(cfg: DictConfig, rank: int, world_size: int):
    """Build train and validation dataloaders."""
    # Dataset
    transforms = VLATransforms(training=True)
    
    train_dataset = RLBenchDataset(
        data_dir=cfg.data.data_dir,
        split='train',
        tasks=cfg.data.tasks,
        action_horizon=cfg.model.action_horizon,
        image_size=cfg.model.vision.image_size,
        cameras=cfg.data.cameras,
        transform=transforms,
    )
    
    # Compute action normalization statistics from training data
    # This normalizes [x,y,z,qx,qy,qz,qw,grip] per-dimension to zero-mean unit-var
    logger.info("Computing action normalization statistics...")
    action_mean, action_std = train_dataset.compute_action_statistics()
    logger.info(f"Action mean: {action_mean}")
    logger.info(f"Action std:  {action_std}")
    
    val_dataset = RLBenchDataset(
        data_dir=cfg.data.data_dir,
        split='val',
        tasks=cfg.data.tasks,
        action_horizon=cfg.model.action_horizon,
        image_size=cfg.model.vision.image_size,
        cameras=cfg.data.cameras,
        transform=VLATransforms(training=False),
    )
    # Share normalization stats with val dataset
    val_dataset.action_mean = action_mean
    val_dataset.action_std = action_std
    
    # Samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def build_sample_queue(cfg: DictConfig, device: torch.device) -> SampleQueue:
    """Build sample queue for drifting loss."""
    action_dim = (
        cfg.model.action_decoder.position_dim +
        cfg.model.action_decoder.rotation_dim +
        cfg.model.action_decoder.gripper_dim
    ) * cfg.model.action_horizon
    
    return SampleQueue(
        queue_size=cfg.data.sample_queue.queue_size,
        num_tasks=len(cfg.data.tasks),
        action_dim=action_dim,
        device=device,
    )


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup distributed training
    rank, world_size, device = setup_distributed()
    
    # Setup logging
    setup_logging(cfg)
    
    # Log config
    if rank == 0:
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    try:
        # Build model
        model = build_model(cfg)
        
        # Build dataloaders
        train_loader, val_loader = build_dataloaders(cfg, rank, world_size)
        
        # Build sample queue
        sample_queue = build_sample_queue(cfg, device)
        
        # Build WandB logger
        wandb_logger = None
        if rank == 0 and cfg.training.wandb.project:
            # Generate meaningful run name
            run_name = generate_run_name(
                model_name=cfg.model.get('name', 'dit_b2'),
                tasks=list(cfg.data.tasks),
                batch_size=cfg.training.batch_size,
                learning_rate=cfg.training.learning_rate,
                hidden_dim=cfg.model.hidden_dim,
                num_layers=cfg.model.transformer.num_layers,
                extra_tags=cfg.training.wandb.get('tags', []),
            )
            
            wandb_logger = WandBLogger(
                LoggerConfig(
                    project=cfg.training.wandb.project,
                    entity=cfg.training.wandb.get('entity'),
                    run_name=run_name,
                    tags=cfg.training.wandb.get('tags', []),
                    mode=cfg.training.wandb.get('mode', 'online'),
                ),
                model_config=OmegaConf.to_container(cfg.model),
                training_config=OmegaConf.to_container(cfg.training),
            )
            wandb_logger.init()
        
        # Build visualization config from wandb settings
        viz_cfg = cfg.training.wandb.get('visualizations', {})
        visualization_config = VisualizationConfig(
            enabled=viz_cfg.get('drifting_field', {}).get('enabled', True),
            # Category A — Training Convergence
            drifting_field_freq=viz_cfg.get('drifting_field', {}).get('frequency', 500),
            drift_trend_freq=viz_cfg.get('drift_trend', {}).get('frequency', 200),
            pred_scatter_freq=viz_cfg.get('prediction_scatter', {}).get('frequency', 500),
            error_radar_freq=viz_cfg.get('error_radar', {}).get('frequency', 500),
            temp_loss_freq=viz_cfg.get('temperature_loss', {}).get('frequency', 500),
            # Category B — Action Quality
            action_dist_freq=viz_cfg.get('action_distribution', {}).get('frequency', 500),
            error_heatmap_freq=viz_cfg.get('error_heatmap', {}).get('frequency', 500),
            trajectory_3d_freq=viz_cfg.get('trajectory_3d', {}).get('frequency', 1000),
            # Category C — Drifting-Specific
            sample_transport_freq=viz_cfg.get('sample_transport', {}).get('frequency', 500),
            mode_coverage_freq=viz_cfg.get('mode_coverage', {}).get('frequency', 2000),
            # General
            num_samples=viz_cfg.get('drifting_field', {}).get('num_samples', 100),
            theme=cfg.training.wandb.get('theme', 'light'),
        )
        
        # Build simulation evaluation config
        sim_eval_cfg = cfg.training.get('simulation_eval', {})
        simulation_eval_config = SimulationEvalConfig(
            enabled=sim_eval_cfg.get('enabled', False),
            num_episodes=sim_eval_cfg.get('num_episodes', 5),
            max_steps=sim_eval_cfg.get('max_steps', 200),
            tasks=sim_eval_cfg.get('tasks', None),
            record_video=sim_eval_cfg.get('record_video', True),
            video_fps=sim_eval_cfg.get('video_fps', 10),
            eval_freq=sim_eval_cfg.get('eval_freq', 5000),
            use_dummy_env=sim_eval_cfg.get('use_dummy_env', False),
        )
        
        # Build trainer
        trainer_config = TrainerConfig(
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            max_steps=cfg.training.max_steps,
            grad_clip=cfg.training.grad_clip,
            grad_accumulation_steps=cfg.training.grad_accumulation_steps,
            batch_size=cfg.training.batch_size,
            ema_decay=cfg.training.ema_decay,
            temperatures=cfg.training.drifting.temperatures,
            cfg_scale_range=tuple(cfg.model.cfg_scale_range),
            cfg_dropout=cfg.model.cfg_dropout,
            checkpoint_dir=cfg.training.checkpoint_dir,
            save_every_n_steps=cfg.training.save_every_n_steps,
            log_every_n_steps=cfg.training.log_every_n_steps,
            eval_every_n_steps=cfg.training.eval_every_n_steps,
            use_amp=cfg.training.use_amp,
            amp_dtype=cfg.training.amp_dtype,
            use_fsdp=cfg.training.use_fsdp,
            visualizations=visualization_config,
            simulation_eval=simulation_eval_config,
        )
        
        trainer = DriftingVLATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_config,
            sample_queue=sample_queue,
            wandb_logger=wandb_logger,
        )
        
        # Resume from checkpoint if exists
        checkpoint_dir = Path(cfg.training.checkpoint_dir)
        latest_ckpt = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
        if latest_ckpt:
            latest = max(latest_ckpt, key=lambda p: int(p.stem.split('_')[-1]))
            trainer.load_checkpoint(str(latest))
            logger.info(f"Resumed from {latest}")
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Finish logging
        if wandb_logger:
            wandb_logger.finish()
        
        logger.info("Training complete!")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()

