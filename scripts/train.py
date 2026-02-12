#!/usr/bin/env python3
"""
Drifting-VLA Training Script
================================

Train with VLM backbone + Drifting DiT.
Supports pre-computed VLM features and multi-dataset training.

Usage:
    # Single GPU toy training (testing)
    python scripts/train.py --config configs/training/baseline.yaml
    
    # Multi-GPU training  
    torchrun --nproc_per_node=8 scripts/train.py --config configs/training/baseline.yaml
    
    # Quick test
    python scripts/train.py --test
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting_vla.models.drifting_vla import DriftingVLA, DriftingVLAConfig
from drifting_vla.data.unified_dataset import (
    UnifiedDataset, collate_unified, create_weighted_sampler,
)
from drifting_vla.data.rlbench_dataset import RLBenchDataset
from drifting_vla.data.dexgraspnet_dataset import DexGraspNetDataset
from drifting_vla.data.lerobot_dataset import LeRobotDataset
from drifting_vla.data.action_mapping import (
    UNIFIED_ACTION_DIM, get_action_mask_tensor, DATASET_EMBODIMENT,
    LEROBOT_DATASETS, DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM,
)
from drifting_vla.data.sample_queue import SampleQueue, GlobalSampleQueue, NegativeSampleQueue
from drifting_vla.training.losses import DriftingLoss
from drifting_vla.training.ema import EMA

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    vlm_model_key: str = 'qwen3vl'
    model_size: str = 'base'    # small/base/large
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    action_horizon: int = 16
    noise_dim: int = 64
    use_flash_attn: bool = False  # Set to False for A40 debugging
    
    # Data
    data_root: str = './data'
    vlm_features_dir: str = './data/vlm_features'
    datasets: List[str] = field(default_factory=lambda: ['rlbench'])
    dataset_weights: Dict[str, float] = field(default_factory=lambda: {'rlbench': 1.0})
    image_size: int = 448
    cameras: List[str] = field(default_factory=lambda: ['front', 'wrist'])
    
    # Training  
    batch_size: int = 16
    learning_rate: float = 4e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 10000
    grad_clip: float = 2.0
    grad_accumulation_steps: int = 2
    
    # Loss
    loss_type: str = 'hybrid'       # 'pure_drift', 'mse', 'hybrid'
    drift_weight: float = 0.1       # Weight for drifting loss in hybrid mode
    temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    n_pos_samples: int = 64
    n_neg_samples: int = 64
    neg_queue_size: int = 2048
    pos_queue_size: int = 256
    
    # EMA
    ema_decay: float = 0.9999
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Distributed
    use_fsdp: bool = False
    
    # Logging
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2000
    checkpoint_dir: str = './checkpoints'
    
    # WandB
    wandb_project: str = 'drifting-vla'
    wandb_mode: str = 'disabled'    # 'online', 'offline', 'disabled'
    wandb_run_name: Optional[str] = None


class DriftingVLATrainer:
    """Trainer for Drifting-VLA."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        
        # Distributed setup
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.is_main = self.rank == 0
        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}' if torch.cuda.is_available() else 'cpu')
        
        # AMP
        if config.amp_dtype == 'bfloat16':
            self.amp_dtype = torch.bfloat16
        elif config.amp_dtype == 'float16':
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32
        
        # Build components
        self._build_model()
        self._build_datasets()
        self._build_optimizer()
        self._build_loss()
        self._build_queues()
        self._init_wandb()
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _build_model(self):
        """Build DriftingVLA model."""
        cfg = self.config
        
        model_config = DriftingVLAConfig(
            vlm_model_key=cfg.vlm_model_key,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            action_horizon=cfg.action_horizon,
            noise_dim=cfg.noise_dim,
            use_flash_attn=cfg.use_flash_attn,
        )
        
        self.model = DriftingVLA(model_config).to(self.device)
        
        if self.distributed and not cfg.use_fsdp:
            self.model = DDP(self.model, device_ids=[self.rank % torch.cuda.device_count()])
        
        # EMA
        unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
        self.ema = EMA(unwrapped, decay=cfg.ema_decay)
        
        if self.is_main:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model: {total:,} params ({trainable:,} trainable)")
    
    def _build_datasets(self):
        """Build unified dataset from configured sources."""
        cfg = self.config
        datasets = {}
        
        for ds_name in cfg.datasets:
            ds_dir = Path(cfg.data_root) / ds_name
            
            try:
                if ds_name == 'rlbench':
                    if not ds_dir.exists():
                        logger.warning(f"RLBench data not found at {ds_dir}, skipping")
                        continue
                    ds = RLBenchDataset(
                        data_dir=str(ds_dir),
                        split='train',
                        image_size=cfg.image_size,
                        cameras=cfg.cameras,
                        action_horizon=cfg.action_horizon,
                    )
                elif ds_name == 'dexgraspnet':
                    if not ds_dir.exists():
                        logger.warning(f"DexGraspNet data not found at {ds_dir}, skipping")
                        continue
                    ds = DexGraspNetDataset(
                        data_dir=str(ds_dir),
                        image_size=cfg.image_size,
                        action_horizon=cfg.action_horizon,
                    )
                elif ds_name in LEROBOT_DATASETS:
                    # LeRobot format datasets (Bridge, ALOHA, DROID, Fractal, BC-Z, etc.)
                    hf_repo = DATASET_HF_REPOS.get(ds_name)
                    if hf_repo is None:
                        logger.warning(f"No HF repo for {ds_name}, skipping")
                        continue
                    
                    ds = LeRobotDataset(
                        repo_id=hf_repo,
                        image_size=cfg.image_size,
                        action_horizon=cfg.action_horizon,
                        max_samples=getattr(cfg, 'max_samples_per_dataset', None) or (50 if getattr(cfg, '_test_mode', False) else None),
                    )
                else:
                    logger.warning(f"Unknown dataset: {ds_name}, skipping")
                    continue
                
                if len(ds) > 0:
                    datasets[ds_name] = ds
                    logger.info(f"  Loaded {ds_name}: {len(ds)} samples")
                else:
                    logger.warning(f"  {ds_name} has 0 samples, skipping")
            except Exception as e:
                logger.error(f"  Failed to load {ds_name}: {e}")
                continue
        
        if not datasets:
            raise RuntimeError("No datasets loaded! Run 'python scripts/download_datasets.py --test' first.")
        
        # VLM features directory
        vlm_dir = cfg.vlm_features_dir if Path(cfg.vlm_features_dir).exists() else None
        
        self.unified_dataset = UnifiedDataset(
            datasets=datasets,
            vlm_features_dir=vlm_dir,
            image_size=cfg.image_size,
            action_horizon=cfg.action_horizon,
            normalize_actions=True,
        )
        
        # DataLoader
        sampler = create_weighted_sampler(self.unified_dataset, cfg.dataset_weights)
        
        self.dataloader = DataLoader(
            self.unified_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=min(4, cfg.batch_size),
            collate_fn=collate_unified,
            pin_memory=True,
            drop_last=True,
        )
        
        self.data_iter = iter(self.dataloader)
        
        if self.is_main:
            logger.info(f"Dataset: {len(self.unified_dataset)} samples, batch_size={cfg.batch_size}")
    
    def _build_optimizer(self):
        """Build optimizer with differential LR."""
        cfg = self.config
        
        param_groups = [
            {'params': self.model.parameters(), 'lr': cfg.learning_rate},
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Cosine LR schedule with warmup
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(cfg.warmup_steps, 1)
            progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _build_loss(self):
        """Build drifting loss function."""
        cfg = self.config
        
        self.loss_fn = DriftingLoss(
            temperatures=cfg.temperatures,
            normalize_features=True,
            normalize_drift=True,
        )
    
    def _build_queues(self):
        """Build sample queues for drifting loss."""
        cfg = self.config
        action_flat_dim = cfg.action_horizon * UNIFIED_ACTION_DIM
        
        self.pos_queue = GlobalSampleQueue(
            queue_size=cfg.pos_queue_size,
            action_dim=action_flat_dim,
            device=torch.device('cpu'),
        )
        
        self.neg_queue = NegativeSampleQueue(
            queue_size=cfg.neg_queue_size,
            action_dim=action_flat_dim,
            device=torch.device('cpu'),
        )
    
    def _init_wandb(self):
        """Initialize WandB logging."""
        self.wandb_run = None
        if self.is_main and self.config.wandb_mode != 'disabled':
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or f"{self.config.model_size}_{time.strftime('%m%d_%H%M')}",
                    config=vars(self.config),
                    mode=self.config.wandb_mode,
                )
                logger.info(f"WandB initialized: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")
    
    def _get_batch(self) -> dict:
        """Get next training batch (with cycling)."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
    
    def _unwrapped_model(self):
        """Get unwrapped model (handles DDP)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    def training_step(self, batch: dict) -> dict:
        """Execute one training step."""
        cfg = self.config
        
        # Move to device
        images = batch['images'].to(self.device)       # [B, V_total, 3, H, W]
        actions_gt = batch['actions'].to(self.device)   # [B, T, 128]
        action_mask = batch['action_mask'].to(self.device)  # [B, 128]
        embodiment_id = batch['embodiment_id'].to(self.device)

        # Proprioception (robot state, 128-dim unified)
        proprio = batch.get('proprio', None)
        if proprio is not None:
            proprio = proprio.to(self.device)

        # Camera/time metadata for positional embeddings
        num_views = batch.get('num_views', 1)
        num_frames = batch.get('num_frames', 1)

        B, T, D = actions_gt.shape

        # VLM encoding: online by default (Pi0-style)
        vlm_features = batch.get('vlm_features', None)
        vlm_pooled = batch.get('vlm_pooled', None)
        vlm_attn_mask = batch.get('vlm_attn_mask', None)
        use_live_vlm = vlm_features is None

        if not use_live_vlm:
            vlm_features = vlm_features.to(self.device)
            vlm_pooled = vlm_pooled.to(self.device)
            if vlm_attn_mask is not None:
                vlm_attn_mask = vlm_attn_mask.to(self.device)
        else:
            model = self._unwrapped_model()
            if not model.vlm_backbone._loaded:
                model.vlm_backbone.load_vlm(self.device)
            vlm_attn_mask = None

        # Sample noise
        noise = torch.randn(B, T, cfg.noise_dim, device=self.device)

        # Sample CFG scale
        cfg_scale = self._sample_cfg_scale(B)

        # Forward pass (with proprio + camera/time metadata)
        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=cfg.use_amp):
            if use_live_vlm:
                language = batch.get('language', [''] * B)
                actions_pred = self.model(
                    images=images,
                    language=language,
                    noise=noise,
                    embodiment_id=embodiment_id,
                    cfg_scale=cfg_scale,
                    proprio=proprio,
                    num_views=num_views,
                    num_frames=num_frames,
                )
            else:
                actions_pred = self.model(
                    vlm_features=vlm_features,
                    vlm_pooled=vlm_pooled,
                    noise=noise,
                    embodiment_id=embodiment_id,
                    cfg_scale=cfg_scale,
                    proprio=proprio,
                    num_views=num_views,
                    num_frames=num_frames,
                    vlm_attn_mask=vlm_attn_mask,
                )  # [B, T, 128]
            
            # Apply action mask to predictions
            mask = action_mask.unsqueeze(1)  # [B, 1, 128]
            actions_pred_masked = actions_pred * mask
            actions_gt_masked = actions_gt * mask
            
            # --- Compute Loss ---
            metrics = {}
            
            if cfg.loss_type in ('mse', 'hybrid'):
                # Masked MSE loss (only on active dims)
                n_active = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1, 1]
                mse_loss = ((actions_pred_masked - actions_gt_masked) ** 2).sum(dim=-1) / n_active.squeeze(-1)
                mse_loss = mse_loss.mean()
                metrics['mse_loss'] = mse_loss.item()
            
            if cfg.loss_type in ('pure_drift', 'hybrid'):
                # Flatten for drifting loss
                pred_flat = actions_pred_masked.reshape(B, -1)
                gt_flat = actions_gt_masked.reshape(B, -1)
                
                # Get positive/negative samples
                try:
                    pos_samples = self.pos_queue.sample(cfg.n_pos_samples).to(self.device)
                except Exception:
                    pos_samples = gt_flat.detach()
                
                if self.neg_queue.count >= cfg.n_neg_samples:
                    neg_samples = self.neg_queue.sample(cfg.n_neg_samples).to(self.device)
                else:
                    neg_samples = gt_flat.detach()
                
                # Compute drifting loss
                loss_output = self.loss_fn(pred_flat, pos_samples, neg_samples)
                drift_loss = loss_output.loss
                metrics['drift_loss'] = drift_loss.item()
                metrics['drift_norm'] = loss_output.drift_norm.item()
                
                # Update queues
                self.neg_queue.push(pred_flat.detach())
                self.pos_queue.add(actions=gt_flat.detach())
            
            # Combine losses
            if cfg.loss_type == 'mse':
                loss = mse_loss
            elif cfg.loss_type == 'pure_drift':
                loss = drift_loss
            else:  # hybrid
                loss = mse_loss + cfg.drift_weight * drift_loss
        
        # Gradient accumulation
        loss_scaled = loss / cfg.grad_accumulation_steps
        loss_scaled.backward()
        
        # Optimizer step
        if (self.global_step + 1) % cfg.grad_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.ema.update()
            metrics['grad_norm'] = grad_norm.item()
        
        metrics['loss'] = loss.item()
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def _sample_cfg_scale(self, batch_size: int) -> torch.Tensor:
        """Sample CFG scale from p(α) ∝ α^{-3}."""
        u = torch.rand(batch_size, device=self.device)
        min_s, max_s = 1.0, 4.0
        k = 3
        alpha = ((max_s**(1-k) - min_s**(1-k)) * u + min_s**(1-k)) ** (1/(1-k))
        return alpha
    
    def train(self):
        """Main training loop."""
        cfg = self.config
        
        if self.is_main:
            logger.info(f"{'='*60}")
            logger.info(f"Drifting-VLA Training")
            logger.info(f"  Model: {cfg.model_size} ({cfg.hidden_dim}d, {cfg.num_layers}L)")
            logger.info(f"  VLM: {cfg.vlm_model_key}")
            logger.info(f"  Datasets: {cfg.datasets}")
            logger.info(f"  Loss: {cfg.loss_type}")
            logger.info(f"  Batch: {cfg.batch_size} × {cfg.grad_accumulation_steps} = {cfg.batch_size * cfg.grad_accumulation_steps}")
            logger.info(f"  Steps: {cfg.max_steps}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"{'='*60}")
        
        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        running_metrics = {}
        
        for step in range(cfg.max_steps):
            self.global_step = step
            
            batch = self._get_batch()
            metrics = self.training_step(batch)
            
            # Update running averages
            running_loss += metrics['loss']
            for k, v in metrics.items():
                if k not in running_metrics:
                    running_metrics[k] = 0.0
                running_metrics[k] += v
            
            # Logging
            if self.is_main and (step + 1) % cfg.log_every == 0:
                avg_loss = running_loss / cfg.log_every
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / elapsed
                
                log_str = (
                    f"Step {step+1}/{cfg.max_steps} | "
                    f"loss={avg_loss:.4f}"
                )
                
                for k in ['mse_loss', 'drift_loss', 'grad_norm']:
                    if k in running_metrics:
                        avg = running_metrics[k] / cfg.log_every
                        log_str += f" | {k}={avg:.4f}"
                
                log_str += f" | lr={metrics.get('lr', 0):.2e} | {steps_per_sec:.1f} steps/s"
                logger.info(log_str)
                
                # WandB
                if self.wandb_run:
                    import wandb
                    log_dict = {f'train/{k}': v / cfg.log_every for k, v in running_metrics.items()}
                    log_dict['train/steps_per_sec'] = steps_per_sec
                    wandb.log(log_dict, step=step+1)
                
                running_loss = 0.0
                running_metrics = {}
            
            # Save checkpoint
            if self.is_main and (step + 1) % cfg.save_every == 0:
                self._save_checkpoint(step + 1)
        
        # Final save
        if self.is_main:
            self._save_checkpoint(cfg.max_steps, final=True)
            elapsed = time.time() - t0
            logger.info(f"Training complete: {cfg.max_steps} steps in {elapsed/3600:.1f} hours")
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped = self._unwrapped_model()
        
        ckpt = {
            'step': step,
            'model_state_dict': unwrapped.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'config': vars(self.config),
        }
        
        name = 'final.pt' if final else f'step_{step}.pt'
        path = ckpt_dir / name
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Drifting-VLA Training')
    
    parser.add_argument('--test', action='store_true', help='Run quick test (30 steps, no wandb)')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    
    # Model
    parser.add_argument('--model-size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--vlm', type=str, default='qwen3vl', choices=['qwen3vl', 'paligemma2'])
    parser.add_argument('--use-flash-attn', action='store_true', default=False)
    parser.add_argument('--skip-vlm', action='store_true', default=False,
                        help='Skip VLM forward (pipeline test only, no VLM download needed)')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--datasets', nargs='+', default=['rlbench'])
    parser.add_argument('--vlm-features-dir', type=str, default='./data/vlm_features')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--loss-type', type=str, default='hybrid', choices=['mse', 'pure_drift', 'hybrid'])
    parser.add_argument('--grad-accumulation', type=int, default=2)
    
    # Logging
    parser.add_argument('--wandb-mode', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb-project', type=str, default='drifting-vla')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--save-every', type=int, default=2000)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per dataset (e.g., 10000 for quick experiments)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed if available
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        if rank != 0:
            logging.getLogger().setLevel(logging.WARNING)
    
    # Build config
    config = TrainConfig()
    
    # Override with args
    config.vlm_model_key = args.vlm
    config.model_size = args.model_size
    config.data_root = args.data_root
    config.datasets = args.datasets
    config.vlm_features_dir = args.vlm_features_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_steps = args.max_steps
    config.loss_type = args.loss_type
    config.grad_accumulation_steps = args.grad_accumulation
    config.wandb_mode = args.wandb_mode
    config.wandb_project = args.wandb_project
    config.log_every = args.log_every
    config.save_every = args.save_every
    config.use_flash_attn = args.use_flash_attn
    config.skip_vlm = args.skip_vlm
    config.max_samples_per_dataset = args.max_samples
    
    # Model size configs
    size_cfgs = {
        'small': (512, 8, 8),
        'base': (768, 12, 12),
        'large': (1024, 24, 16),
    }
    config.hidden_dim, config.num_layers, config.num_heads = size_cfgs[config.model_size]
    
    # Quick test mode
    if args.test:
        config.max_steps = 30
        config.log_every = 10
        config.save_every = 100
        config.wandb_mode = 'disabled'
        config.batch_size = 4
        config.grad_accumulation_steps = 1
        config._test_mode = True
        logger.info("TEST MODE: 30 steps, batch_size=4, no wandb")
    
    # Train
    trainer = DriftingVLATrainer(config)
    trainer.train()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
