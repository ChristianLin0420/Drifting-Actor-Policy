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
from drifting_vla.data.episode_dataset import EpisodeHDF5Dataset
from drifting_vla.data.action_mapping import (
    UNIFIED_ACTION_DIM, get_action_mask_tensor, DATASET_EMBODIMENT,
    LEROBOT_DATASETS, DATASET_HF_REPOS, DATASET_NATIVE_ACTION_DIM,
)
from drifting_vla.data.sample_queue import SampleQueue, GlobalSampleQueue, NegativeSampleQueue
from drifting_vla.training.losses import DriftingLoss
from drifting_vla.training.ema import EMA
from drifting_vla.training.visualizations import VizLogger

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
    dataset_weights: Optional[Dict[str, float]] = None  # None = temperature-balanced sampling
    image_size: int = 448
    cameras: List[str] = field(default_factory=lambda: ['front', 'wrist'])
    # (data_fraction removed — use prepare_data.py --max-episodes instead)
    
    # Training  
    batch_size: int = 16              # Per-GPU; effective = batch × n_gpu × grad_accum
    learning_rate: float = 1e-4       # RDT-1B uses 1e-4 for multi-dataset
    weight_decay: float = 0.05
    warmup_steps: int = 500
    max_steps: int = 10000
    grad_clip: float = 1.0            # Tighter clip for stability
    grad_accumulation_steps: int = 2
    
    # Loss
    loss_type: str = 'hybrid'       # 'pure_drift', 'mse', 'hybrid'
    drift_weight: float = 0.001     # Weight for raw drifting loss in hybrid mode
    temperatures: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.2])
    n_pos_samples: int = 64
    n_neg_samples: int = 64
    neg_queue_size: int = 8192      # Single large global queue
    pos_queue_size: int = 4096      # Single large global queue
    
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
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0)) if self.distributed else 0
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
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
        
        # VLM training mode
        vlm_mode = getattr(cfg, 'vlm_mode', 'frozen')
        vlm_freeze = vlm_mode in ('frozen', 'lora')
        vlm_use_lora = (vlm_mode == 'lora')
        lora_r = getattr(cfg, 'lora_r', 16)
        
        model_config = DriftingVLAConfig(
            vlm_model_key=cfg.vlm_model_key,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            action_horizon=cfg.action_horizon,
            noise_dim=cfg.noise_dim,
            use_flash_attn=cfg.use_flash_attn,
            vlm_freeze=vlm_freeze,
            vlm_use_lora=vlm_use_lora,
        )
        
        if self.is_main:
            logger.info(f"  VLM mode: {vlm_mode} (freeze={vlm_freeze}, lora={vlm_use_lora}, r={lora_r})")
        
        self.model = DriftingVLA(model_config).to(self.device)
        
        # Load VLM eagerly (before DDP + EMA) so LoRA params are in place
        # Otherwise EMA snapshots pre-LoRA shapes → size mismatch on update
        if vlm_use_lora or vlm_mode != 'frozen':
            self.model.vlm_backbone.load_vlm(self.device)
        
        if self.distributed and not cfg.use_fsdp:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                             find_unused_parameters=True)
        
        # EMA (must be after VLM+LoRA loading so param shapes match)
        unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
        self.ema = EMA(unwrapped, decay=cfg.ema_decay)
        
        if self.is_main:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model: {total:,} params ({trainable:,} trainable)")
    
    def _load_vlm_processor(self):
        """Load VLM processor for tokenization in DataLoader workers.
        
        This is REQUIRED for training — without it, batches will have no
        VLM inputs and training will crash. Raises on failure.
        """
        cfg = self.config
        from drifting_vla.models.vlm_backbone import VLM_SPECS
        hf_name = VLM_SPECS[cfg.vlm_model_key]['hf_name']
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(hf_name)
        if self.is_main:
            logger.info(f"Loaded VLM processor: {hf_name}")
        return processor
    
    def _build_datasets(self):
        """Build unified dataset from Episode HDF5 files.
        
        Loads from data/episodes/{dataset_name}/ directories.
        Each dataset has pre-mapped 128-dim actions in HDF5 format.
        VLM tokenization is done in DataLoader workers (parallel), not in model forward.
        """
        cfg = self.config
        episodes_root = Path(getattr(cfg, 'episodes_root', cfg.data_root + '/episodes'))
        
        datasets = {}
        
        # Load VLM processor once for all datasets (tokenization in DataLoader)
        vlm_processor = self._load_vlm_processor()
        
        for ds_name in cfg.datasets:
            ep_dir = episodes_root / ds_name
            
            if not ep_dir.exists() or not (ep_dir / 'metadata.json').exists():
                if self.is_main:
                    logger.warning(f"Episode dir not found: {ep_dir}, skipping {ds_name}")
                continue
            
            try:
                ds = EpisodeHDF5Dataset(
                    episode_dir=str(ep_dir),
                    action_horizon=cfg.action_horizon,
                    num_history_frames=3,
                    image_size=cfg.image_size,
                    vlm_processor=vlm_processor,
                    image_aug=getattr(cfg, 'image_aug', False),
                    cond_mask_prob=getattr(cfg, 'cond_mask_prob', 0.1),
                )
                
                if len(ds) > 0:
                    datasets[ds_name] = ds
                    if self.is_main:
                        logger.info(f"  Loaded {ds_name}: {len(ds)} samples")
                else:
                    if self.is_main:
                        logger.warning(f"  {ds_name} has 0 samples, skipping")
            except Exception as e:
                if self.is_main:
                    logger.error(f"  Failed to load {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not datasets:
            raise RuntimeError(
                "No datasets loaded! Run:\n"
                "  python scripts/prepare_data.py --dataset <name>"
            )
        
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
        """Build optimizer with differential LR for VLM/projector/DiT."""
        cfg = self.config
        unwrapped = self._unwrapped_model()
        
        vlm_lora_params = []
        projector_params = []
        dit_params = []
        
        vlm_lr_scale = getattr(cfg, 'vlm_lr_scale', 0.1)
        
        for name, param in unwrapped.named_parameters():
            if not param.requires_grad:
                continue
            if 'vlm_backbone.vlm' in name:
                vlm_lora_params.append(param)
            elif 'vlm_backbone.proj' in name:
                projector_params.append(param)
            else:
                dit_params.append(param)
        
        param_groups = []
        if vlm_lora_params:
            param_groups.append({
                'params': vlm_lora_params,
                'lr': cfg.learning_rate * vlm_lr_scale,
                'weight_decay': 0.0,
                'name': 'vlm_lora',
            })
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': cfg.learning_rate * 0.5,
                'weight_decay': cfg.weight_decay,
                'name': 'projector',
            })
        if dit_params:
            param_groups.append({
                'params': dit_params,
                'lr': cfg.learning_rate,
                'weight_decay': cfg.weight_decay,
                'name': 'dit',
            })
        
        if self.is_main:
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg['params'])
                logger.info(f"  Optimizer group '{pg.get('name', '?')}': "
                           f"{n_params:,} params, lr={pg['lr']:.1e}")
        
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
        """Build single large global sample queues for drifting loss.
        
        The action mask zeros out inactive dims, so cross-embodiment samples
        naturally have large L2 distance — the kernel exp(-||x-y||/τ) ≈ 0
        between different embodiments. No explicit separation needed.
        A single large queue fills faster and provides more diverse negatives.
        """
        cfg = self.config
        # We store only active dims in the queue (compact representation)
        # The max active dims across all embodiments is ~23*T for behavior1k
        # But we use a fixed large dim and store the full masked-flat vectors
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
        """Initialize WandB logging with proper metric definitions."""
        self.wandb_run = None
        self.viz_logger = VizLogger(self.config)
        if self.is_main and self.config.wandb_mode != 'disabled':
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or f"{self.config.model_size}_{time.strftime('%m%d_%H%M')}",
                    config=vars(self.config),
                    mode=self.config.wandb_mode,
                )
                # Define metric axes so all train/* metrics use the global step
                wandb.define_metric("train/*", step_metric="global_step")
                wandb.define_metric("viz/*", step_metric="global_step")
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

        # Check for VLM ViT encoder inputs (from DataLoader workers)
        vlm_input_ids = batch.get('vlm_input_ids', None)
        vlm_features = batch.get('vlm_features', None)
        use_encoder = vlm_input_ids is not None
        use_precomputed = vlm_features is not None and not use_encoder

        # Prepare forward kwargs
        fwd_kwargs = {}
        if use_encoder:
            # Path 1: Vision-encoder-only — ViT + text embed, skip LLM
            fwd_kwargs['vlm_input_ids'] = vlm_input_ids.to(self.device)
            fwd_kwargs['vlm_attention_mask'] = batch['vlm_attention_mask'].to(self.device)
            pv = batch.get('vlm_pixel_values', None)
            if pv is not None:
                fwd_kwargs['vlm_pixel_values'] = pv.to(self.device)
            thw = batch.get('vlm_image_grid_thw', None)
            if thw is not None:
                fwd_kwargs['vlm_image_grid_thw'] = thw.to(self.device)

            # Ensure VLM visual encoder is loaded
            model = self._unwrapped_model()
            if not model.vlm_backbone._loaded:
                model.vlm_backbone.load_vlm(self.device)
        elif use_precomputed:
            # Path 2: Pre-computed features — offline VLM
            fwd_kwargs['vlm_features'] = vlm_features.to(self.device)
            fwd_kwargs['vlm_pooled'] = batch['vlm_pooled'].to(self.device)
        else:
            raise RuntimeError(
                "No VLM inputs found in batch. Either set vlm_processor on dataset "
                "or provide pre-computed VLM features."
            )

        # Sample noise
        noise = torch.randn(B, T, cfg.noise_dim, device=self.device)

        # Sample CFG scale
        cfg_scale = self._sample_cfg_scale(B)

        # Forward pass
        with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=cfg.use_amp):
            actions_pred = self.model(
                **fwd_kwargs,
                noise=noise,
                embodiment_id=embodiment_id,
                cfg_scale=cfg_scale,
                proprio=proprio,
                num_views=num_views,
                num_frames=num_frames,
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
                # Single global queue drifting loss
                # The action mask zeros inactive dims, so cross-embodiment
                # samples naturally separate via L2 distance in the kernel.
                pred_flat = actions_pred_masked.reshape(B, -1)  # [B, T*128]
                gt_flat = actions_gt_masked.reshape(B, -1)
                
                # Push current batch into global queues
                self.neg_queue.push(pred_flat.detach())
                self.pos_queue.add(actions=gt_flat.detach())
                
                # Compute drift loss when queues have enough samples
                drift_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                loss_output = None
                
                if (self.pos_queue.total_samples() >= cfg.n_pos_samples and
                        self.neg_queue.count >= cfg.n_neg_samples):
                    
                    pos_samples = self.pos_queue.sample(cfg.n_pos_samples).to(self.device)
                    neg_samples = self.neg_queue.sample(cfg.n_neg_samples).to(self.device)
                    
                    try:
                        loss_out = self.loss_fn(pred_flat, pos_samples, neg_samples)
                        if torch.isfinite(loss_out.loss):
                            # Use raw_drift_norm as loss (naturally → 0 at convergence)
                            # instead of normalized loss (always ≈ 1.0)
                            drift_loss = loss_out.raw_drift_norm if loss_out.raw_drift_norm is not None else loss_out.loss
                            metrics['drift_loss'] = drift_loss.item()
                            metrics['drift_norm'] = loss_out.drift_norm.item()
                            if loss_out.raw_drift_norm is not None:
                                metrics['raw_drift_norm'] = loss_out.raw_drift_norm.item()
                            if loss_out.lambda_V is not None:
                                metrics['lambda_V'] = loss_out.lambda_V.item()
                            loss_output = loss_out
                    except Exception:
                        pass
            
            # Combine losses
            if cfg.loss_type == 'mse':
                loss = mse_loss
            elif cfg.loss_type == 'pure_drift':
                loss = drift_loss
            else:  # hybrid
                loss = mse_loss + cfg.drift_weight * drift_loss
            
            # Store for visualization (detached, on CPU to save GPU memory)
            if self.is_main:
                self._last_pred = actions_pred.detach()
                self._last_gt = actions_gt.detach()
                self._last_mask = action_mask.detach()
                if cfg.loss_type in ('pure_drift', 'hybrid') and loss_output is not None:
                    self._last_drift_field = getattr(loss_output, 'drift_field', None)
                    self._last_lambda_V = loss_output.lambda_V.item() if getattr(loss_output, 'lambda_V', None) is not None else None
                    self._last_per_temp = getattr(loss_output, 'per_temp_losses', None)
        
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
            logger.info(f"  Loss: {cfg.loss_type} (drift_weight={cfg.drift_weight})")
            n_gpus = dist.get_world_size() if dist.is_initialized() else 1
            per_gpu = cfg.batch_size
            accum = cfg.grad_accumulation_steps
            global_batch = per_gpu * accum * n_gpus
            logger.info(f"  Batch: {per_gpu}/gpu × {accum} accum × {n_gpus} GPUs = {global_batch} global")
            logger.info(f"  Steps: {cfg.max_steps}")
            logger.info(f"  Device: {self.device} ({n_gpus} GPUs)")
            logger.info(f"{'='*60}")
        
        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        running_metrics = {}
        
        # Track latest batch outputs for visualization
        self._last_pred = None
        self._last_gt = None
        self._last_mask = None
        self._last_drift_field = None
        self._last_lambda_V = None
        self._last_per_temp = None
        
        for step in range(cfg.max_steps):
            self.global_step = step
            
            batch = self._get_batch()
            metrics = self.training_step(batch)
            
            # Update running averages
            running_loss += metrics['loss']
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
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
                
                for k in ['mse_loss', 'drift_loss', 'drift_norm', 'grad_norm']:
                    if k in running_metrics:
                        avg = running_metrics[k] / cfg.log_every
                        log_str += f" | {k}={avg:.4f}"
                
                log_str += f" | lr={metrics.get('lr', 0):.2e} | {steps_per_sec:.1f} steps/s"
                logger.info(log_str)
                
                # WandB scalar metrics
                if self.wandb_run:
                    import wandb
                    log_dict = {f'train/{k}': v / cfg.log_every for k, v in running_metrics.items()}
                    log_dict['train/steps_per_sec'] = steps_per_sec
                    log_dict['train/step'] = step + 1
                    log_dict['global_step'] = step + 1
                    wandb.log(log_dict, step=step+1)
                
                # Track D2 training curve history
                self.viz_logger._train_loss_steps.append(step + 1)
                self.viz_logger._train_loss_vals.append(avg_loss)
                self.viz_logger._train_mse_vals.append(
                    running_metrics.get('mse_loss', avg_loss * cfg.log_every) / cfg.log_every
                    if 'mse_loss' in running_metrics else avg_loss)
                self.viz_logger._train_drift_vals.append(
                    running_metrics.get('drift_norm', 0) / cfg.log_every
                    if 'drift_norm' in running_metrics else 0)

                # D2: Training Curve Dashboard (every 500 steps)
                if self.wandb_run and (step + 1) % 500 == 0:
                    try:
                        from drifting_vla.training.visualizations import create_training_curve_dashboard
                        d2_img = create_training_curve_dashboard(
                            self.viz_logger._train_loss_vals,
                            self.viz_logger._train_mse_vals,
                            self.viz_logger._train_drift_vals,
                            self.viz_logger._train_loss_steps,
                            step + 1,
                        )
                        if d2_img is not None:
                            wandb.log({'viz/D2_training_curves': d2_img}, step=step+1)
                    except Exception as e:
                        logger.debug(f"D2 viz error: {e}")

                running_loss = 0.0
                running_metrics = {}

            # Track D4 data balance (count dataset appearances in batches)
            if self.is_main:
                ds_names_batch = batch.get('dataset_name', [])
                for ds_n in ds_names_batch:
                    self.viz_logger._dataset_batch_counts[ds_n] = \
                        self.viz_logger._dataset_batch_counts.get(ds_n, 0) + 1

            # WandB visualizations (at viz-specific frequencies)
            if self.is_main and self.wandb_run and self._last_pred is not None:
                self.viz_logger.log_training_viz(
                    step=step + 1,
                    actions_pred=self._last_pred,
                    actions_gt=self._last_gt,
                    action_mask=self._last_mask,
                    drift_field=self._last_drift_field,
                    lambda_V=self._last_lambda_V,
                    per_temp_losses=self._last_per_temp,
                    wandb_run=self.wandb_run,
                )
            
            # Evaluation
            if self.is_main and (step + 1) % cfg.eval_every == 0:
                eval_metrics = self._run_evaluation(step + 1)
                if self.wandb_run:
                    import wandb
                    wandb.log(eval_metrics, step=step+1)
            
            # Save checkpoint
            if self.is_main and (step + 1) % cfg.save_every == 0:
                self._save_checkpoint(step + 1)
        
        # Final eval + save
        if self.is_main:
            eval_metrics = self._run_evaluation(cfg.max_steps)
            if self.wandb_run:
                import wandb
                wandb.log(eval_metrics, step=cfg.max_steps)
            self._save_checkpoint(cfg.max_steps, final=True)
            elapsed = time.time() - t0
            logger.info(f"Training complete: {cfg.max_steps} steps in {elapsed/3600:.1f} hours")
    
    @torch.no_grad()
    def _run_evaluation(self, step: int) -> dict:
        """
        Tier 2 evaluation: compute per-dataset action prediction metrics.
        
        Runs the model on the full dataset (no gradient), computes:
          - Per-dataset MSE, MAE
          - Per-dataset positional error (EEF xyz dims)
          - Per-dataset quaternion error (where applicable)
          - Per-dataset coverage (pred range / GT range)
          - Overall aggregated metrics
        
        Returns dict of eval/* metrics for WandB logging.
        """
        cfg = self.config
        self.model.eval()
        
        logger.info(f"[Eval] Running evaluation at step {step}...")
        
        # Collect predictions per dataset
        from drifting_vla.data.action_mapping import (
            DATASET_EMBODIMENT, DATASET_NATIVE_ACTION_DIM, get_action_mask,
            extract_from_unified, EMBODIMENT_GRIPPER_EEF, EMBODIMENT_DEXHAND,
        )
        
        per_dataset = {}  # dataset_name → {pred: [], gt: [], mask: []}
        
        n_eval_batches = min(50, len(self.dataloader))  # Cap eval batches
        eval_iter = iter(self.dataloader)
        
        model = self._unwrapped_model()
        
        for batch_idx in range(n_eval_batches):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(self.dataloader)
                batch = next(eval_iter)
            
            actions_gt = batch['actions'].to(self.device)
            action_mask = batch['action_mask'].to(self.device)
            embodiment_id = batch['embodiment_id'].to(self.device)
            proprio = batch.get('proprio', None)
            if proprio is not None:
                proprio = proprio.to(self.device)

            B, T, D = actions_gt.shape
            noise = torch.randn(B, T, cfg.noise_dim, device=self.device)
            cfg_scale = torch.ones(B, device=self.device) * 2.0  # Fixed CFG for eval
            
            # Build VLM forward kwargs (same logic as training_step)
            fwd_kwargs = {}
            vlm_input_ids = batch.get('vlm_input_ids', None)
            vlm_features = batch.get('vlm_features', None)
            if vlm_input_ids is not None:
                fwd_kwargs['vlm_input_ids'] = vlm_input_ids.to(self.device)
                fwd_kwargs['vlm_attention_mask'] = batch['vlm_attention_mask'].to(self.device)
                pv = batch.get('vlm_pixel_values', None)
                if pv is not None:
                    fwd_kwargs['vlm_pixel_values'] = pv.to(self.device)
                thw = batch.get('vlm_image_grid_thw', None)
                if thw is not None:
                    fwd_kwargs['vlm_image_grid_thw'] = thw.to(self.device)
            elif vlm_features is not None:
                fwd_kwargs['vlm_features'] = vlm_features.to(self.device)
                fwd_kwargs['vlm_pooled'] = batch['vlm_pooled'].to(self.device)
            else:
                continue  # Skip batches without VLM inputs

            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=cfg.use_amp):
                actions_pred = model(
                    **fwd_kwargs,
                    noise=noise, embodiment_id=embodiment_id,
                    cfg_scale=cfg_scale, proprio=proprio,
                    num_views=batch.get('num_views', 1),
                    num_frames=batch.get('num_frames', 1),
                )
            
            # Mask predictions
            mask = action_mask.unsqueeze(1)
            pred_masked = (actions_pred * mask).cpu().numpy()
            gt_masked = (actions_gt * mask).cpu().numpy()
            mask_np = action_mask.cpu().numpy()
            
            # Group by dataset
            ds_names = batch.get('dataset_name', ['unknown'] * B)
            emb_ids = batch['embodiment_id'].cpu().numpy()
            
            for i in range(B):
                ds = ds_names[i]
                if ds not in per_dataset:
                    per_dataset[ds] = {'pred': [], 'gt': [], 'emb_id': emb_ids[i]}
                per_dataset[ds]['pred'].append(pred_masked[i])
                per_dataset[ds]['gt'].append(gt_masked[i])
        
        # Compute metrics per dataset
        eval_metrics = {}
        all_mse = []
        all_mae = []
        
        for ds_name, data in per_dataset.items():
            pred = np.stack(data['pred'])  # [N, T, 128]
            gt = np.stack(data['gt'])      # [N, T, 128]
            emb_id = data['emb_id']
            native_dim = DATASET_NATIVE_ACTION_DIM.get(ds_name, None)
            
            mask_info = get_action_mask(emb_id, native_dim=native_dim)
            active = mask_info.mask
            
            # --- MSE and MAE (on active dims only) ---
            diff = pred - gt  # [N, T, 128]
            active_diff = diff[:, :, active]  # [N, T, n_active]
            
            mse = (active_diff ** 2).mean()
            mae = np.abs(active_diff).mean()
            
            eval_metrics[f'eval/{ds_name}/mse'] = float(mse)
            eval_metrics[f'eval/{ds_name}/mae'] = float(mae)
            all_mse.append(mse)
            all_mae.append(mae)
            
            # --- Per-dim correlation ---
            n_active = int(active.sum())
            correlations = []
            for d in range(n_active):
                g = gt[:, 0, active][:, d]
                p = pred[:, 0, active][:, d]
                if np.std(g) > 1e-8 and np.std(p) > 1e-8:
                    correlations.append(float(np.corrcoef(g, p)[0, 1]))
            if correlations:
                eval_metrics[f'eval/{ds_name}/mean_correlation'] = float(np.mean(correlations))
            
            # --- Coverage (pred range / GT range) ---
            coverages = []
            for d in range(n_active):
                g = gt[:, 0, active][:, d]
                p = pred[:, 0, active][:, d]
                gt_range = g.max() - g.min() + 1e-8
                pred_range = p.max() - p.min()
                coverages.append(min(pred_range / gt_range * 100, 200))
            if coverages:
                eval_metrics[f'eval/{ds_name}/coverage_pct'] = float(np.mean(coverages))
            
            # --- Positional error (for EEF embodiments: first 3 dims = xyz) ---
            if emb_id in (EMBODIMENT_GRIPPER_EEF, EMBODIMENT_DEXHAND):
                try:
                    pred_native = np.stack([extract_from_unified(p[0], emb_id) for p in pred])
                    gt_native = np.stack([extract_from_unified(g[0], emb_id) for g in gt])
                    pos_error = np.linalg.norm(pred_native[:, :3] - gt_native[:, :3], axis=1)
                    eval_metrics[f'eval/{ds_name}/pos_error_m'] = float(pos_error.mean())
                    
                    # Quaternion error (dims 3:7)
                    if pred_native.shape[1] >= 7:
                        q_pred = pred_native[:, 3:7]
                        q_gt = gt_native[:, 3:7]
                        q_pred_norm = q_pred / (np.linalg.norm(q_pred, axis=1, keepdims=True) + 1e-8)
                        q_gt_norm = q_gt / (np.linalg.norm(q_gt, axis=1, keepdims=True) + 1e-8)
                        dot = np.abs(np.sum(q_pred_norm * q_gt_norm, axis=1)).clip(0, 1)
                        quat_error_rad = 2 * np.arccos(dot)
                        eval_metrics[f'eval/{ds_name}/quat_error_deg'] = float(np.degrees(quat_error_rad.mean()))
                except Exception:
                    pass
            
            logger.info(
                f"  [Eval] {ds_name}: MSE={mse:.4f}, MAE={mae:.4f}, "
                f"corr={eval_metrics.get(f'eval/{ds_name}/mean_correlation', 0):.3f}, "
                f"coverage={eval_metrics.get(f'eval/{ds_name}/coverage_pct', 0):.0f}%"
            )
        
        # ── Aggregate: per-embodiment grouping (reduces 60 dataset keys → 6 embodiment keys) ──
        from drifting_vla.data.action_mapping import EMBODIMENT_NAMES
        per_embodiment = {}  # emb_name → {mse: [], mae: [], corr: [], datasets: []}
        
        for ds_name, data in per_dataset.items():
            emb_id = data['emb_id']
            emb_name = EMBODIMENT_NAMES.get(emb_id, f'emb_{emb_id}')
            
            if emb_name not in per_embodiment:
                per_embodiment[emb_name] = {'mse': [], 'mae': [], 'corr': [], 'coverage': [], 'datasets': []}
            
            per_embodiment[emb_name]['mse'].append(eval_metrics.get(f'eval/{ds_name}/mse', 0))
            per_embodiment[emb_name]['mae'].append(eval_metrics.get(f'eval/{ds_name}/mae', 0))
            per_embodiment[emb_name]['corr'].append(eval_metrics.get(f'eval/{ds_name}/mean_correlation', 0))
            per_embodiment[emb_name]['coverage'].append(eval_metrics.get(f'eval/{ds_name}/coverage_pct', 0))
            per_embodiment[emb_name]['datasets'].append(ds_name)
        
        # Log per-embodiment aggregated metrics (6 groups instead of 60 per-dataset)
        for emb_name, agg in per_embodiment.items():
            eval_metrics[f'eval_emb/{emb_name}/mse'] = float(np.mean(agg['mse']))
            eval_metrics[f'eval_emb/{emb_name}/mae'] = float(np.mean(agg['mae']))
            if any(c != 0 for c in agg['corr']):
                eval_metrics[f'eval_emb/{emb_name}/correlation'] = float(np.mean([c for c in agg['corr'] if c != 0]))
            if any(c != 0 for c in agg['coverage']):
                eval_metrics[f'eval_emb/{emb_name}/coverage_pct'] = float(np.mean([c for c in agg['coverage'] if c != 0]))
            eval_metrics[f'eval_emb/{emb_name}/n_datasets'] = len(agg['datasets'])
        
        # Overall aggregate
        if all_mse:
            eval_metrics['eval/overall_mse'] = float(np.mean(all_mse))
            eval_metrics['eval/overall_mae'] = float(np.mean(all_mae))
        
        eval_metrics['eval/step'] = step
        eval_metrics['eval/n_datasets'] = len(per_dataset)
        eval_metrics['eval/n_embodiments'] = len(per_embodiment)
        
        # ── WandB Visual Dashboard: per-dataset bar charts (replaces hard-to-read table) ──
        if self.wandb_run:
            try:
                from drifting_vla.training.visualizations import create_eval_dashboard
                dashboard_data = {}
                for ds_name, data in per_dataset.items():
                    emb_id = data['emb_id']
                    emb_name = EMBODIMENT_NAMES.get(emb_id, f'emb_{emb_id}')
                    dashboard_data[ds_name] = {
                        'mse': eval_metrics.get(f'eval/{ds_name}/mse', 0),
                        'mae': eval_metrics.get(f'eval/{ds_name}/mae', 0),
                        'corr': eval_metrics.get(f'eval/{ds_name}/mean_correlation', 0),
                        'coverage': eval_metrics.get(f'eval/{ds_name}/coverage_pct', 0),
                        'emb_name': emb_name,
                    }
                dashboard_img = create_eval_dashboard(dashboard_data, per_embodiment, step)
                if dashboard_img is not None:
                    eval_metrics['viz/D1_eval_dashboard'] = dashboard_img

                # D3: Per-Dataset Learning Curve (track history + plot)
                for ds_name, data in per_dataset.items():
                    if ds_name not in self.viz_logger._eval_history:
                        self.viz_logger._eval_history[ds_name] = {
                            'steps': [], 'mse': [], 'corr': [],
                        }
                    h = self.viz_logger._eval_history[ds_name]
                    h['steps'].append(step)
                    h['mse'].append(eval_metrics.get(f'eval/{ds_name}/mse', 0))
                    h['corr'].append(eval_metrics.get(f'eval/{ds_name}/mean_correlation', 0))

                from drifting_vla.training.visualizations import (
                    create_per_dataset_learning_curve, create_data_balance_monitor,
                )
                d3_img = create_per_dataset_learning_curve(
                    self.viz_logger._eval_history, step,
                )
                if d3_img is not None:
                    eval_metrics['viz/D3_dataset_learning_curves'] = d3_img

                # D4: Data Balance Monitor
                if self.viz_logger._dataset_batch_counts:
                    dataset_sizes = {
                        ds_name: len(ds) for ds_name, ds in self.unified_dataset.datasets.items()
                    }
                    d4_img = create_data_balance_monitor(
                        self.viz_logger._dataset_batch_counts, dataset_sizes, step,
                    )
                    if d4_img is not None:
                        eval_metrics['viz/D4_data_balance'] = d4_img

            except Exception as e:
                logger.debug(f"Eval dashboard creation failed: {e}")

        # Remove per-dataset flat keys from wandb logging (keep them only in table)
        # This prevents 60+ dataset keys from cluttering the dashboard
        flat_keys_to_remove = [k for k in eval_metrics if k.startswith('eval/') and '/' in k[5:] 
                               and not k.startswith('eval_emb/') and k != 'eval/overall_mse' 
                               and k != 'eval/overall_mae' and k != 'eval/step'
                               and k != 'eval/n_datasets' and k != 'eval/n_embodiments'
                               and k != 'eval/dataset_detail']
        for k in flat_keys_to_remove:
            del eval_metrics[k]
        
        # Log summary
        logger.info(f"[Eval] Overall: MSE={eval_metrics.get('eval/overall_mse', 0):.4f}, "
                     f"MAE={eval_metrics.get('eval/overall_mae', 0):.4f}, "
                     f"datasets={len(per_dataset)}, embodiments={len(per_embodiment)}")
        for emb_name, agg in per_embodiment.items():
            logger.info(f"  [{emb_name}] MSE={np.mean(agg['mse']):.4f}, "
                        f"MAE={np.mean(agg['mae']):.4f}, "
                        f"datasets={agg['datasets']}")
        
        self.model.train()
        return eval_metrics
    
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
    parser.add_argument('--vlm-mode', type=str, default='frozen',
                        choices=['frozen', 'lora', 'full'],
                        help='VLM training: frozen (debug), lora (pre-train), full (post-train)')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank (16=pre-train, 64=post-train)')
    parser.add_argument('--vlm-lr-scale', type=float, default=0.1,
                        help='VLM learning rate = base_lr × this scale')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--episodes-root', type=str, default='./data/episodes',
                        help='Root directory for Episode HDF5 files')
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
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Run evaluation every N steps')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader workers (default 8)')
    parser.add_argument('--eval-batches', type=int, default=50,
                        help='Number of eval batches per evaluation run')
    parser.add_argument('--image-aug', action='store_true', default=False,
                        help='Enable image augmentation (ColorJitter)')
    parser.add_argument('--cond-mask-prob', type=float, default=0.1,
                        help='Condition masking probability for CFG training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize distributed if available
    if 'RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        # Disable NCCL P2P to avoid hangs on some GPU topologies (e.g., A40 PCIe)
        os.environ.setdefault('NCCL_P2P_DISABLE', '1')
        dist.init_process_group(backend='nccl')
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
    config.episodes_root = args.episodes_root
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_steps = args.max_steps
    config.loss_type = args.loss_type
    config.grad_accumulation_steps = args.grad_accumulation
    config.wandb_mode = args.wandb_mode
    config.wandb_project = args.wandb_project
    config.log_every = args.log_every
    config.save_every = args.save_every
    config.eval_every = args.eval_every
    config.use_flash_attn = args.use_flash_attn
    config.vlm_mode = args.vlm_mode
    config.lora_r = args.lora_r
    config.vlm_lr_scale = args.vlm_lr_scale
    config.num_workers = args.num_workers
    config.eval_batches = args.eval_batches
    config.image_aug = args.image_aug
    config.cond_mask_prob = args.cond_mask_prob
    
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
