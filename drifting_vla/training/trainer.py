"""
Drifting-VLA Trainer with FSDP Support
======================================

This module implements the main training loop for Drifting-VLA with:
- FSDP (Fully Sharded Data Parallel) for multi-GPU/multi-node training
- Exponential Moving Average (EMA) for stable evaluation
- Gradient accumulation and mixed precision training
- Checkpointing and resumption
- WandB logging integration
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import numpy as np

from drifting_vla.training.losses import DriftingLoss, CombinedDriftingLoss
from drifting_vla.training.ema import EMA, MultiEMA
from drifting_vla.training.optimizer import create_optimizer, create_scheduler
from drifting_vla.logging.visualizations import (
    plot_drifting_field,
    plot_drift_magnitude_trend,
    plot_prediction_scatter,
    plot_per_dim_error_radar,
    plot_temperature_loss_breakdown,
    plot_action_distribution,
    plot_action_error_heatmap,
    render_trajectory_3d,
    plot_sample_transport,
    plot_mode_coverage,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """
    Configuration for training visualizations.
    
    Attributes:
        enabled: Whether to enable visualizations.
        drifting_field_freq: Freq for drifting field (A1).
        drift_trend_freq: Freq for drift magnitude trend (A2).
        pred_scatter_freq: Freq for prediction scatter (A3).
        error_radar_freq: Freq for per-dim error radar (A4).
        temp_loss_freq: Freq for temperature loss breakdown (A5).
        action_dist_freq: Freq for action distribution (B1).
        error_heatmap_freq: Freq for action error heatmap (B2).
        trajectory_3d_freq: Freq for 3D trajectory (B4).
        sample_transport_freq: Freq for sample transport (C1).
        mode_coverage_freq: Freq for mode coverage (C2).
        num_samples: Number of samples to use in visualizations.
        theme: Visualization theme ('light', 'dark', 'paper').
    """
    enabled: bool = True
    # Category A — Training Convergence
    drifting_field_freq: int = 500
    drift_trend_freq: int = 200
    pred_scatter_freq: int = 500
    error_radar_freq: int = 500
    temp_loss_freq: int = 500
    # Category B — Action Quality
    action_dist_freq: int = 500
    error_heatmap_freq: int = 500
    trajectory_3d_freq: int = 1000
    # Category C — Drifting-Specific
    sample_transport_freq: int = 500
    mode_coverage_freq: int = 2000
    # General
    num_samples: int = 100
    theme: str = 'light'


@dataclass
class SimulationEvalConfig:
    """
    Configuration for simulation-based evaluation.
    
    Attributes:
        enabled: Whether to run simulation evaluation.
        num_episodes: Number of episodes per task.
        max_steps: Maximum steps per episode.
        tasks: Tasks to evaluate on.
        record_video: Whether to record rollout videos.
        video_fps: FPS for recorded videos.
        eval_freq: How often to run simulation eval (steps).
        use_dummy_env: Use DummyEnvironment for testing (no RLBench needed).
    """
    enabled: bool = False  # Disabled by default (requires RLBench)
    num_episodes: int = 5
    max_steps: int = 200
    tasks: Optional[list[str]] = None  # None = use training tasks
    record_video: bool = True
    video_fps: int = 10
    eval_freq: int = 5000  # Run every N training steps
    use_dummy_env: bool = False  # Set to True to test without RLBench


@dataclass
class TrainerConfig:
    """
    Configuration for Drifting-VLA trainer.
    
    Attributes:
        # Optimization
        learning_rate: Base learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Number of warmup steps.
        max_steps: Maximum training steps.
        grad_clip: Gradient clipping norm.
        grad_accumulation_steps: Number of gradient accumulation steps.
        
        # Batch settings
        batch_size: Per-GPU batch size.
        num_workers: DataLoader workers per GPU.
        
        # EMA
        ema_decay: EMA decay rate.
        ema_warmup_steps: Steps before starting EMA.
        
        # Drifting loss
        temperatures: Temperature values for drifting field.
        n_pos_samples: Number of positive samples per batch.
        n_neg_samples: Number of negative samples per batch.
        
        # Checkpointing
        checkpoint_dir: Directory for checkpoints.
        save_every_n_steps: Checkpoint frequency.
        keep_last_n_checkpoints: Number of checkpoints to keep.
        
        # Logging
        log_every_n_steps: Logging frequency.
        eval_every_n_steps: Evaluation frequency.
        
        # Mixed precision
        use_amp: Use automatic mixed precision.
        amp_dtype: AMP dtype ('float16' or 'bfloat16').
        
        # Distributed
        use_fsdp: Use FSDP for distributed training.
        fsdp_shard_size: Minimum size for FSDP sharding.
        
        # Visualizations
        visualizations: VisualizationConfig = None
    """
    # Optimization
    learning_rate: float = 4e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_clip: float = 2.0
    grad_accumulation_steps: int = 1
    
    # Batch settings
    batch_size: int = 32
    num_workers: int = 4
    
    # EMA
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 1000
    
    # Drifting loss
    temperatures: list[float] = None
    n_pos_samples: int = 32
    n_neg_samples: int = 32
    
    # Classifier-free guidance
    cfg_scale_range: tuple[float, float] = (1.0, 4.0)
    cfg_dropout: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 5
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Distributed
    use_fsdp: bool = True
    fsdp_shard_size: int = 100_000_000
    
    # Visualizations
    visualizations: Optional[VisualizationConfig] = None
    
    # Simulation evaluation
    simulation_eval: Optional[SimulationEvalConfig] = None
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.02, 0.05, 0.2]
        if self.visualizations is None:
            self.visualizations = VisualizationConfig()
        if self.simulation_eval is None:
            self.simulation_eval = SimulationEvalConfig()


class DriftingVLATrainer:
    """
    Trainer for Drifting-VLA models.
    
    Handles distributed training with FSDP, mixed precision, EMA,
    checkpointing, and logging.
    
    Args:
        model: DriftingVLA model to train.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        config: TrainerConfig with training settings.
        sample_queue: Optional sample queue for positive samples.
        wandb_logger: Optional WandB logger for metrics.
    
    Example:
        >>> model = DriftingVLA(model_config)
        >>> trainer = DriftingVLATrainer(model, train_loader, val_loader, config)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        sample_queue: Optional[Any] = None,
        wandb_logger: Optional[Any] = None,
    ):
        self.config = config or TrainerConfig()
        self.wandb_logger = wandb_logger
        self.sample_queue = sample_queue
        self.neg_queue = None  # Created after device is known
        
        # Setup distributed training
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main_process = self.rank == 0
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        else:
            self.device = torch.device('cpu')
        
        # Setup model (with FSDP or DDP)
        self.model = self._setup_model(model)
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            num_training_steps=config.max_steps,
            num_warmup_steps=config.warmup_steps,
        )
        
        # Setup loss function
        self.loss_fn = DriftingLoss(
            temperatures=config.temperatures,
            normalize_features=True,
            normalize_drift=True,
        )
        
        # Setup EMA
        self.ema = EMA(
            self._get_unwrapped_model(),
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup_steps,
        )
        
        # Setup AMP
        if config.use_amp:
            dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16}
            self.amp_dtype = dtype_map.get(config.amp_dtype, torch.bfloat16)
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp_dtype == 'float16')
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
        
        # Negative sample queue for drifting loss
        # Stores recent model predictions so negatives come from PREVIOUS steps,
        # not the current batch. This prevents V_attract ≈ V_repel → V ≈ 0.
        from drifting_vla.data.sample_queue import NegativeSampleQueue
        action_dim = (
            getattr(self._get_unwrapped_model().config, 'action_horizon', 16) *
            (getattr(self._get_unwrapped_model().config.action_decoder, 'position_dim', 3) +
             getattr(self._get_unwrapped_model().config.action_decoder, 'rotation_dim', 4) +
             getattr(self._get_unwrapped_model().config.action_decoder, 'gripper_dim', 1))
        )
        self.neg_queue = NegativeSampleQueue(
            queue_size=2048,  # Store ~64 batches of predictions
            action_dim=action_dim,
            device=self.device,
        )
        logger.info(f"Initialized negative sample queue: size=2048, action_dim={action_dim}")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_metric = float('inf')
        
        # Visualization cache (store recent samples for visualization)
        self._viz_cache = {
            'actions_pred': [],       # Flattened [B*T*D] per sample
            'actions_gt': [],         # Flattened GT
            'actions_pred_seq': [],   # Unflattened [T, D] per step (1 sample)
            'actions_gt_seq': [],     # Unflattened GT [T, D] per step
            'drifting_field': [],     # Drift vectors
            'positions': [],          # [B, 3] end-effector positions
            'drift_norms': [],        # Scalar ||V||² per step
            'drift_norm_steps': [],   # Corresponding step numbers
            'per_temp_losses': {},    # Latest per-temperature loss dict
        }
        self._max_viz_cache_size = config.visualizations.num_samples if config.visualizations else 100
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer on rank {self.rank}/{self.world_size}")
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model for distributed training."""
        model = model.to(self.device)
        
        if not self.is_distributed:
            return model
        
        if self.config.use_fsdp:
            try:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                )
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                
                # FSDP mixed precision
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
                
                # Wrap policy: shard modules larger than threshold
                wrap_policy = size_based_auto_wrap_policy(
                    min_num_params=self.config.fsdp_shard_size
                )
                
                model = FSDP(
                    model,
                    auto_wrap_policy=wrap_policy,
                    mixed_precision=mp_policy,
                    device_id=self.device,
                    limit_all_gathers=True,
                )
                logger.info("Using FSDP for distributed training")
            except Exception as e:
                logger.warning(f"FSDP failed: {e}, falling back to DDP")
                model = DDP(model, device_ids=[self.rank])
        else:
            model = DDP(model, device_ids=[self.rank])
            logger.info("Using DDP for distributed training")
        
        return model
    
    def _get_unwrapped_model(self) -> nn.Module:
        """Get the unwrapped model (without FSDP/DDP)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def train(self) -> dict:
        """
        Main training loop.
        
        Returns:
            Dict with final training metrics.
        """
        logger.info(f"Starting training for {self.config.max_steps} steps")
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        # Training metrics
        metrics = {
            'loss': 0.0,
            'mse_loss': 0.0,
            'drift_loss': 0.0,
            'drift_norm': 0.0,
            'raw_drift_norm': 0.0,
            'lambda_V': 0.0,
            'steps': 0,
        }
        
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Training step
            step_metrics = self._training_step(batch)
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                if k in metrics:
                    metrics[k] += v
            metrics['steps'] += 1
            
            # Logging
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics(metrics, start_time)
                metrics = {k: 0.0 for k in metrics}
                metrics['steps'] = 0
                start_time = time.time()
            
            # Visualizations
            if self.config.visualizations.enabled:
                self._log_visualizations()
            
            # Evaluation
            if (self.val_loader is not None and 
                self.global_step % self.config.eval_every_n_steps == 0):
                val_metrics = self.evaluate()
                self._log_eval_metrics(val_metrics)
            
            # Simulation evaluation
            if (self.config.simulation_eval.enabled and 
                self.global_step % self.config.simulation_eval.eval_freq == 0):
                self.evaluate_in_simulation()
            
            # Flush all buffered WandB data for this step in ONE atomic call
            if self.wandb_logger:
                self.wandb_logger.flush_step()
            
            # Checkpointing
            if self.global_step % self.config.save_every_n_steps == 0:
                self.save_checkpoint()
            
            self.global_step += 1
        
        # Final checkpoint
        self.save_checkpoint(is_final=True)
        
        return metrics
    
    def _training_step(self, batch: dict) -> dict:
        """
        Execute one training step.
        
        Args:
            batch: Dict with 'images', 'language', 'actions'.
        
        Returns:
            Dict with step metrics.
        """
        # Move batch to device
        images = batch['images'].to(self.device)
        language = batch['language']
        actions_gt = batch['actions'].to(self.device)  # [B, T, D]
        
        B, T, D = actions_gt.shape
        
        # Sample noise
        noise = torch.randn(
            B, T, self._get_unwrapped_model().config.noise_dim,
            device=self.device
        )
        
        # Sample CFG scale
        cfg_scale = self._sample_cfg_scale(B)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            # Generate actions
            outputs = self.model(
                images, language, noise, cfg_scale,
                return_features=True
            )
            actions_pred = outputs['actions']  # [B, T, D_a]
            
            # Flatten for drifting loss
            actions_flat = actions_pred.view(B, -1)  # [B, T * D_a]
            actions_gt_flat = actions_gt.view(B, -1)
            
            # Get positive samples (from queue or batch GT)
            if self.sample_queue is not None:
                pos_samples = self.sample_queue.sample(self.config.n_pos_samples)
                pos_samples = pos_samples.to(self.device).view(self.config.n_pos_samples, -1)
            else:
                pos_samples = actions_gt_flat
            
            # Negative samples from PREVIOUS steps (not current batch!)
            # Using x.detach() would make V_attract ≈ V_repel → V ≈ 0
            # because negatives would be identical to queries.
            # Instead, sample from a queue of recent model predictions.
            if len(self.neg_queue) >= self.config.n_neg_samples:
                neg_samples = self.neg_queue.sample(self.config.n_neg_samples)
                neg_samples = neg_samples.to(self.device)
            else:
                # Fallback for first few steps when queue is empty:
                # use batch GT as negatives (they're at least different from x)
                neg_samples = actions_gt_flat.detach()
            
            # Push current predictions to negative queue for future steps
            self.neg_queue.push(actions_flat.detach())
            
            # ============================================================
            # HYBRID LOSS: MSE (reliable at any batch) + Drifting (multi-modal)
            # ============================================================
            # 1. Direct MSE supervision — strong gradient signal
            mse_loss = ((actions_pred - actions_gt) ** 2).mean()
            
            # 2. Drifting loss — adds multi-modal structure
            loss_output = self.loss_fn(actions_flat, pos_samples, neg_samples)
            drift_loss = loss_output.loss
            
            # 3. Combined: MSE dominates, drifting adds diversity
            loss = mse_loss + 0.1 * drift_loss
        
        # Gradient accumulation
        loss = loss / self.config.grad_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step (with gradient accumulation)
        if (self.global_step + 1) % self.config.grad_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # Update EMA
            self.ema.update()
        else:
            grad_norm = torch.tensor(0.0)
        
        # Cache samples for visualization
        if self.is_main_process and self.config.visualizations.enabled:
            self._cache_for_visualization(
                actions_pred.detach(),
                actions_gt.detach(),
                loss_output,
            )
        
        return {
            'loss': loss.item() * self.config.grad_accumulation_steps,
            'mse_loss': mse_loss.item(),
            'drift_loss': drift_loss.item(),
            'drift_norm': loss_output.drift_norm.item(),
            'raw_drift_norm': loss_output.raw_drift_norm.item() if loss_output.raw_drift_norm is not None else 0.0,
            'lambda_V': loss_output.lambda_V.item() if loss_output.lambda_V is not None else 0.0,
            'grad_norm': grad_norm.item(),
        }
    
    def _sample_cfg_scale(self, batch_size: int) -> torch.Tensor:
        """Sample CFG scale from distribution."""
        # Sample alpha from p(alpha) ∝ alpha^{-3}
        min_scale, max_scale = self.config.cfg_scale_range
        
        # Inverse CDF sampling for power law
        u = torch.rand(batch_size, device=self.device)
        k = 3  # Power law exponent
        
        alpha = (
            (max_scale ** (1-k) - min_scale ** (1-k)) * u +
            min_scale ** (1-k)
        ) ** (1 / (1-k))
        
        return alpha
    
    def evaluate(self) -> dict:
        """
        Run evaluation on validation set.
        
        Returns:
            Dict with evaluation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_drift_norm = 0.0
        num_batches = 0
        
        # Store first sample for video logging
        first_sample_images = None
        first_sample_actions_pred = None
        first_sample_actions_gt = None
        first_sample_language = None
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                language = batch['language']
                actions_gt = batch['actions'].to(self.device)
                
                B, T, D = actions_gt.shape
                
                # Generate with EMA model
                with self.ema.average_parameters():
                    noise = torch.randn(B, T, 64, device=self.device)
                    actions_pred = self.model(images, language, noise)
                
                # Store first sample for video
                if first_sample_images is None:
                    first_sample_images = images[0].cpu()  # [C, H, W] or [T, C, H, W]
                    first_sample_actions_pred = actions_pred[0].cpu().numpy()  # [T, D]
                    first_sample_actions_gt = actions_gt[0].cpu().numpy()  # [T, D]
                    first_sample_language = language[0] if isinstance(language, list) else language
                
                # Compute validation loss
                actions_flat = actions_pred.view(B, -1)
                actions_gt_flat = actions_gt.view(B, -1)
                
                loss_output = self.loss_fn(
                    actions_flat, actions_gt_flat, actions_flat
                )
                
                total_loss += loss_output.loss.item()
                total_drift_norm += loss_output.drift_norm.item()
                num_batches += 1
        
        self.model.train()
        
        # Log evaluation video if wandb is available.
        # When simulation eval is enabled, the sim produces eval/trajectory_video
        # with real camera frames. Only fall back to dataset-based video if sim is off.
        sim_will_log_video = (
            self.config.simulation_eval.enabled and
            not self.config.simulation_eval.use_dummy_env and
            self.global_step % self.config.simulation_eval.eval_freq == 0
        )
        
        if (self.is_main_process and self.wandb_logger
                and first_sample_images is not None
                and not sim_will_log_video):
            self._log_eval_video(
                first_sample_images,
                first_sample_actions_pred,
                first_sample_actions_gt,
                first_sample_language,
            )
        
        return {
            'val/loss': total_loss / max(num_batches, 1),
            'val/drift_norm': total_drift_norm / max(num_batches, 1),
        }
    
    def _log_metrics(self, metrics: dict, start_time: float) -> None:
        """Log training metrics."""
        if not self.is_main_process:
            return
        
        steps = max(metrics['steps'], 1)
        elapsed = time.time() - start_time
        
        raw_dn = metrics.get('raw_drift_norm', 0) / steps
        lv = metrics.get('lambda_V', 0) / steps
        mse_l = metrics.get('mse_loss', 0) / steps
        drift_l = metrics.get('drift_loss', 0) / steps
        
        log_dict = {
            'train/loss': metrics['loss'] / steps,
            'train/mse_loss': mse_l,             # MSE component (should ↓ toward 0)
            'train/drift_loss': drift_l,          # Drifting component
            'train/drift_norm': metrics['drift_norm'] / steps,
            'train/raw_drift_norm': raw_dn,
            'train/lambda_V': lv,
            'train/grad_norm': metrics.get('grad_norm', 0) / steps,
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'train/step': self.global_step,
            'train/epoch': self.epoch,
            'train/steps_per_sec': steps / elapsed,
        }
        
        if self.wandb_logger:
            self.wandb_logger.log(log_dict, step=self.global_step)
        
        logger.info(
            f"Step {self.global_step}: mse={mse_l:.4f}, drift={drift_l:.4f}, "
            f"loss={log_dict['train/loss']:.4f}, "
            f"lr={log_dict['train/lr']:.2e}"
        )
    
    def _log_eval_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics."""
        if not self.is_main_process:
            return
        
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=self.global_step)
        
        logger.info(f"Eval at step {self.global_step}: {metrics}")
    
    def save_checkpoint(self, is_final: bool = False) -> None:
        """Save training checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'model_state_dict': self._get_unwrapped_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        
        if is_final:
            path = self.checkpoint_dir / 'final_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_step_{self.global_step}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints keeping only the latest N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_step_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.config.keep_last_n_checkpoints:
            old_ckpt = checkpoints.pop(0)
            old_ckpt.unlink()
            logger.info(f"Removed old checkpoint: {old_ckpt}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint for resumption."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self._get_unwrapped_model().load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.ema.load_state_dict(checkpoint['ema_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Resumed from checkpoint {path} at step {self.global_step}")
    
    def _cache_for_visualization(
        self,
        actions_pred: torch.Tensor,
        actions_gt: torch.Tensor,
        loss_output: Any,
    ) -> None:
        """
        Cache samples for visualization.
        
        Caches flattened predictions, ground truth, drifting field vectors,
        positions, drift norms, and per-temperature losses.
        
        Args:
            actions_pred: Predicted actions [B, T, D].
            actions_gt: Ground truth actions [B, T, D].
            loss_output: Output from drifting loss with drift field.
        """
        B = actions_pred.shape[0]
        
        # Flattened predictions and ground truth
        pred_flat = actions_pred.cpu().numpy().reshape(B, -1)
        gt_flat = actions_gt.cpu().numpy().reshape(B, -1)
        self._viz_cache['actions_pred'].extend(pred_flat.tolist())
        self._viz_cache['actions_gt'].extend(gt_flat.tolist())
        
        # Store one unflattened sample per step for error heatmap [T, D]
        self._viz_cache['actions_pred_seq'].append(
            actions_pred[0].cpu().numpy()  # [T, D]
        )
        self._viz_cache['actions_gt_seq'].append(
            actions_gt[0].cpu().numpy()  # [T, D]
        )
        
        # Drifting field vectors
        if hasattr(loss_output, 'drift_field') and loss_output.drift_field is not None:
            self._viz_cache['drifting_field'].extend(
                loss_output.drift_field.cpu().numpy().tolist()
            )
        
        # Raw drift norm and lambda_V (true convergence metrics)
        if hasattr(loss_output, 'raw_drift_norm') and loss_output.raw_drift_norm is not None:
            self._viz_cache['drift_norms'].append(loss_output.raw_drift_norm.item())
            self._viz_cache['drift_norm_steps'].append(self.global_step)
        elif hasattr(loss_output, 'drift_norm'):
            self._viz_cache['drift_norms'].append(loss_output.drift_norm.item())
            self._viz_cache['drift_norm_steps'].append(self.global_step)
        
        # Per-temperature losses (for temp breakdown)
        if hasattr(loss_output, 'per_temp_losses') and loss_output.per_temp_losses is not None:
            self._viz_cache['per_temp_losses'] = {
                k: v.item() for k, v in loss_output.per_temp_losses.items()
            }
        
        # Position component for 3D viz [B, 3]
        positions = actions_pred[:, 0, :3].cpu().numpy()
        self._viz_cache['positions'].extend(positions.tolist())
        
        # Limit cache size (FIFO) for list-type caches
        max_size = self._max_viz_cache_size
        for key in ['actions_pred', 'actions_gt', 'drifting_field', 'positions']:
            if len(self._viz_cache[key]) > max_size:
                self._viz_cache[key] = self._viz_cache[key][-max_size:]
        for key in ['actions_pred_seq', 'actions_gt_seq']:
            if len(self._viz_cache[key]) > 50:
                self._viz_cache[key] = self._viz_cache[key][-50:]
        for key in ['drift_norms', 'drift_norm_steps']:
            if len(self._viz_cache[key]) > 2000:
                self._viz_cache[key] = self._viz_cache[key][-2000:]
    
    def _log_visualizations(self) -> None:
        """
        Generate and log the full visualization suite to WandB.
        
        Visualization Suite:
            A1. Drifting Field — 2D arrows showing drift vectors
            A2. Drift Magnitude Trend — ||V||² over steps
            A3. Prediction Scatter — pred vs GT per action dim
            A4. Per-Dim Error Radar — spider chart of MAE per dim
            A5. Temperature Loss Breakdown — loss per temperature
            B1. Action Distribution — per-dim histograms
            B2. Action Error Heatmap — [T × D] error grid
            B4. 3D Trajectory — end-effector path
            C1. Sample Transport — before/after drifting
            C2. Mode Coverage — GMM clustering
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not self.is_main_process or not self.wandb_logger:
            return
        
        import io
        from PIL import Image
        import wandb as _wandb
        
        viz_cfg = self.config.visualizations
        theme = viz_cfg.theme
        step = self.global_step
        n_samples = viz_cfg.num_samples
        
        # Collect ALL figures into one dict, then buffer into wandb logger.
        # The logger will flush everything atomically in flush_step().
        figures_to_log = {}
        
        def _safe_add(key, gen_fn):
            """Generate a figure, render to Image, add to batch dict."""
            try:
                fig = gen_fn()
                if fig is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150,
                                facecolor=fig.get_facecolor())
                    buf.seek(0)
                    img = Image.open(buf)
                    if img.width > 4096 or img.height > 4096:
                        img = img.resize((min(img.width, 4096), min(img.height, 4096)))
                    figures_to_log[key] = _wandb.Image(img)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate {key}: {e}")
        
        has_pred = len(self._viz_cache['actions_pred']) >= 10
        has_drift = len(self._viz_cache['drifting_field']) >= 10
        
        # ----------------------------------------------------------------
        # A1. Drifting Field
        # ----------------------------------------------------------------
        if step % viz_cfg.drifting_field_freq == 0 and has_pred and has_drift:
            def _gen_drifting_field():
                x = np.array(self._viz_cache['actions_pred'][-n_samples:])
                V = np.array(self._viz_cache['drifting_field'][-n_samples:])
                y_pos = np.array(self._viz_cache['actions_gt'][-n_samples:])
                if x.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    x_2d = pca.fit_transform(x)
                    V_2d = pca.transform(V) - pca.transform(np.zeros_like(V))
                    y_pos_2d = pca.transform(y_pos)
                else:
                    x_2d, V_2d, y_pos_2d = x, V, y_pos
                return plot_drifting_field(
                    x_2d, V_2d, y_pos=y_pos_2d,
                    title=f'A1: Drifting Field (Step {step})', theme=theme)
            _safe_add('viz/A1_drifting_field', _gen_drifting_field)
        
        # ----------------------------------------------------------------
        # A2. Drift Magnitude Trend
        # ----------------------------------------------------------------
        if step % viz_cfg.drift_trend_freq == 0 and len(self._viz_cache['drift_norms']) >= 5:
            def _gen_drift_trend():
                return plot_drift_magnitude_trend(
                    self._viz_cache['drift_norms'],
                    self._viz_cache['drift_norm_steps'],
                    title=f'A2: Drift Magnitude Trend', theme=theme)
            _safe_add('viz/A2_drift_magnitude_trend', _gen_drift_trend)
        
        # ----------------------------------------------------------------
        # A3. Prediction vs GT Scatter
        # ----------------------------------------------------------------
        if step % viz_cfg.pred_scatter_freq == 0 and has_pred:
            def _gen_pred_scatter():
                pred = np.array(self._viz_cache['actions_pred'][-n_samples:])
                gt = np.array(self._viz_cache['actions_gt'][-n_samples:])
                n_dims = min(8, pred.shape[1])
                return plot_prediction_scatter(
                    pred[:, :n_dims], gt[:, :n_dims],
                    title=f'A3: Prediction vs GT (Step {step})', theme=theme)
            _safe_add('viz/A3_prediction_scatter', _gen_pred_scatter)
        
        # ----------------------------------------------------------------
        # A4. Per-Dim Error Radar
        # ----------------------------------------------------------------
        if step % viz_cfg.error_radar_freq == 0 and has_pred:
            def _gen_error_radar():
                pred = np.array(self._viz_cache['actions_pred'][-n_samples:])
                gt = np.array(self._viz_cache['actions_gt'][-n_samples:])
                n_dims = min(8, pred.shape[1])
                return plot_per_dim_error_radar(
                    pred[:, :n_dims], gt[:, :n_dims],
                    title=f'A4: Per-Dim Error (Step {step})', theme=theme)
            _safe_add('viz/A4_per_dim_error_radar', _gen_error_radar)
        
        # ----------------------------------------------------------------
        # A5. Temperature Loss Breakdown
        # ----------------------------------------------------------------
        if step % viz_cfg.temp_loss_freq == 0 and self._viz_cache['per_temp_losses']:
            def _gen_temp_loss():
                return plot_temperature_loss_breakdown(
                    self._viz_cache['per_temp_losses'],
                    title=f'A5: Loss by Temperature (Step {step})', theme=theme)
            _safe_add('viz/A5_temperature_loss', _gen_temp_loss)
        
        # ----------------------------------------------------------------
        # B1. Action Distribution
        # ----------------------------------------------------------------
        if step % viz_cfg.action_dist_freq == 0 and has_pred:
            def _gen_action_dist():
                pred = np.array(self._viz_cache['actions_pred'][-n_samples:])
                gt = np.array(self._viz_cache['actions_gt'][-n_samples:])
                n_dims = min(8, pred.shape[1])
                dim_names = ['pos_x', 'pos_y', 'pos_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'grip'][:n_dims]
                return plot_action_distribution(
                    pred[:, :n_dims], gt[:, :n_dims], dim_names=dim_names,
                    title=f'B1: Action Distribution (Step {step})', theme=theme)
            _safe_add('viz/B1_action_distribution', _gen_action_dist)
        
        # ----------------------------------------------------------------
        # B2. Action Error Heatmap (from most recent unflattened sample)
        # ----------------------------------------------------------------
        if step % viz_cfg.error_heatmap_freq == 0 and len(self._viz_cache['actions_pred_seq']) >= 1:
            def _gen_error_heatmap():
                pred_seq = self._viz_cache['actions_pred_seq'][-1]  # [T, D]
                gt_seq = self._viz_cache['actions_gt_seq'][-1]      # [T, D]
                return plot_action_error_heatmap(
                    pred_seq, gt_seq,
                    title=f'B2: Action Error Heatmap (Step {step})', theme=theme)
            _safe_add('viz/B2_action_error_heatmap', _gen_error_heatmap)
        
        # ----------------------------------------------------------------
        # B4. 3D Trajectory
        # ----------------------------------------------------------------
        if step % viz_cfg.trajectory_3d_freq == 0 and len(self._viz_cache['positions']) >= 10:
            def _gen_trajectory_3d():
                positions = np.array(self._viz_cache['positions'][-50:])
                return render_trajectory_3d(
                    positions, title=f'B4: Predicted Positions (Step {step})', theme=theme)
            _safe_add('viz/B4_trajectory_3d', _gen_trajectory_3d)
        
        # ----------------------------------------------------------------
        # C1. Sample Transport
        # ----------------------------------------------------------------
        if step % viz_cfg.sample_transport_freq == 0 and has_pred and has_drift:
            def _gen_sample_transport():
                x = np.array(self._viz_cache['actions_pred'][-n_samples:])
                V = np.array(self._viz_cache['drifting_field'][-n_samples:])
                y_gt = np.array(self._viz_cache['actions_gt'][-n_samples:])
                # x_after = x + V (drifted positions)
                x_after = x + V
                if x.shape[1] > 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    all_data = np.concatenate([x, x_after, y_gt], axis=0)
                    all_2d = pca.fit_transform(all_data)
                    n = len(x)
                    x_2d = all_2d[:n]
                    x_after_2d = all_2d[n:2*n]
                    y_gt_2d = all_2d[2*n:]
                else:
                    x_2d, x_after_2d, y_gt_2d = x, x_after, y_gt
                return plot_sample_transport(
                    x_2d, x_after_2d, y_gt_2d,
                    title=f'C1: Sample Transport (Step {step})', theme=theme)
            _safe_add('viz/C1_sample_transport', _gen_sample_transport)
        
        # ----------------------------------------------------------------
        # C2. Mode Coverage
        # ----------------------------------------------------------------
        if step % viz_cfg.mode_coverage_freq == 0 and has_pred and len(self._viz_cache['actions_pred']) >= 30:
            def _gen_mode_coverage():
                pred = np.array(self._viz_cache['actions_pred'][-n_samples:])
                return plot_mode_coverage(
                    pred, n_modes=min(5, len(pred) // 5),
                    title=f'C2: Mode Coverage (Step {step})', theme=theme)
            _safe_add('viz/C2_mode_coverage', _gen_mode_coverage)
        
        # ----------------------------------------------------------------
        # Buffer ALL figures into the wandb logger's step buffer.
        # They'll be flushed atomically with training metrics in flush_step().
        # ----------------------------------------------------------------
        if figures_to_log:
            self.wandb_logger.log(figures_to_log, step=step)
            logger.info(f"[VIZ] Buffered {len(figures_to_log)} visualizations at step {step}: {list(figures_to_log.keys())}")
    
    def _log_eval_video(
        self,
        images: torch.Tensor,
        actions_pred: np.ndarray,
        actions_gt: np.ndarray,
        language: str,
    ) -> None:
        """
        Log evaluation video showing input image, predicted vs GT trajectories.
        
        Creates a video with 3 panels:
        - Left: Input image with task description
        - Middle: XY trajectory (fixed axis limits across all frames)
        - Right: 3D trajectory (fixed axis limits, rotating camera)
        
        Args:
            images: Input images [C, H, W] or [T, C, H, W].
            actions_pred: Predicted actions [T, D].
            actions_gt: Ground truth actions [T, D].
            language: Task description string.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        try:
            T = len(actions_pred)
            frames = []
            
            # Handle image format
            if images.dim() == 3:
                img = images.permute(1, 2, 0).numpy()
            else:
                img = images[0].permute(1, 2, 0).numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            
            # Extract position components (first 3 dims)
            pos_pred = actions_pred[:, :3]
            pos_gt = actions_gt[:, :3]
            
            # ---------------------------------------------------------
            # Pre-compute FIXED axis limits from ALL data (both pred+GT)
            # This prevents the plot from jumping between frames
            # ---------------------------------------------------------
            all_x = np.concatenate([pos_pred[:, 0], pos_gt[:, 0]])
            all_y = np.concatenate([pos_pred[:, 1], pos_gt[:, 1]])
            all_z = np.concatenate([pos_pred[:, 2], pos_gt[:, 2]])
            
            x_center = (all_x.max() + all_x.min()) / 2
            y_center = (all_y.max() + all_y.min()) / 2
            z_center = (all_z.max() + all_z.min()) / 2
            
            # Use largest range across all dims + margin for consistent aspect ratio
            ranges = [all_x.max() - all_x.min(),
                      all_y.max() - all_y.min(),
                      all_z.max() - all_z.min()]
            max_range = max(max(ranges), 1e-4) * 1.2  # 20% margin
            half = max_range / 2
            
            xy_xlim = (x_center - half, x_center + half)
            xy_ylim = (y_center - half, y_center + half)
            z_lim = (z_center - half, z_center + half)
            
            # Create frames showing trajectory progression
            for t in range(T):
                fig = plt.figure(figsize=(16, 5), dpi=100, layout='constrained')
                
                # ---- Left: Input image ----
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.imshow(img)
                title_text = f'"{language[:50]}..."' if len(language) > 50 else f'"{language}"'
                ax1.set_title(f'Input Image\n{title_text}', fontsize=10, fontweight='medium')
                ax1.axis('off')
                
                # ---- Middle: 2D XY trajectory (fixed axes) ----
                ax2 = fig.add_subplot(1, 3, 2)
                
                # Full GT trajectory (faded)
                ax2.plot(pos_gt[:, 0], pos_gt[:, 1], '-', color='#06D6A0',
                        alpha=0.3, linewidth=2, label='GT (full)')
                ax2.scatter(pos_gt[:, 0], pos_gt[:, 1], c='#06D6A0', s=30, alpha=0.3)
                
                # Predicted trajectory up to current timestep with gradient
                if t > 0:
                    colors_pred = plt.cm.Blues(np.linspace(0.3, 1.0, t))
                    for i in range(t):
                        ax2.plot(pos_pred[i:i+2, 0], pos_pred[i:i+2, 1],
                                color=colors_pred[i], linewidth=2.5, alpha=0.9)
                
                # Current position markers
                ax2.scatter(pos_pred[t, 0], pos_pred[t, 1], c='#3A86FF', s=150,
                           marker='o', edgecolors='white', linewidths=2,
                           zorder=10, label='Predicted')
                ax2.scatter(pos_gt[t, 0], pos_gt[t, 1], c='#06D6A0', s=150,
                           marker='*', edgecolors='white', linewidths=1.5,
                           zorder=10, label='Ground Truth')
                
                # FIXED axis limits
                ax2.set_xlim(xy_xlim)
                ax2.set_ylim(xy_ylim)
                ax2.set_aspect('equal')
                
                ax2.set_xlabel('X', fontsize=11)
                ax2.set_ylabel('Y', fontsize=11)
                ax2.set_title(f'XY Trajectory (t={t}/{T-1})', fontsize=12, fontweight='medium')
                ax2.legend(loc='upper right', fontsize=9)
                ax2.grid(True, alpha=0.2, linestyle='--')
                
                # ---- Right: 3D trajectory (fixed axes) ----
                ax3 = fig.add_subplot(1, 3, 3, projection='3d')
                
                # Full GT (faded)
                ax3.plot3D(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
                          '-', color='#06D6A0', alpha=0.3, linewidth=2)
                
                # Predicted trajectory up to current timestep
                if t > 0:
                    for i in range(t):
                        color = plt.cm.Blues(0.3 + 0.7 * i / max(t, 1))
                        ax3.plot3D(pos_pred[i:i+2, 0], pos_pred[i:i+2, 1],
                                  pos_pred[i:i+2, 2], color=color,
                                  linewidth=2.5, alpha=0.9)
                
                # Current position markers
                ax3.scatter(*pos_pred[t], c='#3A86FF', s=150, marker='o',
                           edgecolors='white', linewidths=2, zorder=10)
                ax3.scatter(*pos_gt[t], c='#06D6A0', s=150, marker='*',
                           edgecolors='white', linewidths=1.5, zorder=10)
                
                # FIXED 3D axis limits
                ax3.set_xlim(xy_xlim)
                ax3.set_ylim(xy_ylim)
                ax3.set_zlim(z_lim)
                
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                ax3.set_zlabel('Z', fontsize=10)
                ax3.set_title('3D Trajectory', fontsize=12, fontweight='medium')
                ax3.view_init(elev=20, azim=45 + t * 2)
                
                # Render to numpy (layout='constrained' ensures fixed size)
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)
            
            # Log video
            if frames:
                self.wandb_logger.log_video(
                    'eval/trajectory_video', frames, fps=5,
                    caption=f'Step {self.global_step}: {language[:100]}',
                    step=self.global_step)
                logger.info(f"Logged evaluation video at step {self.global_step}")
                
        except Exception as e:
            logger.warning(f"Failed to generate evaluation video: {e}")
    
    def _log_sim_trajectory_video(
        self,
        result: Any,
        task_name: str,
        success_rate: float,
    ) -> None:
        """
        Create and log a combined eval/trajectory_video from simulation rollout.
        
        Combines actual simulation camera frames (left) with trajectory plots
        (XY middle, 3D right) for each timestep. This replaces the dataset-based
        eval video when simulation evaluation is available.
        
        Args:
            result: RolloutResult with .frames, .actions, and .observations.
            task_name: Name of the evaluated task.
            success_rate: Overall success rate for the caption.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not self.wandb_logger:
            return
        
        try:
            sim_frames = result.frames      # List of [H, W, 3] RGB arrays
            actions = result.actions        # List of action arrays [D_a]
            T = len(sim_frames)
            
            if T == 0:
                logger.warning("No simulation frames to log")
                return
            
            # Extract executed positions from actions (first 3 dims = pos)
            if actions:
                positions = np.array([a[:3] if len(a) >= 3 else np.zeros(3) for a in actions])
            else:
                positions = np.zeros((T, 3))
            
            # Pad positions to match frame count (frames include initial state)
            if len(positions) < T:
                pad = np.tile(positions[-1:] if len(positions) > 0 else np.zeros((1, 3)),
                              (T - len(positions), 1))
                positions = np.concatenate([positions, pad], axis=0)
            positions = positions[:T]
            
            # Pre-compute fixed axis limits
            margin = 0.05
            pos_min = positions.min(axis=0)
            pos_max = positions.max(axis=0)
            ranges = pos_max - pos_min
            max_range = max(max(ranges), 1e-4) * 1.3
            centers = (pos_max + pos_min) / 2
            half = max_range / 2
            
            xy_xlim = (centers[0] - half, centers[0] + half)
            xy_ylim = (centers[1] - half, centers[1] + half)
            z_lim = (centers[2] - half, centers[2] + half)
            
            # Determine language description
            if result.observations:
                language = result.observations[0].language
            else:
                language = task_name.replace('_', ' ')
            
            combined_frames = []
            
            for t in range(T):
                fig = plt.figure(figsize=(16, 5), dpi=100, layout='constrained')
                
                # ---- Left: Simulation camera frame ----
                ax1 = fig.add_subplot(1, 3, 1)
                sim_img = sim_frames[t]
                if sim_img.max() <= 1.0:
                    sim_img = (sim_img * 255).astype(np.uint8)
                ax1.imshow(sim_img)
                status = "✓ Success" if result.success else "Running..."
                if t == T - 1 and not result.success:
                    status = "✗ Failed"
                ax1.set_title(
                    f'Simulation: {task_name}\n"{language[:40]}" — {status}',
                    fontsize=10, fontweight='medium'
                )
                ax1.text(0.02, 0.02, f't={t}/{T-1}', transform=ax1.transAxes,
                        fontsize=9, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                ax1.axis('off')
                
                # ---- Middle: XY trajectory (fixed axes) ----
                ax2 = fig.add_subplot(1, 3, 2)
                
                # Full trajectory (faded)
                ax2.plot(positions[:, 0], positions[:, 1], '-', color='#ADB5BD',
                        alpha=0.3, linewidth=1.5, label='Full path')
                
                # Executed trajectory up to current timestep with gradient
                if t > 0:
                    colors_t = plt.cm.Blues(np.linspace(0.3, 1.0, t))
                    for i in range(t):
                        ax2.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                                color=colors_t[i], linewidth=2.5, alpha=0.9)
                
                # Start marker
                ax2.scatter(positions[0, 0], positions[0, 1], c='#06D6A0', s=120,
                           marker='s', edgecolors='white', linewidths=2,
                           zorder=10, label='Start')
                # Current position
                ax2.scatter(positions[t, 0], positions[t, 1], c='#3A86FF', s=150,
                           marker='o', edgecolors='white', linewidths=2,
                           zorder=11, label='Current')
                
                ax2.set_xlim(xy_xlim)
                ax2.set_ylim(xy_ylim)
                ax2.set_aspect('equal')
                ax2.set_xlabel('X', fontsize=11)
                ax2.set_ylabel('Y', fontsize=11)
                ax2.set_title(f'XY Trajectory (t={t}/{T-1})', fontsize=12, fontweight='medium')
                ax2.legend(loc='upper right', fontsize=8)
                ax2.grid(True, alpha=0.2, linestyle='--')
                
                # ---- Right: 3D trajectory (fixed axes, rotating) ----
                ax3 = fig.add_subplot(1, 3, 3, projection='3d')
                
                # Full trajectory (faded)
                ax3.plot3D(positions[:, 0], positions[:, 1], positions[:, 2],
                          '-', color='#ADB5BD', alpha=0.3, linewidth=1.5)
                
                # Executed up to t
                if t > 0:
                    for i in range(t):
                        color = plt.cm.Blues(0.3 + 0.7 * i / max(t, 1))
                        ax3.plot3D(positions[i:i+2, 0], positions[i:i+2, 1],
                                  positions[i:i+2, 2], color=color,
                                  linewidth=2.5, alpha=0.9)
                
                ax3.scatter(*positions[0], c='#06D6A0', s=120, marker='s',
                           edgecolors='white', linewidths=2, zorder=10)
                ax3.scatter(*positions[t], c='#3A86FF', s=150, marker='o',
                           edgecolors='white', linewidths=2, zorder=11)
                
                ax3.set_xlim(xy_xlim)
                ax3.set_ylim(xy_ylim)
                ax3.set_zlim(z_lim)
                ax3.set_xlabel('X', fontsize=10)
                ax3.set_ylabel('Y', fontsize=10)
                ax3.set_zlabel('Z', fontsize=10)
                ax3.set_title('3D Trajectory', fontsize=12, fontweight='medium')
                ax3.view_init(elev=20, azim=45 + t * 3)
                
                # Render to numpy (layout='constrained' ensures fixed size)
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                combined_frames.append(frame)
                plt.close(fig)
            
            # Log as eval/trajectory_video (replaces dataset-based video)
            if combined_frames:
                caption = (
                    f'Step {self.global_step}: {task_name} '
                    f'({"Success" if result.success else "Fail"}, '
                    f'SR={success_rate:.0%}, len={result.episode_length})'
                )
                self.wandb_logger.log_video(
                    'eval/trajectory_video', combined_frames, fps=5,
                    caption=caption, step=self.global_step)
                
                # Also log raw simulation video separately
                self.wandb_logger.log_video(
                    'sim/rollout_video', sim_frames,
                    fps=self.config.simulation_eval.video_fps,
                    caption=caption, step=self.global_step)
                
                logger.info(
                    f"Logged eval/trajectory_video ({len(combined_frames)} frames) "
                    f"and sim/rollout_video ({len(sim_frames)} frames) "
                    f"at step {self.global_step}"
                )
                
        except Exception as e:
            logger.warning(f"Failed to generate simulation trajectory video: {e}")
    
    def evaluate_in_simulation(self) -> dict:
        """
        Run policy evaluation in RLBench simulation.
        
        Executes rollouts in the simulator and measures:
        - Task success rate
        - Average episode length
        - Records rollout videos
        
        Returns:
            Dict with simulation evaluation metrics.
        """
        if not self.config.simulation_eval.enabled:
            return {}
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        try:
            from drifting_vla.envs import RLBenchEnvironment, DummyEnvironment
            from drifting_vla.envs.base_env import EnvConfig
            from drifting_vla.inference.policy import DriftingVLAPolicy, PolicyConfig
            from drifting_vla.inference.rollout import rollout_episode, RolloutConfig
        except ImportError as e:
            logger.warning(f"Cannot run simulation eval - missing dependencies: {e}")
            return {}
        
        sim_cfg = self.config.simulation_eval
        
        # Get tasks
        tasks = sim_cfg.tasks
        if tasks is None:
            # Get tasks from dataloader dataset
            try:
                tasks = list(self.train_loader.dataset.tasks)
            except:
                tasks = ['close_jar']  # Default task
        
        # Select environment class
        if sim_cfg.use_dummy_env:
            logger.info("Using DummyEnvironment for simulation eval (no RLBench)")
            EnvClass = DummyEnvironment
        else:
            logger.info(f"Running simulation evaluation on tasks: {tasks}")
            EnvClass = RLBenchEnvironment
        
        # Get cameras from training dataset
        try:
            cameras = list(self.train_loader.dataset.cameras)
        except:
            cameras = ['front']
        
        # Create environment config
        env_config = EnvConfig(
            task=tasks[0],
            image_size=224,
            cameras=cameras,
            max_episode_length=sim_cfg.max_steps,
            headless=True,
        )
        
        # Create action denormalizer if dataset has normalization stats
        action_normalizer = None
        try:
            ds = self.train_loader.dataset
            if ds.action_mean is not None and ds.action_std is not None:
                from drifting_vla.data.transforms import ActionNormalization
                action_normalizer = ActionNormalization(
                    mean=ds.action_mean, std=ds.action_std
                )
                logger.info("Using action denormalization for simulation eval")
        except Exception:
            pass
        
        # Create policy from current model (using EMA if available)
        policy_config = PolicyConfig(
            cfg_scale=2.0,
            action_horizon=self.config.action_horizon if hasattr(self.config, 'action_horizon') else 16,
            use_ema=False,
            device=str(self.device),
            temporal_ensemble=True,
        )
        
        # Initialize policy with current model and action denormalizer
        with self.ema.average_parameters():
            policy = DriftingVLAPolicy(
                model=self._get_unwrapped_model(),
                config=policy_config,
                action_normalizer=action_normalizer,
            )
        
        rollout_config = RolloutConfig(
            max_steps=sim_cfg.max_steps,
            save_video=sim_cfg.record_video,
            video_fps=sim_cfg.video_fps,
            verbose=False,
        )
        
        all_results = []
        per_task_success = {}
        best_video_result = None  # Full RolloutResult with frames + actions
        best_video_task = None
        
        try:
            env = EnvClass(env_config)
            
            for task in tasks:
                task_successes = []
                
                for ep in range(sim_cfg.num_episodes):
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
                        
                        # Save first successful rollout (or first rollout if none succeed)
                        if sim_cfg.record_video and result.frames:
                            if result.success and best_video_result is None:
                                best_video_result = result
                                best_video_task = task
                            elif best_video_result is None:
                                best_video_result = result
                                best_video_task = task
                        
                        logger.info(
                            f"Sim eval - Task: {task}, Ep: {ep+1}/{sim_cfg.num_episodes}, "
                            f"Success: {result.success}, Length: {result.episode_length}"
                        )
                        
                    except Exception as e:
                        logger.warning(f"Rollout failed for {task} ep {ep}: {e}")
                        task_successes.append(0.0)
                
                per_task_success[task] = np.mean(task_successes) if task_successes else 0.0
            
            env.close()
            
        except Exception as e:
            logger.error(f"Simulation evaluation failed: {e}")
            return {'sim_eval_error': str(e)}
        
        # Compute metrics
        if all_results:
            success_rate = np.mean([r.success for r in all_results])
            avg_length = np.mean([r.episode_length for r in all_results])
        else:
            success_rate = 0.0
            avg_length = 0.0
        
        metrics = {
            'sim/success_rate': success_rate,
            'sim/avg_episode_length': avg_length,
            'sim/num_episodes': len(all_results),
        }
        
        # Add per-task success rates
        for task, rate in per_task_success.items():
            metrics[f'sim/success_{task}'] = rate
        
        # Log metrics to WandB
        if self.wandb_logger:
            self.wandb_logger.log(metrics, step=self.global_step)
            
            # Log combined eval/trajectory_video with sim frames + trajectory plots
            if best_video_result is not None and best_video_result.frames:
                self._log_sim_trajectory_video(
                    best_video_result, best_video_task, success_rate
                )
            else:
                logger.warning("No rollout frames available for video logging")
        
        logger.info(
            f"Simulation eval complete - Success rate: {success_rate:.2%}, "
            f"Avg length: {avg_length:.1f}"
        )
        
        return metrics


