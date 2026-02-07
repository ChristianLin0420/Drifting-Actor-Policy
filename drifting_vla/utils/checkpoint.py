"""
Checkpointing Utilities
=======================

Functions for saving and loading model checkpoints.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Any
import logging
import shutil

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    path: str = 'checkpoint.pt',
    config: Optional[Any] = None,
    additional_state: Optional[dict] = None,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save (can be wrapped in DDP/FSDP).
        optimizer: Optional optimizer state.
        scheduler: Optional LR scheduler state.
        ema: Optional EMA model state.
        step: Current training step.
        epoch: Current epoch.
        path: Save path.
        config: Optional config to save.
        additional_state: Additional state dict entries.
    """
    # Get unwrapped model
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'step': step,
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if additional_state:
        checkpoint.update(additional_state)
    
    # Save to temporary file first, then move (atomic)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_path = path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    shutil.move(str(temp_path), str(path))
    
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    strict: bool = True,
    map_location: str = 'cpu',
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path.
        model: Model to load state into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        ema: Optional EMA to restore.
        strict: Whether to strictly enforce state dict matching.
        map_location: Device mapping for loading.
    
    Returns:
        Checkpoint dict with metadata (step, epoch, config, etc.)
    """
    checkpoint = torch.load(path, map_location=map_location)
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load EMA state
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
    
    logger.info(f"Loaded checkpoint from {path} (step={checkpoint.get('step', 0)})")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """
    Get the latest checkpoint from a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints.
    
    Returns:
        Path to latest checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for numbered checkpoints
    checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
    
    if not checkpoints:
        # Try other patterns
        checkpoints = list(checkpoint_dir.glob('*.pt'))
    
    if not checkpoints:
        return None
    
    # Sort by step number or modification time
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except ValueError:
            return int(p.stat().st_mtime)
    
    latest = max(checkpoints, key=get_step)
    return latest


def export_for_inference(
    checkpoint_path: str,
    output_path: str,
    include_ema: bool = True,
) -> None:
    """
    Export checkpoint for inference (strip optimizer/scheduler state).
    
    Args:
        checkpoint_path: Source checkpoint path.
        output_path: Output path for inference checkpoint.
        include_ema: Whether to use EMA weights if available.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    inference_checkpoint = {
        'config': checkpoint.get('config'),
        'step': checkpoint.get('step', 0),
    }
    
    if include_ema and 'ema_state_dict' in checkpoint:
        # Use EMA weights for inference
        inference_checkpoint['model_state_dict'] = checkpoint['ema_state_dict']['ema_model']
        logger.info("Using EMA weights for inference")
    else:
        inference_checkpoint['model_state_dict'] = checkpoint['model_state_dict']
    
    torch.save(inference_checkpoint, output_path)
    logger.info(f"Exported inference checkpoint to {output_path}")


