"""
Optimizer and Learning Rate Scheduler Configuration
====================================================

This module provides utilities for creating optimizers and learning rate
schedulers for Drifting-VLA training, following best practices from
large-scale model training.

Key features:
- AdamW optimizer with weight decay
- Learning rate warmup
- Cosine annealing schedule
- Layer-wise learning rate decay (optional)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)
from typing import Optional, Iterable
import math


def create_optimizer(
    model: nn.Module,
    lr: float = 4e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    exclude_bias_and_norm: bool = True,
    layer_decay: Optional[float] = None,
) -> AdamW:
    """
    Create AdamW optimizer with proper weight decay configuration.
    
    Following best practices, bias terms and normalization layers are
    excluded from weight decay regularization.
    
    Args:
        model: Model to optimize.
        lr: Base learning rate. Default: 4e-4
        weight_decay: Weight decay coefficient. Default: 0.05
        betas: Adam beta coefficients. Default: (0.9, 0.95)
        eps: Adam epsilon for numerical stability. Default: 1e-8
        exclude_bias_and_norm: If True, exclude bias and norm params
            from weight decay. Default: True
        layer_decay: If provided, apply layer-wise learning rate decay.
            Value should be < 1.0 (e.g., 0.75). Default: None
    
    Returns:
        AdamW optimizer
    
    Example:
        >>> optimizer = create_optimizer(model, lr=4e-4, weight_decay=0.05)
    """
    if layer_decay is not None:
        param_groups = get_layer_decay_param_groups(
            model, lr, weight_decay, layer_decay, exclude_bias_and_norm
        )
    elif exclude_bias_and_norm:
        param_groups = get_param_groups_with_decay(model, weight_decay)
    else:
        param_groups = [{'params': model.parameters(), 'weight_decay': weight_decay}]
    
    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
    
    return optimizer


def get_param_groups_with_decay(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Separate parameters into groups with and without weight decay.
    
    Bias terms and normalization layer parameters are placed in the
    no-decay group to prevent them from being regularized.
    
    Args:
        model: Model to get parameters from.
        weight_decay: Weight decay for non-excluded parameters.
    
    Returns:
        List of parameter group dicts
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should be excluded from weight decay
        if _should_exclude_from_decay(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    return param_groups


def _should_exclude_from_decay(name: str, param: torch.Tensor) -> bool:
    """
    Check if a parameter should be excluded from weight decay.
    
    Excludes:
    - Bias terms
    - LayerNorm/GroupNorm/BatchNorm parameters
    - Embedding layers
    - 1D parameters (usually biases or norms)
    """
    # Bias terms
    if 'bias' in name:
        return True
    
    # Normalization layers
    norm_keywords = ['norm', 'ln', 'layernorm', 'groupnorm', 'batchnorm']
    if any(kw in name.lower() for kw in norm_keywords):
        return True
    
    # Embedding layers
    if 'embed' in name.lower():
        return True
    
    # 1D parameters (typically biases or norms)
    if param.ndim == 1:
        return True
    
    return False


def get_layer_decay_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    exclude_bias_and_norm: bool = True,
) -> list[dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Applies exponentially decaying learning rates from the output
    layer to the input layer. This is particularly useful for
    fine-tuning pre-trained models.
    
    Args:
        model: Model to get parameters from.
        base_lr: Base learning rate (applied to output layer).
        weight_decay: Weight decay coefficient.
        layer_decay: Decay factor per layer (e.g., 0.75).
        exclude_bias_and_norm: Exclude bias/norm from weight decay.
    
    Returns:
        List of parameter group dicts with layer-specific LRs
    """
    # Get layer depths
    layer_depths = _get_layer_depths(model)
    max_depth = max(layer_depths.values()) if layer_depths else 0
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Get layer depth
        depth = layer_depths.get(name, max_depth)
        
        # Compute layer-specific learning rate
        lr = base_lr * (layer_decay ** (max_depth - depth))
        
        # Determine weight decay
        if exclude_bias_and_norm and _should_exclude_from_decay(name, param):
            wd = 0.0
        else:
            wd = weight_decay
        
        param_groups.append({
            'params': [param],
            'lr': lr,
            'weight_decay': wd,
            'name': name,
        })
    
    return param_groups


def _get_layer_depths(model: nn.Module) -> dict[str, int]:
    """
    Infer layer depths from parameter names.
    
    Looks for patterns like 'layer.0', 'blocks.5', etc. to determine
    the depth of each parameter in the network.
    
    Returns:
        Dict mapping parameter names to their layer depths
    """
    import re
    
    depths = {}
    for name, _ in model.named_parameters():
        # Look for layer indices in the name
        matches = re.findall(r'\.(\d+)\.', name)
        if matches:
            # Use the first index found as the depth
            depths[name] = int(matches[0])
        else:
            depths[name] = 0
    
    return depths


def create_scheduler(
    optimizer: AdamW,
    num_training_steps: int,
    num_warmup_steps: int = 1000,
    scheduler_type: str = 'cosine',
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """
    Create learning rate scheduler with warmup.
    
    Implements warmup followed by the specified decay schedule:
    - Linear warmup for num_warmup_steps
    - Cosine/linear decay to min_lr for remaining steps
    
    Args:
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps.
        num_warmup_steps: Number of warmup steps. Default: 1000
        scheduler_type: Type of decay ('cosine', 'linear', 'constant').
            Default: 'cosine'
        min_lr_ratio: Minimum LR as ratio of base LR. Default: 0.1
    
    Returns:
        Learning rate scheduler
    
    Example:
        >>> scheduler = create_scheduler(optimizer, 100000, warmup_steps=5000)
        >>> for batch in dataloader:
        ...     optimizer.step()
        ...     scheduler.step()
    """
    # Warmup scheduler (linear 0 -> 1)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    
    # Main decay scheduler
    num_decay_steps = num_training_steps - num_warmup_steps
    
    if scheduler_type == 'cosine':
        # Cosine annealing to min_lr
        decay_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_decay_steps,
            eta_min=optimizer.defaults['lr'] * min_lr_ratio,
        )
    elif scheduler_type == 'linear':
        # Linear decay to min_lr
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_decay_steps,
        )
    elif scheduler_type == 'constant':
        # Constant LR after warmup
        decay_scheduler = LambdaLR(optimizer, lambda step: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Combine warmup and decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )
    
    return scheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Custom scheduler with linear warmup and cosine decay.
    
    More flexible than SequentialLR for checkpointing and resumption.
    
    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR ratio. Default: 0.0
        last_epoch: Last epoch (for resumption). Default: -1
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list[float]:
        """Compute current learning rates for all parameter groups."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            return [base_lr * factor for base_lr in self.base_lrs]


