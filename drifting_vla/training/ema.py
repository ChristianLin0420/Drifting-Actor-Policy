"""
Exponential Moving Average (EMA) for Model Parameters
======================================================

This module implements EMA for model parameters, which is crucial for
stable training of generative models. The EMA model is used for
evaluation and inference.

EMA update rule:
    theta_ema = decay * theta_ema + (1 - decay) * theta

Typical decay values: 0.999, 0.9995, 0.9999
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class EMA(nn.Module):
    """
    Exponential Moving Average wrapper for neural network models.
    
    Maintains an exponential moving average of model parameters that
    can be used for more stable evaluation and inference. The EMA
    model typically produces smoother outputs than the training model.
    
    Args:
        model: The model to track with EMA.
        decay: EMA decay factor. Higher values = slower update.
            Typical values: 0.999, 0.9995, 0.9999. Default: 0.9999
        warmup_steps: Number of steps before starting EMA updates.
            During warmup, EMA parameters are copied directly. Default: 0
        update_after_step: Start EMA updates after this many steps.
            Default: 0
        device: Device to store EMA parameters. If None, same as model.
    
    Example:
        >>> model = DriftingVLA(config)
        >>> ema = EMA(model, decay=0.9999)
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     ema.update()
        >>> 
        >>> # For evaluation
        >>> with ema.average_parameters():
        ...     output = model(test_batch)
    
    Note:
        The EMA model shares the same architecture as the original model.
        Use ema.ema_model for direct access or the context manager
        average_parameters() for temporary parameter swapping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_after_step: int = 0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        
        # Create EMA copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        
        # Move to specified device if provided
        if device is not None:
            self.ema_model.to(device)
        
        # Step counter
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('decay_value', torch.tensor(decay))
    
    @torch.no_grad()
    def update(self) -> None:
        """
        Update EMA parameters with current model parameters.
        
        Should be called after each optimizer step. The update uses
        the formula: ema = decay * ema + (1 - decay) * current
        """
        self.step += 1
        
        if self.step < self.update_after_step:
            # Copy parameters directly during warmup
            self._copy_parameters()
            return
        
        # Compute effective decay (can implement warmup schedule here)
        decay = self._get_decay()
        
        # Update EMA parameters
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            if model_param.requires_grad:
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def _get_decay(self) -> float:
        """
        Compute the current decay value, potentially with warmup schedule.
        
        Returns:
            decay: Current decay factor
        """
        step = self.step.item()
        
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup of decay
            warmup_decay = (step / self.warmup_steps) * self.decay
            return max(warmup_decay, 0.0)
        
        return self.decay
    
    @torch.no_grad()
    def _copy_parameters(self) -> None:
        """Copy current model parameters to EMA model."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.copy_(model_param.data)
    
    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA parameters to another model.
        
        Args:
            model: Target model to copy parameters to
        """
        for ema_param, target_param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            target_param.data.copy_(ema_param.data)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the EMA model."""
        return self.ema_model(*args, **kwargs)
    
    def state_dict(self):
        """Return state dict including EMA parameters and step counter."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'step': self.step,
            'decay': self.decay_value,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict for EMA model."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.step.copy_(state_dict['step'])
        self.decay_value.copy_(state_dict['decay'])
    
    class _AverageParametersContext:
        """Context manager for temporarily using EMA parameters."""
        
        def __init__(self, ema: 'EMA'):
            self.ema = ema
            self.original_params = None
        
        def __enter__(self):
            # Save original parameters
            self.original_params = [
                p.data.clone() for p in self.ema.model.parameters()
            ]
            # Copy EMA parameters to model
            self.ema.copy_to(self.ema.model)
            return self.ema.model
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original parameters
            for orig, param in zip(
                self.original_params, self.ema.model.parameters()
            ):
                param.data.copy_(orig)
            return False
    
    def average_parameters(self) -> '_AverageParametersContext':
        """
        Context manager that temporarily replaces model parameters with EMA.
        
        Useful for evaluation when you want to use the EMA parameters
        without creating a separate model copy.
        
        Example:
            >>> with ema.average_parameters():
            ...     # Inside this block, model uses EMA parameters
            ...     output = model(input)
            >>> # Outside, model uses original training parameters
        
        Returns:
            Context manager
        """
        return self._AverageParametersContext(self)


class MultiEMA(nn.Module):
    """
    Multiple EMA models with different decay values.
    
    Maintains several EMA copies of the model with different decay
    rates. This allows selecting the best EMA model during evaluation
    or ensembling multiple EMA predictions.
    
    Args:
        model: The model to track with EMA.
        decays: List of decay values. Default: [0.999, 0.9995, 0.9999]
        warmup_steps: Warmup steps before EMA updates.
    
    Example:
        >>> model = DriftingVLA(config)
        >>> multi_ema = MultiEMA(model, decays=[0.999, 0.9995, 0.9999])
        >>> 
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     optimizer.step()
        ...     multi_ema.update()
        >>> 
        >>> # Get best EMA based on validation
        >>> best_ema = multi_ema.get_ema(0.9999)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decays: list[float] = [0.999, 0.9995, 0.9999],
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.decays = decays
        
        # Create EMA for each decay value
        self.ema_models = nn.ModuleDict({
            str(decay): EMA(model, decay=decay, warmup_steps=warmup_steps)
            for decay in decays
        })
    
    def update(self) -> None:
        """Update all EMA models."""
        for ema in self.ema_models.values():
            ema.update()
    
    def get_ema(self, decay: float) -> EMA:
        """
        Get EMA model for specific decay value.
        
        Args:
            decay: Decay value to retrieve
        
        Returns:
            EMA model with specified decay
        """
        return self.ema_models[str(decay)]
    
    def state_dict(self) -> dict:
        """Return state dicts for all EMA models."""
        return {
            str(decay): ema.state_dict()
            for decay, ema in self.ema_models.items()
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dicts for all EMA models."""
        for decay, ema in self.ema_models.items():
            if decay in state_dict:
                ema.load_state_dict(state_dict[decay])


