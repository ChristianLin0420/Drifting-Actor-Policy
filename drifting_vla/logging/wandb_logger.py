"""
WandB Logger for Drifting-VLA
=============================

Comprehensive logging integration with Weights & Biases for:
- Training metrics and loss curves
- Visualization of drifting dynamics
- Model checkpoints and artifacts
- Evaluation results
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Union
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def generate_run_name(
    model_name: str = "dit",
    tasks: Optional[list[str]] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    extra_tags: Optional[list[str]] = None,
    include_timestamp: bool = True,
) -> str:
    """
    Generate a meaningful run name from configuration.
    
    Format: {model}_{tasks}_{key_params}_{timestamp}
    Example: "dit-b_close-jar+stack-blocks_bs16_lr4e-4_0206-1430"
    
    Args:
        model_name: Model identifier (e.g., "dit_b2", "dit_l2").
        tasks: List of task names.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        hidden_dim: Model hidden dimension.
        num_layers: Number of transformer layers.
        extra_tags: Additional tags to include.
        include_timestamp: Whether to add timestamp suffix.
    
    Returns:
        Formatted run name string.
    """
    parts = []
    
    # Model name (shortened)
    model_short = model_name.replace("dit_", "dit-").replace("_", "-")
    parts.append(model_short)
    
    # Tasks (first 2-3, shortened)
    if tasks:
        task_parts = []
        for task in tasks[:3]:
            # Shorten task names
            short = task.replace("_", "-")
            # Take first part if too long
            if len(short) > 12:
                short = short[:10]
            task_parts.append(short)
        tasks_str = "+".join(task_parts)
        if len(tasks) > 3:
            tasks_str += f"+{len(tasks)-3}more"
        parts.append(tasks_str)
    
    # Key hyperparameters
    params = []
    if batch_size:
        params.append(f"bs{batch_size}")
    if learning_rate:
        # Format: 4e-4 instead of 0.0004
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e-")
        params.append(f"lr{lr_str}")
    if hidden_dim:
        params.append(f"d{hidden_dim}")
    if num_layers:
        params.append(f"L{num_layers}")
    
    if params:
        parts.append("_".join(params))
    
    # Extra tags
    if extra_tags:
        parts.extend(extra_tags[:2])
    
    # Timestamp
    if include_timestamp:
        timestamp = datetime.now().strftime("%m%d-%H%M")
        parts.append(timestamp)
    
    return "_".join(parts)


@dataclass
class LoggerConfig:
    """
    Configuration for WandB logger.
    
    Attributes:
        project: WandB project name.
        entity: WandB entity (team/user).
        run_name: Run name (auto-generated if None).
        tags: List of tags for filtering.
        notes: Run description.
        mode: WandB mode ('online', 'offline', 'disabled').
        save_code: Save code to WandB.
        log_freq: Logging frequency (steps).
        log_images: Whether to log images.
        log_videos: Whether to log videos.
        image_log_freq: Image logging frequency.
    """
    project: str = 'drifting-vla'
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ''
    mode: str = 'online'
    save_code: bool = True
    log_freq: int = 100
    log_images: bool = True
    log_videos: bool = True
    image_log_freq: int = 1000


class WandBLogger:
    """
    Weights & Biases logger for Drifting-VLA training.
    
    Provides structured logging with:
    - Automatic metric tracking
    - Visualization logging
    - Model checkpointing
    - Custom dashboards
    
    Args:
        config: LoggerConfig with WandB settings.
        model_config: Optional model configuration to log.
        training_config: Optional training configuration to log.
    
    Example:
        >>> logger = WandBLogger(LoggerConfig(project='drifting-vla'))
        >>> logger.init()
        >>> for step in range(1000):
        ...     logger.log({'train/loss': loss}, step=step)
        >>> logger.finish()
    """
    
    def __init__(
        self,
        config: Optional[LoggerConfig] = None,
        model_config: Optional[dict] = None,
        training_config: Optional[dict] = None,
    ):
        self.config = config or LoggerConfig()
        self.model_config = model_config
        self.training_config = training_config
        
        self.run = None
        self.step = 0
        self._initialized = False
        self._step_buffer = {}  # Accumulates all data for current step
        self._temp_files = []   # Track temp files for cleanup after flush
    
    def init(self) -> None:
        """Initialize WandB run."""
        try:
            import wandb
            
            # Combine configs
            all_config = {}
            if self.model_config:
                all_config['model'] = self.model_config
            if self.training_config:
                all_config['training'] = self.training_config
            
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.run_name,
                tags=self.config.tags,
                notes=self.config.notes,
                config=all_config,
                mode=self.config.mode,
                save_code=self.config.save_code,
            )
            
            self._initialized = True
            
            # Define custom step metrics so WandB knows all viz/* and sim/*
            # and eval/* keys should be plotted against the global training step.
            # This ensures step sliders work correctly even with multiple
            # wandb.log() calls per step.
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="train/step")
            wandb.define_metric("viz/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="train/step")
            wandb.define_metric("sim/*", step_metric="train/step")
            
            logger.info(f"Initialized WandB run: {self.run.name}")
            
        except ImportError:
            logger.warning("wandb not available, logging disabled")
            self._initialized = False
    
    def log(
        self,
        metrics: dict[str, Any],
        step: Optional[int] = None,
        commit: bool = False,
    ) -> None:
        """
        Buffer metrics for the current step.
        
        All data is accumulated in an internal buffer and flushed
        to WandB in a single wandb.log() call when flush_step() is
        called. This ensures WandB gets all data for a step atomically,
        which is required for proper step sliders on image panels.
        
        Args:
            metrics: Dict of metric names to values.
            step: Global step (updates internal counter if provided).
            commit: If True, flush immediately (default: False = buffer).
        """
        if not self._initialized:
            return
        
        if step is not None:
            self.step = step
        
        # Convert tensors to scalars and add to buffer
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.numpy()
            self._step_buffer[key] = value
        
        if commit:
            self.flush_step()
    
    def flush_step(self) -> None:
        """
        Flush all buffered data in a single wandb.log() call.
        
        This ensures WandB receives ALL metrics, images, and videos
        for this step in one atomic call, which is required for the
        step slider to work correctly on image/media panels.
        
        Also cleans up any temp video files after WandB has consumed them.
        """
        if not self._initialized or not self._step_buffer:
            return
        
        import wandb
        import os
        
        wandb.log(self._step_buffer, step=self.step)
        self._step_buffer = {}  # Clear buffer
        
        # Clean up temp video files (WandB has already copied them)
        for temp_path in self._temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        self._temp_files = []
    
    def log_image(
        self,
        key: str,
        image: Union[np.ndarray, torch.Tensor, 'PIL.Image.Image'],
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log image to WandB.
        
        Args:
            key: Metric name for image.
            image: Image array [H, W, C] or [C, H, W].
            caption: Optional caption.
            step: Global step.
        """
        if not self._initialized or not self.config.log_images:
            return
        
        import wandb
        
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Handle CHW format
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = image.transpose(1, 2, 0)
        
        # Normalize if needed
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        wandb.log(
            {key: wandb.Image(image, caption=caption)},
            step=step or self.step,
        )
    
    def log_video(
        self,
        key: str,
        frames: list[np.ndarray],
        fps: int = 10,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log video to WandB.
        
        Args:
            key: Metric name for video.
            frames: List of frame arrays [H, W, C].
            fps: Frames per second.
            caption: Optional caption.
            step: Global step.
        """
        if not self._initialized or not self.config.log_videos:
            return
        
        import wandb
        import tempfile
        import os
        
        if step is not None:
            self.step = step
        
        import tempfile
        import imageio
        
        # Process frames to uint8 [H, W, C]
        processed_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            processed_frames.append(frame)
        
        # Write video to temp file using imageio (reliable)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name
        
        imageio.mimsave(temp_path, processed_frames, fps=fps)
        self._temp_files.append(temp_path)  # Track for cleanup after flush
        
        self._step_buffer[key] = wandb.Video(
            temp_path, fps=fps, caption=caption, format='mp4'
        )
    
    def log_figure(
        self,
        key: str,
        figure: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Log matplotlib figure to WandB.
        
        Renders the figure to a PNG at fixed 150 DPI before logging
        to avoid oversized image errors.
        
        Args:
            key: Metric name.
            figure: Matplotlib figure.
            step: Global step.
        """
        if not self._initialized:
            return
        
        import wandb
        import io
        from PIL import Image
        
        try:
            # Render figure to PNG buffer at fixed DPI with bounded size
            buf = io.BytesIO()
            figure.savefig(buf, format='png', dpi=150,
                          facecolor=figure.get_facecolor())
            buf.seek(0)
            img = Image.open(buf)
            
            # Safety check: ensure image is not too large
            if img.width > 4096 or img.height > 4096:
                img = img.resize((min(img.width, 4096), min(img.height, 4096)))
            
            wandb.log(
                {key: wandb.Image(img)},
                step=step or self.step,
            )
            buf.close()
        except Exception as e:
            logger.warning(f"Failed to log figure {key}: {e}")
    
    def log_histogram(
        self,
        key: str,
        values: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """
        Log histogram to WandB.
        
        Args:
            key: Metric name.
            values: Array of values.
            step: Global step.
        """
        if not self._initialized:
            return
        
        import wandb
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        wandb.log(
            {key: wandb.Histogram(values)},
            step=step or self.step,
        )
    
    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: Optional[int] = None,
    ) -> None:
        """
        Log table to WandB.
        
        Args:
            key: Table name.
            columns: Column names.
            data: Table data (list of rows).
            step: Global step.
        """
        if not self._initialized:
            return
        
        import wandb
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table}, step=step or self.step)
    
    def save_model(
        self,
        model: torch.nn.Module,
        path: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Save model checkpoint as WandB artifact.
        
        Args:
            model: Model to save.
            path: Local path for checkpoint.
            metadata: Optional artifact metadata.
        """
        if not self._initialized:
            return
        
        import wandb
        
        # Save locally
        torch.save(model.state_dict(), path)
        
        # Create artifact
        artifact = wandb.Artifact(
            name=f'model-{self.run.id}',
            type='model',
            metadata=metadata,
        )
        artifact.add_file(path)
        
        wandb.log_artifact(artifact)
        logger.info(f"Saved model artifact: {path}")
    
    def watch(
        self,
        model: torch.nn.Module,
        log_freq: int = 1000,
        log_graph: bool = True,
    ) -> None:
        """
        Watch model for gradient and parameter logging.
        
        Args:
            model: Model to watch.
            log_freq: Logging frequency.
            log_graph: Log computation graph.
        """
        if not self._initialized:
            return
        
        import wandb
        
        wandb.watch(
            model,
            log='all',
            log_freq=log_freq,
            log_graph=log_graph,
        )
    
    def define_metric(
        self,
        name: str,
        step_metric: str = 'step',
        summary: str = 'last',
    ) -> None:
        """
        Define custom metric with specific aggregation.
        
        Args:
            name: Metric name.
            step_metric: X-axis metric.
            summary: Summary type ('last', 'min', 'max', 'mean').
        """
        if not self._initialized:
            return
        
        import wandb
        
        wandb.define_metric(name, step_metric=step_metric, summary=summary)
    
    def finish(self) -> None:
        """Finish WandB run."""
        if not self._initialized:
            return
        
        import wandb
        
        wandb.finish()
        logger.info("Finished WandB run")
    
    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self.run.id if self.run else None
    
    @property
    def run_url(self) -> Optional[str]:
        """Get URL to current run."""
        return self.run.url if self.run else None


