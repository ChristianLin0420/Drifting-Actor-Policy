"""
Base Environment Interface
==========================

Abstract base class for robot manipulation environments.
Provides a consistent interface across RLBench, LIBERO, and CALVIN.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Any
from abc import ABC, abstractmethod


@dataclass
class EnvConfig:
    """
    Configuration for environment.
    
    Attributes:
        task: Task name or list of tasks.
        image_size: Image resolution.
        cameras: Camera names to use.
        action_mode: Action space type ('delta', 'absolute').
        max_episode_length: Maximum steps per episode.
        headless: Run without rendering window.
    """
    task: Union[str, list[str]] = 'reach_target'
    image_size: int = 224
    cameras: list[str] = None
    action_mode: str = 'delta'
    max_episode_length: int = 200
    headless: bool = True
    
    def __post_init__(self):
        if self.cameras is None:
            self.cameras = ['front']


@dataclass
class EnvObservation:
    """
    Observation from environment.
    
    Attributes:
        images: Visual observations [C, H, W] or [V, C, H, W].
        proprio: Proprioceptive state (joint positions, etc.).
        language: Task description.
        success: Whether task is completed.
        done: Whether episode is done.
        info: Additional information dict.
    """
    images: np.ndarray
    proprio: np.ndarray
    language: str
    success: bool = False
    done: bool = False
    info: dict = None
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}
    
    def to_tensor(self, device: torch.device = None) -> dict:
        """Convert to tensor dict."""
        result = {
            'images': torch.from_numpy(self.images).float(),
            'proprio': torch.from_numpy(self.proprio).float(),
            'language': self.language,
        }
        
        if device is not None:
            result['images'] = result['images'].to(device)
            result['proprio'] = result['proprio'].to(device)
        
        return result


class BaseEnvironment(ABC):
    """
    Abstract base class for manipulation environments.
    
    Subclasses must implement:
    - reset(): Reset environment and return observation
    - step(action): Execute action and return observation
    - close(): Clean up environment resources
    
    Optional methods:
    - get_task_names(): Return available tasks
    - render(): Return rendered image
    """
    
    def __init__(self, config: EnvConfig):
        """
        Initialize environment.
        
        Args:
            config: Environment configuration.
        """
        self.config = config
        self.episode_length = 0
        self.current_task = None
    
    @abstractmethod
    def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnvObservation:
        """
        Reset environment to initial state.
        
        Args:
            task: Specific task to reset to (None = random/default).
            seed: Random seed for reproducibility.
        
        Returns:
            Initial observation.
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> EnvObservation:
        """
        Execute action in environment.
        
        Args:
            action: Action array [D_a].
        
        Returns:
            Observation after action.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_task_names(self) -> list[str]:
        """Return list of available task names."""
        if isinstance(self.config.task, list):
            return self.config.task
        return [self.config.task]
    
    def get_action_dim(self) -> int:
        """Return action dimensionality."""
        return 7  # Default: 3 pos + 3 rot + 1 gripper
    
    def get_proprio_dim(self) -> int:
        """Return proprioception dimensionality."""
        return 8  # Default: 7 joints + gripper
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """
        Render current environment state.
        
        Args:
            mode: Render mode ('rgb_array', 'human').
        
        Returns:
            Rendered image if mode='rgb_array'.
        """
        return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class DummyEnvironment(BaseEnvironment):
    """
    Dummy environment for testing without simulation.
    
    Returns random observations and always succeeds after
    a fixed number of steps. Renders synthetic frames for video testing.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.success_at_step = 10
        self._last_action = None
        self._trajectory = []
    
    def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnvObservation:
        """Reset with random initial observation."""
        if seed is not None:
            np.random.seed(seed)
        
        self.episode_length = 0
        self.current_task = task or self.config.task
        self._last_action = None
        self._trajectory = [np.array([0.5, 0.5, 0.5])]  # Start position
        
        return EnvObservation(
            images=np.random.rand(3, self.config.image_size, self.config.image_size).astype(np.float32),
            proprio=np.random.rand(self.get_proprio_dim()).astype(np.float32),
            language=f"complete the {self.current_task} task",
            success=False,
            done=False,
        )
    
    def step(self, action: np.ndarray) -> EnvObservation:
        """Step with random observation."""
        self.episode_length += 1
        self._last_action = action
        
        # Simulate trajectory movement
        if len(action) >= 3:
            new_pos = self._trajectory[-1] + action[:3] * 0.1
            new_pos = np.clip(new_pos, 0, 1)
            self._trajectory.append(new_pos)
        
        success = self.episode_length >= self.success_at_step
        done = success or self.episode_length >= self.config.max_episode_length
        
        return EnvObservation(
            images=np.random.rand(3, self.config.image_size, self.config.image_size).astype(np.float32),
            proprio=np.random.rand(self.get_proprio_dim()).astype(np.float32),
            language=f"complete the {self.current_task} task",
            success=success,
            done=done,
        )
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """
        Render a synthetic visualization frame.
        
        Creates a simple visualization showing:
        - Task name
        - Current step
        - Trajectory so far
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
        
        # Left: Simulated camera view (noise with overlays)
        ax1 = axes[0]
        img = np.random.rand(self.config.image_size, self.config.image_size, 3) * 0.3 + 0.2
        ax1.imshow(img)
        ax1.set_title(f'Task: {self.current_task}', fontsize=12, fontweight='bold')
        ax1.text(
            0.5, 0.9, f'Step: {self.episode_length}',
            transform=ax1.transAxes, ha='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        ax1.axis('off')
        
        # Right: Trajectory visualization
        ax2 = axes[1]
        if len(self._trajectory) > 0:
            traj = np.array(self._trajectory)
            colors = plt.cm.viridis(np.linspace(0, 1, len(traj)))
            
            # Plot trajectory
            for i in range(len(traj) - 1):
                ax2.plot(
                    traj[i:i+2, 0], traj[i:i+2, 1],
                    color=colors[i], linewidth=2
                )
            
            # Current position
            ax2.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, zorder=10, marker='o')
            ax2.scatter(traj[0, 0], traj[0, 1], c='green', s=100, zorder=10, marker='s')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Simulated Trajectory', fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Convert to numpy array (layout='constrained' at figure creation)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return frame
    
    def close(self) -> None:
        """No cleanup needed."""
        pass


