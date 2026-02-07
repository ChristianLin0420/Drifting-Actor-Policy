"""
Environment Rollout Utilities
=============================

Functions for evaluating policies in environments.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
import logging

from drifting_vla.envs.base_env import BaseEnvironment, EnvObservation
from drifting_vla.inference.policy import DriftingVLAPolicy

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """
    Configuration for rollout.
    
    Attributes:
        max_steps: Maximum steps per episode.
        save_video: Whether to save video of rollout.
        video_fps: Frames per second for video.
        verbose: Print progress during rollout.
    """
    max_steps: int = 200
    save_video: bool = False
    video_fps: int = 10
    verbose: bool = False


@dataclass
class RolloutResult:
    """
    Result of a single rollout.
    
    Attributes:
        success: Whether task was completed.
        episode_length: Number of steps taken.
        total_reward: Cumulative reward.
        observations: List of observations.
        actions: List of actions taken.
        frames: List of rendered frames (if save_video).
        info: Additional information.
    """
    success: bool
    episode_length: int
    total_reward: float = 0.0
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def rollout_episode(
    env: BaseEnvironment,
    policy: DriftingVLAPolicy,
    config: Optional[RolloutConfig] = None,
    task: Optional[str] = None,
    seed: Optional[int] = None,
) -> RolloutResult:
    """
    Execute a single rollout episode.
    
    Args:
        env: Environment to rollout in.
        policy: Policy for action generation.
        config: RolloutConfig with settings.
        task: Specific task to evaluate.
        seed: Random seed for reproducibility.
    
    Returns:
        RolloutResult with episode data.
    
    Example:
        >>> env = RLBenchEnvironment(env_config)
        >>> policy = DriftingVLAPolicy.from_checkpoint('model.pt')
        >>> result = rollout_episode(env, policy, task='reach_target')
        >>> print(f"Success: {result.success}, Length: {result.episode_length}")
    """
    config = config or RolloutConfig()
    
    # Reset environment and policy
    obs = env.reset(task=task, seed=seed)
    policy.reset()
    
    result = RolloutResult(
        success=False,
        episode_length=0,
    )
    
    # Collect initial observation
    result.observations.append(obs)
    
    if config.save_video:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            result.frames.append(frame)
    
    # Rollout loop
    for step in range(config.max_steps):
        # Get action from policy
        action = policy.get_action(obs.images, obs.language, obs.proprio)
        result.actions.append(action)
        
        # Execute action
        obs = env.step(action)
        result.observations.append(obs)
        result.episode_length = step + 1
        
        # Save frame
        if config.save_video:
            frame = env.render(mode='rgb_array')
            if frame is not None:
                result.frames.append(frame)
        
        # Check termination
        if obs.done:
            result.success = obs.success
            break
        
        if config.verbose and (step + 1) % 10 == 0:
            logger.info(f"Step {step + 1}/{config.max_steps}")
    
    result.info = {
        'task': task or env.current_task,
        'seed': seed,
    }
    
    return result


def evaluate_policy(
    env: BaseEnvironment,
    policy: DriftingVLAPolicy,
    tasks: Optional[list[str]] = None,
    num_episodes: int = 10,
    config: Optional[RolloutConfig] = None,
) -> dict:
    """
    Evaluate policy across multiple episodes and tasks.
    
    Args:
        env: Environment for evaluation.
        policy: Policy to evaluate.
        tasks: List of tasks to evaluate (None = env default).
        num_episodes: Episodes per task.
        config: Rollout configuration.
    
    Returns:
        Dict with evaluation metrics:
        - success_rate: Overall success rate
        - per_task_success: Success rate per task
        - avg_episode_length: Average episode length
        - results: List of all RolloutResults
    """
    config = config or RolloutConfig()
    tasks = tasks or env.get_task_names()
    
    all_results = []
    per_task_success = {}
    
    for task in tasks:
        task_successes = []
        
        for ep in range(num_episodes):
            result = rollout_episode(
                env=env,
                policy=policy,
                config=config,
                task=task,
                seed=ep,
            )
            
            all_results.append(result)
            task_successes.append(float(result.success))
            
            logger.info(
                f"Task: {task}, Episode: {ep+1}/{num_episodes}, "
                f"Success: {result.success}, Length: {result.episode_length}"
            )
        
        per_task_success[task] = np.mean(task_successes)
    
    # Compute overall metrics
    success_rate = np.mean([r.success for r in all_results])
    avg_length = np.mean([r.episode_length for r in all_results])
    
    return {
        'success_rate': success_rate,
        'per_task_success': per_task_success,
        'avg_episode_length': avg_length,
        'results': all_results,
    }


def save_rollout_video(
    result: RolloutResult,
    output_path: str,
    fps: int = 10,
) -> None:
    """
    Save rollout frames as video.
    
    Args:
        result: RolloutResult with frames.
        output_path: Output video path.
        fps: Frames per second.
    """
    if not result.frames:
        logger.warning("No frames to save")
        return
    
    try:
        import imageio
        
        imageio.mimsave(
            output_path,
            result.frames,
            fps=fps,
        )
        
        logger.info(f"Saved rollout video to {output_path}")
        
    except ImportError:
        logger.warning("imageio not available, cannot save video")


