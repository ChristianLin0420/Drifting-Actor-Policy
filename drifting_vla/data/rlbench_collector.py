"""
RLBench Data Collector
======================

Collect expert demonstrations from RLBench using motion planning.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """
    Configuration for RLBench data collection.
    
    Attributes:
        tasks: List of tasks to collect.
        num_episodes: Number of episodes per task.
        image_size: Image resolution.
        cameras: Camera names to record.
        save_depth: Whether to save depth images.
        output_dir: Output directory for HDF5 files.
    """
    tasks: list[str] = None
    num_episodes: int = 100
    image_size: int = 224
    cameras: list[str] = None
    save_depth: bool = False
    output_dir: str = './data/rlbench'
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ['reach_target']
        if self.cameras is None:
            self.cameras = ['front', 'wrist']


class RLBenchCollector:
    """
    Collect expert demonstrations from RLBench.
    
    Uses RLBench's built-in motion planner to generate successful
    demonstrations for various manipulation tasks.
    
    Args:
        config: CollectorConfig with collection settings.
    
    Example:
        >>> config = CollectorConfig(
        ...     tasks=['reach_target', 'pick_up_cup'],
        ...     num_episodes=100,
        ... )
        >>> collector = RLBenchCollector(config)
        >>> collector.collect()
    
    Note:
        Requires RLBench and CoppeliaSim to be properly installed.
        Run inside a container with X11/Xvfb for headless operation.
    """
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RLBench (lazy import)
        self.env = None
        self.task = None
    
    def _init_rlbench(self) -> None:
        """Initialize RLBench environment."""
        try:
            from rlbench.environment import Environment
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import JointVelocity
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.observation_config import ObservationConfig
            
            # Configure observations
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            
            for camera in self.config.cameras:
                if camera == 'front':
                    obs_config.front_camera.set_all(True)
                elif camera == 'wrist':
                    obs_config.wrist_camera.set_all(True)
                elif camera == 'left_shoulder':
                    obs_config.left_shoulder_camera.set_all(True)
                elif camera == 'right_shoulder':
                    obs_config.right_shoulder_camera.set_all(True)
            
            obs_config.front_camera.image_size = (self.config.image_size, self.config.image_size)
            obs_config.wrist_camera.image_size = (self.config.image_size, self.config.image_size)
            
            # Create environment
            action_mode = MoveArmThenGripper(
                arm_action_mode=JointVelocity(),
                gripper_action_mode=Discrete()
            )
            
            self.env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=True,
            )
            self.env.launch()
            
            logger.info("RLBench environment initialized")
            
        except ImportError as e:
            logger.error(f"Failed to import RLBench: {e}")
            raise
    
    def _get_task(self, task_name: str):
        """Get task class from name."""
        from rlbench.tasks import (
            ReachTarget, PickUpCup, StackBlocks, OpenDrawer, CloseJar,
            PutGroceriesInCupboard, PlaceWineAtRackLocation, 
            SweepToDustpan, TurnTap, PushButton,
        )
        
        task_map = {
            'reach_target': ReachTarget,
            'pick_up_cup': PickUpCup,
            'stack_blocks': StackBlocks,
            'open_drawer': OpenDrawer,
            'close_jar': CloseJar,
            'put_groceries_in_cupboard': PutGroceriesInCupboard,
            'place_wine_at_rack_location': PlaceWineAtRackLocation,
            'sweep_to_dustpan': SweepToDustpan,
            'turn_tap': TurnTap,
            'push_button': PushButton,
        }
        
        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}")
        
        return task_map[task_name]
    
    def collect(self) -> dict[str, int]:
        """
        Collect demonstrations for all configured tasks.
        
        Returns:
            Dict mapping task names to number of collected episodes.
        """
        if self.env is None:
            self._init_rlbench()
        
        results = {}
        
        for task_name in self.config.tasks:
            logger.info(f"Collecting {self.config.num_episodes} episodes for {task_name}")
            
            try:
                num_collected = self._collect_task(task_name)
                results[task_name] = num_collected
                logger.info(f"Collected {num_collected} episodes for {task_name}")
            except Exception as e:
                logger.error(f"Failed to collect {task_name}: {e}")
                results[task_name] = 0
        
        return results
    
    def _collect_task(self, task_name: str) -> int:
        """Collect demonstrations for a single task."""
        # Get task
        task_class = self._get_task(task_name)
        task = self.env.get_task(task_class)
        
        # Create output file
        output_path = self.output_dir / f'{task_name}.hdf5'
        
        num_collected = 0
        
        with h5py.File(output_path, 'w') as f:
            # Store metadata
            f.attrs['task'] = task_name
            f.attrs['image_size'] = self.config.image_size
            f.attrs['cameras'] = self.config.cameras
            
            for ep_idx in range(self.config.num_episodes):
                try:
                    # Get demonstration
                    demo = task.get_demos(1, live_demos=True)[0]
                    
                    # Create episode group
                    ep_group = f.create_group(f'episode_{ep_idx}')
                    
                    # Store episode data
                    self._store_episode(ep_group, demo, task_name)
                    
                    num_collected += 1
                    
                    if (ep_idx + 1) % 10 == 0:
                        logger.info(f"  Collected {ep_idx + 1}/{self.config.num_episodes}")
                    
                except Exception as e:
                    logger.warning(f"Failed episode {ep_idx}: {e}")
                    continue
        
        return num_collected
    
    def _store_episode(self, group: h5py.Group, demo, task_name: str) -> None:
        """Store a single episode to HDF5."""
        # Extract observations
        observations = demo._observations
        T = len(observations)
        
        # Create observations group
        obs_group = group.create_group('observations')
        img_group = obs_group.create_group('images')
        
        # Store images
        for camera in self.config.cameras:
            images = []
            for obs in observations:
                if camera == 'front':
                    img = obs.front_rgb
                elif camera == 'wrist':
                    img = obs.wrist_rgb
                elif camera == 'left_shoulder':
                    img = obs.left_shoulder_rgb
                elif camera == 'right_shoulder':
                    img = obs.right_shoulder_rgb
                else:
                    continue
                images.append(img)
            
            if images:
                img_group.create_dataset(
                    camera,
                    data=np.stack(images),
                    compression='gzip',
                )
        
        # Store depth if requested
        if self.config.save_depth:
            depth_group = obs_group.create_group('depth')
            for camera in self.config.cameras:
                depths = []
                for obs in observations:
                    if camera == 'front':
                        d = obs.front_depth
                    elif camera == 'wrist':
                        d = obs.wrist_depth
                    else:
                        continue
                    depths.append(d)
                
                if depths:
                    depth_group.create_dataset(
                        camera,
                        data=np.stack(depths),
                        compression='gzip',
                    )
        
        # Store proprioception
        proprio = np.array([
            np.concatenate([obs.joint_positions, obs.gripper_open])
            if hasattr(obs, 'gripper_open') and obs.gripper_open is not None
            else obs.joint_positions
            for obs in observations
        ])
        obs_group.create_dataset('proprio', data=proprio)
        
        # Store actions (gripper pose deltas + gripper action)
        actions = []
        for i in range(T - 1):
            obs = observations[i]
            next_obs = observations[i + 1]
            
            # Compute pose delta
            pos_delta = next_obs.gripper_pose[:3] - obs.gripper_pose[:3]
            quat_delta = next_obs.gripper_pose[3:] - obs.gripper_pose[3:]
            gripper = np.array([1.0 if next_obs.gripper_open else 0.0])
            
            action = np.concatenate([pos_delta, quat_delta, gripper])
            actions.append(action)
        
        # Pad last action
        if actions:
            actions.append(actions[-1])
        else:
            actions = [np.zeros(8)]  # pos(3) + quat(4) + gripper(1)
        
        group.create_dataset('actions', data=np.array(actions))
        
        # Store language description
        descriptions = demo.variation_descriptions()
        language = descriptions[0] if descriptions else task_name.replace('_', ' ')
        group.create_dataset('language', data=language)
        
        # Store task name
        group.create_dataset('task', data=task_name)
    
    def close(self) -> None:
        """Close RLBench environment."""
        if self.env is not None:
            self.env.shutdown()
            self.env = None


