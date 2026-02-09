"""
RLBench Environment Wrapper
===========================

Wrapper for RLBench providing a consistent interface for
Drifting-VLA training and evaluation.
"""

import numpy as np
from typing import Optional
import logging

from drifting_vla.envs.base_env import BaseEnvironment, EnvConfig, EnvObservation

logger = logging.getLogger(__name__)


class RLBenchEnvironment(BaseEnvironment):
    """
    RLBench environment wrapper.
    
    Provides a standardized interface for interacting with RLBench tasks.
    
    Args:
        config: EnvConfig with environment settings.
    
    Example:
        >>> config = EnvConfig(task='reach_target', image_size=224)
        >>> env = RLBenchEnvironment(config)
        >>> obs = env.reset()
        >>> for _ in range(100):
        ...     action = policy(obs.images, obs.language)
        ...     obs = env.step(action)
        ...     if obs.done:
        ...         break
        >>> env.close()
    
    Note:
        Requires RLBench and CoppeliaSim to be installed.
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        
        self.env = None
        self.task = None
        self.task_descriptions = {}
        self._last_front_rgb = None  # Store COPIED front camera frame for render()
        
        # Initialize RLBench
        self._init_rlbench()
    
    def _init_rlbench(self) -> None:
        """Initialize RLBench environment."""
        try:
            from rlbench.environment import Environment
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.observation_config import ObservationConfig
            
            # Configure observations
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            # Always enable proprioceptive state
            obs_config.joint_positions = True
            obs_config.joint_velocities = True
            obs_config.gripper_open = True
            obs_config.gripper_pose = True
            
            for camera in self.config.cameras:
                if camera == 'front':
                    obs_config.front_camera.set_all(True)
                    obs_config.front_camera.image_size = (
                        self.config.image_size,
                        self.config.image_size
                    )
                elif camera == 'wrist':
                    obs_config.wrist_camera.set_all(True)
                    obs_config.wrist_camera.image_size = (
                        self.config.image_size,
                        self.config.image_size
                    )
                elif camera == 'left_shoulder':
                    obs_config.left_shoulder_camera.set_all(True)
                elif camera == 'right_shoulder':
                    obs_config.right_shoulder_camera.set_all(True)
            
            # Configure action mode
            if self.config.action_mode == 'delta':
                action_mode = MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=False),
                    gripper_action_mode=Discrete()
                )
            else:
                action_mode = MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
                    gripper_action_mode=Discrete()
                )
            
            # Create environment
            self.env = Environment(
                action_mode=action_mode,
                obs_config=obs_config,
                headless=self.config.headless,
            )
            self.env.launch()
            
            logger.info("RLBench environment initialized")
            
        except ImportError as e:
            logger.warning(f"RLBench not available: {e}")
            self.env = None
        except Exception as e:
            logger.error(f"Failed to initialize RLBench: {e}")
            self.env = None
    
    def _get_task_class(self, task_name: str):
        """
        Get RLBench task class from name.
        
        Dynamically imports tasks to handle version differences in RLBench.
        
        Args:
            task_name: Snake-case task name (e.g. 'close_jar').
        
        Returns:
            RLBench task class.
        
        Raises:
            ValueError: If task is not found.
        """
        import importlib
        
        # Build task map dynamically to handle missing tasks across versions
        task_candidates = {
            'reach_target': 'ReachTarget',
            'pick_up_cup': 'PickUpCup',
            'stack_blocks': 'StackBlocks',
            'open_drawer': 'OpenDrawer',
            'close_jar': 'CloseJar',
            'put_groceries_in_cupboard': 'PutGroceriesInCupboard',
            'place_wine_at_rack_location': 'PlaceWineAtRackLocation',
            'sweep_to_dustpan': 'SweepToDustpan',
            'turn_tap': 'TurnTap',
            'push_button': 'PushButton',
            'slide_block_to_target': 'SlideBlockToTarget',
            'put_item_in_drawer': 'PutItemInDrawer',
            'place_cups': 'PlaceCups',
            'place_shape_in_shape_sorter': 'PlaceShapeInShapeSorter',
            'push_buttons': 'PushButtons',
            'stack_wine': 'StackWine',
            'insert_onto_square_peg': 'InsertOntoSquarePeg',
            'meat_off_grill': 'MeatOffGrill',
        }
        
        if task_name not in task_candidates:
            available = ', '.join(task_candidates.keys())
            raise ValueError(f"Unknown task: {task_name}. Available: {available}")
        
        class_name = task_candidates[task_name]
        
        try:
            module = importlib.import_module('rlbench.tasks')
            task_class = getattr(module, class_name)
            return task_class
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Task '{task_name}' (class {class_name}) not available in "
                f"this RLBench version: {e}"
            )
    
    def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EnvObservation:
        """
        Reset environment with new task instance.
        
        Args:
            task: Task name (None = use configured task).
            seed: Random seed for task variation.
        
        Returns:
            Initial observation.
        """
        if self.env is None:
            # Return dummy observation if RLBench not available
            return self._dummy_observation(task)
        
        # Set task
        task_name = task or self.config.task
        if isinstance(task_name, list):
            task_name = np.random.choice(task_name)
        
        # Get task if different from current
        if task_name != self.current_task:
            task_class = self._get_task_class(task_name)
            self.task = self.env.get_task(task_class)
            self.current_task = task_name
        
        # Reset task — returns (descriptions_list, observation)
        descriptions, obs = self.task.reset()
        # COPY the image — RLBench reuses the same buffer!
        if hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
            self._last_front_rgb = obs.front_rgb.copy()
        else:
            self._last_front_rgb = None
        
        # Cache language description from reset output
        if descriptions:
            self.task_descriptions[task_name] = descriptions[0]
        else:
            self.task_descriptions[task_name] = task_name.replace('_', ' ')
        
        self.episode_length = 0
        
        return self._make_observation(obs, success=False, done=False)
    
    def step(self, action: np.ndarray) -> EnvObservation:
        """
        Execute action in environment.
        
        Sanitizes actions before sending to RLBench:
        - Normalizes quaternion to unit length
        - Clips position to reasonable range
        - Binarizes gripper (open/close)
        
        Action format: [x, y, z, qx, qy, qz, qw, gripper] (8-dim)
        - Position: absolute world coordinates
        - Quaternion: [qx, qy, qz, qw] (Hamilton), must be unit-normalized
        - Gripper: 0.0=close, 1.0=open (Discrete mode)
        
        Args:
            action: Raw action array [8] = [pos(3), quat(4), gripper(1)].
        
        Returns:
            Observation after action.
        """
        if self.env is None:
            return self._dummy_observation(self.current_task, done=True)
        
        self.episode_length += 1
        
        # Sanitize action for RLBench compatibility
        action = self._sanitize_action(action)
        
        # Log first action for debugging
        if self.episode_length == 1:
            logger.info(
                f"Sim action: pos=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}] "
                f"quat=[{action[3]:.3f},{action[4]:.3f},{action[5]:.3f},{action[6]:.3f}] "
                f"grip={action[7]:.0f}"
            )
        
        try:
            obs, reward, done = self.task.step(action)
            success = reward > 0
            # COPY the image — RLBench reuses the same buffer!
            if hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
                self._last_front_rgb = obs.front_rgb.copy()
            if self.episode_length <= 3:
                logger.info(f"  Step {self.episode_length}: action executed OK, reward={reward}")
        except Exception as e:
            # Path planning failures — robot stays in place
            if "path could not be found" in str(e).lower():
                logger.info(f"  Step {self.episode_length}: PATH PLANNING FAILED — robot didn't move")
                obs = self._get_current_obs()
                if obs is not None and hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
                    self._last_front_rgb = obs.front_rgb.copy()
                success = False
                done = False
            else:
                logger.info(f"  Step {self.episode_length}: FAILED — {e}")
                success = False
                done = True
                obs = self._get_current_obs()
                if obs is not None and hasattr(obs, 'front_rgb') and obs.front_rgb is not None:
                    self._last_front_rgb = obs.front_rgb.copy()
        
        # Check max episode length
        if self.episode_length >= self.config.max_episode_length:
            done = True
        
        return self._make_observation(obs, success=success, done=done)
    
    def _make_observation(
        self,
        obs,
        success: bool,
        done: bool,
    ) -> EnvObservation:
        """Convert RLBench observation to EnvObservation."""
        # Extract images
        images = []
        for camera in self.config.cameras:
            if camera == 'front':
                img = obs.front_rgb
            elif camera == 'wrist':
                img = obs.wrist_rgb
            elif camera == 'left_shoulder':
                img = obs.left_shoulder_rgb
            elif camera == 'right_shoulder':
                img = obs.right_shoulder_rgb
            else:
                img = np.zeros((self.config.image_size, self.config.image_size, 3))
            
            # Convert to CHW and normalize
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(img)
        
        if len(images) == 1:
            images = images[0]
        else:
            images = np.stack(images, axis=0)
        
        # Extract proprioception (handle None/missing fields gracefully)
        joint_pos = obs.joint_positions if obs.joint_positions is not None else np.zeros(7)
        joint_pos = np.atleast_1d(np.asarray(joint_pos, dtype=np.float32))
        
        gripper_val = 0.0
        if obs.gripper_open is not None:
            gripper_val = float(np.asarray(obs.gripper_open).item()) if np.asarray(obs.gripper_open).ndim == 0 else float(obs.gripper_open)
        
        proprio = np.concatenate([joint_pos, [gripper_val]]).astype(np.float32)
        
        # Get language
        language = self.task_descriptions.get(
            self.current_task,
            f"complete the {self.current_task} task"
        )
        
        return EnvObservation(
            images=images,
            proprio=proprio,
            language=language,
            success=success,
            done=done,
            info={
                'task': self.current_task,
                'episode_length': self.episode_length,
            }
        )
    
    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Sanitize raw model output into a valid RLBench action.
        
        RLBench expects: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]
        where quaternion must be unit-length and gripper is 0.0 or 1.0.
        
        This ensures even random/untrained model outputs produce valid actions
        that the simulator can execute.
        
        Args:
            action: Raw action array [8] = [pos(3), quat(4), gripper(1)].
        
        Returns:
            Sanitized action array [8].
        """
        action = np.array(action, dtype=np.float64).copy()
        
        # Replace NaN/Inf with zeros
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Clip position to reasonable workspace bounds
        action[:3] = np.clip(action[:3], -2.0, 2.0)
        
        # Normalize quaternion to unit length (indices 3:7)
        quat = action[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-6:
            # Degenerate quaternion — use identity [1, 0, 0, 0]
            action[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            action[3:7] = quat / quat_norm
        
        # Binarize gripper (RLBench Discrete mode: 0.0=close, 1.0=open)
        action[7] = 1.0 if action[7] > 0.5 else 0.0
        
        return action
    
    def _get_current_obs(self):
        """Get current observation (for error recovery)."""
        try:
            return self.env._scene.get_observation()
        except:
            return None
    
    def _dummy_observation(
        self,
        task: Optional[str] = None,
        done: bool = False,
    ) -> EnvObservation:
        """Create dummy observation when RLBench not available."""
        return EnvObservation(
            images=np.zeros((3, self.config.image_size, self.config.image_size), dtype=np.float32),
            proprio=np.zeros(8, dtype=np.float32),
            language=f"complete the {task or 'unknown'} task",
            success=False,
            done=done,
        )
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """
        Render current state as RGB image.
        
        Returns a COPY of the front camera frame (RLBench reuses
        internal buffers, so we must copy on capture).
        
        Returns:
            RGB image [H, W, 3] as uint8, or None if unavailable.
        """
        if self._last_front_rgb is not None:
            return self._last_front_rgb  # Already a copy
        
        return None
    
    def get_action_dim(self) -> int:
        """Get action dimensionality."""
        return 8  # 3 pos + 4 quat + 1 gripper
    
    def close(self) -> None:
        """Shutdown RLBench environment."""
        if self.env is not None:
            try:
                self.env.shutdown()
            except:
                pass
            self.env = None
        
        logger.info("RLBench environment closed")


