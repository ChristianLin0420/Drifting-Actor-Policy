"""
Sample Queue for Drifting Loss
==============================

Maintains a queue of positive samples per task for computing the
drifting loss. This enables efficient sampling of positive examples
during training.
"""

import torch
import numpy as np
from typing import Optional
from collections import defaultdict
import threading
import logging

logger = logging.getLogger(__name__)


class SampleQueue:
    """
    Per-task queue of positive samples for drifting loss.
    
    Maintains a FIFO queue of recent expert actions for each task.
    This provides a diverse set of positive samples for computing
    the attraction term in the drifting field.
    
    Args:
        queue_size: Maximum samples per task.
        num_tasks: Number of tasks (for pre-allocation).
        action_dim: Dimension of flattened action vectors.
        device: Device to store samples on.
    
    Example:
        >>> queue = SampleQueue(queue_size=128, action_dim=160)
        >>> 
        >>> # Add samples during training
        >>> queue.add(task_id=0, actions=expert_actions)
        >>> 
        >>> # Sample for drifting loss
        >>> pos_samples = queue.sample(n=32, task_ids=[0, 1, 2])
    """
    
    def __init__(
        self,
        queue_size: int = 128,
        num_tasks: Optional[int] = None,
        action_dim: int = 160,
        device: torch.device = torch.device('cpu'),
    ):
        self.queue_size = queue_size
        self.action_dim = action_dim
        self.device = device
        
        # Per-task queues
        self.queues = defaultdict(list)
        self.queue_indices = defaultdict(int)
        
        # Pre-allocated storage (if num_tasks known)
        if num_tasks is not None:
            self.storage = torch.zeros(
                num_tasks, queue_size, action_dim,
                device=device
            )
            self.counts = torch.zeros(num_tasks, dtype=torch.long, device=device)
            self.use_storage = True
        else:
            self.storage = None
            self.counts = None
            self.use_storage = False
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.total_added = 0
    
    def add(
        self,
        task_id: int,
        actions: torch.Tensor,
    ) -> None:
        """
        Add samples to the queue for a task.
        
        Args:
            task_id: Task identifier.
            actions: Action samples [N, D] or [D].
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        actions = actions.to(self.device)
        N = actions.shape[0]
        
        with self.lock:
            if self.use_storage:
                # Use pre-allocated storage
                for i in range(N):
                    idx = self.queue_indices[task_id] % self.queue_size
                    self.storage[task_id, idx] = actions[i]
                    self.queue_indices[task_id] += 1
                    self.counts[task_id] = min(
                        self.counts[task_id] + 1,
                        self.queue_size
                    )
            else:
                # Use dynamic lists
                queue = self.queues[task_id]
                for i in range(N):
                    if len(queue) >= self.queue_size:
                        queue.pop(0)
                    queue.append(actions[i])
            
            self.total_added += N
    
    def sample(
        self,
        n: int,
        task_ids: Optional[list[int]] = None,
        uniform_tasks: bool = True,
    ) -> torch.Tensor:
        """
        Sample positive actions from the queue.
        
        Args:
            n: Number of samples to return.
            task_ids: Specific tasks to sample from (None = all).
            uniform_tasks: If True, sample uniformly across tasks.
        
        Returns:
            Sampled actions [n, D].
        """
        with self.lock:
            if self.use_storage:
                return self._sample_from_storage(n, task_ids, uniform_tasks)
            else:
                return self._sample_from_lists(n, task_ids, uniform_tasks)
    
    def _sample_from_storage(
        self,
        n: int,
        task_ids: Optional[list[int]],
        uniform_tasks: bool,
    ) -> torch.Tensor:
        """Sample from pre-allocated storage."""
        # Get valid tasks
        if task_ids is None:
            valid_tasks = torch.where(self.counts > 0)[0]
        else:
            valid_tasks = torch.tensor(
                [t for t in task_ids if self.counts[t] > 0],
                device=self.device
            )
        
        if len(valid_tasks) == 0:
            return torch.zeros(n, self.action_dim, device=self.device)
        
        samples = []
        
        if uniform_tasks:
            # Sample uniformly across tasks
            samples_per_task = n // len(valid_tasks) + 1
            
            for task_id in valid_tasks:
                count = int(self.counts[task_id])
                indices = torch.randint(0, count, (samples_per_task,), device=self.device)
                task_samples = self.storage[task_id, indices]
                samples.append(task_samples)
            
            samples = torch.cat(samples, dim=0)[:n]
        else:
            # Sample proportionally to task sizes
            total_samples = self.counts[valid_tasks].sum()
            probs = self.counts[valid_tasks].float() / total_samples
            
            task_indices = torch.multinomial(probs, n, replacement=True)
            
            for i, task_idx in enumerate(task_indices):
                task_id = valid_tasks[task_idx]
                count = int(self.counts[task_id])
                sample_idx = torch.randint(0, count, (1,)).item()
                samples.append(self.storage[task_id, sample_idx])
            
            samples = torch.stack(samples, dim=0)
        
        return samples
    
    def _sample_from_lists(
        self,
        n: int,
        task_ids: Optional[list[int]],
        uniform_tasks: bool,
    ) -> torch.Tensor:
        """Sample from dynamic lists."""
        # Get valid tasks
        if task_ids is None:
            valid_tasks = [t for t, q in self.queues.items() if len(q) > 0]
        else:
            valid_tasks = [t for t in task_ids if len(self.queues[t]) > 0]
        
        if len(valid_tasks) == 0:
            return torch.zeros(n, self.action_dim, device=self.device)
        
        samples = []
        
        if uniform_tasks:
            samples_per_task = n // len(valid_tasks) + 1
            
            for task_id in valid_tasks:
                queue = self.queues[task_id]
                indices = np.random.randint(0, len(queue), samples_per_task)
                for idx in indices:
                    samples.append(queue[idx])
            
            samples = torch.stack(samples[:n], dim=0)
        else:
            # Pool all samples and sample randomly
            all_samples = []
            for task_id in valid_tasks:
                all_samples.extend(self.queues[task_id])
            
            indices = np.random.randint(0, len(all_samples), n)
            samples = torch.stack([all_samples[i] for i in indices], dim=0)
        
        return samples.to(self.device)
    
    def get_task_counts(self) -> dict[int, int]:
        """Get number of samples per task."""
        with self.lock:
            if self.use_storage:
                return {
                    i: int(self.counts[i])
                    for i in range(len(self.counts))
                    if self.counts[i] > 0
                }
            else:
                return {t: len(q) for t, q in self.queues.items()}
    
    def total_samples(self) -> int:
        """Get total number of samples across all tasks."""
        counts = self.get_task_counts()
        return sum(counts.values())
    
    def clear(self) -> None:
        """Clear all queues."""
        with self.lock:
            if self.use_storage:
                self.storage.zero_()
                self.counts.zero_()
            else:
                self.queues.clear()
            
            self.queue_indices.clear()
            self.total_added = 0


class GlobalSampleQueue(SampleQueue):
    """
    Global sample queue (no per-task separation).
    
    Useful when task distinctions are not important or when
    mixing samples from all tasks is desired.
    """
    
    def __init__(
        self,
        queue_size: int = 1000,
        action_dim: int = 160,
        device: torch.device = torch.device('cpu'),
    ):
        super().__init__(
            queue_size=queue_size,
            num_tasks=1,
            action_dim=action_dim,
            device=device,
        )
    
    def add(self, actions: torch.Tensor, **kwargs) -> None:
        """Add samples (ignores task_id)."""
        super().add(task_id=0, actions=actions)
    
    def sample(self, n: int, **kwargs) -> torch.Tensor:
        """Sample from global queue."""
        return super().sample(n=n, task_ids=[0], uniform_tasks=True)


class NegativeSampleQueue:
    """
    Queue of recent model predictions for negative sampling in the drifting loss.
    
    WHY THIS IS NEEDED:
    Using x.detach() as negatives means y_neg ≈ x, causing V_attract ≈ V_repel,
    which makes the drifting field V ≈ 0. By storing predictions from PREVIOUS
    steps, the negatives are different from the current x, giving a non-zero
    drifting field that provides useful gradient signal.
    
    The paper uses a similar approach: q is the current generator distribution,
    sampled from a pool of recent outputs across many GPUs (effective batch 8192).
    This queue simulates that pool on a single GPU.
    
    Args:
        queue_size: Maximum stored predictions (larger = more diverse negatives).
        action_dim: Dimension of flattened action vectors.
        device: Storage device.
    
    Example:
        >>> neg_queue = NegativeSampleQueue(queue_size=1024, action_dim=128)
        >>> 
        >>> # Each training step:
        >>> neg_samples = neg_queue.sample(64)           # Sample old predictions
        >>> neg_queue.push(actions_pred.detach())         # Store new predictions
    """
    
    def __init__(
        self,
        queue_size: int = 1024,
        action_dim: int = 128,
        device: torch.device = torch.device('cpu'),
    ):
        self.queue_size = queue_size
        self.action_dim = action_dim
        self.device = device
        
        # Ring buffer
        self.buffer = torch.zeros(queue_size, action_dim, device=device)
        self.write_idx = 0
        self.count = 0
    
    def push(self, actions: torch.Tensor) -> None:
        """
        Push model predictions into the buffer.
        
        Args:
            actions: Predicted actions [N, D] (flattened).
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        actions = actions.detach().to(self.device)
        N = actions.shape[0]
        
        for i in range(N):
            self.buffer[self.write_idx % self.queue_size] = actions[i]
            self.write_idx += 1
        
        self.count = min(self.count + N, self.queue_size)
    
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample negative predictions from the buffer.
        
        If the buffer has fewer than n samples, returns what's available
        padded with random noise (which also serves as valid negatives).
        
        Args:
            n: Number of samples.
        
        Returns:
            Negative samples [n, D].
        """
        if self.count == 0:
            # Buffer empty — return random noise as negatives
            return torch.randn(n, self.action_dim, device=self.device)
        
        # Sample random indices from filled portion
        indices = torch.randint(0, self.count, (n,), device=self.device)
        return self.buffer[indices]
    
    def __len__(self) -> int:
        return self.count


