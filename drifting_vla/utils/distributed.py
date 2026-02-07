"""
Distributed Training Utilities
==============================

Helper functions for setting up distributed training with PyTorch.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = 'nccl',
    init_method: str = 'env://',
    timeout_minutes: int = 30,
) -> tuple[int, int, torch.device]:
    """
    Initialize distributed training.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi').
        init_method: Initialization method ('env://' or TCP address).
        timeout_minutes: Timeout for initialization.
    
    Returns:
        Tuple of (rank, world_size, device).
    
    Example:
        >>> rank, world_size, device = setup_distributed()
        >>> model = model.to(device)
        >>> model = DDP(model, device_ids=[rank])
    """
    # Check if already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        return rank, world_size, device
    
    # Check for distributed environment variables
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.info("Distributed environment not detected, running single GPU")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return 0, 1, device
    
    # Get rank and world size
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        backend = 'gloo'  # NCCL requires CUDA
    
    # Initialize process group
    timeout = torch.distributed.distributed_c10d._DEFAULT_PG_TIMEOUT
    if timeout_minutes:
        from datetime import timedelta
        timeout = timedelta(minutes=timeout_minutes)
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )
    
    logger.info(
        f"Initialized distributed training: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, device={device}"
    )
    
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Destroyed distributed process group")


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """
    All-reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce.
        op: Reduction operation (SUM, MEAN, etc.).
    
    Returns:
        Reduced tensor.
    """
    if not dist.is_initialized():
        return tensor
    
    dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather.
    
    Returns:
        List of tensors from all processes.
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return tensors


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source to all processes.
    
    Args:
        tensor: Tensor to broadcast.
        src: Source rank.
    
    Returns:
        Broadcasted tensor.
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


