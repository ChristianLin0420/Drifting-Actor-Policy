#!/usr/bin/env python3
"""Quick NCCL communication test."""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl')
rank = dist.get_rank()
print(f'[rank {rank}] init done, device=cuda:{local_rank}', flush=True)

# Test simple all-reduce
t = torch.ones(1, device=f'cuda:{local_rank}') * (rank + 1)
dist.all_reduce(t)
print(f'[rank {rank}] all_reduce: {t.item()} (expected {dist.get_world_size() * (dist.get_world_size() + 1) / 2})', flush=True)

# Test DDP with a small model
model = nn.Linear(10, 10).to(f'cuda:{local_rank}')
print(f'[rank {rank}] small model created, params={sum(p.numel() for p in model.parameters())}', flush=True)
ddp_model = DDP(model, device_ids=[local_rank])
print(f'[rank {rank}] DDP small model OK', flush=True)

# Test DDP with medium model
class MediumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(768, 768) for _ in range(12)])
    
med_model = MediumModel().to(f'cuda:{local_rank}')
n_params = sum(p.numel() for p in med_model.parameters())
print(f'[rank {rank}] medium model created, params={n_params}', flush=True)
ddp_med = DDP(med_model, device_ids=[local_rank])
print(f'[rank {rank}] DDP medium model OK', flush=True)

dist.destroy_process_group()
print(f'[rank {rank}] ALL TESTS PASSED', flush=True)






