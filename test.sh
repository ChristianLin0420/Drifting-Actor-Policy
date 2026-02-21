#!/bin/bash
# =============================================================================
# Drifting-VLA End-to-End Pipeline
# =============================================================================
# Parallel data preparation + 8-GPU training.
# Data prep automatically skips already-prepared datasets (checks metadata.json).
#
# Usage (host):    bash test.sh
# Usage (Docker):  docker run --gpus all --ipc=host --ulimit memlock=-1 \
#                      -v $(pwd):/workspace -v $(pwd)/data:/workspace/data \
#                      -e WANDB_API_KEY=$WANDB_API_KEY \
#                      -it drifting-vla:pretrain bash test.sh
# =============================================================================
set -e

# =============================================================================
# Step 1: Prepare data (parallel, auto-skip done, RGB-only)
# =============================================================================
# --parallel 3: process 3 datasets simultaneously
# --rgb-only: exclude depth/segmentation images (default)
# --cleanup: free HF cache after each dataset
# Automatically skips datasets with existing metadata.json

python scripts/prepare_data.py \
    --datasets bc_z taco_play \
    --parallel 2 --cleanup

# LeRobot datasets
python scripts/prepare_data.py --dataset droid --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset utaustin_mutex --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset nyu_franka --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset behavior1k_t0000 --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset dexora --max-episodes 25 --force --cleanup

# Non-LeRobot datasets (special handlers)
python scripts/prepare_data.py --dataset rlbench --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset dexgraspnet --max-episodes 25 --force --cleanup
python scripts/prepare_data.py --dataset dexwild --max-episodes 25 --force --cleanup

# =============================================================================
# Step 2: 8-GPU Training (5 datasets, 3 embodiment types)
# =============================================================================
# aloha (bimanual), bc_z (gripper_joint), taco_play (gripper_joint),
# stanford_hydra (delta_eef), cmu_stretch (delta_eef)
# Effective batch: 32/gpu × 2 accum × 8 GPUs = 512 global

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=8 scripts/train.py \
    --datasets aloha bc_z taco_play stanford_hydra cmu_stretch \
    --episodes-root ./data/episodes \
    --batch-size 32 \
    --grad-accumulation 2 \
    --max-steps 10000 \
    --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base \
    --num-workers 8 \
    --image-aug --cond-mask-prob 0.1 \
    --log-every 50 --eval-every 500 --save-every 2500 \
    --wandb-mode online --wandb-project drifting-vla
