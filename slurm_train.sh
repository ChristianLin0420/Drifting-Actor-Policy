#!/bin/bash
# =============================================================================
# Continuable SLURM Training — 15-Node H200 (120 GPUs) via Apptainer
# =============================================================================
# Self-requeuing job that auto-resumes from the latest checkpoint when SLURM
# preempts or the wall-time limit is reached.
#
# Container: christianlin0420/drifting-vla:pretrain
#   Pulled via Apptainer (available at /usr/bin/apptainer on this cluster).
#   The image is cached to APPTAINER_CACHEDIR on first run; subsequent runs
#   use the cached SIF file, so there is no repeated network pull.
#
# How it works:
#   1. SLURM sends SIGUSR1 120 seconds before killing the job
#   2. The Python training script catches the signal, saves a checkpoint, exits
#   3. This script requeues itself via `scontrol requeue`
#   4. On restart, `--resume latest` auto-finds the newest checkpoint
#
# Usage:
#   sbatch slurm_train.sh                   # full training
#   sbatch slurm_train.sh --max-steps 1000  # override max steps
#
# Monitor:
#   tail -f logs/slurm/<jobid>.out
#   squeue -u $USER
# =============================================================================

#SBATCH --job-name=drifting-vla
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --partition=normal
#SBATCH --account=MST113264

set -euo pipefail

# ── Environment ──
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=$((SLURM_NNODES * 8))
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# NCCL tuning for H200 NVLink + InfiniBand
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

# HuggingFace model cache — shared from host so weights are not re-downloaded
# on every run. Bind-mounted into the container at the same path.
export HF_HOME="${HF_HOME:-/work/$USER/.cache/huggingface}"
mkdir -p "$HF_HOME"

# ── Apptainer (Singularity) container settings ──
# The cluster has Apptainer 1.4.3 at /usr/bin/apptainer.
# Pyxis/Enroot is NOT available — we use `apptainer exec --nv` instead.
#
# The image is pulled from Docker Hub on first use and cached as a SIF file
# under APPTAINER_CACHEDIR. Subsequent nodes/jobs reuse the cache.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/work/$USER/.apptainer/cache}"
mkdir -p "$APPTAINER_CACHEDIR"

# Pre-built SIF file (avoids 15 nodes pulling from Docker Hub in parallel).
# Build once on the login node:
#   apptainer pull /work/$USER/.apptainer/cache/drifting-vla-pretrain.sif \
#       docker://christianlin0420/drifting-vla:pretrain
CONTAINER_IMAGE="/work/$USER/.apptainer/cache/drifting-vla-pretrain.sif"
if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "$(date): SIF not found at $CONTAINER_IMAGE — pulling from Docker Hub..."
    apptainer pull "$CONTAINER_IMAGE" docker://christianlin0420/drifting-vla:pretrain
fi

# Bind mounts:
#   $(pwd)                      → /workspace             (project code + configs + scripts)
#   /work/crlc112358/datasets   → /workspace/data/episodes  (on-disk episode HDF5 files)
#   $HF_HOME                    → $HF_HOME               (model weight cache, same path inside)
#   /dev/shm                    → /dev/shm               (shared memory for DataLoader workers)
DATASETS_DIR="/work/crlc112358/datasets"
APPTAINER_BINDS="$(pwd):/workspace,${DATASETS_DIR}:/workspace/data/episodes,${HF_HOME}:${HF_HOME},/dev/shm:/dev/shm"

CKPT_DIR="$(pwd)/checkpoints/pretrain_h200_15n"
LOG_DIR="$(pwd)/logs/slurm"
TORCHRUN_LOG_DIR="$(pwd)/logs/torchrun"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$TORCHRUN_LOG_DIR"

# Inside the container, $(pwd) is bind-mounted to /workspace.
# All paths passed as args to train.py must use container paths.
CONTAINER_CKPT_DIR="/workspace/checkpoints/pretrain_h200_15n"

# ── Auto-resume logic ──
RESUME_ARG=""
if [ -f "$CKPT_DIR/latest.pt" ]; then
    RESUME_ARG="--resume latest"
    echo "$(date): Resuming from $CKPT_DIR/latest.pt"
elif ls "$CKPT_DIR"/step_*.pt 1>/dev/null 2>&1; then
    RESUME_ARG="--resume latest"
    echo "$(date): Resuming from latest step_*.pt in $CKPT_DIR"
else
    echo "$(date): Starting fresh training"
fi

# Allow extra args to be appended (e.g., sbatch slurm_train.sh --max-steps 1000)
EXTRA_ARGS="${@}"

# ── Signal handler: requeue on preemption ──
# The Python script handles SIGUSR1 itself (saves checkpoint + exits).
# This bash handler just requeues the SLURM job so it restarts automatically.
handle_preempt() {
    echo "$(date): Caught signal in launcher — requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap handle_preempt USR1 TERM

echo "============================================================"
echo "  Drifting-VLA Training (SLURM + Apptainer)"
echo "  Image:     $CONTAINER_IMAGE"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Nodes:     $SLURM_NNODES"
echo "  GPUs:      $WORLD_SIZE"
echo "  Master:    $MASTER_ADDR:$MASTER_PORT"
echo "  Ckpt dir:  $CKPT_DIR (→ $CONTAINER_CKPT_DIR in container)"
echo "  Resume:    ${RESUME_ARG:-none (fresh start)}"
echo "============================================================"

# ── Launch distributed training inside the container ──
# Each srun task (one per node) runs `apptainer exec --nv` to enter the
# Docker image, then invokes torchrun for that node's 8 GPUs.
# `--nv` passes through CUDA devices and libraries from the host.
# `--no-home` prevents host ~/.* dotfiles from leaking into the container.
# `--env` forwards the distributed training and NCCL variables.
srun --kill-on-bad-exit=1 \
    apptainer exec \
        --nv \
        --cleanenv \
        --pwd /workspace \
        --bind "$APPTAINER_BINDS" \
        --env "MASTER_ADDR=${MASTER_ADDR},MASTER_PORT=${MASTER_PORT},WORLD_SIZE=${WORLD_SIZE},OMP_NUM_THREADS=${OMP_NUM_THREADS},PYTHONUNBUFFERED=${PYTHONUNBUFFERED},NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE},NCCL_IB_DISABLE=${NCCL_IB_DISABLE},NCCL_DEBUG=${NCCL_DEBUG},NCCL_TIMEOUT=${NCCL_TIMEOUT},HF_HOME=${HF_HOME},HF_TOKEN=${HF_TOKEN},WANDB_API_KEY=e66e164e1c3b7f8c38fcea72427dafb0f4b35b80,MPLCONFIGDIR=/tmp/matplotlib,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        "$CONTAINER_IMAGE" \
        torchrun \
            --nnodes="$SLURM_NNODES" \
            --nproc_per_node=8 \
            --rdzv_id="$SLURM_JOB_ID" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        scripts/train.py \
            --config configs/training/pretrain_available_datasets.yaml \
            --checkpoint-dir "$CONTAINER_CKPT_DIR" \
            --gradient-checkpointing \
            --use-flash-attn \
            $RESUME_ARG \
            $EXTRA_ARGS \
    &

# Wait in background so bash trap can fire on signals
wait $!
EXIT_CODE=$?

echo "$(date): Training process exited with code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): Training completed successfully (all steps done)"
else
    echo "$(date): Training interrupted (exit=$EXIT_CODE) — will resume on requeue"
fi
