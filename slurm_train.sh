#!/bin/bash
# =============================================================================
# Continuable SLURM Training — 15-Node H200 (120 GPUs)
# =============================================================================
# Self-requeuing job that auto-resumes from the latest checkpoint when SLURM
# preempts or the wall-time limit is reached.
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
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --exclusive
# Uncomment and set your partition:
# #SBATCH --partition=h200

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

# HuggingFace cache on fast scratch (adjust to your cluster)
export HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"

CKPT_DIR="./checkpoints/pretrain_h200_15n"
LOG_DIR="./logs/slurm"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

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
echo "  Drifting-VLA Training (SLURM)"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Nodes:     $SLURM_NNODES"
echo "  GPUs:      $WORLD_SIZE"
echo "  Master:    $MASTER_ADDR:$MASTER_PORT"
echo "  Ckpt dir:  $CKPT_DIR"
echo "  Resume:    ${RESUME_ARG:-none (fresh start)}"
echo "============================================================"

# ── Launch distributed training ──
srun --kill-on-bad-exit=1 \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node=8 \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    scripts/train.py \
        --config configs/training/pretrain_h200_15node.yaml \
        --checkpoint-dir "$CKPT_DIR" \
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
