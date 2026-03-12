#!/bin/bash
# =============================================================================
# Mixture Training Test — 17 datasets x 20 episodes, 1000 steps, 8xL40
# =============================================================================
# Validates the full multi-dataset multi-embodiment training pipeline:
#   - Temperature-balanced sampling across 6 embodiment types
#   - Action mask correctness across native dims (2 to 39)
#   - Cross-embodiment drifting loss
#   - Per-embodiment eval metrics
#   - WandB visualization dashboards (D1-D4)
#
# Usage:
#   docker exec -it <container> bash mixture_test.sh
#
# Memory: ~1.7 GB HDF5 + ~5 GB peak during download = ~7 GB peak
# =============================================================================
set -e

DATASETS="aloha"

# DATASETS="aloha bc_z taco_play stanford_hydra cmu_stretch utaustin_mutex \
#           nyu_franka dexora bridgev2 kuka berkeley_fanuc cmu_play_fusion \
#           jaco_play austin_buds austin_sirius columbia_pusht nyu_door"

# Need to check: dexwild, rlbench droid droid behavior1k_t0000-t0049 austin_sailor

EPISODES_ROOT="./data/episodes"
MAX_EPISODES=20
TRAIN_STEPS=1000
GPUS=8

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "  Mixture Training Test"
echo "  Datasets: 17 | Episodes: $MAX_EPISODES each | Steps: $TRAIN_STEPS"
echo "============================================================"

# =============================================================================
# Step 1: Prepare all datasets (skip already-prepared, cleanup cache after each)
# =============================================================================
echo ""
echo "=== Step 1: Preparing datasets ==="
PREPARED=0
SKIPPED=0

for ds in $DATASETS; do
    if [ -f "$EPISODES_ROOT/$ds/metadata.json" ]; then
        echo "  [SKIP] $ds — already prepared"
        SKIPPED=$((SKIPPED + 1))
    else
        echo "  [PREP] $ds — downloading + converting $MAX_EPISODES episodes..."
        python scripts/prepare_data.py \
            --dataset "$ds" \
            --max-episodes $MAX_EPISODES \
            --cleanup \
            --output-root "$EPISODES_ROOT"
        if [ $? -ne 0 ]; then
            echo "  [WARN] $ds preparation failed — skipping"
        else
            PREPARED=$((PREPARED + 1))
        fi
    fi
done

echo ""
echo "  Preparation done: $PREPARED newly prepared, $SKIPPED already cached"
echo ""

# Verify all datasets have metadata
echo "=== Verifying prepared datasets ==="
VALID_DATASETS=""
COUNT=0
for ds in $DATASETS; do
    if [ -f "$EPISODES_ROOT/$ds/metadata.json" ]; then
        VALID_DATASETS="$VALID_DATASETS $ds"
        COUNT=$((COUNT + 1))
        SAMPLES=$(python -c "import json; m=json.load(open('$EPISODES_ROOT/$ds/metadata.json')); print(m['total_samples'])")
        echo "  [$COUNT] $ds — $SAMPLES samples"
    else
        echo "  [MISS] $ds — no data, excluding from training"
    fi
done

if [ $COUNT -eq 0 ]; then
    echo "ERROR: No datasets prepared!"
    exit 1
fi

echo ""
echo "  $COUNT datasets ready for mixture training"
echo ""

# =============================================================================
# Step 2: Mixture training (1000 steps, WandB enabled)
# =============================================================================
echo "=== Step 2: Mixture training ($TRAIN_STEPS steps on $GPUS GPUs) ==="
echo "  Datasets: $COUNT"
echo "  Batch: 4/gpu × 2 accum × $GPUS GPUs = $((4 * 2 * GPUS)) global"
echo "  WandB: online (project: drifting-vla-mixture-test)"
echo ""

torchrun --nproc_per_node=$GPUS scripts/train.py \
    --datasets $VALID_DATASETS \
    --episodes-root "$EPISODES_ROOT" \
    --batch-size 4 \
    --grad-accumulation 2 \
    --max-steps $TRAIN_STEPS \
    --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base \
    --num-workers 4 \
    --image-aug --cond-mask-prob 0.1 \
    --log-every 10 --eval-every 200 --save-every 500 \
    --wandb-mode online --wandb-project drifting-vla-mixture-test

echo ""
echo "============================================================"
echo "  Mixture training complete!"
echo "  Check WandB: drifting-vla-mixture-test"
echo "============================================================"
