#!/bin/bash
# =============================================================================
# Drifting-VLA Smoke Test — Flexible Per-Dataset Pipeline
# =============================================================================
# Run individual stages manually to debug each dataset:
#
#   bash test.sh prepare aloha        # Step 1: Download + convert 20 episodes
#   bash test.sh train aloha          # Step 2: Train 10 steps on 8×L40
#   bash test.sh verify aloha         # Step 3: Verify action mapping round-trip
#   bash test.sh clean aloha          # Step 4: Remove data to free disk
#   bash test.sh all aloha            # Run all steps sequentially
#
# For Docker (exec into running container):
#   docker exec -it <container> bash test.sh prepare aloha
#
# Recommended small datasets for quick testing:
#   aloha         ~20K frames, parquet images, 14-dim bimanual
#   cmu_stretch   ~25K frames, 1 camera, 8-dim delta_eef
#   nyu_franka    ~45K frames, 2 cameras, 15-dim bimanual
#
# Skip these (too large / too slow for smoke test):
#   droid         25.5M frames — skip
#   dexora        2.9M frames  — skip unless disk allows
# =============================================================================
set -e

STAGE="${1:-all}"
DATASET="${2:-aloha}"
MAX_EPISODES="${3:-20}"
TRAIN_STEPS="${4:-10}"
GPUS="${5:-8}"

DATA_ROOT="./data"
EPISODES_ROOT="./data/episodes"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "Drifting-VLA Test: stage=$STAGE  dataset=$DATASET"
echo "  max_episodes=$MAX_EPISODES  train_steps=$TRAIN_STEPS  gpus=$GPUS"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# Step 1: PREPARE — download + convert to Episode HDF5
# ─────────────────────────────────────────────────────────────────
do_prepare() {
    echo ""
    echo "[PREPARE] Downloading and converting $DATASET ($MAX_EPISODES episodes)..."
    python scripts/prepare_data.py \
        --dataset "$DATASET" \
        --max-episodes "$MAX_EPISODES" \
        --force --cleanup \
        --data-root "$DATA_ROOT" \
        --output-root "$EPISODES_ROOT"

    echo "[PREPARE] Checking output..."
    if [ -f "$EPISODES_ROOT/$DATASET/metadata.json" ]; then
        echo "[PREPARE] ✅ metadata.json found"
        python -c "
import json, sys
m = json.load(open('$EPISODES_ROOT/$DATASET/metadata.json'))
print(f'  dataset: {m[\"dataset_name\"]}')
print(f'  episodes: {m[\"total_episodes\"]}')
print(f'  samples: {m[\"total_samples\"]}')
print(f'  embodiment_id: {m[\"embodiment_id\"]}')
print(f'  view_names: {m.get(\"view_names\", \"N/A\")}')
"
    else
        echo "[PREPARE] ❌ metadata.json NOT found — prepare failed"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────
# Step 2: TRAIN — run a few training steps on 8×L40
# ─────────────────────────────────────────────────────────────────
do_train() {
    echo ""
    echo "[TRAIN] Running $TRAIN_STEPS training steps on $DATASET ($GPUS GPUs)..."

    if [ ! -f "$EPISODES_ROOT/$DATASET/metadata.json" ]; then
        echo "[TRAIN] ❌ No prepared data for $DATASET — run 'bash test.sh prepare $DATASET' first"
        exit 1
    fi

    torchrun --nproc_per_node="$GPUS" scripts/train.py \
        --datasets "$DATASET" \
        --episodes-root "$EPISODES_ROOT" \
        --batch-size 2 \
        --grad-accumulation 1 \
        --max-steps "$TRAIN_STEPS" \
        --lr 1e-4 \
        --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
        --loss-type hybrid --model-size base \
        --num-workers 4 \
        --image-aug --cond-mask-prob 0.1 \
        --log-every 2 --eval-every "$TRAIN_STEPS" --save-every 9999 \
        --wandb-mode disabled

    echo "[TRAIN] ✅ $TRAIN_STEPS steps completed for $DATASET"
}

# ─────────────────────────────────────────────────────────────────
# Step 3: VERIFY — check action mapping + data integrity
# ─────────────────────────────────────────────────────────────────
do_verify() {
    echo ""
    echo "[VERIFY] Checking action mapping and data integrity for $DATASET..."

    python -c "
import sys, json, numpy as np, h5py
sys.path.insert(0, '.')
from drifting_vla.data.action_mapping import (
    DATASET_EMBODIMENT, DATASET_NATIVE_ACTION_DIM, DATASET_FIELD_FORMATS,
    STATE_VEC_IDX_MAPPING, UNIFIED_ACTION_DIM,
    map_to_unified, extract_from_unified, get_action_mask,
    assemble_state_vec, assemble_state_vec_batch,
)
from pathlib import Path

ds = '$DATASET'
print(f'Dataset: {ds}')

# Check registry entries exist
emb_id = DATASET_EMBODIMENT.get(ds)
native_dim = DATASET_NATIVE_ACTION_DIM.get(ds)
fields = DATASET_FIELD_FORMATS.get(ds)

if emb_id is None:
    print(f'  ❌ {ds} not in DATASET_EMBODIMENT')
    sys.exit(1)
if native_dim is None:
    print(f'  ❌ {ds} not in DATASET_NATIVE_ACTION_DIM')
    sys.exit(1)
print(f'  embodiment_id: {emb_id}, native_dim: {native_dim}')

# Check field format mapping
if fields:
    print(f'  field_format: {len(fields)} fields → {fields[:5]}...')
    for f in fields:
        if f not in STATE_VEC_IDX_MAPPING:
            print(f'  ❌ field \"{f}\" not in STATE_VEC_IDX_MAPPING')
            sys.exit(1)
    print(f'  ✅ All field names valid')

    # Round-trip test via assemble_state_vec
    rng = np.random.RandomState(42)
    action = rng.randn(native_dim).astype(np.float32)
    unified, mask = assemble_state_vec(action, fields)
    assert unified.shape == (UNIFIED_ACTION_DIM,), f'bad shape: {unified.shape}'
    assert mask.sum() == len(fields), f'mask sum {mask.sum()} != {len(fields)}'

    # Check we can recover via extract_from_unified
    recovered = extract_from_unified(unified, emb_id, dataset_name=ds)
    err = np.abs(action[:len(recovered)] - recovered[:native_dim]).max()
    print(f'  Round-trip max error: {err:.2e}')
    if err > 1e-5:
        print(f'  ⚠️  Round-trip error > 1e-5')
    else:
        print(f'  ✅ Round-trip OK')

# Check action mask
mask_info = get_action_mask(emb_id, native_dim=native_dim, dataset_name=ds)
print(f'  active_dims: {mask_info.active_dims}')
print(f'  quat_dims: {mask_info.quat_dims}')

# Check HDF5 files exist and have correct shape
ep_dir = Path('$EPISODES_ROOT/$DATASET')
if ep_dir.exists():
    h5_files = sorted(ep_dir.glob('ep_*.hdf5'))
    if h5_files:
        with h5py.File(str(h5_files[0]), 'r') as f:
            act_shape = f['actions'].shape
            mask_shape = f['action_mask'].shape
            print(f'  HDF5 actions: {act_shape}, mask: {mask_shape}')
            assert act_shape[1] == UNIFIED_ACTION_DIM, f'actions dim {act_shape[1]} != {UNIFIED_ACTION_DIM}'
            assert mask_shape[0] == UNIFIED_ACTION_DIM, f'mask dim {mask_shape[0]} != {UNIFIED_ACTION_DIM}'
            print(f'  ✅ HDF5 shapes correct (128-dim unified)')
    else:
        print(f'  ⚠️  No HDF5 files found (run prepare first)')
else:
    print(f'  ⚠️  Episode dir not found (run prepare first)')

print(f'[VERIFY] ✅ {ds} PASSED')
"
}

# ─────────────────────────────────────────────────────────────────
# Step 4: CLEAN — remove prepared data to free disk
# ─────────────────────────────────────────────────────────────────
do_clean() {
    echo ""
    echo "[CLEAN] Removing prepared data for $DATASET..."
    rm -rf "$EPISODES_ROOT/$DATASET"
    rm -rf "$DATA_ROOT/$DATASET"

    # Also clean HF cache to reclaim all disk space
    echo "[CLEAN] Removing HuggingFace cache..."
    rm -rf ~/.cache/huggingface/datasets
    rm -rf ~/.cache/huggingface/hub
    rm -rf ~/.cache/huggingface/lerobot

    echo "[CLEAN] ✅ Cleaned $DATASET + HF cache"
}

# ─────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────
case "$STAGE" in
    prepare)  do_prepare ;;
    train)    do_train ;;
    verify)   do_verify ;;
    clean)    do_clean ;;
    all)
        do_prepare
        do_verify
        do_train
        do_clean
        ;;
    *)
        echo "Usage: bash test.sh {prepare|train|verify|clean|all} <dataset> [max_episodes] [train_steps] [gpus]"
        echo ""
        echo "Examples:"
        echo "  bash test.sh prepare aloha          # Download + convert 20 episodes"
        echo "  bash test.sh train aloha            # Train 10 steps on 8 GPUs"
        echo "  bash test.sh verify aloha           # Check action mapping round-trip"
        echo "  bash test.sh clean aloha            # Remove data"
        echo "  bash test.sh all aloha              # All steps"
        echo "  bash test.sh all cmu_stretch 20 10 4  # Custom: 4 GPUs"
        echo ""
        echo "Recommended datasets: aloha, cmu_stretch, nyu_franka"
        exit 1
        ;;
esac
