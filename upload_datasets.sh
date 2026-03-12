#!/bin/bash
# =============================================================================
# Dataset Upload Pipeline — Prepare + Upload to HuggingFace
# =============================================================================
# For each dataset: download → convert to HDF5 → clean cache → upload to HF
#
# Usage:
#   bash upload_datasets.sh                              # All datasets, full
#   bash upload_datasets.sh --max-episodes 1000          # Limited for experiments
#   bash upload_datasets.sh --dataset aloha              # Single dataset
#   bash upload_datasets.sh --dataset aloha --max-episodes 500
#   bash upload_datasets.sh --skip-prepare               # Upload already-prepared only
#
# Prerequisites:
#   huggingface-cli login   (or set HF_TOKEN env var)
#
# Uploads to: christian0420/drifting-vla-{dataset_name}
# =============================================================================
set -e

HF_ORG="christian0420"
HF_PREFIX="drifting-vla"
EPISODES_ROOT="./data/episodes"
DATA_ROOT="./data"

# Parse arguments
MAX_EPISODES=""
SINGLE_DATASET=""
SKIP_PREPARE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        --dataset) SINGLE_DATASET="$2"; shift 2 ;;
        --skip-prepare) SKIP_PREPARE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# droid is too large, so we skip it

# All datasets to process
ALL_DATASETS=(
    # Original datasets
    aloha bc_z taco_play stanford_hydra cmu_stretch utaustin_mutex
    nyu_franka rlbench dexwild dexora
    # Behavior 1K (all 50 tasks)
    $(for i in $(seq -w 0 49); do echo "behavior1k_t00$i"; done)
    # Open X-Embodiment (batch 1)
    bridgev2 kuka berkeley_fanuc cmu_play_fusion jaco_play
    austin_buds austin_sailor austin_sirius columbia_pusht nyu_door
    # ALOHA Static variants
    aloha_static_cups_open aloha_static_vinh_cup aloha_static_vinh_cup_left
    aloha_static_coffee aloha_static_pingpong aloha_static_tape
    aloha_static_pro_pencil aloha_static_candy aloha_static_fork
    aloha_static_velcro aloha_static_battery aloha_static_screw
    aloha_static_towel aloha_static_ziploc
    # ALOHA Mobile variants
    aloha_mobile_cabinet aloha_mobile_chair aloha_mobile_wash_pan
    aloha_mobile_wipe_wine aloha_mobile_elevator aloha_mobile_shrimp
    # Open X-Embodiment (batch 2)
    berkeley_rpt toto stanford_robocook berkeley_mvp kaist_nonprehensile
    ucsd_pick_place ucsd_kitchen asu_table_top utokyo_pr2_fridge
    utokyo_pr2_tabletop utokyo_xarm_bimanual tokyo_u_lsmo
    dlr_sara_grid dlr_sara_pour dlr_edan nyu_rot usc_cloth_sim
    cmu_franka_exploration imperialcollege_sawyer
)

if [ -n "$SINGLE_DATASET" ]; then
    DATASETS=("$SINGLE_DATASET")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "============================================================"
echo "  Dataset Upload Pipeline"
echo "  HF Org: $HF_ORG"
echo "  Datasets: ${#DATASETS[@]}"
echo "  Max episodes: ${MAX_EPISODES:-full}"
echo "  Skip prepare: $SKIP_PREPARE"
echo "============================================================"

TOTAL=${#DATASETS[@]}
SUCCESS=0
FAILED=0
SKIPPED=0

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    idx=$((i + 1))
    repo="${HF_ORG}/${HF_PREFIX}-${ds}"
    ep_dir="${EPISODES_ROOT}/${ds}"

    echo ""
    echo "========================================"
    echo "  [$idx/$TOTAL] $ds → $repo"
    echo "========================================"

    # ── Step 1: Prepare (download + convert) ──
    if [ "$SKIP_PREPARE" = true ]; then
        if [ ! -f "$ep_dir/metadata.json" ]; then
            echo "  [SKIP] No prepared data and --skip-prepare set, skipping"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        echo "  [SKIP] Using existing prepared data"
    elif [ -f "$ep_dir/metadata.json" ] && [ -z "$MAX_EPISODES" ]; then
        echo "  [SKIP] Already prepared (use --max-episodes to re-prepare)"
    else
        echo "  [PREPARE] Downloading + converting..."
        PREP_ARGS="--dataset $ds --data-root $DATA_ROOT --output-root $EPISODES_ROOT --cleanup"
        if [ -n "$MAX_EPISODES" ]; then
            PREP_ARGS="$PREP_ARGS --max-episodes $MAX_EPISODES --force"
        fi
        if ! python scripts/prepare_data.py $PREP_ARGS; then
            echo "  [FAIL] Preparation failed for $ds"
            FAILED=$((FAILED + 1))
            continue
        fi
    fi

    # Verify
    if [ ! -f "$ep_dir/metadata.json" ]; then
        echo "  [FAIL] No metadata.json after preparation"
        FAILED=$((FAILED + 1))
        continue
    fi

    SAMPLES=$(python -c "import json; m=json.load(open('$ep_dir/metadata.json')); print(m['total_samples'])")
    EPISODES=$(python -c "import json; m=json.load(open('$ep_dir/metadata.json')); print(m['total_episodes'])")
    echo "  [INFO] $EPISODES episodes, $SAMPLES samples"

    # ── Step 2: Clean HF cache ──
    echo "  [CLEAN] Removing HF download cache..."
    rm -rf ~/.cache/huggingface/datasets
    rm -rf ~/.cache/huggingface/hub
    rm -rf ~/.cache/huggingface/lerobot
    rm -rf "${DATA_ROOT}/${ds}"  # raw downloaded data (non-LeRobot datasets)

    # ── Step 3: Upload to HuggingFace ──
    echo "  [UPLOAD] Pushing to $repo ..."

    # Create repo if it doesn't exist
    python -c "
from huggingface_hub import HfApi, create_repo
api = HfApi()
try:
    create_repo('$repo', repo_type='dataset', private=False, exist_ok=True)
    print('  Repo ready: $repo')
except Exception as e:
    print(f'  Repo create note: {e}')
"

    # Upload the entire episodes directory
    python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$ep_dir',
    repo_id='$repo',
    repo_type='dataset',
    commit_message='Upload $ds ($EPISODES episodes, $SAMPLES samples)',
)
print('  Upload complete: $repo')
"

    if [ $? -eq 0 ]; then
        echo "  [OK] $ds uploaded to $repo"
        SUCCESS=$((SUCCESS + 1))

        # ── Step 4: Clean converted HDF5 to free disk ──
        echo "  [CLEAN] Removing converted HDF5 ($ep_dir)..."
        rm -rf "$ep_dir"
    else
        echo "  [FAIL] Upload failed for $ds — keeping local data"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Upload Summary"
echo "  Success: $SUCCESS / $TOTAL"
echo "  Failed:  $FAILED"
echo "  Skipped: $SKIPPED"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
