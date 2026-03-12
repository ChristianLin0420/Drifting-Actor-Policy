#!/bin/bash
# =============================================================================
# Download Pre-Converted Datasets from HuggingFace
# =============================================================================
# Pulls HDF5 episodes uploaded by upload_datasets.sh — no conversion needed.
#
# Usage:
#   bash download_from_hf.sh                     # All datasets
#   bash download_from_hf.sh --dataset aloha     # Single dataset
#   bash download_from_hf.sh --list              # List available datasets
#
# Downloads to: ./data/episodes/{dataset_name}/
# =============================================================================
set -e

HF_ORG="christian0420"
HF_PREFIX="drifting-vla"
EPISODES_ROOT="./data/episodes"

SINGLE_DATASET=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) SINGLE_DATASET="$2"; shift 2 ;;
        --list) LIST_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ALL_DATASETS=(
    aloha bc_z taco_play stanford_hydra cmu_stretch utaustin_mutex
    nyu_franka rlbench dexwild dexora droid
    $(for i in $(seq -w 0 49); do echo "behavior1k_t00$i"; done)
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

if [ "$LIST_ONLY" = true ]; then
    echo "Available datasets (${#ALL_DATASETS[@]}):"
    for ds in "${ALL_DATASETS[@]}"; do
        repo="${HF_ORG}/${HF_PREFIX}-${ds}"
        local_dir="${EPISODES_ROOT}/${ds}"
        if [ -f "$local_dir/metadata.json" ]; then
            echo "  ✅ $ds  ($repo)  — already downloaded"
        else
            echo "  ⬇  $ds  ($repo)"
        fi
    done
    exit 0
fi

if [ -n "$SINGLE_DATASET" ]; then
    DATASETS=("$SINGLE_DATASET")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================================"
echo "  Downloading Pre-Converted Datasets"
echo "  Source: $HF_ORG/${HF_PREFIX}-*"
echo "  Target: $EPISODES_ROOT/"
echo "  Datasets: ${#DATASETS[@]}"
echo "============================================================"

mkdir -p "$EPISODES_ROOT"

SUCCESS=0
FAILED=0
SKIPPED=0

for ds in "${DATASETS[@]}"; do
    repo="${HF_ORG}/${HF_PREFIX}-${ds}"
    local_dir="${EPISODES_ROOT}/${ds}"

    if [ -f "$local_dir/metadata.json" ]; then
        echo "  [SKIP] $ds — already exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "  [DOWN] $ds ← $repo"
    mkdir -p "$local_dir"

    if hf download "$repo" \
        --repo-type dataset \
        --local-dir "$local_dir"; then
        echo "  [OK] $ds downloaded"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  [FAIL] $ds — check if repo exists or run 'huggingface-cli login'"
        rmdir "$local_dir" 2>/dev/null || true
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
echo "  Download Summary"
echo "  Success: $SUCCESS | Skipped: $SKIPPED | Failed: $FAILED"
echo "============================================================"

# Show what's ready
echo ""
echo "Ready datasets:"
for ds in "${DATASETS[@]}"; do
    local_dir="${EPISODES_ROOT}/${ds}"
    if [ -f "$local_dir/metadata.json" ]; then
        SAMPLES=$(python -c "import json; print(json.load(open('$local_dir/metadata.json'))['total_samples'])" 2>/dev/null || echo "?")
        echo "  ✅ $ds ($SAMPLES samples)"
    fi
done
