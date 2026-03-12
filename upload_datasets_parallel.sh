#!/bin/bash
# =============================================================================
# Parallel Dataset Upload Pipeline — N Workers with Queue
# =============================================================================
# Parallel version of upload_datasets.sh. N workers each pull from a shared
# queue, processing one dataset at a time: prepare → clean → upload → clean HDF5.
#
# Usage:
#   bash upload_datasets_parallel.sh                        # 3 workers, all datasets, full
#   bash upload_datasets_parallel.sh 5                      # 5 workers
#   bash upload_datasets_parallel.sh 3 --max-episodes 1000  # limited
#   bash upload_datasets_parallel.sh 1                      # sequential fallback
#
# Logs: ./logs/upload/worker_{N}.log (one per worker, continuous)
# Console: live progress with worker status
# Monitor: tail -f logs/upload/worker_1.log
#
# Prerequisites:
#   huggingface-cli login   (or set HF_TOKEN env var)
# =============================================================================
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null)
set -e

HF_ORG="christian0420"
HF_PREFIX="drifting-vla"
EPISODES_ROOT="./data/episodes"
DATA_ROOT="./data"
LOG_DIR="./logs/upload"

# Parse arguments
NUM_WORKERS="${1:-3}"
shift 2>/dev/null || true
MAX_EPISODES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-episodes) MAX_EPISODES="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Skipped: droid, dexwild
# Done: aloha, cmu_stretch

# All datasets
ALL_DATASETS=(
    # Original
    bc_z taco_play stanford_hydra utaustin_mutex
    nyu_franka rlbench dexora
    # Behavior 1K
    $(for i in $(seq -w 0 49); do echo "behavior1k_t00$i"; done)
    # OXE batch 1
    bridgev2 kuka berkeley_fanuc cmu_play_fusion jaco_play
    austin_buds austin_sailor austin_sirius columbia_pusht nyu_door
    # ALOHA Static
    aloha_static_cups_open aloha_static_vinh_cup aloha_static_vinh_cup_left
    aloha_static_coffee aloha_static_pingpong aloha_static_tape
    aloha_static_pro_pencil aloha_static_candy aloha_static_fork
    aloha_static_velcro aloha_static_battery aloha_static_screw
    aloha_static_towel aloha_static_ziploc
    # ALOHA Mobile
    aloha_mobile_cabinet aloha_mobile_chair aloha_mobile_wash_pan
    aloha_mobile_wipe_wine aloha_mobile_elevator aloha_mobile_shrimp
    # OXE batch 2
    berkeley_rpt toto stanford_robocook berkeley_mvp kaist_nonprehensile
    ucsd_pick_place ucsd_kitchen asu_table_top utokyo_pr2_fridge
    utokyo_pr2_tabletop utokyo_xarm_bimanual tokyo_u_lsmo
    dlr_sara_grid dlr_sara_pour dlr_edan nyu_rot
    usc_cloth_sim cmu_franka_exploration imperialcollege_sawyer
)

TOTAL=${#ALL_DATASETS[@]}

mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "============================================================"
echo "  Parallel Upload Pipeline"
echo "  HF Org: $HF_ORG"
echo "  Workers: $NUM_WORKERS"
echo "  Datasets: $TOTAL"
echo "  Max episodes: ${MAX_EPISODES:-full}"
echo "  Logs: $LOG_DIR/worker_{1..$NUM_WORKERS}.log"
echo "============================================================"
echo ""

# ── Queue mechanism using a file + flock ──
QUEUE_FILE=$(mktemp /tmp/upload_queue.XXXXXX)
LOCK_FILE=$(mktemp /tmp/upload_lock.XXXXXX)
STATUS_DIR=$(mktemp -d /tmp/upload_status.XXXXXX)

# Fill queue
for ds in "${ALL_DATASETS[@]}"; do
    echo "$ds" >> "$QUEUE_FILE"
done

# Track results
mkdir -p "$STATUS_DIR"

# ── Worker function ──
process_dataset() {
    local worker_id=$1
    local ds=$2
    local repo="${HF_ORG}/${HF_PREFIX}-${ds}"
    local ep_dir="${EPISODES_ROOT}/${ds}"
    local worker_cache="./cache_worker_${worker_id}"
    local log_file="${LOG_DIR}/worker_${worker_id}.log"

    # Isolate HF cache per worker
    export HF_HOME="${worker_cache}"
    export HUGGINGFACE_HUB_CACHE="${worker_cache}/hub"

    {
        echo ""
        echo "========================================"
        echo "Worker $worker_id: $ds → $repo"
        echo "Started: $(date)"
        echo "========================================"

        # Step 1: Prepare
        if [ -f "$ep_dir/metadata.json" ] && [ -z "$MAX_EPISODES" ]; then
            echo "[SKIP] Already prepared"
        else
            echo "[PREPARE] Downloading + converting..."
            PREP_ARGS="--dataset $ds --data-root $DATA_ROOT --output-root $EPISODES_ROOT --cleanup"
            if [ -n "$MAX_EPISODES" ]; then
                PREP_ARGS="$PREP_ARGS --max-episodes $MAX_EPISODES --force"
            fi
            if ! python scripts/prepare_data.py $PREP_ARGS; then
                echo "[FAIL] Preparation failed"
                return 1
            fi
        fi

        # Verify
        if [ ! -f "$ep_dir/metadata.json" ]; then
            echo "[FAIL] No metadata.json"
            return 1
        fi

        local samples=$(python -c "import json; print(json.load(open('$ep_dir/metadata.json'))['total_samples'])")
        local episodes=$(python -c "import json; print(json.load(open('$ep_dir/metadata.json'))['total_episodes'])")
        echo "[INFO] $episodes episodes, $samples samples"

        # Step 2: Clean download cache
        echo "[CLEAN] Removing download cache..."
        rm -rf "${worker_cache}/datasets" "${worker_cache}/hub" "${worker_cache}/lerobot"
        rm -rf "${DATA_ROOT}/${ds}"

        # Step 3: Upload
        echo "[UPLOAD] Pushing to $repo..."
        python -c "
from huggingface_hub import HfApi, create_repo
api = HfApi()
create_repo('$repo', repo_type='dataset', private=False, exist_ok=True)
api.upload_folder(
    folder_path='$ep_dir',
    repo_id='$repo',
    repo_type='dataset',
    commit_message='Upload $ds ($episodes episodes, $samples samples)',
)
print('[UPLOAD] Complete')
"
        if [ $? -ne 0 ]; then
            echo "[FAIL] Upload failed"
            return 1
        fi

        # Step 4: Clean HDF5
        echo "[CLEAN] Removing HDF5 ($ep_dir)..."
        rm -rf "$ep_dir"

        # Clean worker cache dir
        rm -rf "${worker_cache}"

        echo "[DONE] $ds uploaded successfully"
        echo "Finished: $(date)"

    } >> "$log_file" 2>&1

    return $?
}

worker_loop() {
    local worker_id=$1

    while true; do
        # Atomic read from queue
        local ds
        ds=$(flock "$LOCK_FILE" bash -c '
            if [ -s "'"$QUEUE_FILE"'" ]; then
                head -1 "'"$QUEUE_FILE"'"
                sed -i "1d" "'"$QUEUE_FILE"'"
            fi
        ')

        # Queue empty → exit
        [ -z "$ds" ] && break

        echo "  [W${worker_id}] ⏳ $ds — starting..."

        local start_time=$SECONDS
        if process_dataset "$worker_id" "$ds"; then
            local elapsed=$(( SECONDS - start_time ))
            local mins=$(( elapsed / 60 ))
            local secs=$(( elapsed % 60 ))
            echo "  [W${worker_id}] ✅ $ds — done (${mins}m${secs}s)"
            echo "$ds" >> "$STATUS_DIR/success"
        else
            local elapsed=$(( SECONDS - start_time ))
            local mins=$(( elapsed / 60 ))
            echo "  [W${worker_id}] ❌ $ds — failed (${mins}m, see $LOG_DIR/worker_${worker_id}.log)"
            echo "$ds" >> "$STATUS_DIR/failed"
        fi

        # Progress
        local done_count=0 fail_count=0
        [ -f "$STATUS_DIR/success" ] && done_count=$(wc -l < "$STATUS_DIR/success")
        [ -f "$STATUS_DIR/failed" ] && fail_count=$(wc -l < "$STATUS_DIR/failed")
        local processed=$((done_count + fail_count))
        echo "  [PROGRESS] $processed/$TOTAL done ($done_count success, $fail_count failed)"
    done

    echo "  [W${worker_id}] Worker finished"
}

# ── Launch workers ──
echo "Launching $NUM_WORKERS workers..."
echo ""

PIDS=()
for i in $(seq 1 "$NUM_WORKERS"); do
    worker_loop "$i" &
    PIDS+=($!)
done

# Wait for all workers
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done

# ── Summary ──
echo ""
echo "============================================================"
echo "  Upload Complete"
echo "============================================================"

SUCCESS_COUNT=0
FAIL_COUNT=0
[ -f "$STATUS_DIR/success" ] && SUCCESS_COUNT=$(wc -l < "$STATUS_DIR/success")
[ -f "$STATUS_DIR/failed" ] && FAIL_COUNT=$(wc -l < "$STATUS_DIR/failed")

echo "  Success: $SUCCESS_COUNT / $TOTAL"
echo "  Failed:  $FAIL_COUNT"

if [ -f "$STATUS_DIR/failed" ] && [ -s "$STATUS_DIR/failed" ]; then
    echo ""
    echo "  Failed datasets:"
    while IFS= read -r ds; do
        echo "    ❌ $ds (grep '$ds' $LOG_DIR/worker_*.log)"
    done < "$STATUS_DIR/failed"
fi

echo ""
echo "  Logs: $LOG_DIR/"
echo "============================================================"

# Cleanup temp files
rm -f "$QUEUE_FILE" "$LOCK_FILE"
rm -rf "$STATUS_DIR"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
