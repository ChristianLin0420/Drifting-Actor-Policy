export TMPDIR=~/tmp && mkdir -p $TMPDIR

# ============================================================
# Step 1: Download datasets (10K samples each, with images + language)
# Skips if arrow_data/ or .npy data already exists
# ============================================================
python scripts/download_datasets.py --n-samples 10000 --dataset aloha
python scripts/download_datasets.py --n-samples 10000 --dataset rlbench
python scripts/download_datasets.py --n-samples 10000 --dataset bc_z
python scripts/download_datasets.py --n-samples 10000 --dataset taco_play
python scripts/download_datasets.py --n-samples 10000 --dataset utaustin_mutex
python scripts/download_datasets.py --n-samples 10000 --dataset cmu_stretch
python scripts/download_datasets.py --n-samples 10000 --dataset nyu_franka
python scripts/download_datasets.py --n-samples 10000 --dataset stanford_hydra
python scripts/download_datasets.py --n-samples 10000 --dataset behavior1k_t0000
python scripts/download_datasets.py --n-samples 10000 --dataset dexgraspnet

# ============================================================
# Step 2: Render 8-view RGB for DexGraspNet (requires meshdata)
# ============================================================
python scripts/render_dexgraspnet.py --max-scenes 100 --n-views 8

# ============================================================
# Step 3: 8-GPU training
#
# 10K samples × 9 datasets = ~90K total + DexGraspNet scenes
# Effective batch = 16 × 8 GPUs × 2 accum = 256
# 10K steps at 0.15 steps/s ≈ 18 hours (with live VLM)
# Pre-compute VLM features first for 10× speedup
# ============================================================
torchrun --nproc_per_node=8 scripts/train.py \
    --datasets aloha rlbench bc_z taco_play utaustin_mutex cmu_stretch nyu_franka stanford_hydra behavior1k_t0000 dexgraspnet \
    --max-samples 10000 \
    --batch-size 32 \
    --grad-accumulation 4 \
    --max-steps 10000 \
    --lr 1e-4 \
    --loss-type hybrid \
    --wandb-mode online \
    --log-every 50 \
    --save-every 1000 \
    --eval-every 1000 \
    --data-root ./data
