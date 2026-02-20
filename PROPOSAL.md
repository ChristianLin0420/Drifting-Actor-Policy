# Drifting-VLA: One-Step Multi-Embodiment Robotic Manipulation

**Technical Proposal & Architecture Guide — v3 (Feb 2026)**

---

## Overview

Drifting-VLA is a unified foundation model for robotic manipulation that generates actions in **one forward pass** (1-NFE) using the Drifting model paradigm.

- **6 embodiment types:** Absolute EEF, Joint Position, Bimanual, Bimanual+Mobile, Dexterous Hand, Delta EEF
- **11 dataset families** (63+ sub-datasets): RLBench, DexGraspNet 2.0, DexWild, ALOHA, DROID, BC-Z, TACO Play, UT Austin MUTEX, CMU Stretch, NYU Franka, Stanford Hydra, Behavior 1K ×50
- **VLM backbone:** Qwen3-VL-2B-Instruct — **Vision-Encoder-Only (Pi0 Style)**, skipping the LLM decoder entirely
- **DiT action generator:** 147M trainable params, 128-dim unified action space (RDT-1B compatible semantic mapping)
- **17 WandB visualization panels** across training, analysis, and evaluation
- **10× faster inference** than diffusion policies (50ms vs 500ms)
- **Single-pass data pipeline:** `prepare_data.py` downloads from HuggingFace and converts to HDF5 directly — no intermediate Arrow files
- **RDT-1B aligned:** Semantic named-field action mapping, 6D rotation, image augmentation, classifier-free guidance condition masking, control frequency conditioning, normalized L2 eval metric

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Datasets](#2-datasets)
3. [Data Storage: Episode-Centric HDF5](#3-data-storage-episode-centric-hdf5)
4. [Unified 128-Dim Action Space (RDT-1B Compatible)](#4-unified-128-dim-action-space-rdt-1b-compatible)
5. [Model Architecture](#5-model-architecture)
6. [Loss Function](#6-loss-function)
7. [Training](#7-training)
8. [Visualization Suite](#8-visualization-suite)
9. [Data Pipeline Commands](#9-data-pipeline-commands)
10. [Docker & Training Server Usage](#10-docker--training-server-usage)
11. [Codebase Structure](#11-codebase-structure)
12. [Key Design Choices (vs RDT-1B)](#12-key-design-choices-vs-rdt-1b)
13. [Implementation Status](#13-implementation-status)
14. [References](#14-references)

---

## 1. Environment Setup

### 1.1 Conda Environment

```bash
conda create -n drifting-vla python=3.10 -y
conda activate drifting-vla

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.5.0 --no-build-isolation  # optional, for H100/B200
pip install lerobot datasets "transformers>=4.57" peft accelerate \
    h5py wandb opencv-python-headless matplotlib scikit-learn \
    einops tqdm rich open3d mujoco qwen-vl-utils

cd Drifting-Actor-Policy && pip install -e .
```

### 1.2 System Requirements

| Component | Dev Server | Training Server |
|-----------|-----------|-----------------|
| **GPU** | 1× A40 48GB | 8× A40 48GB (tested) or 8× H100 80GB |
| **Disk** | 500 GB SSD | 5 TB SSD |
| **Python** | 3.10 | 3.10 |
| **CUDA** | 12.1+ | 12.1+ |

---

## 2. Datasets

### 2.1 Dataset Registry (11 Families Integrated)

| # | Dataset | HF Repo | Samples | Act Dim | Embodiment | Views | Temporal | Lang |
|---|---------|---------|---------|---------|------------|-------|----------|------|
| 1 | **rlbench** | `hqfang/rlbench-18-tasks` | 270K | 8 | gripper_eef | 2-5 cams | ✅ | ✅ |
| 2 | **dexgraspnet** | `lhrlhr/DexGraspNet2.0` | 500K | 23 | dex_hand | 8 rendered | ❌ static | ✅ |
| 3 | **aloha** | `lerobot/aloha_sim_transfer_cube_human_image` | 20K | 14 | bimanual | 1 | ✅ | ✅ |
| 4 | **droid** | `lerobot/droid_1.0.1` | 25.5M | 7 | gripper_joint | 3 | ✅ | ✅ |
| 5 | **bc_z** | `lerobot/berkeley_autolab_ur5` | 98K | 7 | gripper_joint | 2 | ✅ | ✅ |
| 6 | **taco_play** | `lerobot/taco_play` | 238K | 7 | gripper_joint | 2 | ✅ | ❌ |
| 7 | **utaustin_mutex** | `lerobot/utaustin_mutex` | 362K | 7 | gripper_delta_eef | 2 | ✅ | ✅ |
| 8 | **cmu_stretch** | `lerobot/cmu_stretch` | 25K | 8 | gripper_delta_eef | 1 | ✅ | ❌ |
| 9 | **nyu_franka** | `lerobot/nyu_franka_play_dataset` | 45K | 15 | bimanual | 2 | ✅ | ❌ |
| 10 | **stanford_hydra** | `lerobot/stanford_hydra_dataset` | 358K | 7 | gripper_delta_eef | 2 | ✅ | ❌ |
| 11-60 | **behavior1k_t0000–t0049** | `lerobot/behavior1k-task{i:04d}` | ~430K each | 23 | bimanual_mobile | 3 RGB | ✅ | ✅ |

**Dexterous hand dataset (integrated, verified):**

| # | Dataset | Source | Episodes | Hand | Act Dims | Temporal | Images |
|---|---------|--------|----------|------|----------|----------|--------|
| 12 | **DexWild** | `YingyangPolarBear/DexWild` | 9.5K | LEAP v2 (16 DOF) + wrist | 23 (7 wrist + 16 finger) | ✅ | ✅ 2 thumb cams |

---

## 3. Data Storage: Episode-Centric HDF5

### 3.1 Single-Pass Pipeline (No Intermediate Files)

**OLD (removed):** `download_datasets.py` → Arrow files → `convert_to_episodes.py` → HDF5
**NEW:** `prepare_data.py` → HDF5 directly (stream from HuggingFace, no Arrow saved to disk)

```bash
# Prepare all datasets (full)
python scripts/prepare_data.py --all --cleanup

# Prepare specific dataset
python scripts/prepare_data.py --dataset aloha

# Quick smoke-test (2 episodes per dataset)
python scripts/prepare_data.py --all --max-episodes 2 --cleanup

# Force re-prepare (overwrite existing)
python scripts/prepare_data.py --dataset rlbench --force
```

The `--cleanup` flag removes HuggingFace cache after each dataset to save disk space — critical for servers with limited storage.

### 3.2 Episode File Layout

```
data/episodes/{dataset_name}/
├── metadata.json                  # Index + stats + schema
├── ep_000000.hdf5
├── ep_000001.hdf5
└── ...

Single episode HDF5:
ep_000000.hdf5
├── images/
│   ├── view_0     [T, H, W, 3] uint8    ← raw RGB
│   └── view_1     [T, H, W, 3] uint8
├── actions        [T, 128] float32       ← 128-dim unified, pre-mapped via named fields
├── action_mask    [128] bool             ← active dims (from DATASET_FIELD_FORMATS)
├── proprio        [T, 128] float32       ← 128-dim unified proprioception
├── language       scalar string
└── attrs:
    ├── dataset_name, embodiment_id, episode_length
    ├── n_views, image_size
    └── action_dim
```

### 3.3 Temporal vs Static Dataset Handling

| | Temporal (rlbench, bc_z, behavior1k, ...) | Static (dexgraspnet) |
|---|---|---|
| **Images** | 3 frames × V views = `[3V, 3, 448, 448]` | 1 frame × 8 views = `[8, 3, 448, 448]` |
| **Actions** | 16 consecutive `[16, 128]` from episode | 1 grasp tiled `[16, 128]` (all identical) |
| **time_id** | 0, 1, 2 (t-1, t, t+1) | 0 for all views (same moment) |
| **camera_id** | 0..V-1 per view | 0..7 for 8 views |

### 3.4 Image Augmentation (RDT-1B Style)

During training, 50% of images receive `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)`. Controlled via `--image-aug` flag.

### 3.5 Condition Masking for Classifier-Free Guidance

With probability `cond_mask_prob` (default 0.1):
- Language → replaced with `""`
- Images → replaced with black frames
- Proprioception → zeroed out

This enables classifier-free guidance at inference time. Controlled via `--cond-mask-prob` flag.

---

## 4. Unified 128-Dim Action Space (RDT-1B Compatible)

### 4.1 Semantic Named-Field Mapping

The 128-dim action space uses **RDT-1B compatible semantic named-field mapping**, extended with dexterous hand slots. Each physical quantity maps to a specific named slot. A single robot can fill **multiple regions simultaneously** (e.g., both joint positions AND EEF positions).

```
128-dim Semantic State Vector Layout:

  [0, 10):    right arm joint positions      arm_joint_{0-9}_pos
  [10, 15):   right gripper joint positions  gripper_joint_{0-4}_pos / gripper_open
  [15, 25):   right arm joint velocities     arm_joint_{0-9}_vel
  [25, 30):   right gripper joint velocities gripper_joint_{0-4}_vel
  [30, 33):   right EEF positions            eef_pos_{x,y,z}
  [33, 39):   right EEF 6D rotation          eef_angle_{0-5}  (continuous, NOT quaternion)
  [39, 42):   right EEF velocities           eef_vel_{x,y,z}
  [42, 45):   right EEF angular velocities   eef_angular_vel_{roll,pitch,yaw}
  [45, 50):   right dex finger joints        right_finger_joint_{0-4}_pos
  [50, 60):   left arm joint positions       left_arm_joint_{0-9}_pos
  [60, 65):   left gripper joint positions   left_gripper_joint_{0-4}_pos
  [65, 75):   left arm joint velocities      left_arm_joint_{0-9}_vel
  [75, 80):   left gripper joint velocities  left_gripper_joint_{0-4}_vel
  [80, 83):   left EEF positions             left_eef_pos_{x,y,z}
  [83, 89):   left EEF 6D rotation           left_eef_angle_{0-5}
  [89, 92):   left EEF velocities            left_eef_vel_{x,y,z}
  [92, 95):   left EEF angular velocities    left_eef_angular_vel_{...}
  [95, 100):  left dex finger joints         left_finger_joint_{0-4}_pos
  [100, 103): base velocities                base_vel_{x,y}, base_angular_vel
  [103, 119): dexterous finger joints (16)   dex_finger_joint_{0-15}_pos
  [119, 128): reserved
```

**162 named fields** in `STATE_VEC_IDX_MAPPING` — fully compatible with RDT-1B's 46 existing datasets.

### 4.2 Per-Dataset Field Format Mapping

Each dataset defines a `DATASET_FIELD_FORMATS` list that maps native action dimensions to named fields:

```python
# Example: DROID (7-dim joint) → fills [0:6] + [10]
DATASET_FIELD_FORMATS['droid'] = [
    'arm_joint_0_pos', ..., 'arm_joint_5_pos', 'gripper_open'
]

# Example: ALOHA (14-dim bimanual) → fills [50:56]+[60] (left) + [0:6]+[10] (right)
DATASET_FIELD_FORMATS['aloha'] = [
    'left_arm_joint_0_pos', ..., 'left_gripper_open',
    'arm_joint_0_pos', ..., 'gripper_open',
]

# Example: DexWild (23-dim) → fills [30:37] (EEF) + [103:119] (16 fingers)
DATASET_FIELD_FORMATS['dexwild'] = [
    'eef_pos_x', 'eef_pos_y', 'eef_pos_z',
    'eef_angle_0', ..., 'eef_angle_3',
    'dex_finger_joint_0_pos', ..., 'dex_finger_joint_15_pos',
]
```

### 4.3 `assemble_state_vec()` — RDT-1B Style Mapping

```python
def assemble_state_vec_batch(actions, field_names):
    """[T, D_native] → [T, 128] + [128] mask, via named fields."""
    for i, name in enumerate(field_names):
        idx = STATE_VEC_IDX_MAPPING[name]
        unified[:, idx] = actions[:, i]
        mask[idx] = 1.0
```

### 4.4 6D Rotation Representation

Rotation conversion utilities included for datasets that report quaternions:

```python
quaternion_to_6d_rotation(quat)   # [x,y,z,w] → [6] (first 2 cols of rotation matrix)
rotation_6d_to_quaternion(rot6d)  # [6] → [x,y,z,w] (via Gram-Schmidt + cross product)
```

Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019.

### 4.5 Six Embodiment Types

| ID | Embodiment | Active Regions | Datasets |
|----|-----------|----------------|----------|
| 0 | `gripper_eef` | EEF `[30:39]` + grip `[10]` | rlbench |
| 1 | `gripper_joint` | Joints `[0:6]` + grip `[10]` | droid, bc_z, taco_play |
| 2 | `bimanual` | Left `[50:60]` + Right `[0:10]` | aloha, nyu_franka |
| 3 | `dex_hand` | EEF `[30:37]` + Fingers `[103:119]` | dexgraspnet, dexwild |
| 4 | `gripper_delta_eef` | Delta EEF `[30:36]` + grip `[10]` | utaustin_mutex, stanford_hydra, cmu_stretch |
| 5 | `bimanual_mobile` | Left+Right arms + Base `[100:103]` | behavior1k ×50 |

### 4.6 Normalization (RDT-1B Style Robust [-1, 1])

```python
action_min = mean - 5 * max(std, 0.01)   # clip outliers at ±5σ
action_max = mean + 5 * max(std, 0.01)
center = (action_min + action_max) / 2
half_range = (action_max - action_min) / 2
normalized = clip((action - center) / half_range, -1.0, 1.0)
```

---

## 5. Model Architecture

### 5.1 System 1: VLM Backbone — Vision-Encoder-Only (Pi0 Style)

```
Pipeline:
  images → Qwen3-VL ViT Encoder (model.visual, ~407M) → [N_vis, 2048]
  text   → embed_tokens lookup (~311M, <1ms)           → [L_text, 2048]
  [visual ⊕ text] → masked pooling                     → [B, 2048]
  → proj_seq(Linear+LN)  → [B, L, 768]  c_seq
  → proj_pool(Linear+LN) → [B, 768]     c_pool
  → + camera_embed(cam_id) + time_embed(frame_id) per-token
```

| Mode | ViT Encoder | LLM Decoder | Trainable VLM Params | Use Case |
|------|------------|------------|---------------------|----------|
| `frozen` | Frozen | ❌ Deleted | 0 | Debug, fast iteration |
| `lora` | LoRA (r=16) | ❌ Deleted | ~5M (1.2%) | **Pre-training (recommended)** |
| `precomputed` | ❌ Not loaded | ❌ Not loaded | 0 | Fastest (offline features) |

### 5.2 System 2: Drifting DiT (147M params)

```
Inputs:
  c_seq:          [B, L, 768]   VLM features with camera+time embeddings
  c_pool:         [B, 768]      Global context
  noise:          [B, T, 64]    Random Gaussian
  embodiment_id:  [B]           0-5
  proprio:        [B, 128]      Proprioception (128-dim unified)
  ctrl_freqs:     [B]           Control frequency in Hz (RDT-1B style)

Pipeline:
  c_global = c_pool + Embed(embodiment_id) + ProprioMLP(proprio) + CFG_embed(α)
           + FreqEmbed(ctrl_freqs)     ← NEW: control frequency conditioning
  noise → NoiseTokenizer → [B, T, 768]
  → CrossAttention(Q=noise, K/V=c_seq, kv_mask=vlm_attn_mask) → [B, T, 768]
  → DiT (12 blocks, adaLN-Zero, RoPE, QK-Norm, SwiGLU) → [B, T, 768]
  → ActionHead MLP(768 → 128) → [B, T, 128] actions
```

**Total trainable (LoRA mode):** LoRA ~5M + Projector 3.2M + DiT 147M = **~155M trainable**

### 5.3 Control Frequency Conditioning (RDT-1B Style)

Different datasets have different control frequencies (e.g., ALOHA=50Hz, DROID=15Hz). The model conditions on `ctrl_freqs` via a learned embedding:

```python
if ctrl_freqs is not None and self.freq_embed is not None:
    c_global = c_global + self.freq_embed(ctrl_freqs.clamp(0, 100).long())
```

Config: `configs/dataset_control_freq.json`

### 5.4 Memory Budget (Per A40 46GB, batch=32/GPU, LoRA mode)

| Component | Memory |
|-----------|--------|
| Qwen3-VL ViT encoder (bf16) | ~1.5 GB |
| Text embed_tokens | ~0.5 GB |
| LoRA adapters (r=16) | ~20 MB |
| DiT base (768d, 12L) | ~0.6 GB |
| ViT activations (B=32, 3 imgs) | ~18 GB |
| DiT activations + optimizer | ~16 GB |
| **Total** | **~37 GB** ✅ |

---

## 6. Loss Function

### 6.1 Hybrid Loss

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{drift}}$$

**Masked MSE:** Only computed on active dims via `action_mask [128]`.

**Drifting Loss:** Per-embodiment pos/neg queues prevent cross-embodiment noise.

Multi-temperature kernels: τ ∈ {0.02, 0.05, 0.2}, double softmax normalized.

---

## 7. Training

### 7.1 Quick Test (Single GPU)

```bash
python scripts/train.py \
    --datasets aloha bc_z rlbench dexwild \
    --episodes-root ./data/episodes \
    --batch-size 4 --grad-accumulation 1 \
    --max-steps 10 --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base \
    --wandb-mode disabled
```

### 7.2 Multi-GPU Pre-training (8×A40, tested & verified)

```bash
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/train.py \
    --datasets aloha bc_z behavior1k_t0000 droid cmu_stretch dexgraspnet \
        dexwild nyu_franka rlbench stanford_hydra taco_play utaustin_mutex \
    --episodes-root ./data/episodes \
    --batch-size 32 --grad-accumulation 2 \
    --max-steps 10000 --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base \
    --num-workers 8 --image-aug --cond-mask-prob 0.1 \
    --log-every 50 --eval-every 500 --save-every 2500 \
    --wandb-mode online --wandb-project drifting-vla
```

**Effective batch:** 32 × 2 = **64** per GPU, × 8 GPUs = **512 global**

### 7.3 CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | `base` | `small` (512d/8L) / `base` (768d/12L) / `large` (1024d/24L) |
| `--vlm-mode` | `frozen` | `frozen` / `lora` / `full` |
| `--lora-r` | 16 | LoRA rank (16=pretrain, 64=posttrain) |
| `--vlm-lr-scale` | 0.1 | VLM LR = base_lr × scale |
| `--batch-size` | 16 | Per-GPU batch size |
| `--grad-accumulation` | 2 | Gradient accumulation steps |
| `--lr` | 4e-4 | Base learning rate |
| `--max-steps` | 10000 | Total training steps |
| `--loss-type` | `hybrid` | `mse` / `pure_drift` / `hybrid` |
| `--datasets` | `['rlbench']` | Space-separated dataset names |
| `--episodes-root` | `./data/episodes` | Episode HDF5 root directory |
| `--num-workers` | 8 | DataLoader workers (RDT-1B standard) |
| `--image-aug` | False | Enable ColorJitter image augmentation |
| `--cond-mask-prob` | 0.1 | CFG condition masking probability |
| `--eval-batches` | 50 | Number of eval batches per evaluation |
| `--wandb-mode` | `disabled` | `online` / `offline` / `disabled` |
| `--log-every` | 50 | Scalar logging interval |
| `--eval-every` | 500 | Evaluation interval |
| `--save-every` | 2000 | Checkpoint save interval |
| `--use-flash-attn` | False | Enable Flash Attention 2 (H100/B200) |

---

## 8. Visualization Suite

### 8.1 Panel Overview (17 Panels)

| Category | Panel | Description | Frequency | What to Look For |
|----------|-------|-------------|-----------|-----------------|
| **A: Drift** | `A1_drifting_field` | Drift vectors + magnitude histogram | 500 steps | Arrows should point toward GT |
| | `A2_drift_magnitude_trend` | λ_V convergence with EMA | 200 steps | Should decrease; plateau = increase drift_weight |
| | `A3_prediction_scatter` | Dense pred vs GT scatter | 500 steps | Points cluster along y=x diagonal |
| | `A4_per_dim_error_bar` | MAE per active dim, color-coded | 500 steps | Identifies worst dims per embodiment |
| | `A5_temperature_loss` | Per-temperature kernel loss | 500 steps | Low-τ/High-τ < 0.5 = fine detail learned |
| **B: Actions** | `B1_action_distribution` | Histogram grid (GT vs Pred) | 500 steps | Coverage → 100%; <20% = mode collapse |
| | `B2_action_error_heatmap` | Dims × timestep heatmap | 500 steps | Reveals temporal error patterns |
| | `B3_per_region_summary` | MAE/Corr/Coverage per region | 1000 steps | Quick embodiment-level quality check |
| | `B4_trajectory_2d` | 4 samples × XY + XZ views | 1000 steps | Gray error lines should shrink over training |
| | `B5_temporal_error_curve` | Error vs timestep + confidence | 500 steps | Error growth >50% = model degrades over horizon |
| **C: Transport** | `C1_sample_transport` | PCA transport map with arrows | 500 steps | Improvement% should be >50% |
| | `C2_mode_coverage` | PCA GT vs Pred scatter | 2000 steps | Pred cloud should overlap GT cloud |
| | `C3_action_smoothness` | Velocity/jerk/FFT analysis | 1000 steps | Pred should match GT smoothness |
| **D: Dashboard** | `D1_eval_dashboard` | Per-dataset MSE/MAE/Corr/Coverage bars | Every eval | Main eval summary |
| | `D2_training_curves` | Loss + MSE + Drift Norm | 500 steps | All should decrease |
| | `D3_dataset_learning_curves` | Per-dataset MSE & Corr over time | Every eval | Which datasets improve vs stagnate |
| | `D4_data_balance` | Configured vs actual sampling | Every eval | Detect sampling imbalance |

### 8.2 Eval Metrics (RDT-1B Aligned)

| Metric | Description | Source |
|--------|-------------|--------|
| `MSE` | Mean squared error on active dims | Standard |
| `MAE` | Mean absolute error on active dims | Standard |
| `Correlation` | Per-dim Pearson correlation | Standard |
| `Coverage %` | pred_range / GT_range × 100 | New |
| **`Normalized L2`** | `RMSE / (state_norm + ε)` per dim, averaged | **RDT-1B style** |

---

## 9. Data Pipeline Commands

### 9.1 Full Pipeline

```bash
# Step 1: Prepare data (single-pass, no Arrow)
python scripts/prepare_data.py --all --cleanup

# Step 2: Validate
python scripts/validate_training.py --all --tier 4

# Step 3: Train
torchrun --nproc_per_node=8 scripts/train.py \
    --datasets aloha bc_z behavior1k_t0000 droid cmu_stretch dexgraspnet \
        dexwild nyu_franka rlbench stanford_hydra taco_play utaustin_mutex \
    --episodes-root ./data/episodes \
    --batch-size 32 --grad-accumulation 2 --max-steps 10000 \
    --vlm-mode lora --lora-r 16 --num-workers 8 \
    --image-aug --cond-mask-prob 0.1 \
    --wandb-mode online --wandb-project drifting-vla
```

### 9.2 Quick Smoke-Test

```bash
python scripts/prepare_data.py --dataset aloha --max-episodes 2 --force
python scripts/train.py --datasets aloha --batch-size 4 --max-steps 10 --test
```

### 9.3 Dataset-by-Dataset (Memory-Managed)

For servers with limited disk, prepare and verify one dataset at a time:

```bash
python scripts/prepare_data.py --dataset aloha --cleanup --force
python scripts/prepare_data.py --dataset bc_z --cleanup --force
python scripts/prepare_data.py --dataset rlbench --cleanup --force
# ... etc
```

---

## 10. Docker & Training Server Usage

### 10.1 Build Docker Image

```bash
docker build -f docker/Dockerfile.pretrain -t drifting-vla:pretrain .
```

### 10.2 Interactive Development

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -it drifting-vla:pretrain bash
```

### 10.3 Data Preparation (Inside Container)

```bash
docker run --gpus all \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    drifting-vla:pretrain \
    python scripts/prepare_data.py --all --cleanup
```

### 10.4 Training (8 GPUs)

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    -v $(pwd):/workspace \
    -v $(pwd)/data:/workspace/data \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    drifting-vla:pretrain \
    torchrun --nproc_per_node=8 scripts/train.py \
        --datasets aloha bc_z behavior1k_t0000 droid cmu_stretch dexgraspnet \
            dexwild nyu_franka rlbench stanford_hydra taco_play utaustin_mutex \
        --episodes-root ./data/episodes \
        --batch-size 32 --grad-accumulation 2 \
        --max-steps 10000 --lr 1e-4 \
        --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
        --loss-type hybrid --model-size base \
        --num-workers 8 --image-aug --cond-mask-prob 0.1 \
        --wandb-mode online --wandb-project drifting-vla
```

### 10.5 Dockerfile Features

- **Base:** `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- **Explicit COPY:** Only source code copied, never `data/` (`.dockerignore` + build-time guard)
- **Entrypoint:** Auto-login WandB via `WANDB_API_KEY` env var
- **Health check:** Verifies all key imports at build time
- **Environment:** All training env vars pre-set (HDF5, NCCL, CUDA allocator)

### 10.6 Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | `1` | Real-time stdout logging |
| `HDF5_USE_FILE_LOCKING` | `FALSE` | Prevent HDF5 lock conflicts in multi-worker DataLoaders |
| `NCCL_P2P_DISABLE` | `1` | Prevent NCCL hangs on PCIe GPU topologies (A40) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce CUDA memory fragmentation |
| `WANDB_API_KEY` | your key | WandB authentication |

---

## 11. Codebase Structure

```
Drifting-Actor-Policy/
├── drifting_vla/
│   ├── models/
│   │   ├── drifting_vla.py          # Main model (VLM+DiT+ctrl_freq conditioning)
│   │   ├── vlm_backbone.py          # Vision-encoder-only VLM (Pi0 style)
│   │   ├── dit.py                   # DiT transformer (adaLN-Zero, RoPE, SwiGLU)
│   │   ├── fusion.py                # Cross-attention with KV masking
│   │   └── action_decoder.py        # NoiseTokenizer
│   ├── data/
│   │   ├── action_mapping.py        # RDT-1B semantic 128-dim + assemble_state_vec
│   │   ├── episode_dataset.py       # HDF5 loader + image_aug + cond_mask
│   │   ├── unified_dataset.py       # Multi-dataset wrapper + collate + sampler
│   │   └── sample_queue.py          # Pos/neg queues for drifting loss
│   └── training/
│       ├── losses.py                # DriftingLoss (multi-temp, per-embodiment)
│       ├── drifting_field.py        # V(a) drift field computation
│       ├── visualizations.py        # 17 WandB panels
│       └── ema.py                   # Exponential moving average
├── scripts/
│   ├── train.py                     # Full training loop (DDP, eval, normalized L2)
│   ├── prepare_data.py              # Single-pass download + convert (no Arrow)
│   ├── validate_training.py         # 4-tier validation suite
│   └── precompute_vlm_features.py   # Offline VLM feature extraction
├── docker/
│   ├── Dockerfile.pretrain          # Training image (RLBench-style, explicit COPY)
│   ├── Dockerfile.base              # Base dependencies
│   └── Dockerfile.rlbench           # RLBench with CoppeliaSim
├── configs/
│   ├── dataset_control_freq.json    # Per-dataset Hz for ctrl_freq conditioning
│   ├── zero2.json                   # DeepSpeed ZeRO-2 config (ready for scaling)
│   ├── training/baseline.yaml
│   └── model/dit_b2.yaml
├── tests/
├── PROPOSAL.md                      # This file
├── pyproject.toml
└── requirements/
```

---

## 12. Key Design Choices (vs RDT-1B)

| Decision | Drifting-VLA | RDT-1B | Rationale |
|----------|-------------|--------|-----------|
| **Action mapping** | Semantic named-field (compatible) | Semantic named-field | Cross-embodiment transfer via shared slots |
| **Rotation** | 6D continuous (conversion utilities) | 6D continuous | Avoids quaternion discontinuities |
| **Velocity channels** | Included in mapping | Included | Joint + EEF velocity for dynamics |
| **Dex hand** | Extended `[45:50]`+`[95:100]`+`[103:119]` | Not supported | 16 DOF finger joints for DexWild/DexGraspNet |
| **VLM** | Qwen3-VL ViT encoder only (Pi0) | SigLIP (frozen, detached) | Fine-tunable with LoRA |
| **Text** | Qwen3-VL tokenizer → embeddings | T5-XXL (4096-dim) | Unified VLM, no separate text encoder |
| **Image augmentation** | ColorJitter (50% training) | ColorJitter + corruption | Generalization |
| **Condition masking** | 10% random drop (lang/img/proprio) | 10% random mask | CFG training |
| **Action generation** | 1-NFE Drifting (one forward pass) | 5-step DDPM diffusion | 10× faster inference |
| **DataLoader** | 8 workers, persistent, prefetch | 8 workers, persistent | Prevent data bottleneck |
| **Eval** | Separate uniform DataLoader + normalized L2 | Separate sample loader + MSE/L2 | Clean evaluation |
| **Ctrl frequency** | `freq_embed(Hz)` conditioning | `ctrl_freqs` conditioning | Time-scale awareness |
| **Data pipeline** | Single-pass (no Arrow) | Producer-consumer (TFRecord → buffer) | Simpler, less disk |
| **Parallelism** | PyTorch DDP (ZeRO-2 ready) | DeepSpeed ZeRO-2 | DDP sufficient at 155M params |

---

## 13. Implementation Status

### 13.1 Completed ✅

| Component | Status | Details |
|-----------|--------|---------|
| RDT-1B semantic action mapping (162 fields) | ✅ | `STATE_VEC_IDX_MAPPING`, `assemble_state_vec_batch()` |
| Per-dataset `DATASET_FIELD_FORMATS` | ✅ | All 11+ datasets with named-field mappings |
| 6D rotation conversion utilities | ✅ | `quaternion_to_6d_rotation()`, `rotation_6d_to_quaternion()` |
| Single-pass data pipeline (`prepare_data.py`) | ✅ | No Arrow intermediates, `--cleanup` flag |
| Image augmentation (ColorJitter) | ✅ | 50% of training images, `--image-aug` flag |
| CFG condition masking | ✅ | `--cond-mask-prob 0.1` for lang/img/proprio |
| Control frequency conditioning | ✅ | `freq_embed` in DriftingVLA, `dataset_control_freq.json` |
| Separate eval DataLoader | ✅ | Uniform sampling, no augmentation/masking |
| Normalized L2 metric | ✅ | RDT-1B style `RMSE / (state_norm + ε)` |
| DataLoader optimization | ✅ | 8 workers, `persistent_workers`, `prefetch_factor=2` |
| DeepSpeed ZeRO-2 config | ✅ | `configs/zero2.json` ready for scaling |
| Vision-encoder-only VLM (Pi0 style) | ✅ | ViT + text embed, LLM decoder deleted |
| 17 WandB visualization panels | ✅ | A1-A5, B1-B5, C1-C3, D1-D4 |
| Multi-GPU DDP training (8×A40) | ✅ | Stable at batch=32, grad_accum=2 |
| Docker image (RLBench-style) | ✅ | Explicit COPY, entrypoint, health check |
| Old scripts removed | ✅ | `download_datasets.py`, `convert_to_episodes.py` deleted |

### 13.2 Training Results (8×A40, LoRA, 14 datasets, 10K steps)

**Run:** `base_0218_1630` — WandB: `crlc112358/drifting-vla/runs/fvkls5id`

**Final evaluation (Step 10000):**

| Embodiment | Datasets | MSE ↓ | MAE ↓ |
|------------|----------|-------|-------|
| gripper_eef | rlbench | 0.0198 | 0.0874 |
| gripper_joint | droid, taco_play, bc_z | 0.0140 | 0.0751 |
| gripper_delta_eef | stanford_hydra, cmu_stretch, utaustin_mutex | 0.0106 | 0.0583 |
| bimanual | aloha, nyu_franka | 0.0063 | 0.0501 |
| bimanual_mobile | behavior1k ×3 | 0.0024 | 0.0267 |
| dex_hand | dexwild, dexgraspnet | 0.0323 | 0.1338 |
| **Overall** | **14 datasets, 6 embodiments** | **0.0127** | **0.0668** |

**Best per-dataset:** utaustin_mutex (corr=0.856), nyu_franka (corr=0.876), aloha (corr=0.834)

---

## 14. References

1. **Drifting:** One-Step Generation via Training-Time Distribution Evolution. arXiv:2602.04770, 2025.
2. **RDT-1B:** Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864, 2024.
3. **Pi0:** A Vision-Language-Action Flow Model. arXiv:2410.24164, 2024.
4. **DexGraspNet 2.0:** Learning Generative Dexterous Grasping. CoRL 2024.
5. **Qwen3-VL:** [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
6. **Open X-Embodiment:** RT-X Models. arXiv:2310.08864, 2023.
7. **Behavior 1K:** OmniGibson Simulation. [HuggingFace](https://huggingface.co/collections/lerobot/behavior-1k)
8. **Diffusion Policy:** Visuomotor Policy Learning via Action Diffusion. RSS 2023.
9. **Octo:** An Open-Source Generalist Robot Policy. arXiv:2405.12213, 2024.
10. **DexWild:** Dexterous Hand Manipulation in the Wild. [HuggingFace](https://huggingface.co/datasets/boardd/dexwild-dataset)
11. **6D Rotation:** Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019.
12. **PEFT (LoRA):** Low-Rank Adaptation of Large Language Models. ICLR 2022.
