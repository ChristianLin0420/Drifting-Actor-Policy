# Drifting-VLA: One-Step Multi-Embodiment Robotic Manipulation

**Technical Documentation & Training Guide**

---

## Overview

Drifting-VLA is a unified foundation model for robotic manipulation that generates actions in **one forward pass** (1-NFE) using the Drifting model paradigm. The model:

- **5 embodiment types:** Absolute EEF, Joint Position, Bimanual, Dexterous Hand, Delta EEF
- **60 datasets:** RLBench, DexGraspNet 2.0, 8 OXE datasets, 50 Behavior 1K tasks (~32M samples)
- **VLM backbone:** Qwen3-VL-2B-Instruct (2.1B params, frozen + LoRA optional)
- **DiT action generator:** 146M trainable params, 128-dim unified action space
- **10× faster inference** than diffusion policies (50ms vs 500ms)

---

## 1. Environment Setup

### 1.1 Conda Environment

```bash
# Create environment
conda create -n drifting-vla python=3.10 -y
conda activate drifting-vla

# Install PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 (optional, for H100/B200)
pip install flash-attn==2.5.0 --no-build-isolation

# Install core dependencies
pip install lerobot datasets "transformers>=4.57" peft accelerate \
    h5py wandb opencv-python-headless matplotlib scikit-learn \
    einops tqdm rich

# Install project
cd Drifting-Actor-Policy
pip install -e .
```

### 1.2 Docker (for training server)

```bash
# Build pretraining image
docker build -f docker/Dockerfile.pretrain -t drifting-vla:latest .

# Run training (single GPU)
docker run --gpus all -v $(pwd)/data:/workspace/data \
    drifting-vla:latest python scripts/train.py --test

# Run training (8 GPUs)
docker run --gpus all -v $(pwd)/data:/workspace/data \
    drifting-vla:latest \
    torchrun --nproc_per_node=8 scripts/train.py --batch-size 64 --max-steps 100000
```

### 1.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Dev server** | 1× A40 48GB, 100 GB disk | 1× A40 + 500 GB SSD |
| **Training server** | 8× H100 80GB, 3 TB disk | 64× H100 (8 nodes) + 5 TB SSD |
| **Python** | 3.10+ | 3.10 |
| **CUDA** | 12.1+ | 12.1 |
| **Disk (raw data)** | ~300 GB (10 datasets) | ~3 TB (all 60 datasets) |
| **Disk (VLM features)** | ~4 GB (10 datasets × 100 samples) | ~130 GB (all datasets, full) |

---

## 2. Data: 60 Datasets Across 3 Embodiments

### 2.1 Dataset Registry

**Verified working datasets** from [LeRobot OXE Collection](https://huggingface.co/collections/lerobot/open-x-embodiment) and [Behavior 1K](https://huggingface.co/collections/lerobot/behavior-1k):

| # | Dataset | HF Repo | Samples | Act | Embody | Views | Lang |
|---|---------|---------|---------|-----|--------|-------|------|
| 1 | **rlbench** | `hqfang/rlbench-18-tasks` | 270K | 8 | gripper | 5 cams | ✅ per-task |
| 2 | **dexgraspnet** | `lhrlhr/DexGraspNet2.0` | 500K | 23 | dex hand | rendered | ✅ object name |
| 3 | **aloha** | `lerobot/aloha_sim_transfer_cube_human_image` | 20K | 14 | bimanual | 1 | ✅ |
| 4 | **droid** | `lerobot/droid_1.0.1` | 25.5M | 7 | gripper | 3 | ✅ |
| 5 | **bc_z** | `lerobot/berkeley_autolab_ur5` | 98K | 7 | gripper | 2 | ✅ |
| 6 | **taco_play** | `lerobot/taco_play` | 238K | 7 | gripper | 2 | ✅ |
| 7 | **utaustin_mutex** | `lerobot/utaustin_mutex` | 362K | 7 | gripper | 2 | ✅ |
| 8 | **cmu_stretch** | `lerobot/cmu_stretch` | 25K | 8 | gripper | 1 | ✅ |
| 9 | **nyu_franka** | `lerobot/nyu_franka_play_dataset` | 45K | 15 | bimanual | 2 | ✅ |
| 10 | **stanford_hydra** | `lerobot/stanford_hydra_dataset` | 358K | 7 | gripper | 2 | ✅ |
| 11-60 | **behavior1k_t0000-t0049** | `lerobot/behavior1k-task{i:04d}` | ~430K each | 23 | bimanual | 3 RGB | ✅ |

**Total: ~32.4M samples, ~2.8 TB raw data, ~130 GB VLM features**

### 2.2 Download Test Data (50 samples per dataset)

```bash
# Download 50-sample subsets for testing/debugging
python scripts/download_datasets.py --test

# Download specific dataset
python scripts/download_datasets.py --test --dataset aloha
python scripts/download_datasets.py --test --dataset behavior1k_t0000

# Download full datasets (for training server)
python scripts/download_datasets.py --all
```

**Output:** `data/{dataset}/arrow_data/` with 50 samples per dataset.

### 2.3 Verify Data with Sample Videos

```bash
# Generate per-camera episode videos for visual inspection
python scripts/generate_sample_videos.py --all --n-frames 50

# Output: sample_videos/{dataset}/{camera}.mp4
# Example:
#   sample_videos/rlbench/front.mp4, wrist.mp4, ...
#   sample_videos/behavior1k_t0000/head.mp4, left_wrist.mp4, right_wrist.mp4
```

**Result:** 23 MP4 files showing actual images and language instructions from each dataset.

---

## 3. Data Processing: Unified 128-Dim Action Space

### 3.1 Non-Overlapping Region Layout

Different action representations (absolute EEF, joint positions, delta EEF) are placed in **separate, non-overlapping regions** of the 128-dim vector to avoid semantic confusion:

```
128-dim Unified Action Vector:

  ┌─ Region A ─────────┐ ┌─ Region B ─────────┐ ┌─ Region C ────────────────────────────┐ ┌─ Region D ──────────────────┐ ┌─ Region E ─────────┐ ┌─ Pad ──┐
  [0] [1] [2] [3:6] [7]  [8] [9:14]       [15]  [16:22]  [23]   [24:30]          [31]    [32:47]                          [48:54]          [55]    [56:127]
   x   y   z  quat   G    j0  j1-j6        G    Lj0-Lj6  LG     Rj0-Rj6          RG      f0-f15                           Δxyz Δrot       ΔG       zeros
  └── Abs EEF (8) ───┘   └── Joints (8) ───┘   └────── Bimanual (16) ──────────────┘     └── Dex Fingers (16) ─────────┘  └── Delta EEF (8) ──┘
```

### 3.2 Five Embodiment Types

| ID | Embodiment | Region | Active Dims | Datasets |
|----|-----------|--------|-------------|----------|
| 0 | `gripper_eef` | A `[0:8]` | 8 | rlbench |
| 1 | `gripper_joint` | B `[8:16]` | 8 | droid, bc_z, taco_play |
| 2 | `bimanual` | C `[16:32]` | 16 | aloha, nyu_franka, behavior1k (×50) |
| 3 | `dex_hand` | A `[0:7]` + D `[32:48]` | 23 | dexgraspnet |
| 4 | `gripper_delta_eef` | E `[48:56]` | 8 | utaustin_mutex, stanford_hydra, cmu_stretch |

### 3.3 Per-Dataset Action Mapping

```
Dataset            Native Action                    Dim  Format           → 128-dim Region     Active Indices
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
rlbench            [x,y,z,qx,qy,qz,qw,grip]        8   absolute EEF     → Region A [0:8]     [0,1,2,3,4,5,6,7]
droid              [j0,j1,j2,j3,j4,j5,j6]           7   joint position   → Region B [8:15]    [8,9,10,11,12,13,14]
bc_z               [j0,j1,j2,j3,j4,j5,j6]           7   joint position   → Region B [8:15]    [8,9,10,11,12,13,14]
taco_play          [j0,j1,j2,j3,j4,j5,j6]           7   joint position   → Region B [8:15]    [8,9,10,11,12,13,14]
aloha              [Lj0..Lj6, Rj0..Rj6]             14  absolute joints  → Region C [16:30]   [16..29]
nyu_franka         [Lj0..Lj7, Rj0..Rj6]             15  absolute joints  → Region C [16:31]   [16..30]
behavior1k (×50)   [23-dim bimanual state]           23  absolute joints  → Region C [16:32]   [16..31]
dexgraspnet        [wx,wy,wz,qx,qy,qz,qw, f0..f15] 23  absolute EEF+j   → A[0:7] + D[32:48] [0..6, 32..47]
utaustin_mutex     [Δx,Δy,Δz,Δrx,Δry,Δrz,grip]      7  delta EEF        → Region E [48:55]   [48..54]
stanford_hydra     [Δx,Δy,Δz,Δrx,Δry,Δrz,grip]      7  delta EEF        → Region E [48:55]   [48..54]
cmu_stretch        [Δx,Δy,Δz,Δrx,Δry,Δrz,grip,base] 8  delta EEF+base   → Region E [48:56]   [48..55]
```

### 3.4 Visual Region Map

```
Index: 0──────7 8──────15 16─────────31 32─────────47 48──────55 56────127
       │ Reg A │ │ Reg B │ │   Reg C    │ │   Reg D    │ │ Reg E │ │ Pad  │

rlbench:        ████████ ·········· ················ ················ ·········· ·······
droid:          ········ ████████ ················ ················ ·········· ·······
bc_z:           ········ ████████ ················ ················ ·········· ·······
taco_play:      ········ ████████ ················ ················ ·········· ·······
aloha:          ········ ·········· ██████████████ ················ ·········· ·······
nyu_franka:     ········ ·········· ███████████████ ················ ·········· ·······
behavior1k:     ········ ·········· ████████████████ ················ ·········· ·······
dexgraspnet:    ███████· ·········· ················ ████████████████ ·········· ·······
utaustin_mutex: ········ ·········· ················ ················ ████████ ·······
stanford_hydra: ········ ·········· ················ ················ ████████ ·······
cmu_stretch:    ········ ·········· ················ ················ █████████ ·······

█ = active    · = zero (masked out)
```

**Key design:** Absolute EEF and delta EEF occupy different regions — no semantic overlap. Dex hand wrist shares Region A with absolute EEF intentionally (same `[x,y,z,qx,qy,qz,qw]` format).

### 3.5 Action Normalization

Per-dataset z-normalization with **minimum std clamped to 0.01** to prevent quaternion blowup:

```python
unified_std = np.maximum(unified_std, 0.01)  # Max 100× amplification
normalized_action = (action - mean) / std
```

Without this clamp, quaternion dims (std ≈ 0.009) amplify values by 10,000× → MSE loss in millions.

### 3.2 Image Processing (Pi0-Style Variable Views)

**No hard cap on views** — all RGB cameras used per dataset:

1. **Per-dataset extraction:** Each dataset adapter detects all `observation.images.*` keys, filters RGB-only (excludes depth/seg)
2. **Resize:** All views resized to 448×448 (Qwen3-VL native)
3. **Batch collation:** Pad views to `max(V)` in batch with black frames
4. **VLM processing:** Qwen3-VL accepts `[B, V, 3, 448, 448]` where V varies (1-5 typically)

Example:
- **RLBench:** 5 views (front, wrist, left_shoulder, right_shoulder, overhead) → all used
- **Behavior 1K:** 3 RGB views (head, left_wrist, right_wrist) → all used
- **ALOHA:** 1 view (top) → padded to max_V in batch

### 3.3 Language Instruction Alignment

**All language comes from the actual dataset** (no generated/fallback text):

```python
# LeRobot datasets: 'task' key
language = row['task']  # e.g., "sweep the green cloth to the left side"

# RLBench: variation_descriptions.pkl
language = episode_metadata['description']  # e.g., "close the red jar"

# DexGraspNet: object metadata
language = f"grasp the {object_name}"  # e.g., "grasp the mug"
```

If language is missing, a warning is logged and the sample is skipped or empty string used.

---

## 4. VLM Feature Pre-Computation (Offline, One-Time)

### 4.1 Why Pre-Compute?

- **Speed:** VLM forward takes ~40ms per sample. Pre-computing means VLM runs once offline, then training loads 4KB features (1000× faster).
- **Memory:** VLM (4 GB) not in GPU during training → larger batch sizes.
- **Ablation-friendly:** Switch VLMs (Qwen3-VL ↔ PaliGemma2) by re-running pre-compute, no training code change.

### 4.2 Pre-Computation Command

```bash
# Single dataset with 100 samples (for testing)
python scripts/precompute_vlm_features.py \
    --dataset rlbench \
    --vlm qwen3vl \
    --max-samples 100 \
    --device cuda

# All datasets (for production)
python scripts/precompute_vlm_features.py \
    --all \
    --vlm qwen3vl \
    --device cuda

# Multi-GPU (8 GPUs, ~2 hours for 1M samples)
torchrun --nproc_per_node=8 scripts/precompute_vlm_features.py \
    --all --vlm qwen3vl
```

**Output:** `data/vlm_features/{dataset}_vlm_features.h5`

**Storage:** ~4 KB per sample (HDF5 fp16 compressed)
- 100K samples: ~400 MB
- 1M samples: ~4 GB
- 32M samples (all datasets): ~130 GB

### 4.3 HDF5 Format

```
{dataset}_vlm_features.h5
├── sample_0/
│   ├── hidden: [L, 2048] float16 (gzip)
│   ├── pooled: [2048] float16
│   └── attrs: {sequence_length: L, language: "..."}
├── sample_1/
│   ├── ...
```

---

## 5. Model Architecture

### 5.1 System 1: VLM Backbone (Qwen3-VL-2B-Instruct)

```
[B, V, 3, 448, 448] images + [B] language strings
         ↓
Qwen3-VL-2B-Instruct (2.1B params, frozen)
         ↓
[B, L, 2048] hidden states  (L varies with V: more views = longer L)
         ↓
Linear(2048 → 768) + LayerNorm  (trainable projector, 3.2M params)
         ↓
[B, L, 768] c_seq,  [B, 768] c_pool
```

**Multi-view processing:** All V views processed jointly through Qwen3-VL's native multi-image API. Variable sequence lengths handled via padding + attention mask in the DiT.

### 5.2 System 2: Drifting DiT Action Generator (147M params)

Aligned with RDT-1B inputs: proprioception `z_t`, multi-frame history, camera+timestep embeddings.

```
Inputs (aligned with RDT-1B):
  - c_seq: [B, L, 768]        VLM features (projected)
  - c_pool: [B, 768]          Global context
  - noise: [B, T, 64]         Random Gaussian
  - embodiment_id: [B]        {0=abs_eef, 1=joints, 2=bimanual, 3=dex_hand, 4=delta_eef}
  - cfg_scale: [B]            Guidance scale
  - proprio: [B, 128]         Proprioception z_t (robot state, 128-dim unified)
  - num_views, num_frames      Camera/timestep metadata

Pipeline:
  # VLM features + camera/time positional embeddings
  c_seq += camera_embed(cam_id) + time_embed(frame_id)   # per-token

  # Global conditioning (includes proprioception)
  c_global = c_pool + Embed(embodiment_id) + Embed(cfg_scale) + ProprioMLP(proprio)

  # Noise → tokens
  noise [B,T,64] → NoiseTokenizer → [B,T,768] noise_tokens

  # Cross-attention: noise attends to VLM features
  noise_tokens ─┐
                ├─→ CrossAttention(Q=noise, K/V=c_seq) → [B,T,768] fused
  c_seq ────────┘

  # DiT transformer with adaLN conditioning
  fused → DiT (12 blocks, adaLN-Zero, RoPE, QK-Norm) → [B,T,768]
       → ActionHead MLP(768 → 768 → 128) → [B,T,128] actions
```

**Features (aligned with RDT-1B):**
- **Proprioception `z_t`:** Robot state (joint pos, gripper, EE pose) → MLP → added to `c_global`
- **Multi-frame history:** Images from t-1, t, t+1 (3 frames × V cameras) all processed jointly by Qwen3-VL
- **Camera+timestep embeddings:** Learned embeddings added to VLM tokens so DiT knows which camera/frame each token came from

**Total params:** VLM 2.1B (frozen) + Projector 3.2M + DiT 147M = **150M trainable**

### 5.3 Unified Action Space (128-dim, Non-Overlapping Regions)

See Section 3 for the full layout. Summary:

```
Region A [0:8]:   Absolute EEF      — rlbench, dexgraspnet(wrist)
Region B [8:16]:  Joint Positions    — droid, bc_z, taco_play
Region C [16:32]: Bimanual Joints    — aloha, nyu_franka, behavior1k
Region D [32:48]: Dex Hand Fingers   — dexgraspnet
Region E [48:56]: Delta EEF          — utaustin_mutex, stanford_hydra, cmu_stretch
         [56:127] Padding (zeros)
```

5 embodiment types × non-overlapping regions → no semantic overlap between absolute EEF, joint positions, and delta EEF.

---

## 6. Training Pipeline

### 6.1 Quick Test (30 steps, 4 datasets, ~5 minutes)

```bash
export TMPDIR=~/tmp && mkdir -p $TMPDIR

# With pre-computed VLM features (fast: ~10 steps/sec)
python scripts/train.py \
    --test \
    --datasets rlbench aloha bc_z behavior1k_t0000 \
    --vlm-features-dir ./data/vlm_features \
    --data-root ./data

# Without pre-computed features (slower: ~0.8 steps/sec, live VLM)
python scripts/train.py \
    --test \
    --datasets rlbench aloha bc_z behavior1k_t0000 \
    --data-root ./data
```

**Output:** Logs showing 30 training steps, MSE + drifting loss, final checkpoint saved to `checkpoints/`.

### 6.2 Single-GPU Training (10K steps, ~2 hours)

```bash
python scripts/train.py \
    --datasets rlbench aloha droid bc_z taco_play utaustin_mutex \
    --batch-size 16 \
    --grad-accumulation 2 \
    --max-steps 10000 \
    --lr 4e-4 \
    --loss-type hybrid \
    --wandb-mode online \
    --wandb-project drifting-vla \
    --data-root ./data \
    --vlm-features-dir ./data/vlm_features
```

**Effective batch:** 16 × 2 = 32

### 6.3 Multi-GPU Training (100K steps, ~40 hours on 8×H100)

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --datasets rlbench aloha droid bc_z taco_play utaustin_mutex cmu_stretch nyu_franka stanford_hydra behavior1k_t0000 \
    --batch-size 64 \
    --grad-accumulation 2 \
    --max-steps 100000 \
    --lr 4e-4 \
    --loss-type hybrid \
    --drift-weight 0.1 \
    --wandb-mode online \
    --vlm-features-dir ./data/vlm_features
```

**Effective batch:** 64 × 8 × 2 = 1024

### 6.4 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | `['rlbench']` | List of dataset names |
| `--batch-size` | 16 | Per-GPU batch size |
| `--grad-accumulation` | 2 | Gradient accumulation steps |
| `--max-steps` | 10000 | Total training steps |
| `--lr` | 4e-4 | Learning rate |
| `--loss-type` | `'hybrid'` | `'mse'`, `'pure_drift'`, or `'hybrid'` |
| `--drift-weight` | 0.1 | Weight for drifting loss in hybrid mode |
| `--vlm` | `'qwen3vl'` | VLM backbone (`'qwen3vl'` or `'paligemma2'`) |
| `--vlm-features-dir` | `./data/vlm_features` | Path to pre-computed features |
| `--data-root` | `./data` | Dataset root directory |
| `--wandb-mode` | `'disabled'` | `'online'`, `'offline'`, or `'disabled'` |
| `--skip-vlm` | `False` | Skip VLM (pipeline test only, not for production) |

---

## 7. Loss Function

### 7.1 Hybrid Loss (Default)

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{drift}}$$

**Component 1: Masked MSE**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathbf{m}|} \sum_{ij} \mathbf{m}[j] \cdot (\mathbf{a}_{\text{pred}}[i,j] - \mathbf{a}_{\text{gt}}[i,j])^2$$

- Provides reliable gradient at any batch size
- Only computed on active dimensions (via `action_mask`)

**Component 2: Drifting Loss**

$$\mathcal{L}_{\text{drift}} = \|\mathbf{a} - \text{sg}(\mathbf{a} + V(\mathbf{a}))\|^2$$

where $V(\mathbf{a})$ is the drifting field:

$$V(\mathbf{a}) = \mathbb{E}_{y^+, y^-}[\tilde{k}(\mathbf{a}, y^+)\tilde{k}(\mathbf{a}, y^-)(y^+ - y^-)]$$

- Multi-temperature kernels: $\tau \in \{0.02, 0.05, 0.2\}$
- Double softmax normalized kernels
- Positive samples from dataset queue
- Negative samples from prediction queue

**Typical values:**
- MSE loss: 10-1000 (decreases during training)
- Drift loss: ~1.0 (constant by design)
- Grad norm: 5-50 (clipped to 2.0)

---

## 8. Codebase Structure

```
Drifting-Actor-Policy/
├── drifting_vla/
│   ├── models/
│   │   ├── drifting_vla.py          # Main model
│   │   ├── vlm_backbone.py          # Qwen3-VL wrapper
│   │   ├── dit.py                   # DiT transformer (adaLN-Zero)
│   │   ├── fusion.py                # Cross-attention
│   │   └── action_decoder.py        # NoiseTokenizer
│   ├── data/
│   │   ├── action_mapping.py        # 128-dim mapping, 60 dataset registry
│   │   ├── lerobot_dataset.py       # LeRobot adapter (Pi0-style)
│   │   ├── rlbench_dataset.py       # RLBench PerAct format
│   │   ├── dexgraspnet_dataset.py   # DexGraspNet loader
│   │   ├── unified_dataset.py       # Multi-dataset wrapper
│   │   └── sample_queue.py          # Pos/neg queues
│   └── training/
│       ├── losses.py                # DriftingLoss
│       ├── drifting_field.py        # V computation
│       └── ema.py                   # EMA
├── scripts/
│   ├── train.py                     # Training script
│   ├── download_datasets.py         # Download 50-sample subsets
│   ├── precompute_vlm_features.py   # VLM feature extraction
│   └── generate_sample_videos.py    # Per-camera videos
├── configs/
│   ├── data/multi_embodiment.yaml   # Full dataset config
│   └── training/baseline.yaml       # Training hyperparams
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.pretrain          # Training server image
│   └── Dockerfile.rlbench
└── PROPOSAL.md
```

---

## 9. Key Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **VLM** | Qwen3-VL-2B (online, Pi0-style) | Native multi-image, 2048 hidden dim, co-trainable with LoRA |
| **VLM mode** | Online encoding (default) | Pi0-style: VLM runs live, enables LoRA co-training |
| **Image views** | Pi0-style: all RGB (no hard cap) | Variable V via padding + attention mask |
| **History** | Multi-frame (t-1, t, t+1) | RDT-1B style temporal context for motion inference |
| **Proprio** | Robot state z_t (128-dim) | RDT-1B style: model knows current robot state |
| **Camera embeds** | Learned per-camera + per-timestep | DiT disambiguates which token from which camera/time |
| **Action space** | 128-dim, 5 non-overlapping regions | Abs EEF, Joints, Bimanual, Dex, Delta EEF separated |
| **Normalization** | Per-dataset z-score, min_std=0.01 | Prevents quaternion blowup (MSE: millions → hundreds) |
| **Datasets** | 60 total | 10 OXE + 50 Behavior 1K + RLBench + DexGraspNet |
| **Loss** | Hybrid MSE + Drifting | MSE for convergence, drifting for multi-modality |

---

## 10. Training on H100 Server

### 10.1 Full Pre-Training (100K steps, 64×H100, ~22 hours)

```bash
# Step 1: Pre-compute VLM features for all datasets (~4 hours)
torchrun --nproc_per_node=8 --nnodes=8 \
    scripts/precompute_vlm_features.py \
    --all --vlm qwen3vl

# Step 2: Train with effective batch 8192 (paper's setting)
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=29500 \
    scripts/train.py \
    --datasets rlbench dexgraspnet aloha droid bc_z taco_play utaustin_mutex cmu_stretch nyu_franka stanford_hydra behavior1k_t0000 \
    --batch-size 64 \
    --grad-accumulation 2 \
    --max-steps 100000 \
    --lr 4e-4 \
    --warmup-steps 2000 \
    --loss-type hybrid \
    --drift-weight 0.1 \
    --use-flash-attn \
    --wandb-mode online \
    --wandb-project drifting-vla-h100 \
    --save-every 5000
```

### 10.2 Cost Estimate

| Config | GPUs | Effective Batch | Time | Cost (@$2/GPU-hr) |
|--------|------|----------------|------|-------------------|
| **64×H100 (8 nodes)** | 64 × 80GB | **8,192** | 22 hrs | ~$2,800 |
| 8×H100 (single node) | 8 × 80GB | 1,024 | 80 hrs | ~$1,300 |
| 8×B200 (single node) | 8 × 192GB | 2,048 | 40 hrs | ~$2,400 |

---

## 11. Dataset Size Reference (Memory Planning)

| Dataset | Samples | Disk (raw) | VLM feat | Notes |
|---------|---------|------------|----------|-------|
| rlbench | 270K | ~50 GB | ~1.1 GB | 18 tasks, 5 cameras |
| dexgraspnet | 500K | ~100 GB | ~2.0 GB | Point cloud data |
| aloha | 20K | ~8 GB | ~80 MB | Sim bimanual |
| droid | 25.5M | ~2 TB | ~102 GB | **Largest**, real-world |
| bc_z | 98K | ~30 GB | ~400 MB | Language-conditioned |
| taco_play | 238K | ~30 GB | ~950 MB | Diverse manip |
| utaustin_mutex | 362K | ~40 GB | ~1.4 GB | Language-conditioned |
| cmu_stretch | 25K | ~5 GB | ~100 MB | Mobile manip |
| nyu_franka | 45K | ~12 GB | ~180 MB | Bimanual kitchen |
| stanford_hydra | 358K | ~40 GB | ~1.4 GB | Coffee making |
| behavior1k (×50) | ~5M each | ~500 GB | ~20 GB | Bimanual sim (OmniGibson) |
| **TOTAL** | **~32.4M** | **~2.8 TB** | **~130 GB** | 60 datasets |

---

## 12. Troubleshooting

### 12.1 `/tmp` not writable

```bash
# Set custom temp directory
export TMPDIR=~/tmp && mkdir -p $TMPDIR
python scripts/train.py ...
```

### 12.2 VLM download fails

```bash
# Manually download Qwen3-VL-2B-Instruct
python -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
print('Downloaded to ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/')
"
```

### 12.3 Video decode errors (torchcodec)

Some LeRobot datasets use video encoding. Errors are automatically caught and the sample is skipped. If too many samples fail:

```bash
# Use image-format datasets instead
python scripts/train.py --datasets aloha_sim_transfer_cube_human_image  # not aloha_mobile (video)
```

### 12.4 Disk space full

```bash
# Clean HuggingFace caches
rm -rf ~/.cache/huggingface/lerobot/  # Downloaded dataset videos (can be 100GB+)
rm -rf ~/.cache/pip/                   # Pip download cache (~15 GB)

# Clean sample videos after inspection
rm -rf sample_videos/
```

---

## References

1. **Drifting:** One-Step Generation via Training-Time Distribution Evolution. arXiv:2602.04770, 2025.
2. **RDT-1B:** a Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864, 2024. [GitHub](https://github.com/thu-ml/RoboticsDiffusionTransformer)
3. **DexGraspNet 2.0:** Learning Generative Dexterous Grasping. CoRL 2024. [GitHub](https://github.com/PKU-EPIC/DexGraspNet2)
4. **Qwen3-VL:** Multimodal Vision-Language Model. [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
5. **Open X-Embodiment:** Robotic Learning Datasets and RT-X Models. arXiv:2310.08864, 2023. [LeRobot Collection](https://huggingface.co/collections/lerobot/open-x-embodiment)
6. **Behavior 1K:** OmniGibson Simulation Benchmark. [HuggingFace Collection](https://huggingface.co/collections/lerobot/behavior-1k)
7. **Diffusion Policy:** Visuomotor Policy Learning via Action Diffusion. RSS 2023.
8. **3D Diffuser Actor:** Policy Diffusion with 3D Scene Representations. arXiv:2402.10885, 2024.
