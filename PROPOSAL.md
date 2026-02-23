# Drifting-VLA: One-Step Multi-Embodiment Robotic Manipulation

**Technical Proposal & Architecture Guide — v5 (Feb 2026)**

---

## Overview

Drifting-VLA is a unified foundation model for robotic manipulation that generates actions in **one forward pass** (1-NFE) using the Drifting model paradigm. Unlike diffusion policies that require 5-50 denoising steps at inference, our model produces a complete action trajectory in a single network evaluation — achieving **10× faster inference** (50ms vs 500ms).

The system combines:
- A **Vision-Language Model** (Qwen3-VL-2B) that processes multi-view images and language instructions
- A **Drifting DiT** (Diffusion Transformer) that generates 128-dimensional unified actions
- A **semantic action mapping** compatible with RDT-1B that supports 7 embodiment types across 13 dataset families

Key numbers:
- **155M trainable parameters** (VLM LoRA 5M + Projector 3.2M + DiT 147M)
- **128-dim unified action space** with 163 named semantic fields
- **7 embodiment types** from single-arm grippers to bimanual dexterous hands
- **13 dataset families** (63+ sub-datasets) totaling ~35M frames

---

## 1. Datasets

### 1.1 Unified Dataset Registry

All datasets are accessed through a single `prepare_data.py` script that handles download, video decoding, and HDF5 conversion in one pass. The table below shows every integrated dataset:

| # | Dataset | HF Repo | Samples | Act Dim | Embodiment | Views | Lang |
|---|---------|---------|---------|---------|------------|-------|------|
| 1 | **aloha** | `lerobot/aloha_sim_..._human_image` | 20K | 14 | bimanual | 1 | ✅ |
| 2 | **bc_z** | `lerobot/berkeley_autolab_ur5` | 98K | 7 | gripper_joint | 2 | ✅ |
| 3 | **taco_play** | `lerobot/taco_play` | 238K | 7 | gripper_joint | 2 | ✅ |
| 4 | **droid** | `lerobot/droid_1.0.1` | 25.5M | 7 | gripper_joint | 3 | ✅ |
| 5 | **stanford_hydra** | `lerobot/stanford_hydra_dataset` | 358K | 7 | gripper_delta_eef | 2 | ✅ |
| 6 | **cmu_stretch** | `lerobot/cmu_stretch` | 25K | 8 | gripper_delta_eef | 1 | ✅ |
| 7 | **utaustin_mutex** | `lerobot/utaustin_mutex` | 362K | 7 | gripper_delta_eef | 2 | ✅ |
| 8 | **nyu_franka** | `lerobot/nyu_franka_play_dataset` | 45K | 15 | bimanual | 2 | ❌ |
| 9 | **rlbench** | `hqfang/rlbench-18-tasks` | 270K | 8 | gripper_eef | 2-5 | ✅ |
| 10 | **dexgraspnet** | `lhrlhr/DexGraspNet2.0` | 500K | 23 | dex_hand | 8 | ✅ |
| 11 | **dexwild** | `YingyangPolarBear/DexWild` | 95K | 23 | dex_hand | 2 | ✅ |
| 12 | **dexora** | `Dexora/Dexora_Real-World_Dataset` | 2.9M | 39 | bimanual_dex | 4 | ✅ |
| 13 | **behavior1k** ×50 | `lerobot/behavior1k-task{i:04d}` | ~430K ea | 23 | bimanual_mobile | 3 | ✅ |

**Embodiment coverage:** Single-arm gripper (EEF, joints, delta), bimanual (gripper, mobile), single dexterous hand (16-DOF), bimanual dexterous hand (24-DOF with head/spine).

### 1.2 Dataset Types and Handling

Each dataset falls into one of four source types, each with a dedicated handler:

| Type | Datasets | Image Source | Handler |
|------|----------|-------------|---------|
| **LeRobot (video)** | bc_z, taco_play, stanford_hydra, droid, cmu_stretch, utaustin_mutex, nyu_franka, behavior1k, dexora | MP4 video files decoded via pyav | `prepare_lerobot()` with `LeRobotDataset` API |
| **LeRobot (parquet)** | aloha | Images embedded in parquet columns | Same `prepare_lerobot()` — auto-detects |
| **HF generic** | rlbench, dexgraspnet | Parquet or snapshot_download | `prepare_rlbench()`, `prepare_dexgraspnet()` |
| **Custom HDF5** | dexwild | Source HDF5 with JPEG image groups | `prepare_dexwild()` |

---

## 2. Data Pipeline

### 2.1 Design Philosophy

The old two-step pipeline (`download_datasets.py` → Arrow files → `convert_to_episodes.py` → HDF5) was replaced with a **single-pass pipeline** that streams data from HuggingFace and writes Episode HDF5 files directly. No intermediate Arrow files are ever saved to disk.

For video-based datasets (the majority), images are stored as MP4 video files on HuggingFace. The pipeline uses the **LeRobot API** (`LeRobotDataset` with `video_backend='pyav'`) which handles video download and decoding automatically.

### 2.2 Batch Video Decoding

The critical performance optimization is **batch video decoding**. Instead of calling `lerobot_ds[i]` per-frame (which opens the video, seeks, decodes 1 frame, closes — ~20ms overhead each time), we:

1. Read **actions, states, and metadata** from parquet (instant, no video I/O)
2. For each episode, open the MP4 file **once** and decode **all frames sequentially**

```
Per-frame (OLD):  open → seek → decode 1 frame → close  ×  66 frames  =  ~1.3s per episode
Batch (NEW):      open → decode ALL 66 frames → close                  =  ~0.1s per episode
Speedup: ~13× per episode
```

The video path follows LeRobot v2 format:
```
{videos_dir}/{camera_key}/chunk-{ep_id//1000:03d}/file-{ep_id%1000:03d}.mp4
```

### 2.3 Image Storage: Original Resolution

Following RDT-1B's approach, images are stored at their **native resolution** in HDF5 (e.g., taco_play: 150×200, bc_z: 480×640). Resizing to the VLM's target size (448×448 for Qwen3-VL) happens at **training time** in the DataLoader:

```
Original [H, W, 3] → Pad to square (black border) → Resize to 448×448 → ColorJitter (50%) → [0,1] float
```

This means:
- **Smaller HDF5 files** (150×200 is 9× less pixels than 448×448)
- **VLM-agnostic storage** — can switch from Qwen3-VL (448) to SigLIP (384) without re-preparing data
- **Preserves information** for future higher-resolution models

### 2.4 Pipeline Commands

```bash
# Prepare multiple datasets in parallel (auto-skips already-done ones)
python scripts/prepare_data.py --datasets aloha bc_z taco_play --parallel 3 --cleanup

# Single dataset with episode limit for testing
python scripts/prepare_data.py --dataset taco_play --max-episodes 25 --force --cleanup

# Key flags:
#   --rgb-only (default True): excludes depth/segmentation/mask image keys
#   --parallel N: run N datasets simultaneously in separate processes
#   --cleanup: remove HF cache after each dataset to save disk
#   --force: re-prepare even if metadata.json already exists
```

### 2.5 Episode HDF5 Layout

Each episode is stored as a self-contained HDF5 file:

```
ep_000000.hdf5
├── images/
│   ├── view_0     [T, H, W, 3] uint8     ← RGB at original resolution
│   └── view_1     [T, H', W', 3] uint8   ← different cameras can have different sizes
├── actions        [T, 128] float32        ← pre-mapped to 128-dim unified space
├── action_mask    [128] bool              ← which dims are active for this dataset
├── proprio        [T, 128] float32        ← proprioception in same 128-dim space
├── language       scalar string           ← task instruction (e.g., "pick up the red cup")
└── attrs:
    ├── dataset_name     (e.g., "taco_play")
    ├── embodiment_id    (e.g., 1 for gripper_joint)
    ├── episode_length   (e.g., 66)
    ├── n_views          (e.g., 2)
    └── action_dim       (e.g., 7 native dims)
```

### 2.6 Training-Time Augmentation

Two augmentations are applied during training (not during data preparation):

**Image Augmentation** (`--image-aug`): 50% of training images receive `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)`, matching RDT-1B's training augmentation.

**Condition Masking for CFG** (`--cond-mask-prob 0.1`): With 10% probability, each condition is independently dropped:
- Language → replaced with empty string `""`
- Images → replaced with black frames (all zeros)
- Proprioception → zeroed out

This trains the model to work with and without each condition, enabling classifier-free guidance at inference time where the model can be steered by adjusting the guidance scale.

---

## 3. Unified 128-Dim Action Space

### 3.1 Design Philosophy

Different robots have different action representations — a 6-DOF arm uses joint positions, a mobile manipulator uses delta end-effector commands, a dexterous hand has 16+ finger joints. To train a single model across all of them, we need a **unified action space**.

Our approach follows **RDT-1B's semantic named-field mapping**: each physical quantity (e.g., "right arm joint 3 position") maps to a specific named slot in a 128-dimensional vector. The key properties are:

1. **Semantic naming**: Every slot has a human-readable name like `arm_joint_3_pos` or `eef_pos_x`
2. **Overlapping regions**: A single robot can fill BOTH joint positions AND EEF positions simultaneously — they're different physical quantities that co-exist
3. **Sparse activation**: Each dataset only fills 7-39 of the 128 dims; the rest are zero and masked out by `action_mask`
4. **RDT-1B compatibility**: Slots `[0:103]` are identical to RDT-1B, ensuring cross-compatibility with their 46 pre-trained datasets

### 3.2 Layout

The 128-dim vector is organized into left/right mirrored regions plus shared regions:

```
RIGHT SIDE (slots 0-49)                         LEFT SIDE (slots 50-99)
───────────────────────────────────────          ───────────────────────────────────────
[0,  10)  Arm joint positions                    [50, 60)  Left arm joint positions
          arm_joint_{0-9}_pos                              left_arm_joint_{0-9}_pos

[10, 15)  Gripper joint positions                [60, 65)  Left gripper joints
          gripper_joint_{0-4}_pos                          left_gripper_joint_{0-4}_pos

[15, 25)  Arm joint velocities                   [65, 75)  Left arm joint velocities
          arm_joint_{0-9}_vel                              left_arm_joint_{0-9}_vel

[25, 30)  Gripper velocities                     [75, 80)  Left gripper velocities
          gripper_joint_{0-4}_vel                          left_gripper_joint_{0-4}_vel

[30, 33)  EEF position (xyz)                     [80, 83)  Left EEF position
          eef_pos_{x,y,z}                                  left_eef_pos_{x,y,z}

[33, 39)  EEF 6D rotation (continuous)           [83, 89)  Left EEF 6D rotation
          eef_angle_{0-5}                                  left_eef_angle_{0-5}

[39, 42)  EEF velocity                           [89, 92)  Left EEF velocity
          eef_vel_{x,y,z}                                  left_eef_vel_{x,y,z}

[42, 45)  EEF angular velocity                   [92, 95)  Left EEF angular velocity
          eef_angular_vel_{roll,pitch,yaw}                 left_eef_angular_vel_*

[45, 48)  Head/spine joints (Dexora)             [95,100)  Reserved
          head_joint_{0,1}, spine_joint

[48, 50)  Reserved


SHARED REGIONS (slots 100-127)
──────────────────────────────────────────────────────────────
[100,103)  Base velocities           base_vel_{x,y}, base_angular_vel
[103,115)  Right dex finger joints   dex_finger_joint_{0-11}_pos    (12 DOF max)
[115,127)  Left dex finger joints    dex_finger_joint_{12-23}_pos   (12 DOF max)
[127,128)  Reserved
```

**163 named fields** total in `STATE_VEC_IDX_MAPPING`. Many fields have aliases (e.g., `arm_joint_0_pos` and `right_arm_joint_0_pos` both map to index 0).

### 3.3 How Velocity Slots Work

Velocity slots `[15:30]`, `[39:45]`, `[65:80]`, `[89:95]` are included for datasets that provide velocity information in their actions or states. Most current datasets only use position slots, leaving velocities as zeros. This is the intended behavior — the `action_mask` tells the model which dims are active.

RDT-1B uses the same approach: for example, `stanford_robocook` maps its 7-dim action to velocity slots (`eef_vel_x, eef_vel_y, eef_vel_z, eef_angular_vel_roll, ...`) while its state uses position slots. The same 128-dim vector can hold BOTH positions and velocities — they're different physical quantities in separate slots.

### 3.4 Dexterous Hand Slots

The `[103:127]` region handles dexterous manipulation. For **single-hand** datasets (DexWild, DexGraspNet), all 16 finger joints map to `dex_finger_joint_{0-15}` → `[103:119]`. For **bimanual dexterous** datasets (Dexora with 12+12=24 finger joints), right hand uses `{0-11}` → `[103:115]` and left hand uses `{12-23}` → `[115:127]`.

This is our extension beyond RDT-1B, which left `[103:128]` as reserved. The extension is backward compatible — no existing RDT-1B dataset uses these slots.

### 3.5 Seven Embodiment Types

| ID | Type | What it controls | Active Slots | Example |
|----|------|-----------------|-------------|---------|
| 0 | `gripper_eef` | Absolute end-effector pose + gripper | EEF `[30:39]` + grip `[10]` | rlbench |
| 1 | `gripper_joint` | Joint positions + gripper | Joints `[0:6]` + grip `[10]` | droid, bc_z, taco_play |
| 2 | `bimanual` | Two arms with grippers | L `[50:56]+[60]` + R `[0:6]+[10]` | aloha, nyu_franka |
| 3 | `dex_hand` | Wrist EEF + finger joints | EEF `[30:37]` + fingers `[103:119]` | dexwild, dexgraspnet |
| 4 | `gripper_delta_eef` | Delta EEF commands | Delta `[30:36]` + grip `[10]` | stanford_hydra, utaustin_mutex |
| 5 | `bimanual_mobile` | Two arms + mobile base | Arms + base `[100:103]` | behavior1k ×50 |
| 6 | `bimanual_dex` | Two arms + dex hands + head | Arms + fingers `[103:127]` + head `[45:48]` | dexora |

### 3.6 Per-Dataset Field Mapping

Each dataset defines a `DATASET_FIELD_FORMATS` list that maps its native action dimensions to named fields:

```python
# DROID: 7-dim joint position → fills [0:6] + [10]
'droid': ['arm_joint_0_pos', 'arm_joint_1_pos', ..., 'arm_joint_5_pos', 'gripper_open']

# ALOHA: 14-dim bimanual → fills left [50:56]+[60] + right [0:6]+[10]
'aloha': ['left_arm_joint_0_pos', ..., 'left_gripper_open',
          'arm_joint_0_pos', ..., 'gripper_open']

# DexWild: 23-dim (7 wrist + 16 fingers) → fills [30:37] + [103:119]
'dexwild': ['eef_pos_x', ..., 'eef_angle_3', 'dex_finger_joint_0_pos', ..., 'dex_finger_joint_15_pos']

# Dexora: 39-dim (6+6 arms + 12+12 fingers + 3 head/spine)
'dexora': ['left_arm_joint_0_pos', ..., 'arm_joint_5_pos',
           'dex_finger_joint_12_pos', ..., 'dex_finger_joint_23_pos',    # left hand
           'dex_finger_joint_0_pos', ..., 'dex_finger_joint_11_pos',     # right hand
           'head_joint_0', 'head_joint_1', 'spine_joint']
```

The `assemble_state_vec_batch()` function performs this mapping: for each field name, it looks up the index in `STATE_VEC_IDX_MAPPING` and places the value there. This is identical to RDT-1B's `assemble_state_vec()` (theirs uses TensorFlow `scatter_nd_update`, ours uses NumPy indexing — same logic).

### 3.7 Normalization

Per-dataset robust [-1, 1] normalization (RDT-1B style):
```
action_min = mean - 5 × max(std, 0.01)
action_max = mean + 5 × max(std, 0.01)
center = (min + max) / 2
half_range = (max - min) / 2
normalized = clip((action - center) / half_range, -1, 1)
```

The `max(std, 0.01)` prevents quaternion dimensions (which have tiny std ≈ 0.009) from being amplified by 10,000× — which would make the MSE loss explode.

---

## 4. Model Architecture

### 4.1 System 1: VLM Backbone — Vision-Encoder-Only (Pi0 Style)

We use Qwen3-VL-2B-Instruct but **skip the 28-layer LLM decoder entirely**. Only two components are used:

1. **ViT Vision Encoder** (`model.visual`, ~407M params): Processes images into visual tokens
2. **Text Embedding Layer** (`model.embed_tokens`, ~311M params): Converts tokenized text into embeddings

This is the "Pi0 style" — the same approach used by Physical Intelligence's Pi0 model with PaliGemma. The LLM decoder is never loaded, saving ~1.4B params of memory.

**Forward pipeline:**
```
Step 1: images [B, N_images, 3, 448, 448]
        → Qwen3-VL ViT encoder → [N_total_patches, 1024]
        → vis_proj(1024 → 2048)  → [N_vis_tokens, 2048]

Step 2: text tokens [B, L_text]
        → embed_tokens lookup    → [B, L_text, 2048]

Step 3: Concatenate visual + text → [B, L_total, 2048]
        → proj_seq(Linear + LayerNorm) → [B, L_total, 768]  = c_seq
        → masked_pooling              → [B, 2048]
        → proj_pool(Linear + LayerNorm) → [B, 768]           = c_pool

Step 4: Per-token camera + time embeddings
        c_seq[:, vis_offset:vis_end] += camera_embed(cam_id) + time_embed(frame_id)
```

**Why skip the LLM decoder?** The LLM's 28 transformer layers are designed for autoregressive text generation — they add ~300ms latency and 6GB memory but don't improve action quality (the DiT handles action generation). By extracting features from the ViT encoder + text embeddings only, we get the visual understanding capability without the overhead.

**Visual projection (`vis_proj`):** Qwen3-VL-2B's ViT outputs 1024-dim tokens, but `embed_tokens` produces 2048-dim. We add a learned `nn.Linear(1024, 2048)` to match dimensions before concatenation.

### 4.2 VLM Training Modes

| Mode | ViT Encoder | What's trained | Memory | Speed | Use Case |
|------|------------|---------------|--------|-------|----------|
| `frozen` | All frozen | Only DiT + projectors | ~25 GB | 1.0 steps/s | Quick debugging |
| `lora` | LoRA adapters (r=16, ~5M) | LoRA + DiT + projectors | ~37 GB | 0.2 steps/s | **Pre-training** |
| `full` | All unfrozen | Everything | ~45 GB | 0.1 steps/s | Post-training (not recommended for pre-train) |

### 4.3 System 2: Drifting DiT Action Generator (147M params)

The DiT takes VLM features and generates action trajectories:

```
Inputs:
  c_seq   [B, L, 768]    — VLM visual + text features (with camera/time embeddings)
  c_pool  [B, 768]       — Pooled global context
  noise   [B, T, 64]     — Random Gaussian noise (T=16 action horizon)
  embodiment_id [B]      — Which robot type (0-6)
  proprio [B, 128]       — Current robot state (128-dim unified)
  ctrl_freqs [B]         — Control frequency in Hz (e.g., 50 for ALOHA, 15 for DROID)

Pipeline:
  1. Build global conditioning:
     c_global = c_pool + Embed(embodiment_id) + ProprioMLP(proprio)
              + CFG_embed(cfg_scale) + FreqEmbed(ctrl_freqs)

  2. Noise → action tokens:
     noise [B, T, 64] → NoiseTokenizer(Linear) → [B, T, 768]

  3. Cross-attention: action tokens attend to VLM features
     CrossAttention(Q=noise_tokens, K/V=c_seq, mask=vlm_attn_mask) → [B, T, 768]

  4. DiT transformer with adaLN-Zero conditioning:
     12 blocks × (SelfAttention + FFN), conditioned on c_global via adaptive LayerNorm
     Features: RoPE positional encoding, QK-Norm, SwiGLU FFN

  5. Action head:
     MLP(768 → 768 → 128) → [B, T, 128]  ← predicted actions
```

**Control frequency conditioning:** Different datasets operate at different frequencies (ALOHA=50Hz, DROID=15Hz, DexWild=30Hz). The model receives `ctrl_freqs` as input and learns a `FreqEmbed` to adapt its temporal predictions. This is the same approach used by RDT-1B.

**1-NFE generation:** Unlike diffusion models that iteratively denoise over 5-50 steps, our model produces the final action in one forward pass. The "noise" input is not a corrupted action — it's a random signal that the model transforms into a coherent action trajectory through learned generation, guided by the drifting loss during training.

---

## 5. Loss Function

### 5.1 Hybrid Loss

```
L = L_MSE + drift_weight × L_drift
```

**Component 1: Masked MSE**

Standard regression loss, but only computed on active dimensions:
```python
mask = action_mask.unsqueeze(1)    # [B, 1, 128]
n_active = mask.sum()
mse = ((pred * mask - gt * mask) ** 2).sum() / n_active
```

This ensures the model isn't penalized for predictions in inactive dimensions (which should be zero).

**Component 2: Drifting Loss**

The drifting loss comes from the paper "Generative Modeling via Drifting" (arXiv:2602.04770). The key idea: compute a **drifting field** V that points generated samples toward real data and away from other generated samples. When the generator has converged (q = p_data), V = 0.

```
V(x) = Σ_τ [ W_pos(x, y+)(y+ - x) - W_neg(x, y-)(y- - x) ]
        ↑ attraction to data      ↑ repulsion from generated

L_drift = ||V(x)||²    ← naturally → 0 at convergence
```

Key implementation details:
- **Multi-temperature kernels**: τ ∈ {0.02, 0.05, 0.2} for multi-scale structure
- **Double softmax normalization**: Row-wise + column-wise → geometric mean
- **Single global queue**: Positive samples (expert actions) and negative samples (model predictions) are stored in a global FIFO queue. Cross-embodiment samples naturally separate via L2 distance in the kernel.
- **Raw drift norm as loss**: We use the un-normalized `||V||²` which naturally decreases to 0 at convergence, rather than the normalized version (which is always ≈ 1.0 by design).

**Why hybrid?** Pure MSE provides reliable point-to-point regression but can cause mode collapse (averaging over multi-modal action distributions). Pure drifting loss captures multi-modality but converges slowly. The hybrid combines both: MSE for fast initial convergence, drifting for distributional matching.

---

## 6. Training

### 6.1 Training Loop Key Features

- **DDP (DistributedDataParallel)**: Standard PyTorch DDP with `find_unused_parameters=True` (needed for the VLM's conditional loading)
- **Mixed precision**: `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
- **Gradient accumulation**: Configurable (default 2 steps)
- **Cosine LR with warmup**: 500-step warmup, then cosine decay
- **Differential learning rates**: VLM LoRA at `lr × 0.1`, projector at `lr × 0.5`, DiT at full `lr`
- **Gradient clipping**: Max norm 1.0
- **EMA**: Exponential moving average of model weights (decay 0.9999)

### 6.2 DataLoader Configuration

- **8 workers** with `persistent_workers=True` and `prefetch_factor=2`
- **Separate eval DataLoader**: Uses uniform random sampling (not weighted), no augmentation, no condition masking — ensures clean evaluation
- **Temperature-balanced sampling**: Datasets are sampled with probability ∝ N^0.5 (square root of dataset size), preventing large datasets from dominating

### 6.3 Global Batch Size Logging

The training banner shows the full effective batch computation:
```
Batch: 32/gpu × 2 accum × 8 GPUs = 512 global
```

### 6.4 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-size` | `base` | Architecture: `small` (512d/8L), `base` (768d/12L), `large` (1024d/24L) |
| `--vlm-mode` | `frozen` | VLM training: `frozen` (fast), `lora` (recommended), `full` (expensive) |
| `--lora-r` | 16 | LoRA rank: 16 for pre-training, 64 for post-training |
| `--vlm-lr-scale` | 0.1 | VLM learning rate = base_lr × this value |
| `--batch-size` | 16 | Per-GPU batch size |
| `--grad-accumulation` | 2 | Gradient accumulation steps |
| `--lr` | 4e-4 | Base learning rate |
| `--max-steps` | 10000 | Total training steps |
| `--loss-type` | `hybrid` | Loss function: `mse`, `pure_drift`, or `hybrid` |
| `--num-workers` | 8 | DataLoader workers |
| `--image-aug` | False | Enable ColorJitter augmentation |
| `--cond-mask-prob` | 0.1 | CFG condition masking probability |
| `--eval-batches` | 50 | Number of batches per evaluation |
| `--use-flash-attn` | False | Enable Flash Attention 2 (H100/H200) |
| `--wandb-mode` | `disabled` | Logging: `online`, `offline`, or `disabled` |

### 6.5 Training Commands

**Single-node (8×A40), 10K steps (~12 hours):**
```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --datasets aloha bc_z taco_play stanford_hydra cmu_stretch \
    --episodes-root ./data/episodes \
    --batch-size 32 --grad-accumulation 2 --max-steps 10000 --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base --num-workers 8 \
    --image-aug --cond-mask-prob 0.1 \
    --wandb-mode online --wandb-project drifting-vla
```

---

## 7. Visualization Suite (17 WandB Panels)

### 7.1 Panel Catalog

Each panel is generated at specific intervals during training and logged to WandB:

| Panel | What It Shows | Frequency | What Good Looks Like |
|-------|--------------|-----------|---------------------|
| **A1: Drifting Field** | Scatter plot with drift arrows showing V direction/magnitude | 500 steps | Arrows point toward GT; high magnitude early, decreasing over training |
| **A2: Drift Magnitude Trend** | Time series of λ_V with EMA smoothing | 200 steps | Decreasing curve; plateau = increase drift_weight |
| **A3: Prediction Scatter** | Dense pred vs GT scatter for up to 12 active dims | 500 steps | Points clustered along y=x diagonal; high R² |
| **A4: Per-Dim Error Bar** | Horizontal bar chart of MAE per active dim, color-coded by region | 500 steps | Uniform low bars; identifies which dims are hardest |
| **A5: Temperature Loss** | Bar chart comparing drift norm at each temperature | 500 steps | Low-τ/High-τ ratio < 0.5 = fine detail learned |
| **B1: Action Distribution** | Histogram grid overlaying GT (blue) and Pred (red) | 500 steps | Coverage → 100%; < 20% = mode collapse |
| **B2: Error Heatmap** | Dims × timesteps heatmap of MAE | 500 steps | Uniform color; hot spots = problem dims/times |
| **B3: Per-Region Summary** | Grouped bars: MAE, Correlation, Coverage per region | 1000 steps | All regions similar quality |
| **B4: Trajectory 2D** | 4 samples × XY + XZ projections, GT vs Pred lines | 1000 steps | Lines overlap; gray error connectors shrink |
| **B5: Temporal Error** | Error vs timestep with confidence band | 500 steps | Flat or slowly increasing; > 50% growth = bad |
| **C1: Transport Map** | PCA scatter with drift arrows, before/after distances | 500 steps | Improvement% > 50% |
| **C2: Mode Coverage** | PCA GT vs Pred scatter | 2000 steps | Pred cloud overlaps GT cloud |
| **C3: Smoothness** | Velocity, jerk, FFT spectrum comparison | 1000 steps | Pred matches GT's dynamics |
| **D1: Eval Dashboard** | Per-dataset MSE/MAE/Correlation/Coverage bars | Every eval | Color-coded by embodiment |
| **D2: Training Curves** | Loss + MSE + Drift Norm over training | 500 steps | All decreasing |
| **D3: Learning Curves** | Per-dataset MSE & Correlation over eval checkpoints | Every eval | All improving; stagnating = data issue |
| **D4: Data Balance** | Configured sampling weight vs actual batch counts | Every eval | Balanced; under-represented datasets = adjust weights |

### 7.2 Eval Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **MSE** | Mean squared error on active dims | Lower = better point-to-point accuracy |
| **MAE** | Mean absolute error on active dims | More interpretable than MSE |
| **Correlation** | Per-dim Pearson r, averaged | 1.0 = perfect linear fit; <0 = anti-correlated |
| **Coverage %** | pred_range / GT_range × 100 | 100% = full distribution covered; 1% = mode collapse |
| **Normalized L2** | RMSE / (state_norm + ε), per dim | RDT-1B style; cross-embodiment comparable |

---

## 8. Docker

### 8.1 Build and Run

```bash
# Build the training image
docker build -f docker/Dockerfile.pretrain -t drifting-vla:pretrain .

# Interactive development (code volume-mounted from host)
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    -v $(pwd):/workspace -v $(pwd)/data:/workspace/data \
    -e WANDB_API_KEY=$WANDB_API_KEY -it drifting-vla:pretrain bash

# Visualize an episode as video
python scripts/visualize_episode.py data/episodes/taco_play/ep_000000.hdf5 -o sample.mp4
```

### 8.2 Dockerfile Features

- **Base**: `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
- **Explicit COPY**: Only source code, never `data/` (enforced by `.dockerignore` + build-time `test ! -d /workspace/data`)
- **Entrypoint**: Auto-login WandB if `WANDB_API_KEY` env var is set
- **Health check**: `RUN python -c "from drifting_vla.models.drifting_vla import DriftingVLA; ..."` — fails the build if imports break
- **Environment**: All training env vars pre-set (`HDF5_USE_FILE_LOCKING=FALSE`, `NCCL_P2P_DISABLE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)

---

## 9. Codebase Structure

```
Drifting-Actor-Policy/
├── drifting_vla/
│   ├── models/
│   │   ├── drifting_vla.py       Main model: VLM + DiT + ctrl_freq + embodiment
│   │   ├── vlm_backbone.py       Vision-encoder-only VLM (Pi0 style) + vis_proj
│   │   ├── dit.py                DiT transformer: adaLN-Zero, RoPE, QK-Norm, SwiGLU
│   │   ├── fusion.py             Cross-attention with KV masking
│   │   └── action_decoder.py     NoiseTokenizer
│   ├── data/
│   │   ├── action_mapping.py     128-dim semantic mapping, 163 fields, 7 embodiments
│   │   ├── episode_dataset.py    HDF5 loader + pad-to-square + augmentation + CFG mask
│   │   ├── unified_dataset.py    Multi-dataset wrapper + collation + weighted sampler
│   │   └── sample_queue.py       Global pos/neg queues for drifting loss
│   └── training/
│       ├── losses.py             DriftingLoss: multi-temp kernels, raw ||V||² loss
│       ├── drifting_field.py     V(a) computation: double softmax, feature normalization
│       ├── visualizations.py     17 WandB panels (A1-A5, B1-B5, C1-C3, D1-D4)
│       └── ema.py                Exponential moving average
├── scripts/
│   ├── train.py                  Full training loop: DDP, eval, normalized L2, global batch log
│   ├── prepare_data.py           Single-pass download+convert: LeRobot API, batch video, parallel
│   └── visualize_episode.py      Episode → MP4 with task instruction + multi-view grid
├── docker/
│   ├── Dockerfile.pretrain       Training image (RLBench-style, explicit COPY, entrypoint)
│   ├── Dockerfile.base           Base dependencies
│   └── Dockerfile.rlbench        RLBench with CoppeliaSim
├── configs/
│   ├── dataset_control_freq.json Per-dataset Hz for ctrl_freq conditioning
│   └── zero2.json                DeepSpeed ZeRO-2 config (ready for multi-node scaling)
└── PROPOSAL.md                   This file
```

---

## 10. Design Choices vs RDT-1B

| Decision | Drifting-VLA | RDT-1B | Why |
|----------|-------------|--------|-----|
| **Action mapping** | Identical `[0:103]` + dex `[103:127]` | `[0:103]` only, `[103:128]` reserved | We extend for dexterous hands while maintaining compatibility |
| **Dex hands** | 24-DOF bimanual (Dexora) + 16-DOF single (DexWild) | Not supported | Novel capability |
| **Head/spine** | `[45:48]` for humanoid-class robots | Reserved | Needed for Dexora's 39-dim action |
| **Action generation** | 1-NFE Drifting (one forward pass) | 5-step DDPM diffusion | 10× faster inference |
| **Drifting loss** | Raw `\|\|V\|\|²` → 0 at convergence, + MSE | Pure diffusion loss (MSE on noise) | Direct convergence signal + distributional matching |
| **VLM** | Qwen3-VL ViT encoder only, LoRA fine-tunable | SigLIP (frozen, detached) | LoRA enables visual adaptation; Qwen3-VL has richer features |
| **Text encoder** | Qwen3-VL embed_tokens (2048-dim) | T5-XXL (4096-dim) separate encoder | Unified VLM, no separate text model needed |
| **Image storage** | Original resolution (resize at training) | Original resolution | Both identical — resize at training time |
| **Image augmentation** | ColorJitter (50%), CFG masking (10%) | ColorJitter + corruption, CFG masking | Similar approach |
| **Data pipeline** | LeRobot API + batch video decode, single-pass | TFRecord producer-consumer with disk buffer | Simpler, no intermediate files |
| **Parallelism** | PyTorch DDP (DeepSpeed ZeRO-2 ready) | DeepSpeed ZeRO-2 | DDP sufficient at 155M params; ZeRO-2 ready for scaling |
| **Eval** | Separate uniform DataLoader + normalized L2 | Separate sample DataLoader + MSE/L2 | Both use clean eval without augmentation |
| **Ctrl frequency** | `freq_embed(Hz)` conditioning | Same `ctrl_freqs` conditioning | Identical approach |

---

## 11. Training Results (8×A40, 14 datasets, 10K steps)

Results from run `base_0218_1630` (WandB: `crlc112358/drifting-vla/runs/fvkls5id`):

| Embodiment | Datasets | MSE ↓ | MAE ↓ |
|------------|----------|-------|-------|
| gripper_eef | rlbench | 0.0198 | 0.0874 |
| gripper_joint | droid, taco_play, bc_z | 0.0140 | 0.0751 |
| gripper_delta_eef | stanford_hydra, cmu_stretch, utaustin_mutex | 0.0106 | 0.0583 |
| bimanual | aloha, nyu_franka | 0.0063 | 0.0501 |
| bimanual_mobile | behavior1k ×3 | 0.0024 | 0.0267 |
| dex_hand | dexwild, dexgraspnet | 0.0323 | 0.1338 |
| **Overall** | **14 datasets, 6 embodiments** | **0.0127** | **0.0668** |

---

## 12. Training Scale & Resource Planning

### 12.1 Scale Tiers

| Tier | Steps | 8×A40 | 64×H200 | Purpose |
|------|-------|-------|---------|---------|
| Smoke test | 100 | 5 min | <1 min | Pipeline verification |
| Quick experiment | 10,000 | 12 hrs | 1.5 hrs | Development iteration |
| Medium pre-train | 100,000 | 6 days | 11 hrs | Ablations, paper results |
| **Full pre-train** | **1,000,000** | **58 days** | **4.6 days** | **Production (RDT-1B scale)** |
| Post-train | 200,000 | 12 days | 22 hrs | LoRA r=64, augmentation, CFG |

### 12.2 Training at 1M Steps (RDT-1B Scale)

We target **1,000,000 pre-training steps** to match RDT-1B's training scale. While our 1-NFE architecture converges faster per-step than diffusion, the full 1M steps ensure:

1. **All 13 dataset families seen sufficiently** — with effective batch 8,192, 1M steps = 8.2B total samples
2. **Drifting loss equilibrium** — V needs many iterations across all embodiments to converge toward zero
3. **Cross-embodiment transfer** — learning shared representations across 7 embodiment types requires extended training
4. **Fair comparison with RDT-1B** — same compute budget enables direct architectural comparison

Post-training adds 200K steps with image augmentation, larger LoRA (r=64), and CFG condition masking.

### 12.3 Multi-Node Configuration (64×H200)

**Phase 1 — Pre-training (1M steps, no augmentation):**
```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$RANK \
    --master_addr=$MASTER --master_port=29500 \
    scripts/train.py \
    --datasets aloha bc_z taco_play stanford_hydra cmu_stretch \
        droid utaustin_mutex nyu_franka dexora dexwild \
    --batch-size 64 --grad-accumulation 2 --max-steps 1000000 --lr 1e-4 \
    --vlm-mode lora --lora-r 16 --vlm-lr-scale 0.1 \
    --loss-type hybrid --model-size base --num-workers 8 --use-flash-attn \
    --log-every 500 --eval-every 10000 --save-every 25000 \
    --wandb-mode online --wandb-project drifting-vla-pretrain
```

**Phase 2 — Post-training (200K steps, augmentation + larger LoRA):**
```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$RANK \
    --master_addr=$MASTER --master_port=29500 \
    scripts/train.py \
    --datasets aloha bc_z taco_play stanford_hydra cmu_stretch \
        droid utaustin_mutex nyu_franka dexora dexwild \
    --batch-size 64 --grad-accumulation 2 --max-steps 200000 --lr 1e-4 \
    --vlm-mode lora --lora-r 64 --vlm-lr-scale 0.05 \
    --loss-type hybrid --model-size base --num-workers 8 --use-flash-attn \
    --image-aug --cond-mask-prob 0.1 \
    --log-every 200 --eval-every 5000 --save-every 10000 \
    --wandb-mode online --wandb-project drifting-vla-posttrain
```

### 12.4 Full Training Time Estimate

| Phase | Steps | Eff. Batch | 8×A40 (1 node) | 64×H200 (8 nodes) | GPU-hrs |
|-------|-------|-----------|----------------|-------------------|---------|
| Pre-train | 1,000,000 | 8,192 | 58 days | **4.6 days** | 7,066 |
| Post-train | 200,000 | 8,192 | 12 days | **22 hrs** | 1,408 |
| **Total** | **1,200,000** | | **70 days** | **~5.5 days** | **8,474** |

Estimated cost at ~$3/GPU-hr (H200): **~$25,400 total**.

Speed estimation basis:
- 8×A40 baseline: ~0.2 steps/s (measured, batch=32/gpu, accum=2)
- 64×H200: ~2.5 steps/s (8× more GPUs × 2× H200 speed × 1.5× Flash Attn, batch=64/gpu)

### 12.5 Data Preparation: Dataset Sizes and Time

Actual download + HDF5 conversion estimates based on verified HuggingFace dataset sizes:

| Dataset | Episodes | Frames | HF Download | HDF5 Output | Est. Prep Time |
|---------|----------|--------|-------------|-------------|---------------|
| aloha | 50 | 20K | ~0.5 GB | ~0.3 GB | 5 min |
| bc_z | 1,000 | 98K | ~76 GB | ~15 GB | 6 hrs |
| taco_play | 3,603 | 238K | ~48 GB | ~8 GB | 10 hrs |
| stanford_hydra | 600 | 358K | ~25 GB | ~5 GB | 8 hrs |
| cmu_stretch | 135 | 25K | ~3 GB | ~0.5 GB | 30 min |
| droid | 76,000 | 25.5M | **~1.7 TB** (RLDS) | ~300 GB | ~3 days |
| utaustin_mutex | 1,500 | 362K | ~30 GB | ~6 GB | 6 hrs |
| nyu_franka | 450 | 45K | ~8 GB | ~2 GB | 2 hrs |
| rlbench | 2,700 | 270K | ~116 GB | ~20 GB | 4 hrs |
| dexgraspnet | 8,270 scenes | 500K grasps | ~50 GB | ~10 GB | 3 hrs |
| dexwild | 9,500 | 95K | ~30 GB | ~8 GB | 4 hrs |
| dexora | 12,200 | 2.9M | ~150 GB (est) | ~50 GB | 12 hrs |
| behavior1k ×50 | ~100 ea | ~430K ea | ~5 GB ea | ~3 GB ea | ~2 hrs ea |
| **Total** | | **~35M** | **~2.8 TB** | **~600 GB** | |

**Wall time with `--parallel 5`:** ~2-3 days (bottleneck: DROID at 1.7 TB).

**Disk requirement:** ~3.4 TB peak (HF cache + HDF5 output). With `--cleanup` flag, HF cache is removed after each dataset, reducing peak to ~2 TB.

**Note:** The DROID dataset is by far the largest (1.7 TB download, 76K episodes). For initial experiments, use `--max-episodes 1000` to limit to ~1% of DROID (~20 GB download).

---

## 13. References

1. **Drifting:** One-Step Generation via Training-Time Distribution Evolution. arXiv:2602.04770, 2025.
2. **RDT-1B:** Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864, 2024.
3. **Pi0:** A Vision-Language-Action Flow Model. arXiv:2410.24164, 2024.
4. **DexGraspNet 2.0:** Learning Generative Dexterous Grasping. CoRL 2024.
5. **Qwen3-VL:** [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
6. **Open X-Embodiment:** RT-X Models. arXiv:2310.08864, 2023.
7. **Dexora:** [HuggingFace](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset)
8. **DexWild:** [HuggingFace](https://huggingface.co/datasets/boardd/dexwild-dataset)
9. **6D Rotation:** Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019.
10. **LoRA:** Hu et al., "Low-Rank Adaptation of Large Language Models", ICLR 2022.
11. **Behavior 1K:** [HuggingFace](https://huggingface.co/collections/lerobot/behavior-1k)
12. **Diffusion Policy:** Visuomotor Policy Learning via Action Diffusion. RSS 2023.
