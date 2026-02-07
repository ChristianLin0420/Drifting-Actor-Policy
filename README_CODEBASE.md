# Drifting-VLA Codebase

Production-ready PyTorch implementation of **Drifting-VLA** — a Vision-Language-Action policy using Drifting Models for **one-step** robotic action generation.

> **Paper reference:** [Drifting: One-Step Generative Model](https://arxiv.org/pdf/2602.04770)

---

## Quick Start

### 1. Build Docker Image

```bash
# Build base image (CUDA 12.1 + PyTorch 2.2)
docker build -f docker/Dockerfile.base -t drifting-vla:base .

# Build RLBench image (adds CoppeliaSim v4.1 + PyRep + RLBench)
docker build -f docker/Dockerfile.rlbench -t drifting-vla:rlbench .
```

### 2. Train (one command)

```bash
docker run --gpus all --name drifting-vla-train \
  -v $(pwd):/workspace --shm-size=16g \
  -e WANDB_API_KEY=your_key \
  drifting-vla:rlbench \
  python scripts/train.py
```

### 3. Interactive Mode

```bash
docker run -it --gpus all --name drifting-vla-dev \
  -v $(pwd):/workspace --shm-size=16g \
  -e WANDB_API_KEY=your_key \
  drifting-vla:rlbench bash

# Inside container — everything is ready:
python scripts/train.py
python scripts/train.py training.learning_rate=1e-3 training.max_steps=500
```

---

## Architecture

```
Image ──→ DINOv2 (frozen) ──┐
                             ├──→ Cross-Attention Fusion ──→ DiT Transformer ──→ Action Decoder ──→ [x,y,z, qx,qy,qz,qw, grip]
Language ──→ CLIP (frozen) ──┘                                    ↑
                                                            Noise ε ~ N(0,I)
```

| Component | Implementation | Params |
|-----------|---------------|--------|
| Vision Encoder | DINOv2-L/14 (frozen) | 304M |
| Language Encoder | CLIP ViT-L/14 (frozen) | 123M |
| Fusion | Cross-Attention (2 layers) | ~12M |
| DiT Transformer | 12 layers, 768-dim, 12 heads | ~110M |
| Action Decoder | MLP (2 layers) | ~2M |
| **Total** | | **574M (147M trainable)** |

**Action format:** `[x, y, z, qx, qy, qz, qw, gripper]` (8-dim)
- Position: absolute world coordinates
- Quaternion: `[qx, qy, qz, qw]` Hamilton convention, auto-normalized
- Gripper: 0.0 = close, 1.0 = open

---

## Drifting Loss

The core training objective evolves the model's output distribution toward expert data via a kernelized drifting field:

```
V(x) = V_attract(x, y_pos) − V_repel(x, y_neg)
Loss = MSE(x, stopgrad(x + V(x)))
```

**Key design choices:**
- Multi-temperature kernels: τ ∈ {0.02, 0.05, 0.2}
- Drift normalization: `E[‖V‖²/D] ≈ 1` (loss stays ~1.0 by design)
- **Convergence metric:** `λ_V = sqrt(E[‖V_raw‖²]/D)` → 0 at equilibrium
- Feature normalization for scale invariance

> **Note:** With drift normalization, the loss is always ≈ 1.0. This is correct per the paper. Monitor `train/lambda_V` and `train/raw_drift_norm` for convergence — these should decrease as the model learns.

---

## WandB Visualization Suite

All visualizations are logged atomically per step with proper step sliders.

### Training Metrics (Charts)

| Key | What It Shows |
|-----|---------------|
| `train/loss` | MSE loss (≈ 1.0 with normalization) |
| `train/lambda_V` | **Primary convergence metric** — should ↓ |
| `train/raw_drift_norm` | Raw drift magnitude before normalization |
| `train/lr` | Learning rate schedule |
| `val/loss` | Validation loss |
| `val/drift_norm` | Validation drift norm |
| `sim/success_rate` | RLBench simulation success rate |

### Visualization Panels (Images with step slider)

| Key | Plot | Interpretation |
|-----|------|----------------|
| `viz/A1_drifting_field` | 2D arrow plot | Arrows shrink = converging |
| `viz/A2_drift_magnitude_trend` | λ_V over steps | Should decrease |
| `viz/A3_prediction_scatter` | Pred vs GT per dim | Points on diagonal = good |
| `viz/A4_per_dim_error_radar` | Spider chart MAE | Spikes = hard dims |
| `viz/A5_temperature_loss` | Bar chart per τ | Low-τ = precision, high-τ = modes |
| `viz/B1_action_distribution` | Histograms | Overlap = good |
| `viz/B2_action_error_heatmap` | [T×D] grid | Dark = high error |
| `viz/B4_trajectory_3d` | 3D path | Smooth = coherent |
| `viz/C1_sample_transport` | Before→after drift | Orange near green = converged |
| `viz/C2_mode_coverage` | GMM clusters | Even bars = no collapse |

### Videos

| Key | Content |
|-----|---------|
| `eval/trajectory_video` | Sim camera + XY/3D trajectory plots |
| `sim/rollout_video` | Raw CoppeliaSim camera |

---

## Configuration

Uses Hydra. Override anything from CLI:

```bash
# Change model
python scripts/train.py model=dit_l2

# Change hyperparameters
python scripts/train.py training.learning_rate=1e-3 training.warmup_steps=50

# Disable simulation eval
python scripts/train.py training.simulation_eval.enabled=false

# Change visualization frequency
python scripts/train.py training.wandb.visualizations.drifting_field.frequency=100
```

### Config Files

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Root config (composes model + training + data) |
| `configs/model/dit_b2.yaml` | DiT-B/2 architecture (574M params) |
| `configs/model/dit_l2.yaml` | DiT-L/2 architecture (larger) |
| `configs/training/baseline.yaml` | Training hyperparameters + WandB + sim eval |
| `configs/data/rlbench.yaml` | Dataset path, tasks, cameras |

---

## Project Structure

```
├── docker/
│   ├── Dockerfile.base          # CUDA 12.1 + PyTorch 2.2 + Flash Attention
│   ├── Dockerfile.rlbench       # + CoppeliaSim 4.1 + RLBench + Xvfb
│   └── scripts/start_xvfb.sh
├── configs/
│   ├── config.yaml              # Root Hydra config
│   ├── model/dit_b2.yaml        # Model architecture
│   ├── training/baseline.yaml   # Training + WandB + sim eval
│   └── data/rlbench.yaml        # Dataset config
├── drifting_vla/
│   ├── models/
│   │   ├── drifting_vla.py      # Main model assembly
│   │   ├── dit.py               # DiT transformer + adaLN-Zero
│   │   ├── vision_encoder.py    # DINOv2 wrapper
│   │   ├── language_encoder.py  # CLIP text encoder
│   │   ├── fusion.py            # Cross-attention fusion
│   │   └── action_decoder.py    # MLP heads → [pos, quat, grip]
│   ├── training/
│   │   ├── drifting_field.py    # Algorithm 2: V(x) computation
│   │   ├── losses.py            # DriftingLoss + λ_V tracking
│   │   ├── trainer.py           # Training loop + viz + sim eval
│   │   ├── ema.py               # Exponential moving average
│   │   └── optimizer.py         # AdamW + cosine schedule
│   ├── envs/
│   │   ├── rlbench_env.py       # RLBench wrapper (action sanitization)
│   │   └── base_env.py          # Abstract env + DummyEnvironment
│   ├── inference/
│   │   ├── policy.py            # One-step inference with temporal ensemble
│   │   └── rollout.py           # Environment rollout utilities
│   ├── logging/
│   │   ├── wandb_logger.py      # Buffered WandB (atomic per-step logging)
│   │   ├── visualizations.py    # 10 plot functions (fixed-size output)
│   │   └── themes.py            # Plot styling
│   └── data/
│       ├── rlbench_dataset.py   # HuggingFace PerAct format loader
│       └── sample_queue.py      # Positive sample queue for drifting loss
├── scripts/
│   ├── train.py                 # Main training entry point
│   ├── simulate_eval.py         # Standalone simulation evaluation
│   └── download_rlbench_data.py # Download from HuggingFace
└── requirements/
    ├── base.txt                 # Core dependencies
    └── rlbench.txt              # RLBench-specific deps
```

---

## Data

Uses pre-generated RLBench demonstrations from HuggingFace:

```bash
python scripts/download_rlbench_data.py --tasks close_jar --output ./data/rlbench
```

Dataset: [`hqfang/rlbench-18-tasks`](https://huggingface.co/datasets/hqfang/rlbench-18-tasks) (PerAct format)

---

## Docker Details

### What's included in `drifting-vla:rlbench`

| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA | 12.1 | GPU compute |
| PyTorch | 2.2 | Deep learning framework |
| Flash Attention | 2 | Efficient attention |
| CoppeliaSim | 4.1.0 | Physics simulator |
| PyRep | 4.1.0.3 | CoppeliaSim Python API |
| RLBench | 1.2.0 | Robot manipulation benchmark |
| Xvfb | auto-start | Headless rendering |
| WandB | auto-login | Experiment tracking (via `WANDB_API_KEY` env) |

### Environment auto-configuration

The entrypoint script automatically:
1. Starts Xvfb on display :99
2. Sets `QT_QPA_PLATFORM_PLUGIN_PATH` for CoppeliaSim
3. Logs into WandB if `WANDB_API_KEY` is set
4. Configures git safe directory

No manual setup needed inside the container.

---

## Citation

```bibtex
@article{drifting2025,
  title={Drifting: One-Step Generation via Training-Time Distribution Evolution},
  year={2025},
  url={https://arxiv.org/abs/2602.04770}
}
```
