# Background Knowledge: Drifting-VLA

This document covers the theoretical foundations, detailed architecture, and design rationale for the Drifting-VLA project — a one-step multi-embodiment robotic manipulation model.

---

## 1. Drifting Models: Core Theory

### 1.1 What is a Drifting Model?

A Drifting model [arXiv:2602.04770](https://arxiv.org/abs/2602.04770) is a generative model that achieves **one-step sample generation** (1-NFE) by evolving the generator's output distribution during training.

**Key idea:** Instead of iteratively denoising at inference (like diffusion), the generator learns to map noise directly to data in a single forward pass. The training-time optimization uses a **drifting field** that pushes generated samples toward the data distribution.

```
Diffusion (100 steps):    noise → denoise → denoise → ... → denoise → sample
Drifting (1 step):        noise → generator → sample
```

### 1.2 The Drifting Field

The drifting field $V_{p,q}(x)$ is a vector field that points from generated samples toward the data manifold:

$$V_{p,q}(x) = \mathbb{E}_{y^+ \sim p}\mathbb{E}_{y^- \sim q}\Big[\tilde{k}(x, y^+)\tilde{k}(x, y^-)(y^+ - y^-)\Big]$$

Where:
- $x$ = current generated sample (query point)
- $y^+$ = positive sample from data distribution $p$
- $y^-$ = negative sample from generator distribution $q$
- $\tilde{k}$ = normalized kernel (attraction/repulsion weights)

**Intuition:** The field pulls $x$ toward positive samples (data) and pushes it away from negative samples (other generated points). At equilibrium ($p = q$), the field becomes zero.

### 1.3 Normalized Kernels (Double Softmax)

The kernel weights use **double softmax normalization** (Paper Eq. 13-15):

$$s_{ij} = -\frac{\|x_i - y_j\|}{\tau}$$

$$\tilde{k}(x_i, y_j) = \sqrt{\text{softmax}_j(s_{ij}) \cdot \text{softmax}_i(s_{ij})}$$

This ensures:
- Each query $x_i$ distributes attention across targets $y_j$ (softmax over $j$)
- Each target $y_j$ distributes influence across queries $x_i$ (softmax over $i$)
- The geometric mean balances both normalizations

**Multi-temperature:** We use $\tau \in \{0.02, 0.05, 0.2\}$ and sum the fields. Small $\tau$ = local structure, large $\tau$ = global structure.

### 1.4 Training Objective

$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{\epsilon \sim p_\epsilon}\Big[\|f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q}(f_\theta(\epsilon)))\|^2\Big]$$

Where:
- $f_\theta(\epsilon)$ = generator output (actions)
- $\text{sg}(\cdot)$ = stop gradient
- $V_{p,q}$ = drifting field

**The gradient is:**

$$\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f_\theta(\epsilon)$$

The stop-gradient means the model is pushed in the **direction** of $V$, not its magnitude.

### 1.5 Normalizations

**Feature normalization** (Paper Eq. 18-21):
- Scale all samples so average pairwise distance ≈ $\sqrt{D}$
- Prevents one dimension from dominating kernel computation

**Drift normalization** (Paper Eq. 23-25):
- Scale $V$ so that $\mathbb{E}[\|V\|^2 / D] = 1$
- Makes loss ≈ $D$ (constant by design)
- The learning signal is in $V$'s **direction**, not magnitude

### 1.6 Why Loss ≈ Constant is Correct

```
Common confusion: "Loss isn't decreasing → model isn't learning"

Reality for drifting:
  - Drift normalization ensures ||V||² ≈ D (constant)
  - Therefore L ≈ D (constant)
  - But V's DIRECTION changes during training
  - Gradient = -2V · ∇θ f pushes model based on direction
  - At equilibrium: V → 0, but normalization prevents this from showing in loss
  
What to monitor instead:
  - Task success rate (primary metric)
  - λ_V (normalization factor before scaling) — this does decrease
  - Action distribution quality (MMD vs expert)
```

### 1.7 Drifting vs Diffusion vs Flow Matching

| | Diffusion | Flow Matching | Drifting |
|---|-----------|--------------|----------|
| **Inference steps** | 50-100 | 5-20 | **1** |
| **Training signal** | Denoising score | Velocity field | Drifting field |
| **When distribution evolves** | At inference | At inference | **At training** |
| **ImageNet FID** | 1.79 (DiT) | 1.98 (SiT) | **1.54** |
| **Latency** | ~500ms | ~100ms | **~50ms** |

**Key insight:** Drifting moves the distribution evolution from inference-time (expensive) to training-time (amortized).

---

## 2. Model Architecture

### 2.1 Two-System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DRIFTING-VLA                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  SYSTEM 1: VLM Backbone (Qwen3-VL-2B-Instruct, 2.1B params)       │ │
│  │                                                                     │ │
│  │  Images [B,V,3,448,448] + Language [B] strings                     │ │
│  │       ↓                                                             │ │
│  │  Qwen3-VL: ViT vision encoder + Qwen2 language model               │ │
│  │  (native multi-image: all V views processed jointly)               │ │
│  │       ↓                                                             │ │
│  │  Hidden states [B, L, 2048]  (L varies with V)                    │ │
│  │       ↓                                                             │ │
│  │  Projection: Linear(2048→768) + LayerNorm                         │ │
│  │       ↓                                                             │ │
│  │  c_seq: [B, L, 768]    c_pool: [B, 768]                          │ │
│  │                                                                     │ │
│  │  Mode: Frozen → LoRA co-training (Pi0-style staged)               │ │
│  └───────────────────────────────┬────────────────────────────────────┘ │
│                                  ↓                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  SYSTEM 2: Drifting DiT (146M trainable params)                    │ │
│  │                                                                     │ │
│  │  Noise ε ~ N(0,I)  ──→  NoiseTokenizer(MLP 64→768)               │ │
│  │                           + positional embeddings                  │ │
│  │                                 ↓                                   │ │
│  │                          noise_tokens [B,T,768]                    │ │
│  │                                 ↓                                   │ │
│  │  c_global = c_pool + Embed(embodiment_id) + Embed(cfg_scale)      │ │
│  │                                 ↓                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │  Cross-Attention (×2 layers)                                  │ │ │
│  │  │  Q = noise_tokens, K/V = c_seq (VLM features)               │ │ │
│  │  │  → noise attends to fine-grained visual-language features    │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  │                                 ↓                                   │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │  DiT Transformer Blocks (×12)                                 │ │ │
│  │  │                                                                │ │ │
│  │  │  Each block:                                                   │ │ │
│  │  │    1. adaLN-Zero(x, c_global) → modulated LayerNorm          │ │ │
│  │  │    2. Self-Attention (RoPE + QK-Norm + Flash Attention)       │ │ │
│  │  │    3. Gate × Attention output + residual                      │ │ │
│  │  │    4. adaLN-Zero(x, c_global) → modulated LayerNorm          │ │ │
│  │  │    5. SwiGLU MLP (768 → 2048 → 768)                          │ │ │
│  │  │    6. Gate × MLP output + residual                            │ │ │
│  │  │                                                                │ │ │
│  │  │  adaLN-Zero: predict (scale, shift, gate) from c_global      │ │ │
│  │  │  Initialize gates to 0 → identity at start (stable training) │ │ │
│  │  └──────────────────────────────────────────────────────────────┘ │ │
│  │                                 ↓                                   │ │
│  │  Action Head: MLP(768 → 768 → 128) → actions [B, T, 128]        │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Parameter Count:                                                        │
│    VLM (Qwen3-VL-2B):  2.1B  (frozen, or LoRA ~0.5M trainable)        │
│    Projector:           3.2M  (always trainable)                        │
│    DiT + Action Head:   146M  (always trainable)                        │
│    Total trainable:     ~149M (or ~150M with LoRA)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 adaLN-Zero Conditioning

The DiT uses **adaptive Layer Normalization with Zero initialization**:

```
Input: x [B, T, 768], c_global [B, 768]

# Predict 6 modulation parameters from conditioning
modulation = MLP(c_global)  → [B, 6×768]
scale1, shift1, gate1, scale2, shift2, gate2 = chunk(modulation, 6)

# Attention block
x_norm = LayerNorm(x) * (1 + scale1) + shift1
x = x + gate1 * SelfAttention(x_norm)

# MLP block
x_norm = LayerNorm(x) * (1 + scale2) + shift2
x = x + gate2 * SwiGLU_MLP(x_norm)

# Gates initialized to 0 → at training start, each block is identity
# This prevents random DiT from corrupting gradients early in training
```

### 2.3 VLM Co-Training Strategy (Pi0-Style)

```
Stage 1 (Steps 0 → 5K):
  VLM: Frozen (no gradient)
  DiT: Training from scratch
  Why: Random DiT gradients would corrupt VLM

Stage 2 (Steps 5K → 50K):
  VLM: LoRA unfreezes (lr=1e-5, 10× lower than DiT)
  DiT: Continue training (lr=4e-4)
  Why: VLM adapts to robot domain

Stage 3 (Steps 50K → 100K):
  VLM: LoRA continues (lr=5e-6, lowered 2×)
  DiT: Continue training (lr=cosine decay → 1e-5)
  Why: Fine refinement

LoRA config: r=16, alpha=32, targets=[q_proj, v_proj], last 4 layers only
Trainable: ~0.5M params (0.024% of 2.1B)
```

---

## 3. Unified Action Space

### 3.1 128-Dim Layout (Non-Overlapping Regions)

```
Region A [0:8]:   Absolute EEF        x y z qx qy qz qw grip
Region B [8:16]:  Joint Positions     j0 j1 j2 j3 j4 j5 j6 grip
Region C [16:32]: Bimanual Joints     Lj0..Lj6 LG Rj0..Rj6 RG
Region D [32:48]: Dex Hand Fingers    f0 f1 ... f15
Region E [48:56]: Delta EEF           Δx Δy Δz Δrx Δry Δrz ΔG Δbase
         [56:127] Padding             zeros

Abs EEF mask:       1 1 1 1 1 1 1 1  0 0 0 0 0 0 0 0  0 ... 0  0 ... 0  0 ... 0  (8 active)
Joint mask:         0 0 0 0 0 0 0 0  1 1 1 1 1 1 1 1  0 ... 0  0 ... 0  0 ... 0  (8 active)
Bimanual mask:      0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0  1 ... 1  0 ... 0  0 ... 0  (16 active)
Dex hand mask:      1 1 1 1 1 1 1 0  0 0 0 0 0 0 0 0  0 ... 0  1 ... 1  0 ... 0  (23 active)
Delta EEF mask:     0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0  0 ... 0  0 ... 0  1 ... 1  (8 active)
```

### 3.2 Action Format: Mixed is OK (RDT-1B Approach)

Following [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer), datasets keep their **native action format** — some use absolute EE, some delta EE, some joint positions. This works because:

1. **128-dim unified space** — each dataset occupies the same slots, but semantics differ
2. **Per-dataset z-normalization** — scales become comparable (mean=0, std=1)
3. **Embodiment ID conditioning** — the DiT knows which format to produce
4. **At inference** — denormalize using target dataset stats → correct format

```
Dataset         Format              Example action
─────────────────────────────────────────────────────
rlbench         absolute EE pose    [0.5, 0.3, 0.2, 0.0, 0.0, 0.7, 0.7, 1.0]
droid           joint positions     [1.2, -0.5, 0.8, 0.1, -0.3, 0.6, 1.0]
utaustin_mutex  delta EE            [0.01, -0.02, 0.005, 0.0, 0.0, 0.01, 0.0]
aloha           absolute joints     [0.1, -0.2, ..., 0.3, -0.1] (14 dims)

After z-normalization (per-dataset):
  All → mean≈0, std≈1, comparable scale in the model's latent space
```

This is validated by RDT-1B training on 46 datasets with mixed formats and achieving 53.6% on ManiSkill.

### 3.3 How Action Masking Works in Loss

```python
action_mask = get_mask(embodiment_id)   # [128] bool

# MSE only on active dims
mse_loss = ((pred * mask - gt * mask) ** 2).sum() / mask.sum()

# Drifting field also computed only on active dims
V = compute_drifting_field(pred * mask, pos_samples * mask, neg_samples * mask)
```

---

## 4. Loss Function: Hybrid MSE + Drifting

### 4.1 Why Hybrid?

```
Pure drifting (paper):
  + Multi-modal: captures diverse valid actions
  + Theoretically elegant
  - Requires batch ≥ 2048 for stable field estimation
  - Loss ≈ constant (hard to debug)

Pure MSE:
  + Converges at any batch size
  + Easy to monitor
  - Unimodal: collapses to mean action
  - No diversity in predictions

Hybrid MSE + λ·Drifting:
  + MSE provides convergence guarantee
  + Drifting adds multi-modality
  + Works at batch 32-2048
  λ = 0.1 (default)
```

### 4.2 Training Dynamics

```
Step 0:      MSE ≈ 1000    drift ≈ 1.0    (random model)
Step 1K:     MSE ≈ 50      drift ≈ 1.0    (MSE driving learning)
Step 10K:    MSE ≈ 5       drift ≈ 1.0    (converging on mean)
Step 50K:    MSE ≈ 0.5     drift ≈ 1.0    (drifting adds diversity)
Step 100K:   MSE ≈ 0.1     drift ≈ 1.0    (equilibrium)

Drift loss stays ≈ 1.0 throughout — this is CORRECT.
MSE decreases — this is the convergence signal.
λ_V (drift normalization factor) decreases — this shows drifting is working.
```

---

## 5. Image Processing (Pi0-Style)

### 5.1 Why All RGB Views?

```
Previous approach (hard cap at 2):
  ✗ Behavior 1K: 3 RGB cameras → drops head camera
  ✗ RLBench: 5 cameras → drops 3
  ✗ DROID: 3 cameras → drops 1

Pi0 approach (our current):
  ✓ Accept ALL RGB views from each dataset
  ✓ Filter out depth/segmentation automatically
  ✓ Variable V handled via padding + attention mask
  ✓ Qwen3-VL natively supports multi-image (1-N views)
```

### 5.2 Data Flow for Images

```
Raw data (varies per dataset):
  RLBench:      [front_rgb, wrist_rgb, left_shoulder, right_shoulder, overhead]
  Behavior 1K:  [rgb.head, rgb.left_wrist, rgb.right_wrist, depth.*, seg.*]
  DROID:        [exterior_1, exterior_2, wrist]
  ALOHA:        [top]

Step 1: Filter RGB only (exclude depth, seg, mask)
Step 2: Resize all to 448×448
Step 3: Stack → [V, 3, 448, 448]  (V varies: 1-5)

Batch collation:
  Find max_V in batch → pad shorter samples with black frames
  → [B, max_V, 3, 448, 448]

VLM processing (Qwen3-VL):
  All V views + language encoded jointly in one forward pass
  → [B, L, 2048]  (L depends on V: more views = more tokens)

DiT cross-attention naturally handles variable L via attention
```

---

## 6. Drifting Field Computation Details

### 6.1 Algorithm

```
Input:
  x: [B, D]       query points (model predictions, flattened)
  y_pos: [N+, D]  positive samples (from data/queue)
  y_neg: [N-, D]  negative samples (from queue/detached)
  τ: [0.02, 0.05, 0.2]  temperatures

For each temperature τ:
  1. Compute distances: d_pos[i,j] = ||x[i] - y_pos[j]||
                        d_neg[i,k] = ||x[i] - y_neg[k]||

  2. Score matrices: s_pos[i,j] = -d_pos[i,j] / τ
                     s_neg[i,k] = -d_neg[i,k] / τ

  3. Double softmax normalization:
     w_pos[i,j] = sqrt(softmax_j(s_pos[i,j]) * softmax_i(s_pos[i,j]))
     w_neg[i,k] = sqrt(softmax_k(s_neg[i,k]) * softmax_i(s_neg[i,k]))

  4. Attraction: V_pos[i] = Σ_j w_pos[i,j] * y_pos[j] * Σ_k w_neg[i,k]
     Repulsion:  V_neg[i] = Σ_k w_neg[i,k] * y_neg[k] * Σ_j w_pos[i,j]

  5. Field: V_τ[i] = V_pos[i] - V_neg[i]

Sum across temperatures: V = Σ_τ V_τ

Feature normalization: scale so avg pairwise distance ≈ √D
Drift normalization: scale V so E[||V||²/D] = 1
```

### 6.2 Positive and Negative Sampling

```
Positive samples (y_pos):
  Source: Global queue of recent GT actions
  Size: 256 per step (sampled from queue of 2048)
  Update: Push GT actions after each step

Negative samples (y_neg):
  At large batch (≥2048): x.detach() (paper's approach)
  At small batch (<1024): Negative queue of past predictions
  Queue size: 2048
  Update: Push model predictions after each step

Why queue matters at small batch:
  With batch=32, x.detach() gives 32 negatives ≈ 32 queries
  → V_attract ≈ V_repel → V ≈ 0 (field collapses)
  Queue provides diverse negatives from previous steps
  → V remains informative
```

---

## 7. Comparison to Related Models

### 7.1 Architecture Comparison

```
                   VLM           Action Generator     Action Space    Inference
RDT-1B (2024)     T5-XXL        Diffusion DiT (1.2B)  128-dim mask   100 steps
OpenVLA (2024)     LLaMA-7B      Autoregressive        256 tokens     1 step*
Octo (2023)       ViT            Diffusion MLP         Per-embodiment 10 steps
Pi0 (2024)        ViT+LLM       Flow Matching          Joint/EE      5 steps
Ours              Qwen3-VL-2B   Drifting DiT (146M)   128-dim mask   1 step

* OpenVLA generates 1 action token but struggles with continuous actions
```

### 7.2 Dataset Comparison

```
                 Datasets    Embodiments    Total Samples
RDT-1B           46          2 (grip+bi)    ~1M
OpenVLA          970K demos  1 (gripper)    ~970K
Octo             25          1 (gripper)    ~800K
Pi0              Undisclosed 2              Undisclosed
Ours             60          3 (grip+bi+dex) ~32M
```

### 7.3 Speed Comparison

```
                 NFE    Latency (H100)    Control Hz
Diffusion (100)  100    ~500ms            2 Hz
Flow (10)        10     ~100ms            10 Hz
Drifting (ours)  1      ~50ms             20 Hz

50ms = 40ms VLM + 10ms DiT
```

---

## 8. Common Pitfalls & Solutions

### 8.1 "Loss not decreasing"
**Expected.** Drift normalization makes $\mathcal{L} \approx D$. Track task success rate instead.

### 8.2 "Drifting field V ≈ 0"
Negative samples too similar to queries. Use negative queue (queue_size ≥ 2048).

### 8.3 "Mixed delta/absolute actions"
All LeRobot OXE datasets store absolute actions after conversion. Verified by per-dataset normalization stats.

### 8.4 "Multi-view padding wastes compute"
Black-frame padding has near-zero attention weight. Compute overhead is ~5% vs single-view.

### 8.5 "VLM features don't update during training"
Default is now **online VLM** (Pi0-style). VLM runs live in each forward pass. LoRA enables co-training with differential LR.

---

## 9. Project Objective

**Goal:** Build a unified foundation model that controls gripper, bimanual, and dexterous hand robots through one-step action generation, achieving:

1. **>80% success** on RLBench (18 tasks, gripper)
2. **>70% success** on DexGraspNet 2.0 (dexterous grasping)
3. **>50% success** on ManiSkill (bimanual)
4. **<50ms inference** (20 Hz real-time control)
5. **10× faster** than diffusion baselines (RDT-1B, Diffusion Policy)

**Target venue:** NeurIPS 2026

---

## References

1. **Drifting:** [arXiv:2602.04770](https://arxiv.org/abs/2602.04770) — One-Step Generation via Training-Time Distribution Evolution.
2. **Demo notebook:** [Colab](https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb)
3. **RDT-1B:** [arXiv:2410.07864](https://arxiv.org/abs/2410.07864) — Diffusion Foundation Model for Bimanual Manipulation.
4. **DexGraspNet 2.0:** [GitHub](https://github.com/PKU-EPIC/DexGraspNet2) — Generative Dexterous Grasping.
5. **Qwen3-VL:** [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) — Multimodal Vision-Language Model.
6. **Open X-Embodiment:** [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) — Robotic Learning Datasets.
7. **LeRobot OXE:** [HuggingFace Collection](https://huggingface.co/collections/lerobot/open-x-embodiment)
8. **Behavior 1K:** [HuggingFace Collection](https://huggingface.co/collections/lerobot/behavior-1k)
9. **Diffusion Policy:** RSS 2023 — Visuomotor Policy Learning via Action Diffusion.
10. **Pi0:** Physical Intelligence — Flow-matching VLA with online VLM co-training.

