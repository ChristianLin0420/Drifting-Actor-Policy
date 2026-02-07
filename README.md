# Drifting Actor Policy

**One-step Vision-Language-Action policy for robotic manipulation using [Drifting Models](https://arxiv.org/abs/2602.04770).**

| | |
|---|---|
| **Inference** | Single forward pass (~12ms on H100) |
| **Architecture** | DINOv2 + CLIP + DiT Transformer (574M params) |
| **Training** | Kernelized drifting field with multi-temperature attraction/repulsion |
| **Evaluation** | RLBench simulation with CoppeliaSim |
| **Logging** | WandB with 10+ visualization panels + simulation video |

> **Codebase documentation:** See [`README_CODEBASE.md`](README_CODEBASE.md) for setup, Docker usage, and configuration.

### Quick Start

```bash
# Build Docker image (includes CoppeliaSim + RLBench + Xvfb)
docker build -f docker/Dockerfile.rlbench -t drifting-vla:rlbench .

# Train (WandB auto-login, Xvfb auto-start, sim eval included)
docker run --gpus all -v $(pwd):/workspace --shm-size=16g \
  -e WANDB_API_KEY=your_key \
  drifting-vla:rlbench python scripts/train.py
```

---

## Research Plan

This document presents a comprehensive research plan to integrate the recently proposed **Drifting Model** paradigm with Vision-Language-Action (VLA) policies for robotic manipulation. The Drifting model achieves state-of-the-art results in both generative modeling (ImageNet 256×256: FID 1.54) and robotic control tasks, demonstrating its potential as a unified framework for VLA applications.

**Key Innovation:** We propose **Drifting-VLA**, a novel architecture that leverages the training-time distribution evolution paradigm of Drifting models to learn multimodal action policies conditioned on vision and language, achieving one-step inference without iterative sampling.

---

## Table of Contents

1. [Mathematical Analysis: Drifting vs. Diffusion, Flow Matching, and MeanFlow](#1-mathematical-analysis)
2. [Proposed Solution: Drifting-VLA Architecture](#2-proposed-solution)
3. [Theoretical and Empirical Justifications](#3-justifications)
4. [Implementation Plan and Schedule](#4-implementation-plan)
5. [Codebase Design](#5-codebase-design)
6. [Experimental Protocol](#6-experimental-protocol)

---

## 1. Mathematical Analysis: Drifting vs. Diffusion, Flow Matching, and MeanFlow {#1-mathematical-analysis}

### 1.1 Core Paradigm Comparison

#### 1.1.1 Diffusion Models

**Forward Process:**
```latex
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
```

**Reverse Process (Inference):**
```latex
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
```

**Training Objective:**
```latex
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
```

**Key Characteristics:**
- **Iteration Location:** Inference time (50-1000 steps)
- **Theoretical Foundation:** Stochastic Differential Equations (SDEs)
- **Training Paradigm:** Denoise noisy samples at various time steps
- **Inference Cost:** $O(N \cdot \text{NFE})$ where NFE = 50-1000

#### 1.1.2 Flow Matching

**Continuous Normalizing Flow:**
```latex
\frac{dx_t}{dt} = v_\theta(x_t, t), \quad x_0 \sim p_{\text{prior}}, \quad x_1 \sim p_{\text{data}}
```

**Training Objective (Conditional Flow Matching):**
```latex
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - u_t(x_t | x_0, x_1) \|^2 \right]
```

where $u_t(x_t | x_0, x_1) = \frac{x_1 - x_0}{1}$ for linear interpolation.

**Key Characteristics:**
- **Iteration Location:** Inference time (typically 10-100 steps)
- **Theoretical Foundation:** Ordinary Differential Equations (ODEs)
- **Training Paradigm:** Match velocity field at interpolated states
- **Inference Cost:** $O(N \cdot \text{NFE})$ where NFE = 10-100

#### 1.1.3 MeanFlow

**MeanFlow Field:**
```latex
V_{\text{MeanFlow}}(x) = \mathbb{E}_{y^+ \sim p_{\text{data}}} [y^+ - x]
```

**Training Objective:**
```latex
\mathcal{L}_{\text{MeanFlow}} = \mathbb{E}_{\epsilon \sim p_\epsilon} \left[ \| f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{\text{MeanFlow}}(f_\theta(\epsilon))) \|^2 \right]
```

**Key Characteristics:**
- **Iteration Location:** Training time (one-step inference!)
- **Theoretical Foundation:** Direct distribution matching
- **Training Paradigm:** Iteratively drift generator outputs toward data
- **Inference Cost:** $O(N)$ - single forward pass

#### 1.1.4 Drifting Models (Proposed)

**Drifting Field Definition:**
```latex
V_{p,q}(x) = \mathbb{E}_{y^+ \sim p} \mathbb{E}_{y^- \sim q} [\tilde{k}(x, y^+) \tilde{k}(x, y^-) (y^+ - y^-)]
```

where the normalized kernel is:
```latex
\tilde{k}(x, y) = \frac{k(x, y)}{Z(x)}, \quad Z(x) = \mathbb{E}_y[k(x, y)]
```

and the base kernel is:
```latex
k(x, y) = \exp\left(-\frac{1}{\tau} \|x - y\|\right)
```

**Decomposition into Attraction and Repulsion:**
```latex
V_{p,q}(x) = V_p^+(x) - V_q^-(x)
```

where:
```latex
V_p^+(x) = \frac{1}{Z_p} \mathbb{E}_{y^+ \sim p} [k(x, y^+)(y^+ - x)] \quad \text{(attraction)}
```
```latex
V_q^-(x) = \frac{1}{Z_q} \mathbb{E}_{y^- \sim q} [k(x, y^-)(y^- - x)] \quad \text{(repulsion)}
```

**Training Objective:**
```latex
\mathcal{L}_{\text{Drifting}} = \mathbb{E}_{\epsilon \sim p_\epsilon} \left[ \| f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q_\theta}(f_\theta(\epsilon))) \|^2 \right]
```

**Equilibrium Condition (Anti-Symmetry):**
```latex
V_{p,q}(x) = -V_{q,p}(x) \quad \Rightarrow \quad p = q \Rightarrow V_{p,q}(x) = 0, \, \forall x
```

**Key Characteristics:**
- **Iteration Location:** Training time (one-step inference!)
- **Theoretical Foundation:** Pushforward distribution evolution + kernel-based attraction/repulsion
- **Training Paradigm:** Evolve $q_i = f_{\theta_i} \# p_\epsilon$ through drifting field
- **Inference Cost:** $O(N)$ - single forward pass
- **Novel Feature:** Explicitly models sample interactions via kernelized positive/negative pairs

### 1.2 Comparative Mathematical Analysis

#### 1.2.1 Distribution Evolution Perspective

| Model | Evolution Domain | Evolution Mechanism | Equilibrium Condition |
|-------|-----------------|---------------------|----------------------|
| **Diffusion** | Inference time | SDE: $dx = -\nabla \log p(x_t) dt + \sqrt{2} dw$ | $x_T \sim p_{\text{data}}$ |
| **Flow Matching** | Inference time | ODE: $\frac{dx}{dt} = v_\theta(x, t)$ | $x_1 \sim p_{\text{data}}$ |
| **MeanFlow** | Training time | Direct drift: $x_{i+1} = x_i + \mathbb{E}[y - x]$ | $\mathbb{E}[y - x] = 0$ |
| **Drifting** | Training time | Kernelized drift: $x_{i+1} = x_i + V_{p,q}(x)$ | $V_{p,q}(x) = 0$ |

#### 1.2.2 Sample Complexity Analysis

**Theorem 1 (Informal):** For Drifting models, the drifting field $V_{p,q}(x)$ provides richer gradient information compared to MeanFlow.

**Intuition:**
- **MeanFlow:** $V_{\text{MeanFlow}}(x) = \mathbb{E}[y - x]$ treats all data points equally
- **Drifting:** $V_{p,q}(x) = \mathbb{E}[\tilde{k}(x, y^+) \tilde{k}(x, y^-) (y^+ - y^-)]$ weights by kernel similarity

The kernel $k(x, y)$ assigns higher weights to nearby samples, providing:
1. **Local gradient information:** More accurate drift in high-density regions
2. **Multi-scale learning:** Can use multiple temperature values $\tau \in \{0.02, 0.05, 0.2\}$
3. **Contrastive structure:** Explicitly models negative samples (generated) vs. positive samples (data)

**Empirical Comparison (from paper):**
- **MeanFlow XL/2:** FID 3.43 (latent), 1-NFE
- **Drifting L/2:** FID 1.54 (latent), 1-NFE
- **Improvement:** 55% reduction in FID with comparable model size

#### 1.2.3 Gradient Flow Analysis

**Proposition 2:** The drifting field gradient can be decomposed as:

```latex
\nabla_x V_{p,q}(x) = \nabla_x V_p^+(x) - \nabla_x V_q^-(x)
```

For Gaussian kernel $k(x, y) = \exp(-\|x - y\|^2 / (2\tau^2))$:

```latex
\nabla_x V_p^+(x) = \mathbb{E}_{y^+ \sim p} \left[ \tilde{k}(x, y^+) \left( I - \frac{(y^+ - x)(y^+ - x)^\top}{\tau^2} \right) \right]
```

This provides:
- **Curvature information:** Second-order term $(y^+ - x)(y^+ - x)^\top$
- **Adaptive step size:** Through kernel normalization $\tilde{k}(x, y^+)$

### 1.3 Connection to Maximum Mean Discrepancy (MMD)

**Drifting Field as MMD Gradient:**

The MMD loss is:
```latex
\mathcal{L}_{\text{MMD}}^2(p, q) = \mathbb{E}_{x, x' \sim q}[\xi(x, x')] - 2\mathbb{E}_{y \sim p, x \sim q}[\xi(x, y)] + \text{const}
```

Its gradient is:
```latex
\frac{\partial \mathcal{L}_{\text{MMD}}^2}{\partial x} = 2\mathbb{E}_{y^- \sim q}\left[\frac{\partial \xi(x, y^-)}{\partial x}\right] - 2\mathbb{E}_{y^+ \sim p}\left[\frac{\partial \xi(x, y^+)}{\partial x}\right]
```

For radial kernel $\xi(x, y) = \xi(\|x - y\|^2)$:
```latex
\frac{\partial \xi(x, y)}{\partial x} = 2\xi'(\|x - y\|^2)(x - y)
```

**Key Difference from Drifting:**
1. **MMD uses unnormalized kernels:** $\xi'(\|x - y\|^2)$ depends on absolute distance
2. **Drifting uses normalized kernels:** $\tilde{k}(x, y) = k(x, y) / Z(x)$

The normalization provides:
- **Scale invariance:** Insensitive to feature magnitude
- **Better multi-scale learning:** Can combine features at different resolutions
- **Stable training:** Avoids vanishing/exploding gradients

---

## 2. Proposed Solution: Drifting-VLA Architecture {#2-proposed-solution}

### 2.1 Problem Formulation

**VLA Task:** Learn a policy $\pi_\theta: (\mathcal{I}, \mathcal{L}) \rightarrow \mathcal{A}$ that maps:
- **Vision input** $\mathcal{I}$: RGB images, depth maps, or multi-view observations
- **Language input** $\mathcal{L}$: Natural language task descriptions
- **Action output** $\mathcal{A}$: Robot action sequences (positions, velocities, gripper states)

**Traditional VLA Approaches:**
1. **Diffusion Policy:** Multi-step denoising at inference
2. **Flow-based Policies:** ODE integration at inference
3. **Autoregressive Policies:** Sequential token generation

**Our Approach:** **Drifting-VLA** - Single-step action generation via training-time distribution evolution

### 2.2 Architecture Design

#### 2.2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Drifting-VLA Policy                      │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Vision    │  │   Language   │  │    Noise     │       │
│  │  Encoder    │  │   Encoder    │  │   Sample     │       │
│  │   (ViT)     │  │    (LLM)     │  │  ε ~ N(0,I)  │       │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                  │
│                  ┌────────▼────────┐                         │
│                  │  Fusion Layer   │                         │
│                  │   (Cross-Attn)  │                         │
│                  └────────┬────────┘                         │
│                           │                                  │
│                  ┌────────▼────────┐                         │
│                  │    Drifting     │                         │
│                  │   Transformer   │                         │
│                  │   (DiT-style)   │                         │
│                  └────────┬────────┘                         │
│                           │                                  │
│                  ┌────────▼────────┐                         │
│                  │  Action Decoder │                         │
│                  │   (MLP + Heads) │                         │
│                  └────────┬────────┘                         │
│                           │                                  │
│                    ┌──────▼──────┐                           │
│                    │   Actions   │                           │
│                    │  a_1,...,a_T │                           │
│                    └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.2 Component Specifications

**Vision Encoder:**
- **Architecture:** Pre-trained Vision Transformer (ViT-L/14) or ConvNeXt-V2
- **Input:** RGB images at resolution 224×224
- **Output:** Visual tokens $\mathbf{v} \in \mathbb{R}^{N_v \times d_v}$
- **Pre-training:** MAE or CLIP (frozen or fine-tuned)

**Language Encoder:**
- **Architecture:** Pre-trained LLM (e.g., LLaMA-2-7B, CLIP text encoder)
- **Input:** Tokenized task description
- **Output:** Language embeddings $\mathbf{l} \in \mathbb{R}^{N_l \times d_l}$
- **Pre-training:** Frozen for initial experiments

**Fusion Layer:**
- **Architecture:** Cross-attention layers
- **Mechanism:**
  ```latex
  \mathbf{h}_{\text{fusion}} = \text{CrossAttn}(\mathbf{Q} = [\mathbf{v}; \mathbf{l}; \epsilon], \mathbf{K} = [\mathbf{v}; \mathbf{l}], \mathbf{V} = [\mathbf{v}; \mathbf{l}])
  ```
- **Output:** Fused tokens $\mathbf{h} \in \mathbb{R}^{(N_v + N_l + N_\epsilon) \times d}$

**Drifting Transformer:**
- **Architecture:** DiT-style transformer with:
  - **Depth:** 24 layers (L/2 variant)
  - **Hidden dimension:** 1024
  - **Attention heads:** 16
  - **Conditioning:** adaLN-zero for language/vision conditioning
  - **Positional encoding:** RoPE (Rotary Position Embedding)
  - **Normalization:** RMSNorm + QK-Norm
  - **Activation:** SwiGLU

**Action Decoder:**
- **Architecture:** MLP + specialized heads for:
  - **Position:** 3D Cartesian coordinates
  - **Orientation:** Quaternion or 6D rotation representation
  - **Gripper:** Binary (open/close) or continuous aperture
  - **Velocity:** Joint velocities (optional)
- **Output:** Action sequence $\mathbf{a} = [a_1, ..., a_T] \in \mathbb{R}^{T \times d_a}$
  - $T$: Action horizon (e.g., 16-32 steps)
  - $d_a$: Action dimensionality (e.g., 7-10D)

#### 2.2.3 Training with Drifting Loss

**Positive Samples:** Expert demonstrations from dataset
```latex
\mathbf{a}^+ \sim p_{\text{data}}(\mathbf{a} | \mathcal{I}, \mathcal{L})
```

**Negative Samples:** Generated actions from current policy
```latex
\mathbf{a}^- \sim q_\theta(\mathbf{a} | \mathcal{I}, \mathcal{L}) = f_\theta(\epsilon, \mathcal{I}, \mathcal{L}) \text{ where } \epsilon \sim \mathcal{N}(0, I)
```

**Drifting Field (in action space):**
```latex
V_{p,q}(\mathbf{a} | \mathcal{I}, \mathcal{L}) = \mathbb{E}_{\mathbf{a}^+ \sim p} \mathbb{E}_{\mathbf{a}^- \sim q} [\tilde{k}(\mathbf{a}, \mathbf{a}^+) \tilde{k}(\mathbf{a}, \mathbf{a}^-) (\mathbf{a}^+ - \mathbf{a}^-)]
```

**Training Loss:**
```latex
\mathcal{L}_{\text{Drifting-VLA}} = \mathbb{E}_{\epsilon, \mathcal{I}, \mathcal{L}} \left[ \| \mathbf{a} - \text{sg}(\mathbf{a} + V_{p,q}(\mathbf{a} | \mathcal{I}, \mathcal{L})) \|^2 \right]
```

where $\mathbf{a} = f_\theta(\epsilon, \mathcal{I}, \mathcal{L})$ and sg denotes stop-gradient.

**Multi-Temperature Kernel:**
```latex
k_\tau(\mathbf{a}, \mathbf{a}') = \exp\left(-\frac{\|\mathbf{a} - \mathbf{a}'\|^2}{\tau}\right), \quad \tau \in \{0.02, 0.05, 0.2\}
```

#### 2.2.4 Feature-Space Drifting (Optional Enhancement)

Following the image generation results, we can compute drifting loss in a learned feature space:

**Feature Encoder:** $\phi: \mathbb{R}^{T \times d_a} \rightarrow \mathbb{R}^{d_\phi}$
- **Architecture:** 1D CNN or Transformer encoder
- **Training:** Self-supervised (MAE) or supervised (action prediction auxiliary task)

**Feature-Space Loss:**
```latex
\mathcal{L}_{\text{feature}} = \sum_{j} \mathbb{E}_{\epsilon, \mathcal{I}, \mathcal{L}} \left[ \| \phi_j(\mathbf{a}) - \text{sg}(\phi_j(\mathbf{a}) + V_{p,q}(\phi_j(\mathbf{a}))) \|^2 \right]
```

where $\phi_j$ denotes features at different temporal scales.

### 2.3 Classifier-Free Guidance for VLA

**CFG for Language Conditioning:**

During training, mix negative samples:
```latex
\tilde{q}(\mathbf{a} | \mathcal{I}, \mathcal{L}) = (1 - \gamma) q_\theta(\mathbf{a} | \mathcal{I}, \mathcal{L}) + \gamma q_\theta(\mathbf{a} | \mathcal{I}, \emptyset)
```

This leads to:
```latex
q_\theta(\mathbf{a} | \mathcal{I}, \mathcal{L}) = \alpha p_{\text{data}}(\mathbf{a} | \mathcal{I}, \mathcal{L}) - (\alpha - 1) p_{\text{data}}(\mathbf{a} | \mathcal{I}, \emptyset)
```

where $\alpha = \frac{1}{1 - \gamma}$.

**Implementation:**
1. Sample $\alpha \sim p(\alpha)$ with $p(\alpha) \propto \alpha^{-3}$
2. Condition network on $\alpha$: $f_\theta(\epsilon, \mathcal{I}, \mathcal{L}, \alpha)$
3. At inference: Choose $\alpha$ for task performance vs. diversity trade-off

---

## 3. Theoretical and Empirical Justifications {#3-justifications}

### 3.1 Theoretical Advantages

#### 3.1.1 One-Step Inference Efficiency

**Theorem 3 (Computational Complexity):**

For VLA policies generating action sequences of length $T$:

| Method | Inference Cost | Parameters |
|--------|---------------|------------|
| Diffusion Policy | $O(N \cdot K \cdot T)$ | $K$ = 10-100 diffusion steps |
| Flow Matching | $O(N \cdot K \cdot T)$ | $K$ = 10-50 ODE steps |
| Autoregressive | $O(N \cdot T)$ | Sequential generation |
| **Drifting-VLA** | $O(N \cdot T)$ | **Single forward pass** |

where $N$ is the network size (number of FLOPs per forward pass).

**Real-time Control Benefits:**
- **Latency:** ~10-50ms vs. 100-500ms for multi-step methods
- **Throughput:** 100-200 Hz control frequency achievable
- **Energy:** 10-100× reduction in compute for edge deployment

#### 3.1.2 Mode Coverage and Multi-Modality

**Proposition 4 (Mode Coverage):** The drifting field $V_{p,q}(x)$ prevents mode collapse through explicit repulsion from negative samples.

**Proof Sketch:**
1. If $q$ collapses to a single mode $\mu_1$ of bimodal $p = 0.5\mathcal{N}(\mu_1, \Sigma) + 0.5\mathcal{N}(\mu_2, \Sigma)$
2. Then samples $x \sim q$ are near $\mu_1$
3. But $V_p^+(x)$ has significant component from $\mu_2$: 
   ```latex
   V_p^+(x) \approx 0.5 k(x, \mu_1)(\mu_1 - x) + 0.5 k(x, \mu_2)(\mu_2 - x)
   ```
4. Since $\mu_2$ is far from $x$, attraction from $\mu_2$ is non-zero
5. This prevents equilibrium at mode collapse

**Empirical Evidence:** Figure 3 in paper shows recovery from collapsed initialization.

**VLA Application:** Multi-modal action distributions (e.g., grasp from left or right) are preserved.

#### 3.1.3 Sample Efficiency

**Proposition 5 (Gradient Quality):** The kernelized drifting field provides higher-quality gradients than mean-field methods.

**Intuition:**
- **MeanFlow:** Global averaging loses local structure
  ```latex
  V_{\text{MeanFlow}}(x) = \mathbb{E}_{y \sim p}[y - x]
  ```
- **Drifting:** Kernel weighting preserves local geometry
  ```latex
  V_{\text{Drifting}}(x) = \mathbb{E}_{y^+, y^-}[\tilde{k}(x, y^+)\tilde{k}(x, y^-)(y^+ - y^-)]
  ```

**Expected Benefit:** Faster convergence, fewer training samples needed.

### 3.2 Empirical Advantages

#### 3.2.1 State-of-the-Art Performance

**Image Generation Results (from paper):**
- **ImageNet 256×256 (latent):** FID 1.54 (vs. iMeanFlow 1.72, AdvFlow 2.38)
- **ImageNet 256×256 (pixel):** FID 1.61 (vs. PixelDiT 1.61, SiD2 1.38)
- **One-step generation:** Competitive with multi-step diffusion models

**Robotics Control Results (from paper):**

| Task | Diffusion Policy (100-NFE) | Drifting Policy (1-NFE) |
|------|---------------------------|-------------------------|
| Lift (Visual) | 1.00 | 1.00 |
| Can (Visual) | 0.97 | 0.99 |
| ToolHang (State) | 0.30 | **0.38** |
| PushT (Visual) | 0.84 | 0.86 |
| BlockPush Phase 1 | 0.36 | **0.56** |

**Implications:**
1. **Matching or exceeding** Diffusion Policy with 100× fewer NFE
2. **Better performance** on some tasks (ToolHang, BlockPush)
3. **Robust across modalities** (state and visual observations)

#### 3.2.2 Training Stability

**Observation from Paper:**
- No adversarial training (unlike GANs)
- No mode collapse in 2D experiments (Figure 3)
- Stable training with large batch sizes (8192 effective batch)

**Advantage for VLA:**
- Robotics datasets are often small (1k-100k demonstrations)
- Stable training crucial for data-efficient learning
- No hyperparameter sensitivity to adversarial dynamics

#### 3.2.3 Scalability

**Computational Requirements:**
- **Training:** Parallelizable across unlimited H100 nodes
- **Batch size:** Larger = better (positive/negative sample estimation)
- **Data loading:** Efficient sample queue (128 per class, 1000 unconditional)

**Scaling Laws (Expected):**
- Model size: B/2 → L/2 improves FID 1.75 → 1.54
- Training epochs: 100 → 1280 improves FID 3.36 → 1.75
- Feature encoder: Standard MAE → Large MAE improves FID significantly

### 3.3 Why Drifting is Superior to Alternatives for VLA

#### 3.3.1 vs. Diffusion Policy

| Aspect | Diffusion Policy | Drifting-VLA |
|--------|-----------------|--------------|
| **Inference Speed** | 10-100 steps | **1 step** |
| **Real-time Control** | 10-20 Hz | **100-200 Hz** |
| **Theoretical Foundation** | SDE (complex) | **Pushforward evolution (intuitive)** |
| **Multi-modality** | Good | **Better (explicit repulsion)** |
| **Training Complexity** | Moderate | **Low (no timestep scheduling)** |

#### 3.3.2 vs. Flow Matching

| Aspect | Flow Matching | Drifting-VLA |
|--------|--------------|--------------|
| **Inference Speed** | 10-50 steps | **1 step** |
| **Training Paradigm** | Match velocity at interpolated points | **Evolve distribution at training** |
| **Gradient Quality** | Approximates ODE trajectory | **Direct distribution matching** |
| **Implementation** | ODE solvers needed | **Simple MSE loss** |

#### 3.3.3 vs. MeanFlow

| Aspect | MeanFlow | Drifting-VLA |
|--------|----------|--------------|
| **Gradient Information** | Global mean (uniform weighting) | **Kernelized (local weighting)** |
| **Performance** | FID 3.43 (ImageNet) | **FID 1.54 (ImageNet)** |
| **Sample Efficiency** | Requires many positives | **Better with same budget** |
| **Feature Space** | Not explored | **Multi-scale features** |

#### 3.3.4 vs. Autoregressive (RT-X, PALM-E)

| Aspect | Autoregressive VLA | Drifting-VLA |
|--------|-------------------|--------------|
| **Inference Speed** | Sequential (slow) | **Parallel (fast)** |
| **Latency** | $O(T)$ sequential | **$O(1)$ parallel** |
| **Multi-modality** | Sampling from logits | **Explicit attraction/repulsion** |
| **Discrete Bottleneck** | Action quantization | **Continuous actions** |

---

## 4. Implementation Plan and Schedule {#4-implementation-plan}

### 4.1 Timeline Overview (97 Days Total)

```
┌─────────────┬─────────────┬──────────────┬──────────────┬──────────────┬─────────────┐
│  Phase 1    │  Phase 2    │   Phase 3    │   Phase 4    │   Phase 5    │  Phase 6    │
│  Setup      │  Baseline   │  Feature     │  Scaling     │  Ablations   │  Writing    │
│  (7 days)   │ (14 days)   │  (21 days)   │  (21 days)   │  (14 days)   │  (20 days)  │
└─────────────┴─────────────┴──────────────┴──────────────┴──────────────┴─────────────┘
Day 0-7       Day 8-21      Day 22-42      Day 43-63      Day 64-77      Day 78-97

Total: 97 days
```

### 4.2 Detailed Phase Breakdown

#### Phase 1: Setup and Infrastructure (Days 1-7)

**Objectives:**
1. Set up codebase and development environment
2. Prepare datasets
3. Implement core Drifting model components
4. Verify reproduction of paper results

**Tasks:**

**Days 1-2: Environment Setup**
- [ ] Clone and set up distributed training framework (PyTorch + DeepSpeed/FSDP)
- [ ] Configure H100 cluster (8-16 nodes × 8 GPUs = 64-128 GPUs)
- [ ] Set up experiment tracking (Weights & Biases / MLflow)
- [ ] Prepare data storage and loading infrastructure

**Days 3-4: Data Preparation**
- [ ] Download and preprocess robotics datasets:
  - RT-1 dataset (130k demonstrations, 700 tasks)
  - Bridge V2 dataset (60k demonstrations)
  - CALVIN dataset (for multi-task evaluation)
  - RLBench (for simulation experiments)
- [ ] Implement efficient data loader with:
  - Multi-resolution image loading
  - Language tokenization
  - Action normalization/standardization
  - Sample queue for positive/negative sampling

**Days 5-7: Core Implementation**
- [ ] Implement Drifting model components:
  - Drifting field computation (Algorithm 2 from paper)
  - Kernel functions with multiple temperatures
  - Feature normalization
  - Drift normalization
- [ ] Verify correctness with 2D toy experiments (Figure 3 reproduction)
- [ ] Implement multi-GPU distributed training
- [ ] Set up logging and checkpointing

**Deliverables:**
- ✓ Working distributed training infrastructure
- ✓ Preprocessed datasets ready for training
- ✓ Verified Drifting implementation on toy tasks

#### Phase 2: Baseline Drifting-VLA (Days 8-21)

**Objectives:**
1. Implement baseline Drifting-VLA architecture
2. Train on single task (e.g., pick-and-place)
3. Achieve basic functionality and verify one-step inference

**Tasks:**

**Days 8-10: Architecture Implementation**
- [ ] Vision encoder:
  - Pre-trained ViT-L/14 or ConvNeXt-V2
  - Feature extraction at multiple scales
  - Frozen vs. fine-tuned experiments
- [ ] Language encoder:
  - CLIP text encoder or small LLM (LLaMA-2-7B)
  - Tokenization and embedding layers
- [ ] Fusion layer:
  - Cross-attention mechanism
  - Positional embeddings for modalities
- [ ] Drifting Transformer:
  - DiT-B/2 architecture (starting small)
  - adaLN-zero conditioning
  - RoPE, RMSNorm, QK-Norm
- [ ] Action decoder:
  - MLP heads for position, orientation, gripper
  - Action sequence generation (T=16)

**Days 11-14: Training Pipeline**
- [ ] Implement training loop:
  - Batch sampling (Nc=64 tasks, Npos=32, Nneg=32)
  - Positive sample queue (size 128 per task)
  - CFG conditioning with $\alpha$ sampling
  - EMA for model parameters
- [ ] Loss computation:
  - Action-space drifting loss
  - Multiple temperature kernels
  - Feature/drift normalization
- [ ] Optimization:
  - AdamW optimizer
  - Learning rate warmup (5k steps)
  - Gradient clipping (max_norm=2.0)

**Days 15-18: Single-Task Training**
- [ ] Train on RLBench "reach_target" task:
  - 10k demonstrations
  - 50 epochs (fast iteration)
  - Batch size 2048 (32 GPUs × 64 batch/GPU)
  - Expected training time: ~8 hours
- [ ] Evaluate metrics:
  - Success rate (target: >90%)
  - Action distribution quality (MMD vs. expert)
  - Inference latency (target: <20ms)

**Days 19-21: Debugging and Iteration**
- [ ] Diagnose failure modes:
  - Check equilibrium: is $\|V\|^2$ decreasing?
  - Visualize action distributions (PCA, t-SNE)
  - Analyze kernel behavior (are similarities meaningful?)
- [ ] Hyperparameter tuning:
  - Temperature values $\tau$
  - Batch size allocation (Nc, Npos, Nneg)
  - Learning rate
- [ ] Iterate until baseline works

**Deliverables:**
- ✓ Baseline Drifting-VLA architecture
- ✓ Successful single-task training (>90% success rate)
- ✓ Verified one-step inference (<20ms latency)

#### Phase 3: Feature-Space Drifting (Days 22-42)

**Objectives:**
1. Implement feature-space drifting for action sequences
2. Pre-train action feature encoder
3. Evaluate improvement over baseline

**Tasks:**

**Days 22-25: Feature Encoder Design**
- [ ] Action feature encoder architectures:
  - **Option 1:** 1D ResNet (temporal convolutions)
  - **Option 2:** Transformer encoder (temporal attention)
  - **Option 3:** Hybrid (Conv + Attention)
- [ ] Multi-scale feature extraction:
  - Extract features at different temporal resolutions
  - Global pooling (mean, std)
  - Local patches (2-step, 4-step windows)

**Days 26-30: Self-Supervised Pre-Training**
- [ ] Implement Masked Action Modeling (MAM):
  - Mask 50% of action sequence (random 2-step patches)
  - Reconstruct masked actions with encoder-decoder
  - Loss: MSE on masked regions
- [ ] Pre-train on combined dataset:
  - All available robotics demonstrations (~200k)
  - Batch size 4096, 100 epochs
  - Expected training time: ~24 hours (64 GPUs)
- [ ] Variants to try:
  - Different mask ratios (25%, 50%, 75%)
  - Different encoder widths (256, 512, 768)
  - Longer pre-training (200, 500 epochs)

**Days 31-35: Feature-Space Drifting Training**
- [ ] Implement multi-scale feature loss:
  - Compute drifting field at each feature scale
  - Feature normalization (Equation 18-21 from paper)
  - Drift normalization (Equation 23-25 from paper)
  - Sum across scales and temperatures
- [ ] Train Drifting-VLA with feature loss:
  - Same single-task setup (RLBench reach_target)
  - Compare: action-space only vs. feature-space only vs. both
  - Expected improvement: +5-10% success rate

**Days 36-39: Multi-Task Training**
- [ ] Train on 10 RLBench tasks:
  - Reach, push, pick, stack, open drawer, etc.
  - 5k demonstrations per task (50k total)
  - 100 epochs, batch size 4096
  - Expected training time: ~48 hours (64 GPUs)
- [ ] Evaluate generalization:
  - Zero-shot to new object instances
  - Zero-shot to new language descriptions

**Days 40-42: Ablation Studies**
- [ ] Ablate feature encoder components:
  - Pre-training epochs: {50, 100, 200, 500}
  - Encoder width: {256, 512, 768}
  - Architecture: ResNet vs. Transformer
- [ ] Ablate feature-space loss:
  - Number of scales: {1, 2, 4}
  - Feature normalization: on vs. off
  - Drift normalization: on vs. off

**Deliverables:**
- ✓ Pre-trained action feature encoder
- ✓ Feature-space drifting implementation
- ✓ Multi-task Drifting-VLA model (10 tasks)
- ✓ Ablation results quantifying feature-space benefits

#### Phase 4: Scaling to Full VLA (Days 43-63)

**Objectives:**
1. Scale to large datasets (RT-1 + Bridge)
2. Scale model size (DiT-L/2)
3. Achieve SOTA performance on established benchmarks

**Tasks:**

**Days 43-46: Large-Scale Data Preparation**
- [ ] Combine datasets:
  - RT-1: 130k demonstrations, 700 tasks
  - Bridge V2: 60k demonstrations
  - Total: ~190k demonstrations
- [ ] Implement efficient data loading:
  - Distributed sampling across GPUs
  - Pre-fetching and caching
  - Balanced sampling across tasks
- [ ] Language augmentation:
  - Paraphrase task descriptions (GPT-4)
  - Multi-lingual translations (for robustness)

**Days 47-50: Model Scaling**
- [ ] Implement DiT-L/2 architecture:
  - Hidden dimension: 1024
  - Depth: 24 layers
  - Parameters: ~400M (excluding encoders)
- [ ] Optimize for large-scale training:
  - FSDP (Fully Sharded Data Parallel)
  - Activation checkpointing
  - Mixed precision (FP16/BF16)
  - Flash Attention 2
- [ ] Estimate training cost:
  - Model FLOPs: ~500 GFLOPs per sample
  - Dataset size: 190k × 100 epochs = 19M samples
  - Total FLOPs: ~10^19
  - H100 cluster (128 GPUs): ~7-10 days

**Days 51-60: Large-Scale Training**
- [ ] Train DiT-L/2 Drifting-VLA:
  - Batch size: 8192 (128 GPUs × 64 batch/GPU)
  - Epochs: 100
  - Learning rate: 4e-4 with 10k warmup
  - CFG: $\alpha \in [1, 4]$, $p(\alpha) \propto \alpha^{-5}$
  - EMA decay: {0.999, 0.9995, 0.9998, 0.9999}
- [ ] Monitor training:
  - Drifting loss $\|V\|^2$
  - Success rate on validation tasks
  - Action distribution divergence (MMD, KL)
- [ ] Checkpointing:
  - Save every 10 epochs
  - Keep top-5 by validation success rate

**Days 61-63: Evaluation and Benchmarking**
- [ ] RT-1 benchmark:
  - 6 real-world tasks
  - Compare to RT-1, RT-2 baselines
  - Target: >85% average success rate
- [ ] Bridge V2 benchmark:
  - Diverse manipulation tasks
  - Compare to Diffusion Policy baseline
  - Target: Match or exceed Diffusion Policy
- [ ] CALVIN benchmark:
  - Long-horizon multi-task evaluation
  - Measure: average task chain length
  - Target: >3.0 tasks (SOTA is ~2.8)
- [ ] Inference latency:
  - Measure on single H100
  - Target: <15ms for 1-step generation

**Deliverables:**
- ✓ Large-scale Drifting-VLA model (400M params)
- ✓ Trained on 190k demonstrations, 700+ tasks
- ✓ Benchmark results on RT-1, Bridge, CALVIN
- ✓ Verified real-time inference capability

#### Phase 5: Ablations and Analysis (Days 64-77)

**Objectives:**
1. Comprehensive ablation studies
2. Theoretical analysis and visualizations
3. Comparison with baselines

**Tasks:**

**Days 64-68: Architecture Ablations**
- [ ] Model size: B/2 vs. L/2 vs. XL/2
- [ ] Vision encoder: ViT vs. ConvNeXt vs. DINOv2
- [ ] Language encoder: CLIP vs. LLaMA-2 vs. frozen vs. fine-tuned
- [ ] Fusion mechanism: Cross-attention vs. concatenation vs. FiLM

**Days 69-72: Training Ablations**
- [ ] Batch size allocation:
  - (Nc, Npos, Nneg) ∈ {(64,32,32), (128,64,64), (256,128,64)}
  - Effect on sample efficiency
- [ ] CFG schedule:
  - $p(\alpha)$ distributions: uniform, $\alpha^{-3}$, $\alpha^{-5}$
  - Inference $\alpha$ sweep: {1.0, 1.5, 2.0, 3.0, 4.0}
- [ ] Feature encoder:
  - Pre-training epochs: {100, 500, 1000}
  - Encoder width: {256, 512, 768, 1024}
  - Fine-tuning vs. frozen

**Days 73-75: Comparison with Baselines**
- [ ] Implement and train baselines on same data:
  - **Diffusion Policy** (100-NFE)
  - **Flow Matching Policy** (50-NFE)
  - **Autoregressive Policy** (GPT-style)
- [ ] Metrics:
  - Success rate (primary)
  - Inference latency
  - Sample efficiency (performance vs. data size)
  - Multi-modality (action diversity)

**Days 76-77: Analysis and Visualizations**
- [ ] Gradient flow analysis:
  - Visualize $\nabla_\theta \mathcal{L}$ throughout training
  - Compare gradient norms across methods
- [ ] Equilibrium analysis:
  - Plot $\|V\|^2$ over training
  - Correlate with success rate
- [ ] Mode coverage:
  - Visualize action distributions (PCA, t-SNE)
  - Measure multi-modality (GMM fitting)
- [ ] Kernel behavior:
  - Heatmaps of $k(a, a')$ for action pairs
  - Effect of temperature $\tau$

**Deliverables:**
- ✓ Comprehensive ablation study (15+ experiments)
- ✓ Head-to-head comparison with 3 baselines
- ✓ Theoretical analysis and visualizations
- ✓ Insights for future work

#### Phase 6: Paper Writing (Days 78-97)

**Objectives:**
1. Write full NeurIPS paper
2. Prepare supplementary materials
3. Internal review and polish

**Tasks:**

**Days 78-82: Drafting**
- [ ] Abstract (300 words)
- [ ] Introduction (2 pages):
  - Motivation: one-step VLA for real-time control
  - Contributions: Drifting-VLA, feature-space drifting, SOTA results
- [ ] Related Work (1.5 pages):
  - Diffusion/Flow-based VLA
  - One-step generative models
  - VLA architectures
- [ ] Method (4 pages):
  - Background: Drifting models (Section 1.1-1.3)
  - Drifting-VLA architecture (Section 2.2)
  - Training with feature-space loss (Section 2.2.3-2.2.4)
  - CFG for VLA (Section 2.3)

**Days 83-87: Results and Analysis**
- [ ] Experiments (3 pages):
  - Main results: RT-1, Bridge, CALVIN benchmarks (tables + plots)
  - Ablation studies (tables)
  - Comparison with baselines (table + bar charts)
- [ ] Analysis (1 page):
  - Equilibrium dynamics (Figure 4 style)
  - Mode coverage (Figure 3 style)
  - Inference latency (table)
- [ ] Discussion (0.5 pages):
  - Limitations
  - Future work

**Days 88-91: Figures and Tables**
- [ ] Create all figures:
  - Architecture diagram (Figure in Section 2.2.1)
  - Training dynamics (loss, success rate over time)
  - Qualitative results (robot execution videos → frames)
  - Ablation bar charts
  - Comparison with baselines
- [ ] Create all tables:
  - Main results (RT-1, Bridge, CALVIN)
  - Ablations (architecture, training, feature encoder)
  - Inference latency comparison
- [ ] Caption writing and polish

**Days 92-94: Supplementary Materials**
- [ ] Appendix A: Implementation details
  - Architecture specifications (Table 8 style)
  - Training hyperparameters
  - Evaluation protocols
- [ ] Appendix B: Additional experiments
  - More ablations
  - Qualitative examples
  - Failure case analysis
- [ ] Appendix C: Theoretical proofs
  - Proposition 4 (mode coverage)
  - Proposition 5 (gradient quality)
- [ ] Code release preparation:
  - Clean up codebase
  - Write README
  - Prepare demo scripts

**Days 95-97: Review and Polish**
- [ ] Internal review:
  - Co-authors read and comment
  - Address feedback
- [ ] Proofreading:
  - Grammar and style
  - Consistency checks
  - Reference formatting
- [ ] Final checks:
  - NeurIPS format compliance
  - Page limit (9 pages + unlimited references)
  - Supplementary material organization
- [ ] Submission preparation:
  - PDF generation
  - Supplementary zip file
  - Abstract for OpenReview

**Deliverables:**
- ✓ Complete NeurIPS paper (9 pages)
- ✓ Supplementary materials (20+ pages)
- ✓ Code repository (ready for release)
- ✓ Submission package

### 4.3 Risk Mitigation

**Risk 1: Training Instability**
- **Mitigation:** Start with small model (B/2), verify convergence before scaling
- **Fallback:** Use action-space loss only (no feature encoder) if unstable

**Risk 2: Underperformance vs. Baselines**
- **Mitigation:** Extensive ablations to identify bottlenecks
- **Fallback:** Hybrid model (Drifting + auxiliary diffusion loss)

**Risk 3: Dataset Issues**
- **Mitigation:** Prepare multiple datasets in parallel
- **Fallback:** Simulation-only experiments (RLBench, CALVIN)

**Risk 4: Compute Constraints**
- **Mitigation:** Optimize batch size and gradient accumulation
- **Fallback:** Reduce model size (B/2 instead of L/2)

**Risk 5: Time Pressure**
- **Mitigation:** Parallel experiments across H100 nodes
- **Fallback:** Reduce scope (fewer tasks, fewer ablations)

---

## 5. Codebase Design {#5-codebase-design}

### 5.1 Repository Structure

```
drifting-vla/
├── README.md
├── setup.py
├── requirements.txt
├── configs/
│   ├── model/
│   │   ├── dit_b2.yaml
│   │   ├── dit_l2.yaml
│   │   └── dit_xl2.yaml
│   ├── training/
│   │   ├── baseline.yaml
│   │   ├── feature_space.yaml
│   │   └── large_scale.yaml
│   ├── data/
│   │   ├── rt1.yaml
│   │   ├── bridge.yaml
│   │   └── rlbench.yaml
│   └── experiment/
│       ├── single_task.yaml
│       ├── multi_task.yaml
│       └── benchmark.yaml
├── drifting_vla/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dit.py                 # DiT transformer
│   │   ├── vision_encoder.py      # ViT, ConvNeXt
│   │   ├── language_encoder.py    # CLIP, LLaMA
│   │   ├── fusion.py              # Cross-attention fusion
│   │   ├── action_decoder.py      # MLP heads for actions
│   │   ├── feature_encoder.py     # Action feature encoder (MAM)
│   │   └── drifting_vla.py        # Main model class
│   ├── training/
│   │   ├── __init__.py
│   │   ├── drifting_field.py      # Algorithm 2: compute V
│   │   ├── losses.py              # Drifting loss, feature loss
│   │   ├── trainer.py             # Training loop
│   │   ├── optimizer.py           # AdamW + scheduler
│   │   └── ema.py                 # Exponential moving average
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Base dataset class
│   │   ├── rt1_dataset.py         # RT-1 specific
│   │   ├── bridge_dataset.py      # Bridge specific
│   │   ├── rlbench_dataset.py     # RLBench specific
│   │   ├── transforms.py          # Augmentations
│   │   └── sample_queue.py        # Positive sample queue
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Success rate, MMD, etc.
│   │   ├── benchmarks.py          # RT-1, Bridge, CALVIN
│   │   └── visualize.py           # Action distribution plots
│   └── utils/
│       ├── __init__.py
│       ├── distributed.py         # FSDP, DDP helpers
│       ├── logging.py             # WandB, tensorboard
│       ├── checkpoint.py          # Save/load model
│       └── config.py              # Config management
├── scripts/
│   ├── pretrain_feature_encoder.py
│   ├── train_drifting_vla.py
│   ├── evaluate_benchmark.py
│   ├── ablation_study.py
│   └── visualize_results.py
├── tests/
│   ├── test_drifting_field.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
└── notebooks/
    ├── 01_toy_experiments.ipynb
    ├── 02_action_distribution_viz.ipynb
    └── 03_results_analysis.ipynb
```

### 5.2 Key Module Specifications

#### 5.2.1 `drifting_field.py`

```python
"""
Core drifting field computation (Algorithm 2 from paper)
"""

import torch
import torch.nn.functional as F

def compute_drifting_field(
    x: torch.Tensor,          # [N, D] generated samples
    y_pos: torch.Tensor,       # [N_pos, D] positive samples (expert actions)
    y_neg: torch.Tensor,       # [N_neg, D] negative samples (generated actions)
    temperatures: list[float] = [0.02, 0.05, 0.2],
    normalize_features: bool = True,
    normalize_drift: bool = True,
) -> torch.Tensor:
    """
    Compute drifting field V(x) = V^+(x) - V^-(x)
    
    Args:
        x: Generated samples (batch)
        y_pos: Positive samples from data distribution
        y_neg: Negative samples from generated distribution
        temperatures: List of temperature values for multi-scale kernels
        normalize_features: Apply feature normalization (Eq. 18-21)
        normalize_drift: Apply drift normalization (Eq. 23-25)
        
    Returns:
        V: Drifting field [N, D]
    """
    N, D = x.shape
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]
    
    # Feature normalization (optional)
    if normalize_features:
        scale = compute_feature_normalization_scale(x, y_pos, y_neg, D)
        x = x / scale
        y_pos = y_pos / scale
        y_neg = y_neg / scale
    
    # Compute drifting field across multiple temperatures
    V_total = torch.zeros_like(x)
    
    for tau in temperatures:
        # Compute pairwise distances
        dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
        dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]
        
        # Ignore self in negatives (if y_neg is x)
        if N == N_neg and torch.allclose(y_neg, x):
            dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6
        
        # Compute logits for softmax
        logit_pos = -dist_pos / tau  # [N, N_pos]
        logit_neg = -dist_neg / tau  # [N, N_neg]
        
        # Concatenate for joint normalization
        logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]
        
        # Double softmax normalization (over rows and cols)
        A_row = F.softmax(logit, dim=-1)   # Normalize over y-axis
        A_col = F.softmax(logit, dim=-2)   # Normalize over x-axis
        A = torch.sqrt(A_row * A_col)      # Geometric mean
        
        # Split back into positive and negative
        A_pos = A[:, :N_pos]  # [N, N_pos]
        A_neg = A[:, N_pos:]  # [N, N_neg]
        
        # Compute weights with cross-normalization
        W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
        W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]
        
        # Compute drift
        drift_pos = W_pos @ y_pos  # [N, D]
        drift_neg = W_neg @ y_neg  # [N, D]
        V = drift_pos - drift_neg   # [N, D]
        
        # Drift normalization (optional)
        if normalize_drift:
            V = normalize_drift_field(V, D)
        
        V_total = V_total + V
    
    return V_total


def compute_feature_normalization_scale(
    x: torch.Tensor, 
    y_pos: torch.Tensor, 
    y_neg: torch.Tensor, 
    D: int
) -> torch.Tensor:
    """
    Compute normalization scale such that E[||x - y||] ≈ √D
    (Equation 21 from paper)
    """
    # Concatenate all samples
    all_samples = torch.cat([x, y_pos, y_neg], dim=0)  # [N + N_pos + N_neg, D]
    
    # Compute pairwise distances (subsample for efficiency)
    N_total = all_samples.shape[0]
    if N_total > 1000:
        # Subsample to avoid memory issues
        idx = torch.randperm(N_total, device=all_samples.device)[:1000]
        all_samples = all_samples[idx]
    
    dist = torch.cdist(all_samples, all_samples)  # [N_sub, N_sub]
    avg_dist = dist.mean()
    
    # Scale such that avg_dist ≈ √D
    scale = avg_dist / torch.sqrt(torch.tensor(D, dtype=dist.dtype, device=dist.device))
    
    return scale.detach()  # Stop gradient


def normalize_drift_field(V: torch.Tensor, D: int) -> torch.Tensor:
    """
    Normalize drift such that E[||V||^2 / D] ≈ 1
    (Equation 25 from paper)
    """
    # Compute normalization factor
    norm_squared = (V ** 2).sum(dim=1).mean()  # E[||V||^2]
    lambda_V = torch.sqrt(norm_squared / D)
    
    # Normalize
    V_normalized = V / (lambda_V + 1e-8)  # Add epsilon for stability
    
    return V_normalized
```

#### 5.2.2 `losses.py`

```python
"""
Drifting loss and feature-space loss
"""

import torch
import torch.nn as nn

class DriftingLoss(nn.Module):
    """
    Drifting loss: L = E[||x - sg(x + V(x))||^2]
    """
    def __init__(
        self,
        temperatures: list[float] = [0.02, 0.05, 0.2],
        normalize_features: bool = True,
        normalize_drift: bool = True,
    ):
        super().__init__()
        self.temperatures = temperatures
        self.normalize_features = normalize_features
        self.normalize_drift = normalize_drift
        
    def forward(
        self, 
        x: torch.Tensor,           # [B, D] generated samples
        y_pos: torch.Tensor,       # [N_pos, D] positive samples
        y_neg: torch.Tensor,       # [N_neg, D] negative samples
    ) -> dict:
        """
        Compute drifting loss
        
        Returns:
            dict with keys:
                - 'loss': scalar loss value
                - 'drift_norm': ||V||^2 (for monitoring)
        """
        from drifting_vla.training.drifting_field import compute_drifting_field
        
        # Compute drifting field
        V = compute_drifting_field(
            x, y_pos, y_neg,
            temperatures=self.temperatures,
            normalize_features=self.normalize_features,
            normalize_drift=self.normalize_drift,
        )
        
        # Target: x + V (with stop-gradient)
        x_target = (x + V).detach()
        
        # Loss: MSE between x and x_target
        loss = ((x - x_target) ** 2).mean()
        
        # Monitor drift norm
        drift_norm = (V ** 2).sum(dim=1).mean()
        
        return {
            'loss': loss,
            'drift_norm': drift_norm,
        }


class FeatureSpaceLoss(nn.Module):
    """
    Multi-scale feature-space drifting loss
    """
    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_scales: list[str] = ['global', 'patch_2', 'patch_4'],
        temperatures: list[float] = [0.02, 0.05, 0.2],
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.feature_scales = feature_scales
        self.temperatures = temperatures
        self.loss_fn = DriftingLoss(temperatures=temperatures)
        
    def forward(
        self,
        actions: torch.Tensor,       # [B, T, D_a] generated action sequences
        actions_pos: torch.Tensor,   # [N_pos, T, D_a] positive actions
        actions_neg: torch.Tensor,   # [N_neg, T, D_a] negative actions
    ) -> dict:
        """
        Compute feature-space drifting loss at multiple scales
        
        Returns:
            dict with keys:
                - 'loss': total loss (sum over scales)
                - 'loss_{scale}': loss at each scale
                - 'drift_norm': average drift norm
        """
        # Extract features at multiple scales
        features = self.feature_encoder.extract_multi_scale(actions)
        features_pos = self.feature_encoder.extract_multi_scale(actions_pos)
        features_neg = self.feature_encoder.extract_multi_scale(actions_neg)
        
        total_loss = 0.0
        total_drift_norm = 0.0
        losses = {}
        
        for scale in self.feature_scales:
            # Get features at this scale
            feat = features[scale]           # [B, D_feat]
            feat_pos = features_pos[scale]   # [N_pos, D_feat]
            feat_neg = features_neg[scale]   # [N_neg, D_feat]
            
            # Compute drifting loss in feature space
            loss_dict = self.loss_fn(feat, feat_pos, feat_neg)
            
            # Accumulate
            total_loss = total_loss + loss_dict['loss']
            total_drift_norm = total_drift_norm + loss_dict['drift_norm']
            losses[f'loss_{scale}'] = loss_dict['loss'].item()
        
        # Average drift norm
        avg_drift_norm = total_drift_norm / len(self.feature_scales)
        
        return {
            'loss': total_loss,
            'drift_norm': avg_drift_norm,
            **losses,
        }
```

#### 5.2.3 `trainer.py`

```python
"""
Main training loop for Drifting-VLA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import wandb

class DriftingVLATrainer:
    """
    Trainer for Drifting-VLA models
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = device
        
        # EMA model
        from drifting_vla.training.ema import EMA
        self.ema = EMA(model, decay=cfg.ema_decay)
        
        # Logging
        self.step = 0
        self.epoch = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_drift_norm = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['image'].to(self.device)         # [B, C, H, W]
            language = batch['language'].to(self.device)     # [B, L]
            actions_pos = batch['actions'].to(self.device)   # [B, T, D_a]
            
            # Sample noise
            B = images.shape[0]
            T, D_a = actions_pos.shape[1], actions_pos.shape[2]
            noise = torch.randn(B, T, D_a, device=self.device)
            
            # Sample CFG scale alpha
            alpha = self.sample_cfg_alpha(B)
            
            # Forward: generate actions
            actions = self.model(noise, images, language, alpha)  # [B, T, D_a]
            
            # Sample negative actions (reuse batch as negatives)
            actions_neg = actions.detach()
            
            # Sample additional positive samples from queue
            actions_pos_extra = self.sample_from_queue(batch['task_id'], self.cfg.N_pos)
            actions_pos_all = torch.cat([actions_pos, actions_pos_extra], dim=0)
            
            # Sample unconditional negatives for CFG
            if self.cfg.use_cfg:
                actions_uncond = self.sample_unconditional(self.cfg.N_uncond)
                # Weight unconditional samples by CFG schedule
                # (implementation details in full code)
            
            # Compute loss
            loss_dict = self.loss_fn(actions, actions_pos_all, actions_neg)
            loss = loss_dict['loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            
            # Update EMA
            self.ema.update()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_drift_norm += loss_dict['drift_norm'].item()
            
            if batch_idx % self.cfg.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/drift_norm': loss_dict['drift_norm'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.step,
                })
            
            self.step += 1
        
        # Epoch-level logging
        avg_loss = total_loss / len(self.train_loader)
        avg_drift_norm = total_drift_norm / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'drift_norm': avg_drift_norm,
        }
    
    def validate(self):
        """Validation loop"""
        self.model.eval()
        self.ema.eval()
        
        total_success = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                language = batch['language'].to(self.device)
                actions_gt = batch['actions'].to(self.device)
                
                # Generate actions (one-step inference)
                B, T, D_a = actions_gt.shape
                noise = torch.randn(B, T, D_a, device=self.device)
                alpha = torch.ones(B, device=self.device) * self.cfg.val_cfg_alpha
                
                actions_pred = self.ema.model(noise, images, language, alpha)
                
                # Compute success rate (task-specific metric)
                success = self.compute_success(actions_pred, actions_gt, batch['task_id'])
                
                total_success += success.sum().item()
                total_samples += B
        
        success_rate = total_success / total_samples
        
        wandb.log({
            'val/success_rate': success_rate,
            'val/epoch': self.epoch,
        })
        
        return success_rate
    
    def sample_cfg_alpha(self, B: int) -> torch.Tensor:
        """Sample CFG scale alpha from p(alpha) ∝ alpha^{-k}"""
        # Implementation: inverse CDF sampling
        # (details in full code)
        pass
    
    def sample_from_queue(self, task_ids: torch.Tensor, N: int) -> torch.Tensor:
        """Sample positive examples from queue"""
        # Implementation: sample from per-task queues
        # (details in full code)
        pass
    
    def compute_success(self, actions_pred, actions_gt, task_ids):
        """Compute task-specific success metric"""
        # Implementation: depends on task (e.g., position error < threshold)
        # (details in full code)
        pass
```

### 5.3 Distributed Training Setup

**Multi-Node Training with FSDP:**

```python
"""
scripts/train_drifting_vla.py
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def create_model_fsdp(cfg, device):
    """Create FSDP-wrapped model"""
    # Create model
    model = DriftingVLA(cfg)
    
    # FSDP wrapping policy (wrap layers > 100M params)
    wrap_policy = size_based_auto_wrap_policy(min_num_params=100_000_000)
    
    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device,
        limit_all_gathers=True,
    )
    
    return model

def main():
    # Setup
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Load config
    cfg = load_config('configs/training/large_scale.yaml')
    
    # Create model
    model = create_model_fsdp(cfg, device)
    
    # Create dataloaders (with DistributedSampler)
    train_loader = create_dataloader(cfg, split='train', distributed=True)
    val_loader = create_dataloader(cfg, split='val', distributed=True)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # Create loss
    loss_fn = DriftingLoss(cfg)
    
    # Create trainer
    trainer = DriftingVLATrainer(
        model, train_loader, val_loader,
        optimizer, scheduler, loss_fn, cfg, device
    )
    
    # Training loop
    for epoch in range(cfg.epochs):
        train_metrics = trainer.train_epoch()
        
        if epoch % cfg.val_interval == 0:
            val_metrics = trainer.validate()
        
        if local_rank == 0:  # Only rank 0 saves
            save_checkpoint(model, optimizer, epoch, val_metrics)
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
```

**Launch script:**

```bash
#!/bin/bash
# scripts/launch_distributed.sh

# Configuration
NUM_NODES=16
GPUS_PER_NODE=8
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))  # 128 GPUs

# Master node
MASTER_ADDR="node001"
MASTER_PORT=29500

# Launch on each node
for NODE_RANK in $(seq 0 $((NUM_NODES - 1))); do
    NODE_NAME="node$(printf '%03d' $((NODE_RANK + 1)))"
    
    ssh $NODE_NAME "
        cd /path/to/drifting-vla && \
        torchrun \
            --nnodes=$NUM_NODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            scripts/train_drifting_vla.py \
            --config configs/training/large_scale.yaml
    " &
done

wait
```

---

## 6. Experimental Protocol {#6-experimental-protocol}

### 6.1 Datasets

#### RT-1 Dataset
- **Size:** 130k demonstrations
- **Tasks:** 700 tasks across 13 skills
- **Robots:** 13 robots in office/kitchen environments
- **Observations:** RGB images (640×512), proprioception
- **Actions:** 7-DOF end-effector control + gripper
- **Language:** Natural language task descriptions

#### Bridge V2 Dataset
- **Size:** 60k demonstrations
- **Tasks:** Diverse manipulation in home environments
- **Observations:** RGB-D images, wrist camera
- **Actions:** 7-DOF delta positions + gripper
- **Language:** Free-form task descriptions

#### CALVIN Dataset (for multi-task evaluation)
- **Size:** 24k episodes
- **Tasks:** 34 manipulation tasks in simulated kitchen
- **Observations:** RGB images (200×200), proprioception
- **Actions:** 7-DOF joint velocities + gripper
- **Language:** Task descriptions for long-horizon chains

#### RLBench (for ablations and simulation)
- **Size:** Customizable (we use 5-10k per task)
- **Tasks:** 100+ manipulation tasks
- **Observations:** RGB-D, multiple camera views
- **Actions:** 7-DOF joint positions/velocities + gripper
- **Language:** Templated task descriptions

### 6.2 Evaluation Metrics

#### Primary Metric: Success Rate
```python
def compute_success_rate(predictions, ground_truth, task_type):
    """
    Task-specific success criteria
    
    Examples:
    - Reach: ||pos_final - pos_target|| < 5cm
    - Pick: object lifted > 5cm for > 1s
    - Place: ||pos_final - pos_target|| < 3cm and stable
    """
    success = []
    for pred, gt, task in zip(predictions, ground_truth, task_type):
        if task == 'reach':
            success.append(np.linalg.norm(pred[-1, :3] - gt[-1, :3]) < 0.05)
        elif task == 'pick':
            # Check if object lifted
            success.append(check_object_lifted(pred))
        # ... more task types
    return np.mean(success)
```

#### Secondary Metrics

**1. Action Distribution Quality (MMD):**
```python
def compute_action_mmd(actions_pred, actions_gt, kernel='rbf', bandwidth=1.0):
    """
    Maximum Mean Discrepancy between predicted and ground-truth action distributions
    Lower is better (0 = perfect match)
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Flatten action sequences
    X = actions_pred.reshape(actions_pred.shape[0], -1)  # [N, T*D]
    Y = actions_gt.reshape(actions_gt.shape[0], -1)
    
    # Compute kernels
    XX = rbf_kernel(X, X, gamma=1/(2*bandwidth**2))
    YY = rbf_kernel(Y, Y, gamma=1/(2*bandwidth**2))
    XY = rbf_kernel(X, Y, gamma=1/(2*bandwidth**2))
    
    # MMD^2
    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return np.sqrt(max(mmd_squared, 0))
```

**2. Inference Latency:**
```python
def benchmark_latency(model, device, batch_size=1, num_trials=1000):
    """
    Measure end-to-end inference latency
    """
    model.eval()
    
    # Dummy inputs
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    language = torch.randint(0, 1000, (batch_size, 20), device=device)
    noise = torch.randn(batch_size, 16, 7, device=device)
    alpha = torch.ones(batch_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(noise, images, language, alpha)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_trials):
        _ = model(noise, images, language, alpha)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_latency = (end - start) / num_trials * 1000  # ms
    return avg_latency
```

**3. Multi-Modality Score:**
```python
def compute_multimodality_score(actions_pred, num_modes=5):
    """
    Measure diversity of predicted actions via GMM fitting
    Higher is better (captures more modes)
    """
    from sklearn.mixture import GaussianMixture
    
    # Flatten actions
    X = actions_pred.reshape(actions_pred.shape[0], -1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=num_modes, covariance_type='full')
    gmm.fit(X)
    
    # Compute BIC (lower = better fit, but penalizes complexity)
    bic = gmm.bic(X)
    
    # Compute entropy of mode assignment (higher = more diverse)
    probs = gmm.predict_proba(X)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1).mean()
    
    return {
        'bic': bic,
        'entropy': entropy,
        'num_modes_effective': np.exp(entropy),  # Effective number of modes
    }
```

### 6.3 Baseline Comparisons

#### Diffusion Policy
```python
# Reproduce Diffusion Policy on same data
# Use official implementation: https://github.com/columbia-ai-robotics/diffusion_policy

# Key hyperparameters:
# - Diffusion steps: 100
# - DDPM noise schedule
# - U-Net architecture
# - Same vision/language encoders
```

#### Flow Matching Policy
```python
# Implement Flow Matching baseline
# Use rectified flow formulation

# Key hyperparameters:
# - ODE steps: 50
# - Euler solver
# - Same model architecture
# - Linear interpolation path
```

#### Autoregressive Policy (GPT-style)
```python
# Implement autoregressive baseline
# Discretize actions into tokens

# Key hyperparameters:
# - Vocabulary size: 1024 (k-means on actions)
# - GPT-2 architecture
# - Causal masking
# - Same vision/language encoders
```

### 6.4 Ablation Study Design

**Architecture Ablations:**
```yaml
ablations:
  model_size:
    - name: "B/2"
      params: 133M
      hidden_dim: 768
      depth: 12
    - name: "L/2"
      params: 463M
      hidden_dim: 1024
      depth: 24
    - name: "XL/2"
      params: 1.2B
      hidden_dim: 1536
      depth: 32
  
  vision_encoder:
    - name: "ViT-L/14"
      source: "CLIP"
      frozen: True
    - name: "ConvNeXt-V2-L"
      source: "MAE"
      frozen: False
    - name: "DINOv2-L"
      source: "DINOv2"
      frozen: True
  
  language_encoder:
    - name: "CLIP-text"
      size: 63M
      frozen: True
    - name: "LLaMA-2-7B"
      size: 7B
      frozen: True
    - name: "LLaMA-2-7B-ft"
      size: 7B
      frozen: False
```

**Training Ablations:**
```yaml
ablations:
  batch_allocation:
    - {Nc: 64, Npos: 32, Nneg: 32}
    - {Nc: 128, Npos: 64, Nneg: 64}
    - {Nc: 256, Npos: 128, Nneg: 64}
  
  cfg_schedule:
    - name: "uniform"
      p_alpha: "uniform([1, 4])"
    - name: "alpha^-3"
      p_alpha: "alpha^-3"
    - name: "alpha^-5"
      p_alpha: "alpha^-5"
    - name: "50% no-cfg"
      p_alpha: "50% at alpha=1, 50% alpha^-3"
  
  feature_encoder:
    - pretrain_epochs: [100, 500, 1000]
    - encoder_width: [256, 512, 768, 1024]
    - architecture: ["ResNet", "Transformer", "Hybrid"]
  
  temperatures:
    - [0.02]
    - [0.05]
    - [0.2]
    - [0.02, 0.05, 0.2]
```

### 6.5 Expected Results

**Main Benchmark Results (Target):**

| Benchmark | Metric | Diffusion Policy | Drifting-VLA (ours) | Improvement |
|-----------|--------|-----------------|-------------------|-------------|
| RT-1 (6 tasks) | Success Rate | 83% | **87%** | +4% |
| Bridge V2 | Success Rate | 76% | **79%** | +3% |
| CALVIN | Avg Chain Length | 2.8 | **3.2** | +14% |
| RLBench (10 tasks) | Success Rate | 92% | **94%** | +2% |

**Inference Latency:**

| Method | NFE | Latency (ms) | Throughput (Hz) |
|--------|-----|-------------|----------------|
| Diffusion Policy | 100 | 450 | 2.2 |
| Flow Matching | 50 | 230 | 4.3 |
| **Drifting-VLA** | **1** | **12** | **83** |

**Ablation Results (Expected Trends):**

1. **Model Size:** L/2 outperforms B/2 by ~3-5% success rate
2. **Feature Encoder:** Pre-training 1000 epochs > 100 epochs (+2-3%)
3. **CFG Schedule:** $p(\alpha) \propto \alpha^{-5}$ optimal for success rate
4. **Batch Size:** Larger (Nc=256) better than smaller (Nc=64) (+1-2%)
5. **Multi-Temperature:** Using 3 temperatures better than 1 (+1-2%)

---

## Conclusion

This comprehensive research proposal outlines a complete plan to integrate **Drifting Models** with **Vision-Language-Action (VLA)** policies. The key innovations are:

1. **Mathematical Rigor:** Deep analysis of Drifting vs. Diffusion/Flow/MeanFlow
2. **Architectural Design:** Complete Drifting-VLA architecture with feature-space learning
3. **Theoretical Justification:** Equilibrium analysis, mode coverage, gradient quality
4. **Empirical Validation:** Extensive experiments on RT-1, Bridge, CALVIN benchmarks
5. **Implementation Plan:** Detailed 97-day schedule with risk mitigation
6. **Codebase Design:** Production-ready code architecture for distributed training

With unlimited H100 compute and 97 days, this plan is **feasible and ambitious**, targeting a high-quality NeurIPS 2026 submission that could establish Drifting-VLA as the new state-of-the-art for real-time robotic control.

**Next Steps:**
1. Review this proposal with co-authors
2. Begin Phase 1 (Setup) immediately
3. Iterate on architecture design based on early experiments
4. Maintain flexibility to pivot if unexpected challenges arise

### Current Status

- [x] Phase 1: Codebase, Docker, data pipeline (complete)
- [x] Drifting loss with multi-temperature kernels, feature/drift normalization
- [x] DiT-B/2 model with DINOv2 + CLIP encoders
- [x] RLBench simulation evaluation with CoppeliaSim v4.1
- [x] 10+ WandB visualization panels with step sliders
- [x] Simulation video recording (camera + trajectory overlay)
- [x] Action sanitization (quaternion normalization, gripper binarization)
- [x] Convergence tracking via λ_V and raw_drift_norm
- [ ] Phase 2-6: Scaling, ablations, paper writing