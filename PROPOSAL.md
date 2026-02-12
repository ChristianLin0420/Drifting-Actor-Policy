# Drifting-VLA v2: A Unified One-Step Foundation Model for Multi-Embodiment Robotic Manipulation

**Technical Proposal for NeurIPS 2026 Submission**

---

## Abstract

We propose **Drifting-VLA v2**, a unified vision-language-action foundation model capable of controlling multiple robot embodiments (parallel grippers, bimanual arms, and dexterous hands) through **one-step action generation**. Building upon the recently proposed Drifting model paradigm, which achieves state-of-the-art results in generative modeling (ImageNet FID 1.54) and robotic control, we integrate a frozen Vision-Language Model (VLM) backbone with a Drifting-based action generation head. This architecture enables: (1) **10× faster inference** compared to diffusion-based policies (1 forward pass vs. 100 denoising steps), (2) **unified handling** of heterogeneous embodiments through a fixed-dimensional action space with learned masking, and (3) **superior visual grounding** via joint vision-language reasoning from pre-trained VLMs. We validate our approach on RLBench (parallel gripper), DexGraspNet 2.0 (dexterous hand), and bimanual manipulation benchmarks, demonstrating that training-time distribution evolution via drifting fields provides a scalable alternative to inference-time iterative generation.

**Keywords:** Vision-Language-Action Models, Drifting Models, Multi-Embodiment Manipulation, Foundation Models, One-Step Generation

---

## 1. Introduction

### 1.1 Motivation

Recent advances in robotic manipulation have been driven by two parallel trends: (1) **foundation models** that leverage large-scale pre-training across diverse datasets and embodiments [RT-X, RDT-1B, OpenVLA], and (2) **generative action models** that capture multi-modal action distributions via diffusion or flow-based methods [Diffusion Policy, 3D Diffuser Actor]. However, existing approaches face a fundamental trade-off:

- **Diffusion-based policies** (RDT-1B, Diffusion Policy) achieve strong performance but require 50-100 denoising steps at inference, limiting control frequency to 2-10 Hz.
- **Autoregressive VLAs** (OpenVLA, RT-2) generate actions in one pass but struggle with multi-modal distributions and show poor generalization on manipulation benchmarks (4.8% success on ManiSkill vs. RDT's 53.6%).
- **Existing foundation models** (RDT-1B) support only parallel grippers and bimanual arms, lacking dexterous hand capabilities critical for complex manipulation.

The recently proposed **Drifting model** [arxiv:2602.04770] offers a compelling alternative: by evolving the generator's output distribution toward the data distribution during training (via a kernelized drifting field), the model learns to generate high-quality samples in **one forward pass** at inference. This paradigm has achieved state-of-the-art results in image generation (FID 1.54 on ImageNet) and robotic control tasks, demonstrating 100× fewer function evaluations than diffusion models while maintaining comparable performance.

### 1.2 Research Questions

This proposal addresses the following questions:

1. **Can drifting models scale to multi-embodiment manipulation?** The drifting paper demonstrates single embodiments (parallel gripper) on single tasks. We investigate whether a unified drifting architecture can handle gripper, bimanual, and dexterous hand control simultaneously across dozens of tasks.

2. **Do VLM backbones improve manipulation performance?** We hypothesize that joint vision-language reasoning from pre-trained VLMs (PaliGemma2, Qwen3-VL) provides better scene understanding than separate vision (DINOv2/SigLIP) and language (CLIP/T5) encoders used in prior work.

3. **Can drifting models work at smaller batch sizes?** The drifting paper uses batch size 8192 (128 GPUs). We investigate modifications (negative sample queue, hybrid loss) that enable effective training with batch 512-2048 (1-8 GPUs), making drifting accessible without massive compute.

4. **How do drifting models compare to diffusion baselines?** We provide the first comprehensive comparison of drifting vs diffusion (RDT-1B, Diffusion Policy) on robotic benchmarks, measuring both success rate and inference latency across multiple embodiments.

### 1.3 Contributions

We extend the Drifting model paradigm from image generation to multi-embodiment robotic manipulation with the following novel contributions:

1. **VLM-Conditioned Drifting Architecture:** We are the first to integrate frozen Vision-Language Models (PaliGemma2-3B) as the perception backbone for drifting-based action generation. This two-system design (frozen VLM + trainable Drifting DiT) enables parameter-efficient scaling across embodiments while leveraging pre-trained visual-language understanding.

2. **Unified Action Space for Multi-Embodiment Control:** Extending RDT-1B's approach, we design a fixed 128-dimensional action representation with learned masking that unifies gripper (8-dim), bimanual (16-dim), and dexterous hand (23-dim) control in a single model — the first drifting-based model to support dexterous hands.

3. **Small-Batch Drifting via Hybrid Loss:** We propose two algorithmic modifications that enable drifting models to train effectively with batch sizes 512-2048 (vs. the paper's 8192):
   - **Negative Sample Queue:** Store past predictions to ensure negatives differ from current queries
   - **Hybrid MSE + Drifting Loss:** Add direct supervision (MSE) to stabilize learning when drifting field estimation is noisy

   These modifications are ablated in Phase 5 to determine if they remain beneficial at large batch (8×B200 GPUs).

4. **Multi-Dataset Pre-training Strategy:** We demonstrate training across 5 heterogeneous datasets (RLBench, DexGraspNet 2.0, Bridge, ALOHA, RT-1) with different embodiments, action representations (absolute vs delta), and image formats, using per-dataset normalization and weighted sampling.

5. **Comprehensive Empirical Validation:** We evaluate on RLBench (18 tasks, gripper), DexGraspNet 2.0 (dexterous grasping), and ManiSkill (bimanual), providing the first comprehensive benchmark of drifting models for robotic manipulation across multiple embodiments.

6. **Open-Source Implementation:** Production-ready PyTorch codebase with Docker containers (CoppeliaSim, IsaacGym), WandB logging with 10+ visualization panels, simulation evaluation pipelines, and detailed documentation for reproducibility.

---

## 2. Related Work

### 2.1 Vision-Language-Action Models

**Autoregressive VLAs** (RT-1, RT-2, OpenVLA) treat robot control as a sequence prediction problem, tokenizing actions and generating them autoregressively conditioned on vision and language. While these models benefit from pre-trained LLM backbones, they struggle with continuous action spaces and multi-modal distributions. OpenVLA achieves only 4.8% average success on ManiSkill benchmarks despite 7B parameters and training on 970K demonstrations.

**Diffusion-based VLAs** (RDT-1B, Diffusion Policy) model actions as samples from a learned distribution via iterative denoising. RDT-1B, a 1.2B parameter diffusion transformer pre-trained on 46 datasets, achieves 53.6% on ManiSkill but requires 100 denoising steps (~500ms on H100), limiting real-time control. Our work aims to match RDT's performance with 1-step generation (~50ms).

### 2.2 Drifting Models

The Drifting model [arxiv:2602.04770] introduces a training-time distribution evolution paradigm where the generator's output distribution $q_\theta = f_\theta \# p_{\text{prior}}$ is iteratively evolved toward the data distribution $p_{\text{data}}$ via a drifting field.

**Drifting Field Definition (Paper Eq. 5-8):**

The drifting field is defined as a weighted combination of sample differences:

$$V_{p,q}(x) = \mathbb{E}_{y^+ \sim p}\mathbb{E}_{y^- \sim q}\Big[\tilde{k}(x, y^+)\tilde{k}(x, y^-)(y^+ - y^-)\Big]$$

where $\tilde{k}(x, y)$ is a normalized kernel with double softmax normalization over both positive and negative samples jointly. This can be decomposed as:

$$V_{p,q}(x) = V_p^+(x) - V_q^-(x)$$

where the cross-normalized attraction and repulsion terms are:

$$V_p^+(x) = \mathbb{E}_{y^+}[\tilde{k}(x, y^+) y^+] \cdot \mathbb{E}_{y^-}[\tilde{k}(x, y^-)]$$

$$V_q^-(x) = \mathbb{E}_{y^-}[\tilde{k}(x, y^-) y^-] \cdot \mathbb{E}_{y^+}[\tilde{k}(x, y^+)]$$

Note the critical detail: each term is weighted by the *sum* of the other distribution's kernel weights.

**Training Objective (Paper Eq. 4):**

$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{\epsilon \sim p_\epsilon}[\|f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q}(f_\theta(\epsilon)))\|^2]$$

With drift normalization (Paper Eq. 23-25), the field is scaled such that $\mathbb{E}[\|V\|^2/D] = 1$, which makes the **loss ≈ D** (constant). This is by design — the loss value itself is not the convergence metric. Instead, convergence is measured by task-specific metrics (FID for images, success rate for robotics).

**Key Insight:** The gradient of the loss is:

$$\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f_\theta(\epsilon)$$

The stop-gradient on the target means the gradient pushes $f_\theta(\epsilon)$ in the direction of the drifting field $V$. Even though the loss magnitude stays constant (~D), the *direction* of $V$ changes during training, providing useful gradient signal that evolves the distribution toward equilibrium.

**Equilibrium Condition:** When $p = q$ (the pushforward distribution matches data), the drifting field becomes zero: $V_{p,p}(x) = 0$ for all $x$, by anti-symmetry.

**Empirical Results:** Prior work demonstrated FID 1.54 on ImageNet 256×256 (latent-space generation) and FID 1.61 (pixel-space), achieving state-of-the-art among one-step methods. For robotics, the paper shows competitive performance on manipulation tasks (ToolHang, BlockPush) with 1-NFE vs Diffusion Policy's 100-NFE.

### 2.3 Multi-Embodiment Manipulation

**RDT-1B** addresses multi-embodiment control via a fixed 128-dimensional action space where different embodiments occupy different slots (with zero-padding for unused dimensions). This design enables a single DiT backbone to handle grippers and bimanual arms. However, RDT lacks dexterous hand support and relies on 100-step diffusion.

**DexGraspNet 2.0** tackles dexterous grasping in cluttered scenes using point-cloud-based graspness prediction and diffusion-based grasp pose generation. The model outputs wrist pose (7-dim) + joint angles (16-dim) for Allegro/LEAP hands, achieving strong grasping performance in simulation. However, it is specialized for grasping and does not generalize to full manipulation tasks.

Our work combines the best of both: RDT's unified action space design with Drifting's one-step generation, extended to dexterous hands via integration with DexGraspNet 2.0 data.

---

## 3. Method

### 3.0 Relationship to Original Drifting Model

**What we preserve from [arxiv:2602.04770]:**
- ✅ Core drifting field formulation: $V_{p,q}(x) = \mathbb{E}[\tilde{k}(x,y^+)\tilde{k}(x,y^-)(y^+-y^-)]$
- ✅ Double softmax normalized kernels
- ✅ Multi-temperature kernels ($\tau \in \{0.02, 0.05, 0.2\}$)
- ✅ Feature normalization (scale to $\sqrt{D}$ average distance)
- ✅ Drift normalization (scale to $\mathbb{E}[\|V\|^2/D] = 1$)
- ✅ Training objective: $\mathcal{L} = \|x - \text{sg}(x + V)\|^2$
- ✅ One-step inference paradigm
- ✅ DiT architecture with adaLN-Zero
- ✅ EMA for evaluation

**What we extend/modify:**
- 🔧 **VLM conditioning:** Paper uses class labels; we use PaliGemma2 VLM features for vision-language tasks
- 🔧 **Multi-embodiment:** Paper shows single embodiment; we extend to gripper/bimanual/dex hand
- 🔧 **Negative sampling:** Paper uses x.detach() with batch 8192; we add neg_queue for batch 512-2048
- 🔧 **Hybrid loss:** Paper uses pure drifting; we add MSE component for small-batch stability (ablate in Phase 5)
- 🔧 **Action space:** Paper uses 7-10 dim actions; we use 128-dim unified space with masking
- 🔧 **Multi-dataset training:** Paper trains per-task; we jointly train across 5 datasets

**Validation Strategy:** In Phase 1 (toy case), we first reproduce the paper's pure drifting loss on a single task to verify our implementation is correct. Then in Phase 5 (ablations), we determine which modifications are necessary vs optional.

### 3.1 Problem Formulation

We formulate multi-embodiment manipulation as learning a conditional distribution $p_\theta(\mathbf{a} | \mathcal{I}, \mathcal{L}, e)$ where:
- $\mathcal{I} \in \mathbb{R}^{V \times 3 \times H \times W}$: Multi-view RGB images
- $\mathcal{L}$: Natural language task description
- $e \in \{0, 1, 2\}$: Embodiment type (gripper, bimanual, dexterous hand)
- $\mathbf{a} \in \mathbb{R}^{T \times 128}$: Action sequence (T timesteps, 128-dim fixed space)

The model must satisfy:
1. **One-step generation:** $\mathbf{a} = f_\theta(\epsilon, \mathcal{I}, \mathcal{L}, e)$ where $\epsilon \sim \mathcal{N}(0, I)$
2. **Embodiment-specific constraints:** Active action dimensions determined by $e$
3. **Multi-modality:** Capture diverse valid action modes (e.g., grasp from left or right)

### 3.2 Architecture Overview

#### 3.2.0 Visual Architecture Diagrams

**Figure 1: Complete System Architecture**

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         DRIFTING-VLA v2 ARCHITECTURE                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                          SYSTEM 1: VLM BACKBONE                          │  │
│  │                        (Frozen + Optional LoRA)                          │  │
│  │                                                                           │  │
│  │   ┌──────────┐    ┌──────────┐                                          │  │
│  │   │ Image 1  │    │ Image 2  │   "close the red jar"                    │  │
│  │   │ (front)  │    │ (wrist)  │          ↓                                │  │
│  │   │ 448×448  │    │ 448×448  │    ┌──────────┐                          │  │
│  │   └────┬─────┘    └────┬─────┘    │Tokenizer │                          │  │
│  │        │               │           └────┬─────┘                          │  │
│  │        └───────┬───────┘                │                                │  │
│  │                ↓                         ↓                                │  │
│  │   ┌──────────────────────────────────────────────────┐                  │  │
│  │   │         PaliGemma2-3B (3.0B params)               │                  │  │
│  │   │                                                    │                  │  │
│  │   │   SigLIP-So400m        Gemma-2B                  │                  │  │
│  │   │   Vision Encoder       Language Model            │                  │  │
│  │   │   (400M params)        (2.0B params)             │                  │  │
│  │   │         ↓                    ↓                    │                  │  │
│  │   │   ┌─────────────────────────────────┐            │                  │  │
│  │   │   │  Joint Vision-Language Fusion   │            │                  │  │
│  │   │   │  (Transformer Layers)            │            │                  │  │
│  │   │   └─────────────┬───────────────────┘            │                  │  │
│  │   │                 ↓                                  │                  │  │
│  │   │         Hidden States h_VLM                       │                  │  │
│  │   │         [L, 2048] where L ≈ 1025 + L_lang        │                  │  │
│  │   └──────────────────────┬────────────────────────────┘                  │  │
│  │                          ↓                                                │  │
│  │              ┌───────────────────────┐                                   │  │
│  │              │   Feature Projector   │                                   │  │
│  │              │   Linear: 2048 → 768  │                                   │  │
│  │              │   LayerNorm           │                                   │  │
│  │              └───────────┬───────────┘                                   │  │
│  │                          ↓                                                │  │
│  │              c_seq: [L, 768]   c_pool: [768]                            │  │
│  └──────────────────────────┼──────────────────────────────────────────────┘  │
│                             ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    SYSTEM 2: DRIFTING DiT ACTION HEAD                    │  │
│  │                         (Trainable: 123M params)                          │  │
│  │                                                                           │  │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │  │
│  │   │   Noise ε    │     │ Embodiment   │     │  CFG Scale   │           │  │
│  │   │ [T,64]       │     │   ID (0/1/2) │     │   α ~ p(α)   │           │  │
│  │   │ N(0,I)       │     │              │     │              │           │  │
│  │   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘           │  │
│  │          ↓                     ↓                     ↓                    │  │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │  │
│  │   │Noise→Token  │     │  Embedding   │     │  MLP: 1→768  │           │  │
│  │   │MLP: 64→768  │     │  [3, 768]    │     │              │           │  │
│  │   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘           │  │
│  │          ↓                     └────────┬────────────┘                   │  │
│  │   [T, 768] + Pos_Embed                 ↓                                │  │
│  │       noise_tokens             c_global = c_pool + e_emb + cfg_emb     │  │
│  │          ↓                               ↓                               │  │
│  │   ┌──────────────────────────────────────────────────┐                 │  │
│  │   │     Cross-Attention Layers (×2)                   │                 │  │
│  │   │                                                    │                 │  │
│  │   │     Q = noise_tokens                              │                 │  │
│  │   │     K, V = c_seq (VLM features)                  │                 │  │
│  │   │                                                    │                 │  │
│  │   │     Attn(Q,K,V) allows noise to attend to        │                 │  │
│  │   │     fine-grained visual-language features        │                 │  │
│  │   └──────────────────┬───────────────────────────────┘                 │  │
│  │                      ↓                                                   │  │
│  │           fused_tokens: [T, 768]                                        │  │
│  │                      ↓                                                   │  │
│  │   ┌─────────────────────────────────────────────────────────┐          │  │
│  │   │           DiT Transformer Blocks (×12)                   │          │  │
│  │   │                                                           │          │  │
│  │   │   Each block:                                            │          │  │
│  │   │   ┌──────────────────────────────────────┐              │          │  │
│  │   │   │ 1. adaLN-Zero Normalization          │              │          │  │
│  │   │   │    (conditioned on c_global)         │              │          │  │
│  │   │   │ 2. Self-Attention (RoPE + QK-Norm)   │              │          │  │
│  │   │   │    with Flash Attention (bf16)       │              │          │  │
│  │   │   │ 3. adaLN-Zero Normalization          │              │          │  │
│  │   │   │ 4. SwiGLU MLP (ratio 4.0)            │              │          │  │
│  │   │   └──────────────────────────────────────┘              │          │  │
│  │   │                                                           │          │  │
│  │   │   Hidden: 768-dim, Heads: 12, Params: ~9M per block    │          │  │
│  │   └───────────────────────┬─────────────────────────────────┘          │  │
│  │                           ↓                                              │  │
│  │              transformer_out: [T, 768]                                  │  │
│  │                           ↓                                              │  │
│  │   ┌───────────────────────────────────────────────────┐                │  │
│  │   │            Unified Action Head                     │                │  │
│  │   │                                                     │                │  │
│  │   │   MLP: 768 → 768 → 128                            │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   a_raw: [T, 128] (all dims)                      │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   Apply action_mask (zero inactive dims)          │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   Normalize quaternions (dims 3-6, etc.)          │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   a_final: [T, 128]                               │                │  │
│  │   │                                                     │                │  │
│  │   │   Active dims vary by embodiment:                 │                │  │
│  │   │   • Gripper: dims 0-7 (8 active)                  │                │  │
│  │   │   • Bimanual: dims 0-15 (16 active)               │                │  │
│  │   │   • Dex Hand: dims 0-6, 16-31 (23 active)         │                │  │
│  │   └───────────────────────────────────────────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Total Trainable Parameters: 123.6M                                            │
│  GPU Memory (bf16): ~250 MB (model) + batch activations                       │
│  Inference Latency: ~50ms on B200 (1 VLM forward + 1 DiT forward)             │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Figure 2: Training Pipeline Flow**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                  │
│                                                                               │
│  OFFLINE (One-time pre-processing):                                          │
│  ┌─────────────┐                                                             │
│  │  Raw Data   │  →  Resize to 448×448  →  Tokenize lang  →  [images, text]│
│  │ (5 datasets)│                                                             │
│  └──────┬──────┘                                                             │
│         ↓                                                                     │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │  VLM Feature Extraction (PaliGemma2-3B)               │                   │
│  │  • Process each sample once                           │                   │
│  │  • Extract hidden_states[-1]: [L, 2048]              │                   │
│  │  • Save to HDF5 (fp16 compressed)                     │                   │
│  │  • Time: ~2 hours for 1M samples on 8 GPUs           │                   │
│  │  • Storage: ~4 GB total                               │                   │
│  └──────────────────────┬───────────────────────────────┘                   │
│                         ↓                                                     │
│         [sample_id] → vlm_features.h5                                        │
│                                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                               │
│  ONLINE (Each training step):                                                │
│  ┌──────────────┐                                                            │
│  │ Sample batch │ ← Weighted sampling from [RLBench, DexGrasp, Bridge, ...]│
│  │ B = 128      │                                                            │
│  └──────┬───────┘                                                            │
│         ↓                                                                     │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │ Load from disk:                                      │                    │
│  │ • vlm_features[sample_ids]  → [B, L, 2048] fp16    │                    │
│  │ • actions[sample_ids]       → [B, T, 128] fp32     │                    │
│  │ • action_mask[embodiment]   → [128] bool            │                    │
│  │ • Conversion to GPU takes ~5ms                       │                    │
│  └─────────────────────┬───────────────────────────────┘                    │
│                        ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Project VLM features                                     │               │
│  │  • Linear: 2048 → 768                                    │               │
│  │  • c_seq: [B, L, 768], c_pool: [B, 768]                │               │
│  └────────────────────────┬─────────────────────────────────┘               │
│                           ↓                                                   │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │  Model Forward                                              │             │
│  │  • Sample noise ε ~ N(0, I_{T×64})                         │             │
│  │  • Sample CFG α ~ p(α) ∝ α^{-3}                           │             │
│  │  • actions_pred = model(ε, c_seq, c_pool, e, α)          │             │
│  │  • Time: ~30ms on 8×B200 (FSDP parallel)                  │             │
│  └────────────────────────┬───────────────────────────────────┘             │
│                           ↓                                                   │
│  actions_pred: [B, T, 128]                                                   │
│         ↓                              ↓                                      │
│  Normalize actions_gt          Sample from queues                           │
│  using dataset stats            • pos_queue.sample(128)                     │
│         ↓                        • neg_queue.sample(128)                     │
│         ↓                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Compute Hybrid Loss                                     │                │
│  │  • MSE: ||pred - gt||² on active dims only             │                │
│  │  • Drifting: ||pred - (pred + V)||² where              │                │
│  │    V = attract(pos) - repel(neg)                        │                │
│  │  • L = MSE + 0.1 × Drifting                             │                │
│  └────────────────────┬────────────────────────────────────┘                │
│                       ↓                                                       │
│  L.backward() → Grad clip → Optimizer step → EMA update                     │
│         ↓                                                                     │
│  Update queues:                                                              │
│  • neg_queue.push(actions_pred.detach())  ← Store for next step            │
│  │ pos_queue.push(actions_gt)                                               │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Figure 3: Inference Pipeline Flow**

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE (Real-time)                           │
│                                                                         │
│  Camera Observations                 Task Instruction                  │
│  ┌────────┐  ┌────────┐             "grasp the red cup"               │
│  │ front  │  │ wrist  │                     ↓                          │
│  │448×448 │  │448×448 │                Tokenize                       │
│  └───┬────┘  └───┬────┘                     ↓                          │
│      └────────┬───┘                    [token_ids]                     │
│               ↓                              ↓                          │
│  ┌────────────────────────────────────────────────────┐                │
│  │         VLM Forward (Live, not pre-computed)        │                │
│  │         PaliGemma2-3B on GPU                        │                │
│  │         Time: ~40ms                                  │                │
│  └────────────────────┬───────────────────────────────┘                │
│                       ↓                                                 │
│          hidden_states: [L, 2048]                                      │
│                       ↓                                                 │
│  ┌────────────────────────────────┐                                    │
│  │  Project: 2048 → 768            │                                    │
│  └────────────────┬───────────────┘                                    │
│                   ↓                                                     │
│       c_seq: [L, 768], c_pool: [768]                                  │
│                   ↓                                                     │
│  ┌─────────────────────────────────────────────┐                       │
│  │  DiT Forward (Single Pass)                   │                       │
│  │  • Sample ε ~ N(0, I)                       │                       │
│  │  • Add embodiment_id, cfg_scale=2.0         │                       │
│  │  • Cross-attend to c_seq                    │                       │
│  │  • 12 DiT blocks                             │                       │
│  │  • Action head: 768 → 128                   │                       │
│  │  Time: ~10ms on B200                        │                       │
│  └────────────────┬────────────────────────────┘                       │
│                   ↓                                                     │
│       actions_pred: [T, 128] (normalized space)                        │
│                   ↓                                                     │
│  ┌───────────────────────────────────┐                                 │
│  │  Denormalize using dataset stats   │                                 │
│  │  a = a_norm * std + mean           │                                 │
│  └─────────────────┬─────────────────┘                                 │
│                    ↓                                                    │
│       actions: [T, 128] (absolute world coordinates)                   │
│                    ↓                                                    │
│  ┌────────────────────────────────────────┐                            │
│  │  Extract embodiment-specific action     │                            │
│  │  • Gripper: dims[0:8]                  │                            │
│  │  • Bimanual: dims[0:16]                │                            │
│  │  • Dex hand: dims[0:7, 16:32]          │                            │
│  └────────────────┬───────────────────────┘                            │
│                   ↓                                                     │
│       action_t: [8] or [16] or [23]                                   │
│                   ↓                                                     │
│  ┌────────────────────────────────┐                                    │
│  │  Execute in Robot/Simulator     │                                    │
│  │  • RLBench: CoppeliaSim         │                                    │
│  │  • DexGrasp: IsaacGym           │                                    │
│  │  • Real: Physical robot         │                                    │
│  └────────────────────────────────┘                                    │
│                                                                         │
│  Total Latency: 40ms (VLM) + 10ms (DiT) = 50ms                        │
│  Control Frequency: 20 Hz (real-time capable)                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Figure 4: Data Processing Flow**

```
Raw Dataset → Standardization → Unified Format → VLM Features → Training

┌──────────────┐
│  RLBench     │  front_rgb [128×128] ──┐
│  close_jar   │  wrist_rgb [128×128] ──┼→ Resize → [2,3,448,448]
│  Episode 42  │  actions [T,8] abs     │           ↓
│  Timestep 5  │  language: "close jar" │    ┌──────────────┐
└──────────────┘                         │    │ Unified      │
                                         │    │ Sample       │
┌──────────────┐                         │    │              │
│ DexGraspNet  │  point_cloud [N,3] ────┼→ Render → [4,3,448,448]
│ Scene 1024   │  grasp_pose [23] abs   │    │ images: ✓    │
│ Grasp 7      │  (wrist + 16 joints)   │    │ actions: ✓   │
└──────────────┘                         │    │ language: ✓  │
                                         │    └──────┬───────┘
┌──────────────┐                         │           ↓
│  Bridge V2   │  wrist_img [256×256] ──┼→ Resize    Sample
│  Task 89     │  actions [T,7] DELTA   │    +       ↓
│  Episode 12  │  Integrate to absolute │    ┌──────────────────┐
│  Timestep 3  │  language: "pick cup"  │    │  PaliGemma2-3B   │
└──────────────┘                         │    │  (one forward)   │
                                         │    │  40ms on GPU     │
┌──────────────┐                         │    └──────┬───────────┘
│  ALOHA       │  4 cameras [480×640] ──┼→ Resize           ↓
│  Bimanual    │  actions [T,16] joint  │    vlm_features: [L,2048]
│  Task: fold  │  language: "fold ..."  │           ↓
└──────────────┘                         │    Save to HDF5
                                         │           ↓
All map to:                              │    ┌──────────────────┐
• Images: [V, 3, 448, 448]              │    │ Disk Storage     │
• Actions: [T, 128] (padded)            │    │ • images.h5      │
• Mask: [128] (active dims)             │    │ • actions.h5     │
• Embodiment: {0,1,2}                   │    │ • vlm_features.h5│
└─────────────────────────────────────────────┴──────────────────┘

During Training:
Load [images.h5, actions.h5, vlm_features.h5] → Batch → GPU → Model
```

### 3.2 Detailed Architecture Components

#### 3.2.1 System 1: VLM Backbone (Frozen + LoRA)

We leverage pre-trained Vision-Language Models to extract rich multimodal features that encode both visual scene understanding and language instruction grounding in a unified representation.

**Supported VLM Backbones:**

The codebase implements a unified `VLMBackbone` interface supporting both models:

| | `google/paligemma2-3b-mix-448` | `Qwen/Qwen3-VL-2B-Instruct` |
|---|---|---|
| **Architecture** | SigLIP-So400m + Gemma-2B | ViT + Qwen2.5-2B |
| **Total params** | 3.0B | 2.3B |
| **Vision encoder** | ViT-So400m/14 (400M) | ViT-600M with dynamic resolution |
| **Patch size** | 14×14 | 14×14 |
| **Image resolution** | 448×448 (fixed) | Dynamic (min 224, max 1280) |
| **Visual tokens per image** | 1025 (fixed) | Variable (~256-1024) |
| **Language model** | Gemma-2B | Qwen2.5-2B |
| **Hidden dimension** | 2048 | 1536 |
| **Instruction following** | Good | Excellent (instruct-tuned) |
| **Visual grounding** | Excellent | Good |
| **Multi-image** | Single image per forward | Native multi-image support |

**Default:** PaliGemma2-3B (best visual grounding for manipulation).

**Qwen3-VL-2B advantages:** Smaller (2.3B vs 3B), native multi-image support (no need for per-view independent forward passes), dynamic resolution (can process different-sized inputs efficiently).

**Implementation:**

```python
class VLMBackbone(nn.Module):
    """Unified VLM backbone supporting PaliGemma2 and Qwen3-VL."""
    
    SUPPORTED_MODELS = {
        'paligemma2': {
            'name': 'google/paligemma2-3b-mix-448',
            'hidden_dim': 2048,
            'image_size': 448,
            'multi_image_native': False,
        },
        'qwen3vl': {
            'name': 'Qwen/Qwen3-VL-2B-Instruct',
            'hidden_dim': 1536,
            'image_size': 448,  # Use 448 for consistency
            'multi_image_native': True,
        },
    }
    
    def __init__(self, model_key='paligemma2', dit_hidden_dim=768):
        super().__init__()
        cfg = self.SUPPORTED_MODELS[model_key]
        self.model_key = model_key
        self.vlm = AutoModelForCausalLM.from_pretrained(
            cfg['name'], torch_dtype=torch.bfloat16
        )
        self.vlm.eval()
        
        # Projection adapts to VLM's hidden dim
        self.proj_seq = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], dit_hidden_dim),
            nn.LayerNorm(dit_hidden_dim),
        )
        self.proj_pool = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], dit_hidden_dim),
            nn.LayerNorm(dit_hidden_dim),
        )
    
    def forward(self, images, language_tokens, views_per_sample=None):
        """
        Args:
            images: [B, V, 3, 448, 448] multi-view images
            language_tokens: input_ids [B, L_lang]
            views_per_sample: [B] int tensor, number of views per sample
        Returns:
            c_seq: [B, L_total, dit_hidden_dim]
            c_pool: [B, dit_hidden_dim]
        """
        if self.model_key == 'qwen3vl':
            # Qwen3-VL: native multi-image in one forward
            return self._forward_qwen3vl(images, language_tokens, views_per_sample)
        else:
            # PaliGemma2: vision per view, language appended once
            return self._forward_paligemma2(images, language_tokens)
    
    def _forward_paligemma2(self, images, language_tokens):
        """
        PaliGemma2 multi-view processing:
        1. Run SigLIP vision encoder on each view independently
        2. Concatenate visual tokens from all views
        3. Append language tokens ONCE (not per view)
        4. Run through Gemma-2B language model for joint VL fusion
        """
        B, V, C, H, W = images.shape
        
        # Step 1: Extract visual tokens per view (SigLIP only)
        images_flat = images.view(B * V, C, H, W)
        with torch.no_grad():
            vision_outputs = self.vlm.vision_model(images_flat)
            visual_tokens = vision_outputs.last_hidden_state  # [B*V, 1025, 2048]
        
        # Step 2: Reshape and concatenate across views
        N_vis = visual_tokens.shape[1]  # 1025 per view
        visual_tokens = visual_tokens.view(B, V * N_vis, -1)  # [B, V*1025, 2048]
        
        # Step 3: Append language tokens (encoded once, not per view)
        lang_embeds = self.vlm.get_input_embeddings()(language_tokens)  # [B, L_lang, 2048]
        combined = torch.cat([visual_tokens, lang_embeds], dim=1)  # [B, V*1025+L_lang, 2048]
        
        # Step 4: Joint VL fusion through Gemma-2B layers
        h = self.vlm.language_model(inputs_embeds=combined).last_hidden_state
        
        # Project to DiT dim
        c_seq = self.proj_seq(h)
        c_pool = self.proj_pool(h.mean(dim=1))
        return c_seq, c_pool
```

Switching backbone at config level:

```yaml
# configs/model/drifting_v2.yaml
vlm:
  model_key: 'paligemma2'   # or 'qwen3vl'
```

**Multi-View Processing:**

For multi-view inputs $\mathcal{I} = \{I_1, ..., I_V\}$ where $V$ is the number of camera views:

```python
# Option A: Process each view independently, concatenate features
features = []
for view in images:  # images: [V, 3, 448, 448]
    h_v = vlm(pixel_values=view, input_ids=lang_tokens)  # [1025+L_lang, 2048]
    features.append(h_v)
features = torch.cat(features, dim=0)  # [V×(1025+L_lang), 2048]

# Option B: Tile images into 2×2 grid (for V=4 views)
# → Single VLM forward pass, spatial relationships preserved
grid = tile_images(images, layout=(2,2))  # [896, 896, 3] → rescale to 448×448
features = vlm(pixel_values=grid, input_ids=lang_tokens)  # [1025+L_lang, 2048]
```

**Multi-View Processing for PaliGemma2 (Vision-only per view, language appended once):**

Each camera view is processed through PaliGemma2's **vision encoder only** (SigLIP) independently. The visual tokens from all views are concatenated, then the **language tokens are appended once** at the end. This avoids redundantly encoding the same instruction V times:

```
View 1 (front)  ──→ SigLIP Vision Encoder ──→ v1: [1025, 2048]  (visual tokens)
View 2 (wrist)  ──→ SigLIP Vision Encoder ──→ v2: [1025, 2048]  (visual tokens)
                                                     ↓
                                    torch.cat([v1, v2], dim=0)
                                                     ↓
                                v_concat: [2050, 2048]  (all visual tokens)
                                                     ↓
                   Append language tokens: [v_concat ; lang_tokens]
                                                     ↓
                   h_all: [2050 + L_lang, 2048]  (joint VL representation)
                                                     ↓
                   Pass through Gemma-2B language model layers
                   (cross-attention between visual + language tokens)
                                                     ↓
                   h_final: [2050 + L_lang, 2048]  (fused multi-view + language)
```

**Why vision-only per view, language once:**
- SigLIP vision encoder processes each image independently (no cross-image attention)
- Language instruction is the SAME for all views — encoding it V times wastes compute
- Appending language after visual concatenation lets Gemma-2B jointly attend to ALL views + instruction
- The language model layers perform cross-attention between multi-view visual tokens and language tokens

**Why independent vision processing (not image tiling):**
- PaliGemma2's SigLIP was pre-trained on single images — tiling would break spatial attention patterns
- Each view has independent spatial structure (front = global scene, wrist = local detail)
- Independent processing preserves per-view spatial features
- The subsequent Gemma-2B layers learn to fuse multi-view information with language

**Handling Variable Number of Views:**

Datasets provide 1-4 cameras. Within a batch, samples may have different view counts:

```python
# During pre-computation: process each view independently
view_features = []
for v in range(V):  # V varies: RLBench=2, Bridge=1, ALOHA=3
    h_v = vlm(images[v], lang_tokens)   # [1025+L_lang, 2048]
    view_features.append(h_v)

h_concat = torch.cat(view_features, dim=0)  # [V×L_per_view, 2048]

# During training: pad to MAX_TOKENS and use attention mask
MAX_VLM_TOKENS = 4 * 1060  # Supports up to 4 views
h_padded = pad_to_length(h_concat, MAX_VLM_TOKENS)     # [MAX_VLM_TOKENS, 2048]
attn_mask = create_padding_mask(actual_length=V*L_per_view, max_length=MAX_VLM_TOKENS)
```

The cross-attention in the DiT naturally handles variable-length key/value sequences via the attention mask.

**Projection to DiT dimension:**

$$\mathbf{c}_{\text{seq}} = \text{LayerNorm}(\text{Linear}_{2048 \to 768}(\mathbf{h}_{\text{concat}})) \in \mathbb{R}^{L_{\text{total}} \times 768}$$

$$\mathbf{c}_{\text{pool}} = \text{LayerNorm}(\text{Linear}_{2048 \to 768}(\text{mean}(\mathbf{h}_{\text{concat}}, \text{dim}=0))) \in \mathbb{R}^{768}$$

**VLM Co-Training Strategy (Staged):**

We adopt a **staged training recipe** rather than keeping the VLM frozen throughout:

```
Stage 1 (Steps 0 → 5K):    VLM frozen, DiT trains from scratch
                             → Prevents random DiT gradients from corrupting VLM

Stage 2 (Steps 5K → 50K):  VLM LoRA unfreezes, co-trains with DiT
                             → VLM adapts to robot domain

Stage 3 (Steps 50K → 100K): Continue co-training, lower VLM LR by 2×
                             → Fine refinement
```

**Differential learning rates:**

```python
optimizer = AdamW([
    {'params': vlm_lora.parameters(), 'lr': 1e-5},      # VLM LoRA: 10× lower
    {'params': feature_proj.parameters(), 'lr': 4e-4},    # Projector: standard
    {'params': dit.parameters(), 'lr': 4e-4},              # DiT: standard
    {'params': action_head.parameters(), 'lr': 4e-4},      # Action head: standard
])
```

| Component | LR | Trainable Params | Stage Active |
|-----------|-----|-----------------|-------------|
| VLM base | 0 (frozen) | 0 | Always |
| VLM LoRA | 1e-5 | ~0.5M | Stage 2+ |
| Feature Projector | 4e-4 | 3.2M | Always |
| DiT + Action Head | 4e-4 | 123M | Always |

**LoRA Configuration:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    layers_to_transform=[-4, -3, -2, -1],  # Last 4 layers only
)

vlm = get_peft_model(vlm, lora_config)
# Trainable: 16 × 2 × 4 × 2048 ≈ 0.5M (0.017% of 3B)
```

**Training Phases and VLM Usage:**

| Phase | VLM Usage | Purpose |
|-------|-----------|---------|
| Phase 0 (Pre-compute) | Run once offline, save features to HDF5 | Fast iteration on DiT |
| Phase 1-2 (Toy/Multi-emb) | Load pre-computed features, VLM not in GPU | Validate architecture |
| Phase 3+ (Scale-up) | Load VLM live with LoRA, co-train | Adapt VLM to robot domain |
| Inference | Run VLM live (~40ms per observation) | Real-time control |

#### 3.2.2 System 2: Drifting DiT Action Generator

The action generation module is a Transformer-based architecture that processes random noise conditioned on VLM features to produce action sequences in a single forward pass.

**Architecture Overview:**

| Layer Type | Count | Hidden Dim | Num Heads | Parameters |
|------------|-------|------------|-----------|------------|
| Noise Tokenizer | 1 MLP (3-layer) | 64→768→768 | — | ~1.2M |
| Timestep Embedding | Learned | 768 | — | 16×768 = 12K |
| Embodiment Embedding | Learned | 768 | — | 3×768 = 2.3K |
| CFG Scale MLP | 2-layer | 1→768→768 | — | ~1.2M |
| VLM Sequence Projector | Linear + LN | 2048→768 | — | 1.6M |
| VLM Pooled Projector | Linear + LN | 2048→768 | — | 1.6M |
| Cross-Attention Layers | 2 | 768 | 12 | ~9.4M |
| DiT Blocks | 12 | 768 | 12 | ~108M |
| Action Head | MLP (2-layer) | 768→768→128 | — | ~0.6M |
| **Total** | | | | **~123.6M** |

**Input Representations:**

**1. Noise Tokenization:**

Random noise $\epsilon \sim \mathcal{N}(0, I_{T \times 64})$ is projected to token space:

$$\text{MLP}_{\text{noise}}(\epsilon) = \text{Linear}_3(\text{SiLU}(\text{Linear}_2(\text{SiLU}(\text{Linear}_1(\epsilon)))))$$

where $\text{Linear}_1: 64 \to 768$, $\text{Linear}_2: 768 \to 768$, $\text{Linear}_3: 768 \to 768$.

Then add learned positional embeddings:

$$\mathbf{x}_0 = \text{MLP}_{\text{noise}}(\epsilon) + \text{Embed}_{\text{pos}}(\text{arange}(T)) \in \mathbb{R}^{T \times 768}$$

**2. Global Conditioning Vector:**

The global conditioning aggregates task context, embodiment type, and CFG scale:

$$\mathbf{c}_{\text{global}} = \mathbf{c}_{\text{pool}} + \mathbf{e}_{\text{emb}} + \mathbf{c}_{\text{cfg}}$$

where:
- $\mathbf{c}_{\text{pool}} = \text{Proj}_{\text{pool}}(\text{mean}(\mathbf{h}_{\text{VLM}})) \in \mathbb{R}^{768}$ (task context)
- $\mathbf{e}_{\text{emb}} = \text{Embed}_e(\text{embodiment\_id}) \in \mathbb{R}^{768}$ (embodiment: 0/1/2)
- $\mathbf{c}_{\text{cfg}} = \text{MLP}_{\text{cfg}}(\alpha) \in \mathbb{R}^{768}$ where $\alpha$ is CFG scale

**Cross-Attention Layers:**

Two cross-attention layers allow noise tokens to attend to fine-grained VLM features:

$$\mathbf{Q} = \mathbf{x}_0 \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{c}_{\text{seq}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{c}_{\text{seq}} \mathbf{W}_V$$

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{x}_1 = \mathbf{x}_0 + \text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$

This is repeated for 2 layers, with layer normalization and residual connections.

**DiT Transformer Blocks:**

Each of the 12 DiT blocks follows the adaLN-Zero design from "Scalable Diffusion Models with Transformers":

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \gamma_1 \cdot \text{Attn}(\text{Norm}(\mathbf{x}_i; \mathbf{c}_{\text{global}})) + \gamma_2 \cdot \text{MLP}(\text{Norm}(\mathbf{x}_i; \mathbf{c}_{\text{global}}))$$

where:
- $\text{Norm}(\mathbf{x}; \mathbf{c}) = \text{LayerNorm}(\mathbf{x}) \cdot (1 + \text{scale}(\mathbf{c})) + \text{shift}(\mathbf{c})$ (adaptive normalization)
- $\gamma_1, \gamma_2, \text{scale}, \text{shift}$ are predicted from $\mathbf{c}_{\text{global}}$ via small MLPs
- Self-attention uses RoPE (Rotary Position Embedding) for temporal modeling
- QK-Norm: $\mathbf{Q}' = \text{LayerNorm}(\mathbf{Q})$, $\mathbf{K}' = \text{LayerNorm}(\mathbf{K})$ before attention
- MLP uses SwiGLU activation: $\text{SwiGLU}(x) = (\mathbf{W}_1 x \odot \text{SiLU}(\mathbf{W}_2 x)) \mathbf{W}_3$

**Detailed DiTBlock Specification:**

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4.0):
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)  # scale_1, shift_1, gate_1, scale_2, shift_2, gate_2
        )
        
        # Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)  # No affine (replaced by adaLN)
        self.attn = MultiheadAttention(hidden_dim, num_heads, qk_norm=True, rope=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio))
    
    def forward(self, x, c):
        # c: [B, 768] global conditioning
        # Predict modulation parameters
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Attention block
        x_norm = self.norm1(x) * (1 + scale_1.unsqueeze(1)) + shift_1.unsqueeze(1)
        x = x + gate_1.unsqueeze(1) * self.attn(x_norm)
        
        # MLP block
        x_norm = self.norm2(x) * (1 + scale_2.unsqueeze(1)) + shift_2.unsqueeze(1)
        x = x + gate_2.unsqueeze(1) * self.mlp(x_norm)
        
        return x
```

**Action Head:**

Final MLP projects transformer output to 128-dimensional action space:

$$\mathbf{a}_{\text{raw}} = \text{Linear}_2(\text{SiLU}(\text{Linear}_1(\mathbf{x}_L))) \in \mathbb{R}^{T \times 128}$$

where $\text{Linear}_1: 768 \to 768$ and $\text{Linear}_2: 768 \to 128$.

**Action Masking and Normalization:**

$$\mathbf{a}_{\text{masked}} = \mathbf{a}_{\text{raw}} \odot \mathbf{m}_e$$

where $\mathbf{m}_e \in \{0,1\}^{128}$ is the embodiment-specific mask. Inactive dimensions are forced to zero.

For quaternion dimensions (e.g., dims 3-6 for gripper wrist orientation):

$$\mathbf{q}_{\text{norm}}[j] = \frac{\mathbf{a}_{\text{masked}}[j]}{\sqrt{\sum_{k \in \text{quat\_dims}} \mathbf{a}_{\text{masked}}[k]^2} + \epsilon}$$

This ensures valid unit quaternions for end-effector orientations.

**Training efficiency:** VLM features are pre-computed offline and stored on disk. During training, the VLM is never loaded into GPU memory, saving ~6 GB per GPU. The DiT (123M params) + gradients + optimizer states fit comfortably in 192 GB, enabling batch_size=128 per GPU.

### 3.3 Unified Action Space Design

Following RDT-1B's proven approach, we adopt a fixed 128-dimensional action vector:

```
Dimensions:  [0-6]    [7]     [8-14]  [15]    [16-31]      [32-127]
             arm1_pose grip1   arm2    grip2   joints       padding
             
Gripper (8 active):
  [x, y, z, qx, qy, qz, qw, gripper_open, 0, ..., 0]
  mask = [1,1,1,1,1,1,1,1, 0,...,0]

Bimanual (16 active):
  [left_j1..j7, left_grip, right_j1..j7, right_grip, 0, ..., 0]
  mask = [1,...,1 (×16), 0,...,0]

Dexterous Hand (23 active):
  [wrist_x,y,z,qx,qy,qz,qw, 0, 0,...,0, joint1..joint16, 0,...,0]
  mask = [1,...,1 (×7), 0, 0,..., 1,...,1 (×16), 0,...,0]
```

**Rationale:** Zero-padding is computationally wasteful (dexterous hand uses only 18% of 128 dims) but simplifies the architecture significantly. RDT-1B demonstrated this approach scales to 46 datasets. The action mask ensures the loss and gradient only backpropagate through active dimensions, preventing the model from learning spurious correlations with padding.

### 3.4 Loss Function

#### 3.4.1 Pure Drifting Loss (Paper's Formulation)

The canonical drifting loss from [arxiv:2602.04770] is:

$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{\epsilon}[\|f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q}(f_\theta(\epsilon)))\|^2]$$

where the drifting field $V_{p,q}$ is computed from positive samples $\{y^+\} \sim p_{\text{data}}$ and negative samples $\{y^-\} = \{f_\theta(\epsilon_i).detach()\}_{i=1}^B$ (predictions from the current batch, detached).

**Paper's Training Setup:**
- Effective batch size: **8192** (128 GPUs × 64 per GPU)
- Positive samples: 128 per class from sample queue (for conditional generation)
- Negative samples: Current batch predictions (8192 samples)
- The large batch provides sufficient diversity in negatives

**Key Property:** With drift normalization, the loss magnitude is approximately constant ($\mathcal{L} \approx D$ where $D$ is action dimensionality). This is **correct by design** — the loss value itself is not the convergence metric. Instead:
- The *direction* of the drifting field $V$ evolves during training
- Convergence is measured by task performance (success rate, FID, etc.)
- The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ pushes the model toward equilibrium

**Challenge for Single-GPU Training:** With batch size 32-128 (typical for single GPU), using $y^- = x.detach()$ causes the drifting field to have insufficient diversity. The 32 negative samples are too similar to the 32 query samples, leading to noisy field estimates.

#### 3.4.2 Our Modification: Hybrid MSE + Drifting Loss

To enable effective training on smaller batches (512-2048 vs paper's 8192), we propose a hybrid formulation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}}$$

**Component 1: Masked MSE Loss (Our Addition)**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathbf{m}_e|} \sum_{i,j} \mathbf{m}_e[j] \cdot (\mathbf{a}_{\text{pred}}[i,j] - \mathbf{a}_{\text{gt}}[i,j])^2$$

where $|\mathbf{m}_e|$ is the number of active dimensions. This component:
- Provides a direct, reliable gradient signal at ANY batch size
- Ensures the model learns to predict expert actions accurately
- Acts as a "warm start" for the drifting field to build upon

**Component 2: Drifting Field Loss (Paper's Formulation with Negative Queue)**

The drifting field is computed following the paper's formulation on active action dimensions:

$$\mathbf{a}_{\text{active}} = \mathbf{a}_{\text{pred}} \odot \mathbf{m}_e$$

$$V(\mathbf{a}_{\text{active}}) = \mathbb{E}_{\mathbf{a}^+, \mathbf{a}^-}[\tilde{k}(\mathbf{a}_{\text{active}}, \mathbf{a}^+)\tilde{k}(\mathbf{a}_{\text{active}}, \mathbf{a}^-)(mathbf{a}^+ - \mathbf{a}^-)]$$

The normalized kernel uses double softmax normalization (Paper Eq. 13-15):

$$k(\mathbf{a}, \mathbf{a}') = \exp\left(-\frac{\|\mathbf{a} - \mathbf{a}'\|}{\tau}\right)$$

$$\tilde{k}(x_i, y_j) = \sqrt{\text{softmax}_j(s_{ij}) \cdot \text{softmax}_i(s_{ij})}$$

where $s_{ij} = -\|x_i - y_j\| / \tau$ (note: L2 norm, not squared). Multi-temperature kernels $\tau \in \{0.02, 0.05, 0.2\}$ are summed.

After computing $V$, feature normalization (Paper Eq. 18-21) and drift normalization (Paper Eq. 23-25) are applied:

- Feature normalization: scale samples so average pairwise distance ≈ $\sqrt{D}$
- Drift normalization: scale $V$ so that $\mathbb{E}[\|V\|^2 / D] = 1$

The drifting loss is:

$$\mathcal{L}_{\text{drift}} = \|\mathbf{a}_{\text{active}} - \text{sg}(\mathbf{a}_{\text{active}} + V(\mathbf{a}_{\text{active}}))\|^2$$

After normalization, this loss is approximately constant ($\mathcal{L}_{\text{drift}} \approx D$). **This is correct and expected** — the gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ provides the learning signal through the direction of $V$, not its magnitude.

**Our Modifications for Small-Batch Training:**

1. **Negative Sample Queue:** The paper uses $\mathbf{a}^- = \mathbf{a}_{\text{pred}}.detach()$ (current batch) with **batch size 8192**. For single-GPU training (batch 32-512), this provides insufficient diversity. We introduce a `NegativeSampleQueue` storing predictions from previous steps, ensuring negatives differ from queries even with small batches.

2. **Hybrid Loss:** The paper trains with pure drifting loss at large batch (8192). For smaller batches (512-2048), we add an MSE component:
   - MSE provides a reliable gradient when drifting field estimation is noisy
   - Weight $\lambda_{\text{drift}} = 0.1$ ensures drifting still contributes multi-modal structure
   - This is an ablation — we compare pure vs hybrid in Phase 5

**Justification:** The paper's algorithm (x.detach() as negatives, pure drifting loss) is optimal for large-scale multi-GPU training (batch 8192). Our modifications (negative queue, hybrid loss) enable convergence on limited compute (single GPU, batch 32-512) while preserving the drifting paradigm's core benefits. These are **practical engineering contributions**, not fundamental algorithmic changes.

**Critical Understanding: Why Loss ≈ Constant is Correct**

A common misunderstanding is that "loss not decreasing means the model isn't learning." For drifting models, this is WRONG:

From the paper (Section 3.4, Eq. 23-25):
- Drift normalization scales $V$ so that $\mathbb{E}[\|V\|^2 / D] = 1$
- Therefore, $\mathcal{L}_{\text{drift}} = \|V_{\text{normalized}}\|^2 \approx D$ (constant)
- The loss magnitude doesn't change, but the **direction of $V$** evolves
- The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ pushes the model based on V's direction
- As training progresses, $V$ points more accurately toward the data manifold
- At equilibrium, $V \to 0$ (which would make loss → 0, but in practice we stop before perfect convergence)

**Convergence Metrics for Drifting Models:**
- For image generation: FID, IS (not the drifting loss value)
- For robotics: Success rate, MMD (not the drifting loss value)
- Optional: Track $\lambda_V$ (the normalization factor before scaling) — this does decrease as $V_{\text{raw}}$ shrinks

In our development, we initially misinterpreted constant loss as lack of convergence, leading to the hybrid loss modification. Phase 5 ablations will determine if this modification is beneficial (for multi-modality) or unnecessary (for convergence).

#### 3.4.2 Action Normalization

Per-dataset, per-dimension normalization to zero-mean unit-variance:

$$\mathbf{a}_{\text{norm}}[j] = \frac{\mathbf{a}_{\text{raw}}[j] - \mu_{\text{dataset}}[j]}{\sigma_{\text{dataset}}[j] + \epsilon}$$

This is critical because:
- Position dimensions (meters) have magnitudes ~0.1-1.0
- Quaternion dimensions are unit-normalized, magnitudes ~0.1-1.0
- Joint angles (radians) have magnitudes ~0-3.14
- Without normalization, MSE is dominated by position errors

Statistics $\mu, \sigma$ are computed once per dataset from the training split.

---

## 4. Data Pipeline

### 4.1 Dataset Sources

| Dataset | Embodiment | Samples | Action Format | Image Format | URL |
|---------|-----------|---------|--------------|-------------|-----|
| RLBench (18 tasks) | Parallel gripper | ~270K | Absolute EE pose + grip (8-dim) | Multi-view RGB 128×128 | [HuggingFace](https://huggingface.co/datasets/hqfang/rlbench-18-tasks) |
| DexGraspNet 2.0 | Dexterous hand | ~500K | Wrist pose + joint angles (23-dim) | Point cloud → rendered RGB | [HuggingFace](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0) |
| Bridge V2 | WidowX gripper | 60K | Delta EE (7-dim) | Wrist RGB 256×256 | [Project](https://rail-berkeley.github.io/bridgedata/) |
| ALOHA | Bimanual | 50K | Absolute joint pos (16-dim) | Multi-view RGB | [GitHub](https://github.com/tonyzhaozh/aloha) |
| RT-1 | Everyday Robots | 130K | Delta EE (7-dim) | RGB 320×256 | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/rt_1) |

**Total:** ~1.01M demonstrations across 5 datasets and 3 embodiment types.

### 4.2 Complete Data Processing Pipeline

The data preprocessing pipeline transforms raw demonstrations from heterogeneous sources into a unified format compatible with our model. This section provides the complete end-to-end processing workflow.

#### Step 1: Image Standardization and Multi-View Handling

All images resized to **448×448** (PaliGemma2's native resolution):

```python
def preprocess_images(images: np.ndarray, dataset_name: str, cameras: List[str]) -> np.ndarray:
    """
    Standardize images across datasets.
    
    Args:
        images: Raw images, format varies by dataset:
                - RLBench: dict[camera_name] → [H, W, 3] uint8
                - DexGraspNet: rendered from point cloud [H, W, 3]
                - Bridge: [H, W, 3] uint8 (wrist camera)
                - ALOHA: dict[cam] → [H, W, 3] (bimanual setup)
        dataset_name: Source dataset identifier
        cameras: List of camera names to use
    
    Returns:
        images: [V, 3, 448, 448] float32 in [0, 1]
                V = len(cameras), standardized across datasets
    """
    images_standardized = []
    
    # Extract cameras in consistent order
    for cam in cameras:
        if isinstance(images, dict):
            img = images.get(cam, None)
        else:
            img = images  # Single-view datasets
        
        if img is None:
            # Missing camera → black frame
            img = np.zeros((448, 448, 3), dtype=np.uint8)
        
        # Resize to 448×448 with anti-aliasing
        if img.shape[0] != 448 or img.shape[1] != 448:
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # CHW format
        img = img.transpose(2, 0, 1)
        
        images_standardized.append(img)
    
    return np.stack(images_standardized, axis=0)  # [V, 3, 448, 448]
```

**Camera Mapping Across Datasets:**

| Dataset | Native Cameras | Mapped to Standard |
|---------|---------------|-------------------|
| RLBench | `front_rgb`, `wrist_rgb`, `left_shoulder_rgb` | `front`, `wrist` |
| DexGraspNet | Rendered from point cloud (4 angles) | `front`, `side`, `top`, `wrist` |
| Bridge V2 | Single wrist camera | `wrist` |
| ALOHA | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` | `front`, `wrist_left`, `wrist_right` |
| RT-1 | Multiple angles per episode | `front` (primary) |

**Standardization:** We use `front` + `wrist` as the canonical 2-view setup. Datasets lacking a camera are filled with black frames (the VLM learns to ignore these).

#### Step 2: Action Conversion to Absolute Representation

Datasets use heterogeneous action representations. We standardize to **absolute end-effector pose** (or absolute joint positions for bimanual/dex hand):

**Action Format by Dataset:**

| Dataset | Native Format | Conversion Needed |
|---------|--------------|-------------------|
| RLBench | Absolute EE pose [7] + grip [1] | None ✓ |
| DexGraspNet | Absolute wrist [7] + joint angles [16] | None ✓ |
| Bridge V2 | **Delta EE [6]** + grip [1] | Integrate deltas → absolute |
| RT-1 | **Delta EE [6]** + grip [1] | Integrate deltas → absolute |
| ALOHA | Absolute joint positions [7+7] + grips [2] | None ✓ |

**Delta to Absolute Integration:**

For datasets storing delta actions (position delta $\Delta \mathbf{p}$, rotation delta as axis-angle or delta quaternion):

```python
def delta_to_absolute(delta_actions: np.ndarray, 
                      initial_pose: np.ndarray,
                      rotation_format: str = 'quat') -> np.ndarray:
    """
    Integrate delta actions to absolute end-effector poses.
    
    Args:
        delta_actions: [T, 7] — [delta_x, delta_y, delta_z, 
                                   delta_qx/rx, delta_qy/ry, delta_qz/rz, (delta_qw), 
                                   gripper]
        initial_pose: [7] — [x0, y0, z0, qx0, qy0, qz0, qw0, grip0]
        rotation_format: 'quat' (quaternion delta) or 'axis_angle'
    
    Returns:
        absolute_actions: [T, 8] — [x, y, z, qx, qy, qz, qw, gripper]
    """
    absolute_actions = []
    current_pose = initial_pose.copy()
    
    for delta in delta_actions:
        # Integrate position
        current_pose[:3] += delta[:3]
        
        # Integrate rotation
        if rotation_format == 'quat':
            # Delta is small quaternion change
            delta_quat = delta[3:7]
            delta_quat = delta_quat / (np.linalg.norm(delta_quat) + 1e-8)
            current_pose[3:7] = quaternion_multiply(current_pose[3:7], delta_quat)
            current_pose[3:7] /= np.linalg.norm(current_pose[3:7])
        elif rotation_format == 'axis_angle':
            # Delta is axis-angle [rx, ry, rz]
            delta_angle = np.linalg.norm(delta[3:6])
            if delta_angle > 1e-6:
                delta_axis = delta[3:6] / delta_angle
                delta_quat = axis_angle_to_quat(delta_axis, delta_angle)
                current_pose[3:7] = quaternion_multiply(current_pose[3:7], delta_quat)
                current_pose[3:7] /= np.linalg.norm(current_pose[3:7])
        
        # Gripper state
        if len(delta) > 6:
            current_pose[7] = delta[6] if len(delta) == 7 else delta[7]
        
        absolute_actions.append(current_pose.copy())
    
    return np.array(absolute_actions)
```

**Initial Pose Extraction:**

For delta-action datasets, the initial pose is extracted from the first observation in the episode:

```python
# Bridge V2 / RT-1: extract from episode metadata
initial_pose = episode.observations[0].gripper_pose  # [x,y,z,qx,qy,qz,qw,grip]

# If not available: use dataset mean pose
initial_pose = DATASET_MEAN_POSES[dataset_name]
```

#### Step 3: Action Mapping to 128-dim Unified Space

Each embodiment's native action is mapped to a fixed 128-dim vector following a standardized layout:

**Unified Action Layout:**

```
Index:    [0-2]   [3-6]    [7]     [8-14]  [15]    [16-31]          [32-127]
Field:    pos1    quat1   grip1    pos2    grip2   joint_angles     padding
Type:     xyz     xyzw    binary   xyz/j   binary  θ1..θ16          zeros
Units:    meters  unit    {0,1}    m/rad   {0,1}   radians          —
```

**Per-Embodiment Mapping Functions:**

```python
def map_to_unified_gripper(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gripper: [x, y, z, qx, qy, qz, qw, gripper_open]
    
    Args:
        action: [8] or [T, 8] — native gripper action
    
    Returns:
        unified: [128] or [T, 128] — unified action
        mask: [128] — active dimension mask
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:8] = action
        mask = np.zeros(128, dtype=bool)
        mask[:8] = True
    else:  # [T, 8]
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :8] = action
        mask = np.zeros(128, dtype=bool)
        mask[:8] = True
    
    return unified, mask


def map_to_unified_bimanual(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bimanual: [left_j1..j7, left_gripper, right_j1..j7, right_gripper]
    
    ALOHA uses joint positions (not EE pose). We map to slots 0-15.
    
    Args:
        action: [16] or [T, 16] — native bimanual action
    
    Returns:
        unified: [128] or [T, 128]
        mask: [128]
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:16] = action
        mask = np.zeros(128, dtype=bool)
        mask[:16] = True
    else:
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :16] = action
        mask = np.zeros(128, dtype=bool)
        mask[:16] = True
    
    return unified, mask


def map_to_unified_dexhand(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dexterous Hand: [wrist_x, y, z, qx, qy, qz, qw, joint1..joint16]
    
    Wrist pose goes to slots 0-6.
    Joint angles go to slots 16-31.
    Slots 7-15 are unused (reserved for grip/bimanual).
    
    Args:
        action: [23] or [T, 23] — [wrist_pose(7), joints(16)]
    
    Returns:
        unified: [128] or [T, 128]
        mask: [128]
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:7] = action[:7]     # Wrist pose
        unified[16:32] = action[7:]  # Joint angles
        mask = np.zeros(128, dtype=bool)
        mask[:7] = True
        mask[16:32] = True
    else:
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :7] = action[:, :7]
        unified[:, 16:32] = action[:, 7:]
        mask = np.zeros(128, dtype=bool)
        mask[:7] = True
        mask[16:32] = True
    
    return unified, mask


# Embodiment-to-function mapping
EMBODIMENT_MAPPERS = {
    0: map_to_unified_gripper,
    1: map_to_unified_bimanual,
    2: map_to_unified_dexhand,
}

# Reverse mapping (for inference)
def extract_from_unified(unified_action: np.ndarray, 
                        embodiment_id: int) -> np.ndarray:
    """Extract embodiment-specific action from 128-dim unified vector."""
    if embodiment_id == 0:  # Gripper
        return unified_action[:8]
    elif embodiment_id == 1:  # Bimanual
        return unified_action[:16]
    elif embodiment_id == 2:  # Dex hand
        wrist = unified_action[:7]
        joints = unified_action[16:32]
        return np.concatenate([wrist, joints])  # [23]
```

#### Step 4: VLM Feature Pre-computation (Offline)

**Motivation:** PaliGemma2-3B requires ~6 GB GPU memory and ~40ms per forward pass. Pre-computing features offline allows:
1. Training without VLM in GPU memory → larger batch sizes
2. Faster data loading (read 4KB features vs process 448×448 images)
3. Easy ablation of different VLMs (re-run pre-computation, no training code change)

**Complete Pre-computation Script:**

```python
# scripts/precompute_vlm_features.py

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

def precompute_vlm_features(
    dataset_name: str,
    model_name: str = "google/paligemma2-3b-mix-448",
    batch_size: int = 32,
    num_gpus: int = 8,
):
    """
    Pre-compute VLM features for a dataset.
    
    Time estimate: ~40ms per sample per GPU
    For 1M samples across 8 GPUs: 1M / (8 × 25 samples/sec) ≈ 83 minutes
    """
    # Load VLM
    processor = AutoProcessor.from_pretrained(model_name)
    vlm = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    vlm.eval()
    
    # Load dataset
    dataset = load_dataset(dataset_name)  # Returns UnifiedDataset
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # HDF5 file for storing features
    output_file = f'./data/vlm_features/{dataset_name}_paligemma2.h5'
    h5_file = h5py.File(output_file, 'w')
    
    # Process in batches
    for batch_idx, batch in enumerate(tqdm(loader)):
        images = batch['images']        # [B, V, 3, 448, 448]
        languages = batch['language']   # List[str]
        sample_ids = batch['sample_id'] # List[int]
        
        # Flatten multi-view: [B, V, 3, 448, 448] → [B×V, 3, 448, 448]
        B, V = images.shape[:2]
        images_flat = images.reshape(B * V, 3, 448, 448)
        
        # Replicate language for each view
        langs_flat = [lang for lang in languages for _ in range(V)]
        
        # Process through VLM
        inputs = processor(
            text=langs_flat,
            images=images_flat,
            return_tensors="pt",
            padding=True,
        ).to('cuda')
        
        with torch.no_grad():
            outputs = vlm(
                **inputs,
                output_hidden_states=True,
            )
        
        # Extract last hidden state
        hidden = outputs.hidden_states[-1]  # [B×V, L, 2048]
        
        # Reshape: [B×V, L, 2048] → [B, V, L, 2048]
        L, D = hidden.shape[1], hidden.shape[2]
        hidden = hidden.reshape(B, V, L, D)
        
        # Aggregate views (mean across V)
        hidden_agg = hidden.mean(dim=1)  # [B, L, 2048]
        pooled = hidden_agg.mean(dim=1)  # [B, 2048]
        
        # Save to HDF5 (fp16 for storage efficiency)
        for i, sample_id in enumerate(sample_ids):
            grp = h5_file.create_group(f'sample_{sample_id}')
            grp.create_dataset('hidden', data=hidden_agg[i].cpu().half().numpy(), compression='gzip')
            grp.create_dataset('pooled', data=pooled[i].cpu().half().numpy())
            grp.attrs['sequence_length'] = L
            grp.attrs['hidden_dim'] = D
    
    h5_file.close()
    print(f"Saved to {output_file}")
    print(f"Size: {os.path.getsize(output_file) / 1e9:.2f} GB")
```

**HDF5 Storage Format:**

```
vlm_features_{dataset}.h5
├── sample_0/
│   ├── hidden: [L, 2048] float16 (gzip compressed)
│   ├── pooled: [2048] float16
│   └── attrs: {sequence_length: L, hidden_dim: 2048}
├── sample_1/
│   ├── ...
└── ...
```

**Loading During Training:**

```python
class PrecomputedVLMDataset:
    def __init__(self, base_dataset, vlm_features_path):
        self.base_dataset = base_dataset
        self.vlm_h5 = h5py.File(vlm_features_path, 'r')
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_id = sample['sample_id']
        
        # Load pre-computed VLM features
        hidden = torch.from_numpy(self.vlm_h5[f'sample_{sample_id}/hidden'][:]).float()
        pooled = torch.from_numpy(self.vlm_h5[f'sample_{sample_id}/pooled'][:]).float()
        
        sample['vlm_features'] = hidden
        sample['vlm_pooled'] = pooled
        
        return sample
```

**Storage estimate:** ~4KB per sample (L=1050, 2048-dim, fp16) → 1M samples = ~4 GB total.

#### Step 5: DexGraspNet 2.0 Specific Processing

DexGraspNet 2.0 provides point clouds (not images). We render multi-view RGB images:

```python
def render_pointcloud_to_images(
    pointcloud: np.ndarray,  # [N, 3] xyz
    colors: np.ndarray,       # [N, 3] rgb
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],  # [(pos, rot), ...]
) -> np.ndarray:
    """
    Render point cloud from multiple camera views.
    
    Uses Open3D for fast GPU-accelerated rendering.
    
    Args:
        pointcloud: [N, 3] point positions
        colors: [N, 3] RGB colors
        camera_poses: List of (position, rotation) tuples for each view
    
    Returns:
        images: [V, 448, 448, 3] uint8
    """
    import open3d as o3d
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    images = []
    for cam_pos, cam_rot in camera_poses:
        # Set up virtual camera
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=448, height=448)
        vis.add_geometry(pcd)
        
        # Set camera parameters
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = create_extrinsic_matrix(cam_pos, cam_rot)
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img) * 255).astype(np.uint8)
        
        images.append(img)
        vis.destroy_window()
    
    return np.array(images)  # [V, 448, 448, 3]


# Standard camera poses for DexGraspNet rendering
DEXGRASP_CAMERA_POSES = {
    'front': (pos=[0.5, 0, 0.3], lookat=[0,0,0]),
    'side':  (pos=[0, 0.5, 0.3], lookat=[0,0,0]),
    'top':   (pos=[0, 0, 0.8], lookat=[0,0,0]),
    'wrist': (pos=[0.1, 0, 0.15], lookat=[0,0,0.05]),  # Close-up
}
```

**Grasp Pose to Action:**

DexGraspNet stores grasp poses as:
- Wrist SE(3) pose: 4×4 transformation matrix
- Joint angles: [θ1, ..., θ16] for Allegro hand

```python
def dexgrasp_pose_to_action(grasp_pose_dict):
    """Convert DexGraspNet grasp to our action format."""
    # Extract wrist pose from 4×4 matrix
    T = grasp_pose_dict['wrist_transform']  # [4, 4]
    pos = T[:3, 3]  # Translation
    rot_mat = T[:3, :3]  # Rotation matrix
    quat = rotation_matrix_to_quaternion(rot_mat)  # [qx, qy, qz, qw]
    
    # Joint angles
    joints = grasp_pose_dict['joint_angles']  # [16]
    
    # Concatenate
    action = np.concatenate([pos, quat, joints])  # [23]
    
    return action
```

### 4.3 Data Loading During Training

```python
class UnifiedDataLoader:
    def __init__(self, datasets, sampling_weights):
        self.datasets = {
            'rlbench': RLBenchDataset(...),
            'dexgraspnet': DexGraspNetDataset(...),
            'bridge': BridgeDataset(...),
            'aloha': ALOHADataset(...),
            'rt1': RT1Dataset(...),
        }
        self.weights = sampling_weights  # e.g., [0.3, 0.3, 0.2, 0.1, 0.1]
    
    def __iter__(self):
        while True:
            # Sample dataset according to weights
            dataset_name = np.random.choice(
                list(self.datasets.keys()),
                p=self.weights
            )
            
            # Sample batch from chosen dataset
            batch = self.datasets[dataset_name].sample(batch_size)
            
            # Load pre-computed VLM features from disk
            vlm_features = load_vlm_features(batch['sample_ids'])
            
            # Normalize actions using dataset-specific stats
            actions_norm = normalize_actions(batch['actions'], dataset_name)
            
            yield {
                'vlm_features': vlm_features,
                'vlm_pooled': vlm_pooled,
                'embodiment_id': EMBODIMENT_MAP[dataset_name],
                'action_mask': ACTION_MASK_MAP[dataset_name],
                'actions': actions_norm,
                'dataset_name': dataset_name,
            }
```

---

## 5. Training Procedure

### 5.1 Optimization

**Optimizer:** AdamW with weight decay 0.05, differential LR (Section 3.2.1)

**Learning rate schedule:**
- Warmup: Linear 0 → 4e-4 over 2,000 steps
- Main: Cosine decay 4e-4 → 1e-5 over 98,000 steps
- Total: 100,000 steps
- VLM LoRA unfreezes at step 5,000 with LR 1e-5

**Compute Configurations (system supports both):**

| Setting | 8×B200 (192GB each) | 64×H100 (80GB each) |
|---------|---------------------|---------------------|
| Per-GPU batch | 128 | 64 |
| GPUs | 8 | 64 |
| Grad accumulation | 2 | 2 |
| **Effective batch** | **2,048** | **8,192** |
| Flash Attention | ✅ (bf16 auto-cast) | ✅ (native) |
| FSDP | ✅ | ✅ (ZeRO-3) |
| VLM in GPU (Phase 3+) | ✅ (fits in 192GB) | ✅ (FSDP sharded) |
| Training time (100K steps) | ~40 hrs | ~22 hrs |

**Primary target: 64×H100** (effective batch 8192, matching the Drifting paper exactly).

With batch 8192 on 64×H100:
- Can use **paper's exact formulation** (pure drifting loss, x.detach() as negatives)
- Negative sample queue becomes optional (ablate in Phase 5)
- Hybrid MSE loss becomes optional (ablate in Phase 5)

**Launch command (64×H100, 8 nodes × 8 GPUs):**

```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  scripts/train.py \
  training.batch_size=64 \
  training.grad_accumulation_steps=2 \
  training.num_workers=8 \
  training.drifting.n_pos_samples=256 \
  training.drifting.n_neg_samples=256
```

**Launch command (8×B200, single node):**

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  training.batch_size=128 \
  training.grad_accumulation_steps=2
```

**Mixed precision:** BF16 with Flash Attention 2 (auto-cast fix in dit.py)

**Gradient clipping:** Max norm 2.0

### 5.2 Negative Sample Queue

To prevent drifting field collapse ($V \approx 0$), we maintain a queue of recent model predictions:

```python
neg_queue = NegativeSampleQueue(size=4096, action_dim=128)

# Each training step:
neg_samples = neg_queue.sample(n=128)  # Sample from previous steps
neg_queue.push(actions_pred.detach())   # Store current predictions

# Drifting loss uses neg_samples (NOT actions_pred.detach())
```

Queue size 4096 stores ~2 batches of predictions, ensuring negatives are distinct from current queries.

### 5.3 Positive Sample Queue

Per-task queues for expert actions:

```python
pos_queue = SampleQueue(queue_size=256, num_tasks=len(all_tasks), action_dim=128)

# Each step:
pos_queue.add(task_id=batch['task_id'], actions=batch['actions'])
pos_samples = pos_queue.sample(n=128, task_ids=batch['task_id'])
```

### 5.4 Training Algorithm

**Algorithm: Drifting-VLA v2 Training (64×H100, Effective Batch 8192)**

At 64×H100 with effective batch 8192, we adopt the **paper's canonical formulation** as default, with our modifications available as configuration flags for ablation.

```
Algorithm: Drifting-VLA v2 Training

Input:  Datasets D = {D_1, ..., D_K}, sampling weights w
        Compute: 64×H100, effective batch B = 8192
Output: Trained model θ

--- OFFLINE (one-time, ~4 hours on 8 GPUs) ---
1: For each dataset D_k:
2:     For each sample (images, language, actions):
3:         For each view v in images:
4:             h_v ← PaliGemma2(view_v, language)  # Independent per view
5:         h_concat ← cat([h_1, ..., h_V], dim=0)   # Variable V per dataset
6:         Save h_concat to HDF5

--- ONLINE (100K steps, ~22 hours on 64×H100) ---
7:  Initialize DiT θ, VLM LoRA (frozen until step 5K)
8:  Initialize optimizer (differential LR), pos_queue, neg_queue
9:  for step = 1 to 100,000 do
10:     Sample dataset D_k ~ Categorical(w)
11:     Sample batch B ~ D_k, size B=8192
12:     Load vlm_features, actions, embodiment_id, action_mask from B
13:     
14:     # Normalize GT actions (per-dataset, per-dim)
15:     actions_gt ← (actions - μ_k) / (σ_k + ε)
16:     
17:     # Forward through DiT
18:     noise ← N(0, I_{B×T×64})
19:     α ← sample_cfg_scale()                    # CFG: α ~ p(α) ∝ α^{-3}
20:     c_seq ← proj(vlm_features)                # [B, L, 768]
21:     c_pool ← proj_pool(mean(vlm_features))    # [B, 768]
22:     actions_pred ← DiT(noise, c_seq, c_pool, embodiment_id, α)  # [B, T, 128]
23:     
24:     # Sample positive/negative for drifting field
25:     y_pos ← pos_queue.sample(n=256)
26:     y_neg ← actions_pred.detach()              # Paper's approach (works at batch 8192)
27:     
28:     # Compute drifting field V (paper's formulation)
29:     V ← compute_drifting_field(
30:         x=actions_pred, y_pos=y_pos, y_neg=y_neg,
31:         τ=[0.02, 0.05, 0.2],
32:         normalize_features=True, normalize_drift=True,
33:         action_mask=action_mask                 # Only compute on active dims
34:     )
35:     
36:     # Loss (default: pure drifting; flag: +MSE hybrid)
37:     L_drift ← ||actions_pred - sg(actions_pred + V)||²
38:     if hybrid_mode:
39:         L_mse ← masked_mse(actions_pred, actions_gt, action_mask)
40:         L ← L_mse + λ_drift × L_drift
41:     else:
42:         L ← L_drift                            # Paper's default
43:     
44:     # Optimization
45:     L.backward()
46:     clip_grad_norm(θ, max_norm=2.0)
47:     optimizer.step()                            # Differential LR
48:     ema.update(θ)
49:     
50:     # Update queues
51:     pos_queue.push(actions_gt)
52:     
53:     # Unfreeze VLM LoRA at step 5K (staged training)
54:     if step == 5000:
55:         unfreeze(vlm_lora, lr=1e-5)
56: end for
```

**Configuration Flags for Ablation:**
- `hybrid_mode`: True adds MSE component (default: False at batch 8192)
- `use_neg_queue`: True uses queue instead of x.detach() (default: False at batch 8192)
- `λ_drift`: Weight for drifting loss in hybrid mode (default: 0.1)
- These are tested systematically in Phase 5

---

## 6. Evaluation Protocol

### 6.1 Benchmarks

#### 6.1.1 RLBench (Parallel Gripper)

**Tasks:** 18 manipulation tasks (close_jar, open_drawer, stack_blocks, ...)

**Metric:** Success rate (%) over 25 episodes per task

**Baselines:** RDT-1B, Diffusion Policy, 3D Diffuser Actor, OpenVLA

**Evaluation:** CoppeliaSim simulation with our RLBench environment wrapper

#### 6.1.2 DexGraspNet 2.0 (Dexterous Hand)

**Tasks:** Grasping diverse objects in cluttered scenes

**Metric:** Grasp success rate (%) in IsaacGym simulation

**Baselines:** DexGraspNet 2.0 baseline, GraspTTA, IsaGrasp

**Evaluation:** IsaacGym with Allegro hand model

#### 6.1.3 ManiSkill (Multi-Task)

**Tasks:** 5 benchmark tasks (PegInsertion, PickCube, StackCube, PlugCharger, PushCube)

**Metric:** Success rate (%) over 250 trials (10 seeds × 25 trials)

**Baselines:** RDT-1B (53.6%), Diffusion Policy (30.2%), OpenVLA (4.8%)

**Evaluation:** ManiSkill simulation environment

### 6.2 Ablation Studies

| Ablation | Variants | Purpose |
|----------|----------|---------|
| VLM backbone | PaliGemma2 vs Qwen3-VL vs DINOv2+CLIP | Validate VLM benefit |
| Loss weight | $\lambda_{\text{drift}} \in \{0, 0.05, 0.1, 0.2\}$ | Tune MSE/drift balance |
| Batch size | 512, 1024, 2048, 4096 | Find minimal for drifting |
| Action space | 128-dim vs 64-dim vs multi-head | Validate padding approach |
| Negative sampling | Queue vs x.detach() vs random | Validate queue necessity |
| Multi-embodiment | Joint vs separate training | Transfer learning analysis |

### 6.3 Metrics

**Primary:** Task success rate (%)

**Secondary:**
- Inference latency (ms per action)
- Action distribution quality (MMD vs expert)
- Multi-modality score (GMM entropy)
- Generalization: zero-shot to new objects/scenes

---

## 7. Implementation Requirements

### 7.1 Hardware

**Development:** 1× NVIDIA A40 (48GB) for single-GPU debugging and simulation evaluation

**Training (supported configurations):**

| Config | GPUs | Effective Batch | Training Time (100K) | Cost |
|--------|------|----------------|---------------------|------|
| 8×B200 (single node) | 8 × 192GB | 2,048 | ~40 hrs | ~$3,000 |
| **64×H100 (8 nodes)** | 64 × 80GB | **8,192** | **~22 hrs** | ~$4,500 |
| 8×H100 (single node) | 8 × 80GB | 1,024 | ~80 hrs | ~$3,000 |

**Primary: 64×H100** — matches the Drifting paper's batch size (8192), enabling faithful reproduction.

**Fallback: 8×B200** — effective batch 2048, requires hybrid loss modification.

**Evaluation:** A40 (dev) with CoppeliaSim + RLBench for gripper eval; H100 node with IsaacGym for DexGraspNet eval

### 7.2 Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.2+ | Deep learning framework |
| Flash Attention | 2.5+ | Efficient attention (now with auto-cast fix) |
| Transformers | 4.45+ | PaliGemma2-3B and Qwen3-VL-2B loading |
| PEFT | 0.10+ | LoRA adapters |
| DeepSpeed | 0.14+ | FSDP / ZeRO optimization |
| WandB | Latest | Experiment tracking |
| CoppeliaSim | 4.1.0 | RLBench simulation |
| IsaacGym | 4.0 | DexGraspNet evaluation |

### 7.3 Docker Images

**Base image:** `drifting-vla:base` (CUDA 12.1, PyTorch 2.2, Flash Attn)

**RLBench image:** `drifting-vla:rlbench` (+ CoppeliaSim 4.1, PyRep, RLBench, Xvfb)

**DexGrasp image:** `drifting-vla:dexgrasp` (+ IsaacGym, Open3D, point cloud tools)

### 7.4 Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **VLM backbone** | PaliGemma2-3B (frozen + LoRA) | Best visual grounding at 448×448; 3B fits in memory; LoRA enables task adaptation with <1% params |
| **Why not end-to-end VLM?** | VLMs output text tokens, not continuous actions | Need separate action generation head |
| **Why pre-compute VLM features?** | Saves 6GB GPU memory during training | Enables larger batch sizes; VLM runs once offline per sample |
| **Action space** | Fixed 128-dim with masking (RDT-style) | Proven to work; simpler than multi-head; easy to add new embodiments |
| **Why not multi-head decoder?** | Adds complexity without clear benefit | RDT showed padding works; multi-head requires embodiment-specific tuning |
| **Loss design** | Hybrid MSE + Drifting | MSE ensures convergence; drifting adds multi-modality; neither alone is sufficient |
| **Negative sampling** | Queue of past predictions | Prevents V≈0 self-cancellation; critical for small batch |
| **Why Drifting not Diffusion?** | 1-step vs 100-step inference | 10× faster; enables real-time control (20-100 Hz) |
| **Action normalization** | Per-dataset, per-dim | Prevents position dominating rotation in MSE |
| **Multi-view** | Front + wrist (2 views) | Matches 3D Diffuser Actor baseline; provides depth cues |

---

## 8. Expected Results

### 8.1 Quantitative Targets

| Benchmark | Metric | RDT-1B (100-NFE) | Ours (1-NFE) Target |
|-----------|--------|------------------|---------------------|
| RLBench (18 tasks) | Success rate | ~85% (estimated) | **>80%** |
| DexGraspNet 2.0 | Grasp success | ~75% (baseline) | **>70%** |
| ManiSkill (5 tasks) | Success rate | 53.6% | **>50%** |
| Inference latency | ms per action | ~500ms | **<50ms** |

### 8.2 Ablation Expectations

**VLM Backbone Ablation:**

| Config | Model | Hidden Dim | Multi-Image | Hypothesis |
|--------|-------|-----------|-------------|-----------|
| A | PaliGemma2-3B | 2048 | Independent per view | Best grounding (default) |
| B | Qwen3-VL-2B | 1536 | Native multi-image | Better multi-view, smaller model |
| C | DINOv2 + CLIP (v1) | 1024 | Independent per view | Baseline (no joint VL reasoning) |

Expected: A ≈ B > C by +5-10% success rate

**Loss Formulation Ablations:**
- **At effective batch 2048 (8×B200):**
  - Pure drifting (paper) vs Hybrid (ours): Hypothesis: similar performance (paper's formulation should work at this scale)
  - Negative queue vs x.detach() (paper): Hypothesis: x.detach() sufficient at large batch
  - $\lambda_{\text{drift}} \in \{0, 0.05, 0.1, 0.2, 1.0\}$: Hypothesis: pure drifting ($\lambda=1.0$) achieves best multi-modality
  
- **At effective batch 512 (1 GPU):**
  - Pure drifting vs Hybrid: Hypothesis: hybrid gives +15-25% (pure drifting noisy at small batch)
  - Negative queue vs x.detach(): Hypothesis: queue gives +10-15% (x.detach() causes field collapse)

**Multi-Embodiment Ablations:**
- Joint training vs separate per-embodiment models: -2-5% per task vs single-task (acceptable trade-off for unified model)
- Pre-train on gripper → fine-tune on dex hand: +5-10% vs train-from-scratch (transfer learning benefit)
- Embodiment token: +3-5% vs no embodiment conditioning (helps the model specialize)

---

## 9. Development Schedule

### Phase 0: Infrastructure Setup (Week 1)

**Days 1-2: Environment & Docker**
- Set up 8×B200 node with CUDA 12.1, PyTorch 2.2
- Build Docker images (base, rlbench, dexgrasp)
- Verify FSDP, Flash Attention, multi-GPU work
- Set up WandB project

**Days 3-4: Data Download**
- Download RLBench (18 tasks) → ~50 GB
- Download DexGraspNet 2.0 → ~100 GB
- Download Bridge V2, ALOHA, RT-1 → ~700 GB
- Verify data integrity

**Days 5-7: VLM Feature Pre-computation**
- Implement `scripts/precompute_vlm_features.py`
- Pre-compute PaliGemma2 features for all datasets
- Verify features load correctly during training
- **Deliverable:** ~4 GB HDF5 files with VLM features

---

### Phase 1: Toy Case Validation (Week 2)

**Days 8: Verify Paper's Formulation (Critical First Step)**

Before any modifications, we verify our drifting field implementation matches the paper:

- Implement 2D toy example from [demo notebook](https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb)
- Reproduce Figure 3 from paper (Gaussian mixture → drifting field evolution)
- Verify:
  - Double softmax normalization produces correct kernel weights
  - Feature normalization scales distances to ~$\sqrt{D}$
  - Drift normalization produces $\mathbb{E}[\|V\|^2/D] = 1$
  - Loss stays ~constant while distribution evolves (as expected)
- **Success criteria:** 2D toy example converges to correct distribution with pure drifting loss

This sanity check ensures our understanding is correct before scaling to robotics.

**Days 9-10: VLM Integration (Single Task)**
- Implement `VLMBackbone` with pre-computed feature loading
- Replace DINOv2+CLIP in current codebase
- Train on RLBench close_jar only (15K samples, 1 task)
- **Try pure drifting first** (paper's approach) at batch 512
- If it doesn't converge, add hybrid loss
- **Target:** Success rate > 70% in 5K steps (~1 hour)
- **Success criteria:** Model learns to grasp (qualitative video check)

**Days 11-12: Action Space Unification**
- Implement 128-dim action mapping
- Implement action mask in loss computation
- Verify masked MSE works correctly
- Test on RLBench (8-dim gripper in 128-dim space)

**Days 13-14: Ablation (VLM Backbone)**
- Compare PaliGemma2 vs Qwen3-VL features
- Measure MSE convergence speed, final success rate
- **Deliverable:** Table showing VLM > DINOv2+CLIP

---

### Phase 2: Multi-Embodiment (Weeks 3-4)

**Days 15-17: DexGraspNet Integration**
- Implement `DexGraspNetDataset` (point cloud → rendered images)
- Implement 23-dim action mapping (wrist + joints)
- Pre-compute VLM features for DexGraspNet
- Verify data loading works

**Days 18-21: Joint Training (2 Embodiments)**
- Train on RLBench (gripper) + DexGraspNet (dex hand) jointly
- Sampling weights: 50% each
- 20K steps (~7 hours on 8×B200)
- **Target:** Both losses decrease; gripper MSE < 0.1, dex MSE < 0.2
- **Success criteria:** No catastrophic interference between embodiments

**Days 22-24: Embodiment Token Ablation**
- Train with vs without embodiment embedding
- Measure per-embodiment success rates
- **Deliverable:** Proof that embodiment conditioning helps

**Days 25-28: Add Bimanual (ALOHA)**
- Implement ALOHA dataset loader
- 16-dim action mapping
- Train on 3 embodiments jointly
- **Target:** All three losses decrease
- **Deliverable:** First unified gripper+bimanual+dex model

---

### Phase 3: Scale-Up & Pre-training (Weeks 5-7)

**Days 29-32: Full Dataset Integration**
- Add Bridge V2 and RT-1 datasets
- Implement delta→absolute action conversion
- Tune sampling weights (start with uniform, adjust based on loss)
- **Target:** All 5 datasets load without errors

**Days 33-42: Large-Scale Pre-training**
- Train on all datasets for 100K steps
- Effective batch 2,048
- Monitor per-dataset losses on WandB
- Checkpoint every 5K steps
- **Time:** ~40 hours (~2 days)
- **Target:** All dataset losses converge; no single dataset dominates

**Days 43-49: Hyperparameter Tuning**
- Tune $\lambda_{\text{drift}} \in \{0.05, 0.1, 0.2\}$
- Tune sampling weights based on loss curves
- Tune n_pos_samples, n_neg_samples
- **Deliverable:** Optimal hyperparameters for final training

---

### Phase 4: Evaluation & Baselines (Weeks 8-9)

**Days 50-52: RLBench Evaluation**
- Evaluate on 18 tasks × 25 episodes
- Compare vs RDT-1B, Diffusion Policy, 3D Diffuser Actor
- Measure success rate and inference latency
- **Target:** >80% success, <50ms latency

**Days 53-55: DexGraspNet Evaluation**
- Set up IsaacGym with Allegro hand
- Evaluate on test scenes
- Compare vs DexGraspNet 2.0 baseline
- **Target:** >70% grasp success

**Days 56-58: ManiSkill Evaluation**
- Set up ManiSkill environment
- Evaluate on 5 tasks × 250 trials
- Compare vs RDT-1B (53.6%), Diffusion Policy (30.2%)
- **Target:** >50% average success

**Days 59-63: Baseline Implementations**
- Implement Diffusion Policy on our data (if needed)
- Run RDT-1B evaluation (use their released checkpoint)
- Ensure fair comparison (same data, same evaluation protocol)

---

### Phase 5: Ablations & Analysis (Weeks 10-11)

**Days 64-66: Architecture Ablations**
- VLM backbone: PaliGemma2 vs Qwen3-VL vs DINOv2+CLIP
- DiT depth: 6L vs 12L vs 24L
- Action space: 64-dim vs 128-dim vs 256-dim

**Days 67-69: Loss Ablations (Critical for Validating Our Modifications)**

**Experiment 1: Pure Drifting (Paper) vs Hybrid (Ours) at Different Batch Sizes**

| Config | Batch Size | Negatives | Loss | Expected Result |
|--------|-----------|-----------|------|----------------|
| Paper baseline | 2048 (8 GPU) | x.detach() | Pure drifting | Baseline performance |
| Our modification | 2048 (8 GPU) | neg_queue | Hybrid MSE+drift | Similar or slightly better |
| Paper at small batch | 512 (2 GPU) | x.detach() | Pure drifting | Poor (noisy field) |
| Our fix | 512 (2 GPU) | neg_queue | Hybrid MSE+drift | Recovers performance |

**Experiment 2: Negative Sampling Strategy**

Fix batch=2048, vary negative source:
- x.detach() (paper): Baseline
- neg_queue (ours): Hypothesis: similar at large batch, better at small batch
- Random noise: Control (should be worst)

**Experiment 3: Loss Component Weights**

Fix batch=2048, vary $\lambda_{\text{drift}}$:
- $\lambda = 0$: MSE only (no drifting)
- $\lambda = 0.1$: Our default
- $\lambda = 1.0$: Pure drifting (paper)
- $\lambda = 0.05, 0.2$: Intermediate

Measure: Success rate + action diversity (GMM entropy)

**Days 70-72: Multi-Embodiment Analysis**
- Joint training vs separate per-embodiment models
- Transfer learning: pre-train on gripper → fine-tune on dex hand
- Embodiment token ablation

**Days 73-77: Visualization & Analysis**
- Drifting field evolution over training
- Action distribution diversity (GMM analysis)
- Failure case analysis per embodiment
- **Deliverable:** 10+ plots for paper

---

### Phase 6: Paper Writing (Weeks 12-14)

**Days 78-82: Drafting**
- Abstract (300 words)
- Introduction (2 pages)
- Related Work (1.5 pages)
- Method (4 pages): VLM backbone, Drifting DiT, unified action space, hybrid loss
- **Deliverable:** Draft sections

**Days 83-87: Experiments Section**
- Main results tables (RLBench, DexGraspNet, ManiSkill)
- Ablation tables
- Inference latency comparison
- Qualitative results (robot execution frames)
- **Deliverable:** Complete results section

**Days 88-91: Figures & Polish**
- Architecture diagram
- Training dynamics plots
- Drifting field visualization
- Success rate bar charts
- **Deliverable:** All figures camera-ready

**Days 92-94: Supplementary Materials**
- Implementation details
- Additional ablations
- Failure case analysis
- Hyperparameter tables
- **Deliverable:** 20+ page appendix

**Days 95-97: Final Review**
- Co-author feedback
- Proofreading
- NeurIPS format compliance
- **Deliverable:** Submission-ready PDF

---

## 9.5 Lessons Learned: Common Pitfalls When Implementing Drifting Models

During our development of Drifting-VLA v1, we encountered several implementation issues that led to misunderstandings of the drifting paradigm. We document these to help future implementers:

### Pitfall 1: "Loss Not Decreasing = Model Not Learning" ❌

**Misconception:** With drift normalization, $\mathcal{L}_{\text{drift}} \approx D$ (constant). We initially thought this meant the model wasn't learning and added MSE loss to "fix" it.

**Reality:** Constant loss is **correct and expected** per the paper. The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ provides learning signal through V's **direction**, not magnitude. Convergence is measured by task metrics (success rate), not loss value.

**Resolution:** Track task performance (success rate, FID) as the primary metric. Optionally track $\lambda_V$ (the normalization factor before scaling) which does decrease.

### Pitfall 2: Using x.detach() as Negatives at Small Batch ❌

**Misconception:** The paper uses `y_neg = x.detach()`, so we did the same with batch_size=32.

**Reality:** With batch=32, the 32 negatives are nearly identical to the 32 queries → drifting field V ≈ 0. The paper uses batch=8192 where 8192 negatives provide sufficient diversity.

**Resolution:** For batch <1024, use a negative sample queue storing predictions from previous steps. For batch ≥2048, x.detach() likely suffices (ablate in Phase 5).

### Pitfall 3: Incorrect Action Space Alignment ❌

**Misconception:** Dataset stores absolute poses, simulator uses delta actions (or vice versa).

**Reality:** Misalignment causes the robot to not move (path planning fails, or moves to wrong location). Must verify dataset format matches simulator's action_mode.

**Resolution:** Explicitly check: print first action, verify robot moves in simulation, compare GT trajectory vs predicted trajectory in 3D plots.

### Pitfall 4: Forgetting to Normalize Actions Per-Dimension ❌

**Misconception:** Actions are already in a reasonable range, no normalization needed.

**Reality:** Position (meters, ~0.3-1.5) dominates rotation (quaternion, ~0-1) and gripper (binary) in MSE. The model fits position well but rotation poorly.

**Resolution:** Per-dimension normalization to zero-mean unit-variance. This makes all dims equally important in the loss.

### Pitfall 5: Single Camera Provides Insufficient Information ❌

**Misconception:** DINOv2 is so powerful that one camera view is enough.

**Reality:** Manipulation tasks require depth/proximity cues. Single front camera can't infer 3D workspace. Success rates are 20-30% lower than multi-view baselines (3D Diffuser Actor, RDT).

**Resolution:** Use ≥2 cameras (front + wrist). Multi-view is standard in all SOTA manipulation models.

These lessons inform our phased development plan — Phase 1 (toy case) catches these issues early before committing to expensive large-scale training.

---

## 10. Risk Analysis & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| VLM features too generic for manipulation | Medium | High | Add LoRA fine-tuning; ablate against DINOv2 |
| Drifting loss doesn't converge with 128-dim actions | Low | High | Hybrid MSE already works; increase batch if needed |
| Multi-embodiment interference | Medium | Medium | Tune sampling weights; add per-embodiment batch norm |
| DexGraspNet point clouds hard to render | Low | Medium | Use Open3D rendering; or add PointNet branch |
| 100K steps insufficient for convergence | Medium | Medium | Extend to 200K if needed (~4 days) |

### 10.2 Computational Risks

| Risk | Mitigation |
|------|------------|
| B200 node unavailable | Fall back to 8×H100 (2× slower, still ~4 days) |
| OOM with batch 128 per GPU | Reduce to 64, increase grad_accum to 4 |
| VLM feature pre-compute takes too long | Parallelize across 8 GPUs (~2 hours total) |

### 10.3 Timeline Risks

| Risk | Mitigation |
|------|------------|
| Phase 2 takes longer than 2 weeks | Reduce scope: train only gripper + dex hand (skip bimanual) |
| Baseline comparisons delayed | Use RDT's released checkpoints; skip re-training baselines |
| Paper writing takes >3 weeks | Start writing introduction/related work during Phase 4 |

---

## 11. Success Criteria

### 11.1 Minimum Viable Product (MVP)

By end of Week 7 (Phase 3):
- ✅ Model trains on 3+ datasets without errors
- ✅ MSE converges to < 0.15 on all datasets
- ✅ Gripper success > 70% on RLBench
- ✅ Dex hand grasp success > 60% on DexGraspNet
- ✅ Inference latency < 50ms

### 11.2 Target for Paper Submission

By end of Week 11 (Phase 5):
- ✅ RLBench success > 80% (within 5% of RDT-1B)
- ✅ ManiSkill success > 50% (within 5% of RDT-1B)
- ✅ DexGraspNet success > 70%
- ✅ 10× faster inference than RDT (verified)
- ✅ All ablations complete
- ✅ Qualitative videos showing robot execution

### 11.3 Stretch Goals

- Bimanual manipulation benchmark (if time permits)
- Real-world deployment on physical robot
- Open-source model release (HuggingFace)

---

## 12. Budget Estimate

### 12.1 Compute Costs

**Primary configuration: 64×H100 (8 nodes)**

| Phase | GPU-Hours (64×H100) | Wall Time | Cost (@$2/GPU-hr) |
|-------|---------------------|-----------|-------------------|
| Phase 0 (setup + pre-compute) | 64 × 4 hrs = 256 | 4 hrs | $512 |
| Phase 1 (toy case) | 64 × 2 hrs = 128 | 2 hrs | $256 |
| Phase 2 (multi-embodiment) | 64 × 12 hrs = 768 | 12 hrs | $1,536 |
| Phase 3 (pre-training, 100K steps) | 64 × 22 hrs = 1,408 | 22 hrs | $2,816 |
| Phase 4 (evaluation) | 8 × 20 hrs = 160 | 20 hrs | $320 |
| Phase 5 (ablations, 5 runs) | 64 × 50 hrs = 3,200 | 50 hrs | $6,400 |
| **Total** | **5,920** | **~5 days active** | **$11,840** |

**Fallback: 8×B200 (single node)**

| Phase | GPU-Hours | Wall Time | Cost (@$7.50/hr) |
|-------|-----------|-----------|-------------------|
| Phase 0-5 combined | 8 × 200 hrs = 1,600 | ~8 days | $12,000 |

### 12.2 Personnel

- 1 PhD student (full-time, 14 weeks)
- 1 Advisor (guidance, 2 hours/week)
- Optional: 1 Undergraduate for data preprocessing

### 12.3 Total Budget

- Compute (64×H100): ~$12,000
- Storage (datasets + features): ~$200 (1 TB cloud)
- Personnel: Covered by lab funding
- **Total: ~$12,200**

---

## 13. Conclusion

Drifting-VLA v2 represents a principled approach to unifying multi-embodiment robotic manipulation through one-step action generation. By combining frozen VLM backbones (for rich visual-language understanding) with Drifting-based action generation (for fast, multi-modal inference), we address key limitations of existing foundation models:

1. **Speed:** 10× faster than diffusion-based policies (RDT-1B, Diffusion Policy)
2. **Generality:** First model to handle gripper, bimanual, and dexterous hand in one architecture
3. **Efficiency:** 127M trainable parameters vs RDT's 1.2B (9× fewer)
4. **Scalability:** Proven training pipeline from toy case (1 task, 1 hour) to full scale (5 datasets, 2 days)

The phased development plan ensures early validation at each stage, minimizing risk of wasted compute on a broken pipeline. With **64×H100 GPUs** (effective batch 8192, matching the Drifting paper exactly) and **14 weeks**, this project is **feasible and ambitious**, targeting a high-quality NeurIPS 2026 submission that advances the state-of-the-art in robotic foundation models.

The codebase supports both 64×H100 (8 nodes) and 8×B200 (1 node) configurations, with automatic adjustment of batch size, gradient accumulation, and loss formulation. The 64×H100 configuration enables faithful reproduction of the Drifting paper's training regime while extending to multi-embodiment robotics.

---

## References

1. Drifting: One-Step Generation via Training-Time Distribution Evolution. arXiv:2602.04770, 2025.
2. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864, 2024. [GitHub](https://github.com/thu-ml/RoboticsDiffusionTransformer)
3. DexGraspNet 2.0: Learning Generative Dexterous Grasping. CoRL 2024. [GitHub](https://github.com/PKU-EPIC/DexGraspNet2)
4. PaliGemma 2: A Family of Versatile VLMs for Transfer. arXiv:2412.03555, 2024.
5. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864, 2023.
6. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023.
7. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations. arXiv:2402.10885, 2024.




**Technical Proposal for NeurIPS 2026 Submission**

---

## Abstract

We propose **Drifting-VLA v2**, a unified vision-language-action foundation model capable of controlling multiple robot embodiments (parallel grippers, bimanual arms, and dexterous hands) through **one-step action generation**. Building upon the recently proposed Drifting model paradigm, which achieves state-of-the-art results in generative modeling (ImageNet FID 1.54) and robotic control, we integrate a frozen Vision-Language Model (VLM) backbone with a Drifting-based action generation head. This architecture enables: (1) **10× faster inference** compared to diffusion-based policies (1 forward pass vs. 100 denoising steps), (2) **unified handling** of heterogeneous embodiments through a fixed-dimensional action space with learned masking, and (3) **superior visual grounding** via joint vision-language reasoning from pre-trained VLMs. We validate our approach on RLBench (parallel gripper), DexGraspNet 2.0 (dexterous hand), and bimanual manipulation benchmarks, demonstrating that training-time distribution evolution via drifting fields provides a scalable alternative to inference-time iterative generation.

**Keywords:** Vision-Language-Action Models, Drifting Models, Multi-Embodiment Manipulation, Foundation Models, One-Step Generation

---

## 1. Introduction

### 1.1 Motivation

Recent advances in robotic manipulation have been driven by two parallel trends: (1) **foundation models** that leverage large-scale pre-training across diverse datasets and embodiments [RT-X, RDT-1B, OpenVLA], and (2) **generative action models** that capture multi-modal action distributions via diffusion or flow-based methods [Diffusion Policy, 3D Diffuser Actor]. However, existing approaches face a fundamental trade-off:

- **Diffusion-based policies** (RDT-1B, Diffusion Policy) achieve strong performance but require 50-100 denoising steps at inference, limiting control frequency to 2-10 Hz.
- **Autoregressive VLAs** (OpenVLA, RT-2) generate actions in one pass but struggle with multi-modal distributions and show poor generalization on manipulation benchmarks (4.8% success on ManiSkill vs. RDT's 53.6%).
- **Existing foundation models** (RDT-1B) support only parallel grippers and bimanual arms, lacking dexterous hand capabilities critical for complex manipulation.

The recently proposed **Drifting model** [arxiv:2602.04770] offers a compelling alternative: by evolving the generator's output distribution toward the data distribution during training (via a kernelized drifting field), the model learns to generate high-quality samples in **one forward pass** at inference. This paradigm has achieved state-of-the-art results in image generation (FID 1.54 on ImageNet) and robotic control tasks, demonstrating 100× fewer function evaluations than diffusion models while maintaining comparable performance.

### 1.2 Research Questions

This proposal addresses the following questions:

1. **Can drifting models scale to multi-embodiment manipulation?** The drifting paper demonstrates single embodiments (parallel gripper) on single tasks. We investigate whether a unified drifting architecture can handle gripper, bimanual, and dexterous hand control simultaneously across dozens of tasks.

2. **Do VLM backbones improve manipulation performance?** We hypothesize that joint vision-language reasoning from pre-trained VLMs (PaliGemma2, Qwen3-VL) provides better scene understanding than separate vision (DINOv2/SigLIP) and language (CLIP/T5) encoders used in prior work.

3. **Can drifting models work at smaller batch sizes?** The drifting paper uses batch size 8192 (128 GPUs). We investigate modifications (negative sample queue, hybrid loss) that enable effective training with batch 512-2048 (1-8 GPUs), making drifting accessible without massive compute.

4. **How do drifting models compare to diffusion baselines?** We provide the first comprehensive comparison of drifting vs diffusion (RDT-1B, Diffusion Policy) on robotic benchmarks, measuring both success rate and inference latency across multiple embodiments.

### 1.3 Contributions

We extend the Drifting model paradigm from image generation to multi-embodiment robotic manipulation with the following novel contributions:

1. **VLM-Conditioned Drifting Architecture:** We are the first to integrate frozen Vision-Language Models (PaliGemma2-3B) as the perception backbone for drifting-based action generation. This two-system design (frozen VLM + trainable Drifting DiT) enables parameter-efficient scaling across embodiments while leveraging pre-trained visual-language understanding.

2. **Unified Action Space for Multi-Embodiment Control:** Extending RDT-1B's approach, we design a fixed 128-dimensional action representation with learned masking that unifies gripper (8-dim), bimanual (16-dim), and dexterous hand (23-dim) control in a single model — the first drifting-based model to support dexterous hands.

3. **Small-Batch Drifting via Hybrid Loss:** We propose two algorithmic modifications that enable drifting models to train effectively with batch sizes 512-2048 (vs. the paper's 8192):
   - **Negative Sample Queue:** Store past predictions to ensure negatives differ from current queries
   - **Hybrid MSE + Drifting Loss:** Add direct supervision (MSE) to stabilize learning when drifting field estimation is noisy

   These modifications are ablated in Phase 5 to determine if they remain beneficial at large batch (8×B200 GPUs).

4. **Multi-Dataset Pre-training Strategy:** We demonstrate training across 5 heterogeneous datasets (RLBench, DexGraspNet 2.0, Bridge, ALOHA, RT-1) with different embodiments, action representations (absolute vs delta), and image formats, using per-dataset normalization and weighted sampling.

5. **Comprehensive Empirical Validation:** We evaluate on RLBench (18 tasks, gripper), DexGraspNet 2.0 (dexterous grasping), and ManiSkill (bimanual), providing the first comprehensive benchmark of drifting models for robotic manipulation across multiple embodiments.

6. **Open-Source Implementation:** Production-ready PyTorch codebase with Docker containers (CoppeliaSim, IsaacGym), WandB logging with 10+ visualization panels, simulation evaluation pipelines, and detailed documentation for reproducibility.

---

## 2. Related Work

### 2.1 Vision-Language-Action Models

**Autoregressive VLAs** (RT-1, RT-2, OpenVLA) treat robot control as a sequence prediction problem, tokenizing actions and generating them autoregressively conditioned on vision and language. While these models benefit from pre-trained LLM backbones, they struggle with continuous action spaces and multi-modal distributions. OpenVLA achieves only 4.8% average success on ManiSkill benchmarks despite 7B parameters and training on 970K demonstrations.

**Diffusion-based VLAs** (RDT-1B, Diffusion Policy) model actions as samples from a learned distribution via iterative denoising. RDT-1B, a 1.2B parameter diffusion transformer pre-trained on 46 datasets, achieves 53.6% on ManiSkill but requires 100 denoising steps (~500ms on H100), limiting real-time control. Our work aims to match RDT's performance with 1-step generation (~50ms).

### 2.2 Drifting Models

The Drifting model [arxiv:2602.04770] introduces a training-time distribution evolution paradigm where the generator's output distribution $q_\theta = f_\theta \# p_{\text{prior}}$ is iteratively evolved toward the data distribution $p_{\text{data}}$ via a drifting field.

**Drifting Field Definition (Paper Eq. 5-8):**

The drifting field is defined as a weighted combination of sample differences:

$$V_{p,q}(x) = \mathbb{E}_{y^+ \sim p}\mathbb{E}_{y^- \sim q}\Big[\tilde{k}(x, y^+)\tilde{k}(x, y^-)(y^+ - y^-)\Big]$$

where $\tilde{k}(x, y)$ is a normalized kernel with double softmax normalization over both positive and negative samples jointly. This can be decomposed as:

$$V_{p,q}(x) = V_p^+(x) - V_q^-(x)$$

where the cross-normalized attraction and repulsion terms are:

$$V_p^+(x) = \mathbb{E}_{y^+}[\tilde{k}(x, y^+) y^+] \cdot \mathbb{E}_{y^-}[\tilde{k}(x, y^-)]$$

$$V_q^-(x) = \mathbb{E}_{y^-}[\tilde{k}(x, y^-) y^-] \cdot \mathbb{E}_{y^+}[\tilde{k}(x, y^+)]$$

Note the critical detail: each term is weighted by the *sum* of the other distribution's kernel weights.

**Training Objective (Paper Eq. 4):**

$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{\epsilon \sim p_\epsilon}[\|f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q}(f_\theta(\epsilon)))\|^2]$$

With drift normalization (Paper Eq. 23-25), the field is scaled such that $\mathbb{E}[\|V\|^2/D] = 1$, which makes the **loss ≈ D** (constant). This is by design — the loss value itself is not the convergence metric. Instead, convergence is measured by task-specific metrics (FID for images, success rate for robotics).

**Key Insight:** The gradient of the loss is:

$$\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f_\theta(\epsilon)$$

The stop-gradient on the target means the gradient pushes $f_\theta(\epsilon)$ in the direction of the drifting field $V$. Even though the loss magnitude stays constant (~D), the *direction* of $V$ changes during training, providing useful gradient signal that evolves the distribution toward equilibrium.

**Equilibrium Condition:** When $p = q$ (the pushforward distribution matches data), the drifting field becomes zero: $V_{p,p}(x) = 0$ for all $x$, by anti-symmetry.

**Empirical Results:** Prior work demonstrated FID 1.54 on ImageNet 256×256 (latent-space generation) and FID 1.61 (pixel-space), achieving state-of-the-art among one-step methods. For robotics, the paper shows competitive performance on manipulation tasks (ToolHang, BlockPush) with 1-NFE vs Diffusion Policy's 100-NFE.

### 2.3 Multi-Embodiment Manipulation

**RDT-1B** addresses multi-embodiment control via a fixed 128-dimensional action space where different embodiments occupy different slots (with zero-padding for unused dimensions). This design enables a single DiT backbone to handle grippers and bimanual arms. However, RDT lacks dexterous hand support and relies on 100-step diffusion.

**DexGraspNet 2.0** tackles dexterous grasping in cluttered scenes using point-cloud-based graspness prediction and diffusion-based grasp pose generation. The model outputs wrist pose (7-dim) + joint angles (16-dim) for Allegro/LEAP hands, achieving strong grasping performance in simulation. However, it is specialized for grasping and does not generalize to full manipulation tasks.

Our work combines the best of both: RDT's unified action space design with Drifting's one-step generation, extended to dexterous hands via integration with DexGraspNet 2.0 data.

---

## 3. Method

### 3.0 Relationship to Original Drifting Model

**What we preserve from [arxiv:2602.04770]:**
- ✅ Core drifting field formulation: $V_{p,q}(x) = \mathbb{E}[\tilde{k}(x,y^+)\tilde{k}(x,y^-)(y^+-y^-)]$
- ✅ Double softmax normalized kernels
- ✅ Multi-temperature kernels ($\tau \in \{0.02, 0.05, 0.2\}$)
- ✅ Feature normalization (scale to $\sqrt{D}$ average distance)
- ✅ Drift normalization (scale to $\mathbb{E}[\|V\|^2/D] = 1$)
- ✅ Training objective: $\mathcal{L} = \|x - \text{sg}(x + V)\|^2$
- ✅ One-step inference paradigm
- ✅ DiT architecture with adaLN-Zero
- ✅ EMA for evaluation

**What we extend/modify:**
- 🔧 **VLM conditioning:** Paper uses class labels; we use PaliGemma2 VLM features for vision-language tasks
- 🔧 **Multi-embodiment:** Paper shows single embodiment; we extend to gripper/bimanual/dex hand
- 🔧 **Negative sampling:** Paper uses x.detach() with batch 8192; we add neg_queue for batch 512-2048
- 🔧 **Hybrid loss:** Paper uses pure drifting; we add MSE component for small-batch stability (ablate in Phase 5)
- 🔧 **Action space:** Paper uses 7-10 dim actions; we use 128-dim unified space with masking
- 🔧 **Multi-dataset training:** Paper trains per-task; we jointly train across 5 datasets

**Validation Strategy:** In Phase 1 (toy case), we first reproduce the paper's pure drifting loss on a single task to verify our implementation is correct. Then in Phase 5 (ablations), we determine which modifications are necessary vs optional.

### 3.1 Problem Formulation

We formulate multi-embodiment manipulation as learning a conditional distribution $p_\theta(\mathbf{a} | \mathcal{I}, \mathcal{L}, e)$ where:
- $\mathcal{I} \in \mathbb{R}^{V \times 3 \times H \times W}$: Multi-view RGB images
- $\mathcal{L}$: Natural language task description
- $e \in \{0, 1, 2\}$: Embodiment type (gripper, bimanual, dexterous hand)
- $\mathbf{a} \in \mathbb{R}^{T \times 128}$: Action sequence (T timesteps, 128-dim fixed space)

The model must satisfy:
1. **One-step generation:** $\mathbf{a} = f_\theta(\epsilon, \mathcal{I}, \mathcal{L}, e)$ where $\epsilon \sim \mathcal{N}(0, I)$
2. **Embodiment-specific constraints:** Active action dimensions determined by $e$
3. **Multi-modality:** Capture diverse valid action modes (e.g., grasp from left or right)

### 3.2 Architecture Overview

#### 3.2.0 Visual Architecture Diagrams

**Figure 1: Complete System Architecture**

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         DRIFTING-VLA v2 ARCHITECTURE                            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                          SYSTEM 1: VLM BACKBONE                          │  │
│  │                        (Frozen + Optional LoRA)                          │  │
│  │                                                                           │  │
│  │   ┌──────────┐    ┌──────────┐                                          │  │
│  │   │ Image 1  │    │ Image 2  │   "close the red jar"                    │  │
│  │   │ (front)  │    │ (wrist)  │          ↓                                │  │
│  │   │ 448×448  │    │ 448×448  │    ┌──────────┐                          │  │
│  │   └────┬─────┘    └────┬─────┘    │Tokenizer │                          │  │
│  │        │               │           └────┬─────┘                          │  │
│  │        └───────┬───────┘                │                                │  │
│  │                ↓                         ↓                                │  │
│  │   ┌──────────────────────────────────────────────────┐                  │  │
│  │   │         PaliGemma2-3B (3.0B params)               │                  │  │
│  │   │                                                    │                  │  │
│  │   │   SigLIP-So400m        Gemma-2B                  │                  │  │
│  │   │   Vision Encoder       Language Model            │                  │  │
│  │   │   (400M params)        (2.0B params)             │                  │  │
│  │   │         ↓                    ↓                    │                  │  │
│  │   │   ┌─────────────────────────────────┐            │                  │  │
│  │   │   │  Joint Vision-Language Fusion   │            │                  │  │
│  │   │   │  (Transformer Layers)            │            │                  │  │
│  │   │   └─────────────┬───────────────────┘            │                  │  │
│  │   │                 ↓                                  │                  │  │
│  │   │         Hidden States h_VLM                       │                  │  │
│  │   │         [L, 2048] where L ≈ 1025 + L_lang        │                  │  │
│  │   └──────────────────────┬────────────────────────────┘                  │  │
│  │                          ↓                                                │  │
│  │              ┌───────────────────────┐                                   │  │
│  │              │   Feature Projector   │                                   │  │
│  │              │   Linear: 2048 → 768  │                                   │  │
│  │              │   LayerNorm           │                                   │  │
│  │              └───────────┬───────────┘                                   │  │
│  │                          ↓                                                │  │
│  │              c_seq: [L, 768]   c_pool: [768]                            │  │
│  └──────────────────────────┼──────────────────────────────────────────────┘  │
│                             ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    SYSTEM 2: DRIFTING DiT ACTION HEAD                    │  │
│  │                         (Trainable: 123M params)                          │  │
│  │                                                                           │  │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │  │
│  │   │   Noise ε    │     │ Embodiment   │     │  CFG Scale   │           │  │
│  │   │ [T,64]       │     │   ID (0/1/2) │     │   α ~ p(α)   │           │  │
│  │   │ N(0,I)       │     │              │     │              │           │  │
│  │   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘           │  │
│  │          ↓                     ↓                     ↓                    │  │
│  │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │  │
│  │   │Noise→Token  │     │  Embedding   │     │  MLP: 1→768  │           │  │
│  │   │MLP: 64→768  │     │  [3, 768]    │     │              │           │  │
│  │   └──────┬───────┘     └──────┬───────┘     └──────┬───────┘           │  │
│  │          ↓                     └────────┬────────────┘                   │  │
│  │   [T, 768] + Pos_Embed                 ↓                                │  │
│  │       noise_tokens             c_global = c_pool + e_emb + cfg_emb     │  │
│  │          ↓                               ↓                               │  │
│  │   ┌──────────────────────────────────────────────────┐                 │  │
│  │   │     Cross-Attention Layers (×2)                   │                 │  │
│  │   │                                                    │                 │  │
│  │   │     Q = noise_tokens                              │                 │  │
│  │   │     K, V = c_seq (VLM features)                  │                 │  │
│  │   │                                                    │                 │  │
│  │   │     Attn(Q,K,V) allows noise to attend to        │                 │  │
│  │   │     fine-grained visual-language features        │                 │  │
│  │   └──────────────────┬───────────────────────────────┘                 │  │
│  │                      ↓                                                   │  │
│  │           fused_tokens: [T, 768]                                        │  │
│  │                      ↓                                                   │  │
│  │   ┌─────────────────────────────────────────────────────────┐          │  │
│  │   │           DiT Transformer Blocks (×12)                   │          │  │
│  │   │                                                           │          │  │
│  │   │   Each block:                                            │          │  │
│  │   │   ┌──────────────────────────────────────┐              │          │  │
│  │   │   │ 1. adaLN-Zero Normalization          │              │          │  │
│  │   │   │    (conditioned on c_global)         │              │          │  │
│  │   │   │ 2. Self-Attention (RoPE + QK-Norm)   │              │          │  │
│  │   │   │    with Flash Attention (bf16)       │              │          │  │
│  │   │   │ 3. adaLN-Zero Normalization          │              │          │  │
│  │   │   │ 4. SwiGLU MLP (ratio 4.0)            │              │          │  │
│  │   │   └──────────────────────────────────────┘              │          │  │
│  │   │                                                           │          │  │
│  │   │   Hidden: 768-dim, Heads: 12, Params: ~9M per block    │          │  │
│  │   └───────────────────────┬─────────────────────────────────┘          │  │
│  │                           ↓                                              │  │
│  │              transformer_out: [T, 768]                                  │  │
│  │                           ↓                                              │  │
│  │   ┌───────────────────────────────────────────────────┐                │  │
│  │   │            Unified Action Head                     │                │  │
│  │   │                                                     │                │  │
│  │   │   MLP: 768 → 768 → 128                            │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   a_raw: [T, 128] (all dims)                      │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   Apply action_mask (zero inactive dims)          │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   Normalize quaternions (dims 3-6, etc.)          │                │  │
│  │   │   ↓                                                 │                │  │
│  │   │   a_final: [T, 128]                               │                │  │
│  │   │                                                     │                │  │
│  │   │   Active dims vary by embodiment:                 │                │  │
│  │   │   • Gripper: dims 0-7 (8 active)                  │                │  │
│  │   │   • Bimanual: dims 0-15 (16 active)               │                │  │
│  │   │   • Dex Hand: dims 0-6, 16-31 (23 active)         │                │  │
│  │   └───────────────────────────────────────────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Total Trainable Parameters: 123.6M                                            │
│  GPU Memory (bf16): ~250 MB (model) + batch activations                       │
│  Inference Latency: ~50ms on B200 (1 VLM forward + 1 DiT forward)             │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Figure 2: Training Pipeline Flow**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING PIPELINE                                  │
│                                                                               │
│  OFFLINE (One-time pre-processing):                                          │
│  ┌─────────────┐                                                             │
│  │  Raw Data   │  →  Resize to 448×448  →  Tokenize lang  →  [images, text]│
│  │ (5 datasets)│                                                             │
│  └──────┬──────┘                                                             │
│         ↓                                                                     │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │  VLM Feature Extraction (PaliGemma2-3B)               │                   │
│  │  • Process each sample once                           │                   │
│  │  • Extract hidden_states[-1]: [L, 2048]              │                   │
│  │  • Save to HDF5 (fp16 compressed)                     │                   │
│  │  • Time: ~2 hours for 1M samples on 8 GPUs           │                   │
│  │  • Storage: ~4 GB total                               │                   │
│  └──────────────────────┬───────────────────────────────┘                   │
│                         ↓                                                     │
│         [sample_id] → vlm_features.h5                                        │
│                                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                               │
│  ONLINE (Each training step):                                                │
│  ┌──────────────┐                                                            │
│  │ Sample batch │ ← Weighted sampling from [RLBench, DexGrasp, Bridge, ...]│
│  │ B = 128      │                                                            │
│  └──────┬───────┘                                                            │
│         ↓                                                                     │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │ Load from disk:                                      │                    │
│  │ • vlm_features[sample_ids]  → [B, L, 2048] fp16    │                    │
│  │ • actions[sample_ids]       → [B, T, 128] fp32     │                    │
│  │ • action_mask[embodiment]   → [128] bool            │                    │
│  │ • Conversion to GPU takes ~5ms                       │                    │
│  └─────────────────────┬───────────────────────────────┘                    │
│                        ↓                                                      │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Project VLM features                                     │               │
│  │  • Linear: 2048 → 768                                    │               │
│  │  • c_seq: [B, L, 768], c_pool: [B, 768]                │               │
│  └────────────────────────┬─────────────────────────────────┘               │
│                           ↓                                                   │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │  Model Forward                                              │             │
│  │  • Sample noise ε ~ N(0, I_{T×64})                         │             │
│  │  • Sample CFG α ~ p(α) ∝ α^{-3}                           │             │
│  │  • actions_pred = model(ε, c_seq, c_pool, e, α)          │             │
│  │  • Time: ~30ms on 8×B200 (FSDP parallel)                  │             │
│  └────────────────────────┬───────────────────────────────────┘             │
│                           ↓                                                   │
│  actions_pred: [B, T, 128]                                                   │
│         ↓                              ↓                                      │
│  Normalize actions_gt          Sample from queues                           │
│  using dataset stats            • pos_queue.sample(128)                     │
│         ↓                        • neg_queue.sample(128)                     │
│         ↓                              ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │  Compute Hybrid Loss                                     │                │
│  │  • MSE: ||pred - gt||² on active dims only             │                │
│  │  • Drifting: ||pred - (pred + V)||² where              │                │
│  │    V = attract(pos) - repel(neg)                        │                │
│  │  • L = MSE + 0.1 × Drifting                             │                │
│  └────────────────────┬────────────────────────────────────┘                │
│                       ↓                                                       │
│  L.backward() → Grad clip → Optimizer step → EMA update                     │
│         ↓                                                                     │
│  Update queues:                                                              │
│  • neg_queue.push(actions_pred.detach())  ← Store for next step            │
│  │ pos_queue.push(actions_gt)                                               │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Figure 3: Inference Pipeline Flow**

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE (Real-time)                           │
│                                                                         │
│  Camera Observations                 Task Instruction                  │
│  ┌────────┐  ┌────────┐             "grasp the red cup"               │
│  │ front  │  │ wrist  │                     ↓                          │
│  │448×448 │  │448×448 │                Tokenize                       │
│  └───┬────┘  └───┬────┘                     ↓                          │
│      └────────┬───┘                    [token_ids]                     │
│               ↓                              ↓                          │
│  ┌────────────────────────────────────────────────────┐                │
│  │         VLM Forward (Live, not pre-computed)        │                │
│  │         PaliGemma2-3B on GPU                        │                │
│  │         Time: ~40ms                                  │                │
│  └────────────────────┬───────────────────────────────┘                │
│                       ↓                                                 │
│          hidden_states: [L, 2048]                                      │
│                       ↓                                                 │
│  ┌────────────────────────────────┐                                    │
│  │  Project: 2048 → 768            │                                    │
│  └────────────────┬───────────────┘                                    │
│                   ↓                                                     │
│       c_seq: [L, 768], c_pool: [768]                                  │
│                   ↓                                                     │
│  ┌─────────────────────────────────────────────┐                       │
│  │  DiT Forward (Single Pass)                   │                       │
│  │  • Sample ε ~ N(0, I)                       │                       │
│  │  • Add embodiment_id, cfg_scale=2.0         │                       │
│  │  • Cross-attend to c_seq                    │                       │
│  │  • 12 DiT blocks                             │                       │
│  │  • Action head: 768 → 128                   │                       │
│  │  Time: ~10ms on B200                        │                       │
│  └────────────────┬────────────────────────────┘                       │
│                   ↓                                                     │
│       actions_pred: [T, 128] (normalized space)                        │
│                   ↓                                                     │
│  ┌───────────────────────────────────┐                                 │
│  │  Denormalize using dataset stats   │                                 │
│  │  a = a_norm * std + mean           │                                 │
│  └─────────────────┬─────────────────┘                                 │
│                    ↓                                                    │
│       actions: [T, 128] (absolute world coordinates)                   │
│                    ↓                                                    │
│  ┌────────────────────────────────────────┐                            │
│  │  Extract embodiment-specific action     │                            │
│  │  • Gripper: dims[0:8]                  │                            │
│  │  • Bimanual: dims[0:16]                │                            │
│  │  • Dex hand: dims[0:7, 16:32]          │                            │
│  └────────────────┬───────────────────────┘                            │
│                   ↓                                                     │
│       action_t: [8] or [16] or [23]                                   │
│                   ↓                                                     │
│  ┌────────────────────────────────┐                                    │
│  │  Execute in Robot/Simulator     │                                    │
│  │  • RLBench: CoppeliaSim         │                                    │
│  │  • DexGrasp: IsaacGym           │                                    │
│  │  • Real: Physical robot         │                                    │
│  └────────────────────────────────┘                                    │
│                                                                         │
│  Total Latency: 40ms (VLM) + 10ms (DiT) = 50ms                        │
│  Control Frequency: 20 Hz (real-time capable)                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Figure 4: Data Processing Flow**

```
Raw Dataset → Standardization → Unified Format → VLM Features → Training

┌──────────────┐
│  RLBench     │  front_rgb [128×128] ──┐
│  close_jar   │  wrist_rgb [128×128] ──┼→ Resize → [2,3,448,448]
│  Episode 42  │  actions [T,8] abs     │           ↓
│  Timestep 5  │  language: "close jar" │    ┌──────────────┐
└──────────────┘                         │    │ Unified      │
                                         │    │ Sample       │
┌──────────────┐                         │    │              │
│ DexGraspNet  │  point_cloud [N,3] ────┼→ Render → [4,3,448,448]
│ Scene 1024   │  grasp_pose [23] abs   │    │ images: ✓    │
│ Grasp 7      │  (wrist + 16 joints)   │    │ actions: ✓   │
└──────────────┘                         │    │ language: ✓  │
                                         │    └──────┬───────┘
┌──────────────┐                         │           ↓
│  Bridge V2   │  wrist_img [256×256] ──┼→ Resize    Sample
│  Task 89     │  actions [T,7] DELTA   │    +       ↓
│  Episode 12  │  Integrate to absolute │    ┌──────────────────┐
│  Timestep 3  │  language: "pick cup"  │    │  PaliGemma2-3B   │
└──────────────┘                         │    │  (one forward)   │
                                         │    │  40ms on GPU     │
┌──────────────┐                         │    └──────┬───────────┘
│  ALOHA       │  4 cameras [480×640] ──┼→ Resize           ↓
│  Bimanual    │  actions [T,16] joint  │    vlm_features: [L,2048]
│  Task: fold  │  language: "fold ..."  │           ↓
└──────────────┘                         │    Save to HDF5
                                         │           ↓
All map to:                              │    ┌──────────────────┐
• Images: [V, 3, 448, 448]              │    │ Disk Storage     │
• Actions: [T, 128] (padded)            │    │ • images.h5      │
• Mask: [128] (active dims)             │    │ • actions.h5     │
• Embodiment: {0,1,2}                   │    │ • vlm_features.h5│
└─────────────────────────────────────────────┴──────────────────┘

During Training:
Load [images.h5, actions.h5, vlm_features.h5] → Batch → GPU → Model
```

### 3.2 Detailed Architecture Components

#### 3.2.1 System 1: VLM Backbone (Frozen + LoRA)

We leverage pre-trained Vision-Language Models to extract rich multimodal features that encode both visual scene understanding and language instruction grounding in a unified representation.

**Supported VLM Backbones:**

The codebase implements a unified `VLMBackbone` interface supporting both models:

| | `google/paligemma2-3b-mix-448` | `Qwen/Qwen3-VL-2B-Instruct` |
|---|---|---|
| **Architecture** | SigLIP-So400m + Gemma-2B | ViT + Qwen2.5-2B |
| **Total params** | 3.0B | 2.3B |
| **Vision encoder** | ViT-So400m/14 (400M) | ViT-600M with dynamic resolution |
| **Patch size** | 14×14 | 14×14 |
| **Image resolution** | 448×448 (fixed) | Dynamic (min 224, max 1280) |
| **Visual tokens per image** | 1025 (fixed) | Variable (~256-1024) |
| **Language model** | Gemma-2B | Qwen2.5-2B |
| **Hidden dimension** | 2048 | 1536 |
| **Instruction following** | Good | Excellent (instruct-tuned) |
| **Visual grounding** | Excellent | Good |
| **Multi-image** | Single image per forward | Native multi-image support |

**Default:** PaliGemma2-3B (best visual grounding for manipulation).

**Qwen3-VL-2B advantages:** Smaller (2.3B vs 3B), native multi-image support (no need for per-view independent forward passes), dynamic resolution (can process different-sized inputs efficiently).

**Implementation:**

```python
class VLMBackbone(nn.Module):
    """Unified VLM backbone supporting PaliGemma2 and Qwen3-VL."""
    
    SUPPORTED_MODELS = {
        'paligemma2': {
            'name': 'google/paligemma2-3b-mix-448',
            'hidden_dim': 2048,
            'image_size': 448,
            'multi_image_native': False,
        },
        'qwen3vl': {
            'name': 'Qwen/Qwen3-VL-2B-Instruct',
            'hidden_dim': 1536,
            'image_size': 448,  # Use 448 for consistency
            'multi_image_native': True,
        },
    }
    
    def __init__(self, model_key='paligemma2', dit_hidden_dim=768):
        super().__init__()
        cfg = self.SUPPORTED_MODELS[model_key]
        self.model_key = model_key
        self.vlm = AutoModelForCausalLM.from_pretrained(
            cfg['name'], torch_dtype=torch.bfloat16
        )
        self.vlm.eval()
        
        # Projection adapts to VLM's hidden dim
        self.proj_seq = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], dit_hidden_dim),
            nn.LayerNorm(dit_hidden_dim),
        )
        self.proj_pool = nn.Sequential(
            nn.Linear(cfg['hidden_dim'], dit_hidden_dim),
            nn.LayerNorm(dit_hidden_dim),
        )
    
    def forward(self, images, language_tokens, views_per_sample=None):
        """
        Args:
            images: [B, V, 3, 448, 448] multi-view images
            language_tokens: input_ids [B, L_lang]
            views_per_sample: [B] int tensor, number of views per sample
        Returns:
            c_seq: [B, L_total, dit_hidden_dim]
            c_pool: [B, dit_hidden_dim]
        """
        if self.model_key == 'qwen3vl':
            # Qwen3-VL: native multi-image in one forward
            return self._forward_qwen3vl(images, language_tokens, views_per_sample)
        else:
            # PaliGemma2: vision per view, language appended once
            return self._forward_paligemma2(images, language_tokens)
    
    def _forward_paligemma2(self, images, language_tokens):
        """
        PaliGemma2 multi-view processing:
        1. Run SigLIP vision encoder on each view independently
        2. Concatenate visual tokens from all views
        3. Append language tokens ONCE (not per view)
        4. Run through Gemma-2B language model for joint VL fusion
        """
        B, V, C, H, W = images.shape
        
        # Step 1: Extract visual tokens per view (SigLIP only)
        images_flat = images.view(B * V, C, H, W)
        with torch.no_grad():
            vision_outputs = self.vlm.vision_model(images_flat)
            visual_tokens = vision_outputs.last_hidden_state  # [B*V, 1025, 2048]
        
        # Step 2: Reshape and concatenate across views
        N_vis = visual_tokens.shape[1]  # 1025 per view
        visual_tokens = visual_tokens.view(B, V * N_vis, -1)  # [B, V*1025, 2048]
        
        # Step 3: Append language tokens (encoded once, not per view)
        lang_embeds = self.vlm.get_input_embeddings()(language_tokens)  # [B, L_lang, 2048]
        combined = torch.cat([visual_tokens, lang_embeds], dim=1)  # [B, V*1025+L_lang, 2048]
        
        # Step 4: Joint VL fusion through Gemma-2B layers
        h = self.vlm.language_model(inputs_embeds=combined).last_hidden_state
        
        # Project to DiT dim
        c_seq = self.proj_seq(h)
        c_pool = self.proj_pool(h.mean(dim=1))
        return c_seq, c_pool
```

Switching backbone at config level:

```yaml
# configs/model/drifting_v2.yaml
vlm:
  model_key: 'paligemma2'   # or 'qwen3vl'
```

**Multi-View Processing:**

For multi-view inputs $\mathcal{I} = \{I_1, ..., I_V\}$ where $V$ is the number of camera views:

```python
# Option A: Process each view independently, concatenate features
features = []
for view in images:  # images: [V, 3, 448, 448]
    h_v = vlm(pixel_values=view, input_ids=lang_tokens)  # [1025+L_lang, 2048]
    features.append(h_v)
features = torch.cat(features, dim=0)  # [V×(1025+L_lang), 2048]

# Option B: Tile images into 2×2 grid (for V=4 views)
# → Single VLM forward pass, spatial relationships preserved
grid = tile_images(images, layout=(2,2))  # [896, 896, 3] → rescale to 448×448
features = vlm(pixel_values=grid, input_ids=lang_tokens)  # [1025+L_lang, 2048]
```

**Multi-View Processing for PaliGemma2 (Vision-only per view, language appended once):**

Each camera view is processed through PaliGemma2's **vision encoder only** (SigLIP) independently. The visual tokens from all views are concatenated, then the **language tokens are appended once** at the end. This avoids redundantly encoding the same instruction V times:

```
View 1 (front)  ──→ SigLIP Vision Encoder ──→ v1: [1025, 2048]  (visual tokens)
View 2 (wrist)  ──→ SigLIP Vision Encoder ──→ v2: [1025, 2048]  (visual tokens)
                                                     ↓
                                    torch.cat([v1, v2], dim=0)
                                                     ↓
                                v_concat: [2050, 2048]  (all visual tokens)
                                                     ↓
                   Append language tokens: [v_concat ; lang_tokens]
                                                     ↓
                   h_all: [2050 + L_lang, 2048]  (joint VL representation)
                                                     ↓
                   Pass through Gemma-2B language model layers
                   (cross-attention between visual + language tokens)
                                                     ↓
                   h_final: [2050 + L_lang, 2048]  (fused multi-view + language)
```

**Why vision-only per view, language once:**
- SigLIP vision encoder processes each image independently (no cross-image attention)
- Language instruction is the SAME for all views — encoding it V times wastes compute
- Appending language after visual concatenation lets Gemma-2B jointly attend to ALL views + instruction
- The language model layers perform cross-attention between multi-view visual tokens and language tokens

**Why independent vision processing (not image tiling):**
- PaliGemma2's SigLIP was pre-trained on single images — tiling would break spatial attention patterns
- Each view has independent spatial structure (front = global scene, wrist = local detail)
- Independent processing preserves per-view spatial features
- The subsequent Gemma-2B layers learn to fuse multi-view information with language

**Handling Variable Number of Views:**

Datasets provide 1-4 cameras. Within a batch, samples may have different view counts:

```python
# During pre-computation: process each view independently
view_features = []
for v in range(V):  # V varies: RLBench=2, Bridge=1, ALOHA=3
    h_v = vlm(images[v], lang_tokens)   # [1025+L_lang, 2048]
    view_features.append(h_v)

h_concat = torch.cat(view_features, dim=0)  # [V×L_per_view, 2048]

# During training: pad to MAX_TOKENS and use attention mask
MAX_VLM_TOKENS = 4 * 1060  # Supports up to 4 views
h_padded = pad_to_length(h_concat, MAX_VLM_TOKENS)     # [MAX_VLM_TOKENS, 2048]
attn_mask = create_padding_mask(actual_length=V*L_per_view, max_length=MAX_VLM_TOKENS)
```

The cross-attention in the DiT naturally handles variable-length key/value sequences via the attention mask.

**Projection to DiT dimension:**

$$\mathbf{c}_{\text{seq}} = \text{LayerNorm}(\text{Linear}_{2048 \to 768}(\mathbf{h}_{\text{concat}})) \in \mathbb{R}^{L_{\text{total}} \times 768}$$

$$\mathbf{c}_{\text{pool}} = \text{LayerNorm}(\text{Linear}_{2048 \to 768}(\text{mean}(\mathbf{h}_{\text{concat}}, \text{dim}=0))) \in \mathbb{R}^{768}$$

**VLM Co-Training Strategy (Staged):**

We adopt a **staged training recipe** rather than keeping the VLM frozen throughout:

```
Stage 1 (Steps 0 → 5K):    VLM frozen, DiT trains from scratch
                             → Prevents random DiT gradients from corrupting VLM

Stage 2 (Steps 5K → 50K):  VLM LoRA unfreezes, co-trains with DiT
                             → VLM adapts to robot domain

Stage 3 (Steps 50K → 100K): Continue co-training, lower VLM LR by 2×
                             → Fine refinement
```

**Differential learning rates:**

```python
optimizer = AdamW([
    {'params': vlm_lora.parameters(), 'lr': 1e-5},      # VLM LoRA: 10× lower
    {'params': feature_proj.parameters(), 'lr': 4e-4},    # Projector: standard
    {'params': dit.parameters(), 'lr': 4e-4},              # DiT: standard
    {'params': action_head.parameters(), 'lr': 4e-4},      # Action head: standard
])
```

| Component | LR | Trainable Params | Stage Active |
|-----------|-----|-----------------|-------------|
| VLM base | 0 (frozen) | 0 | Always |
| VLM LoRA | 1e-5 | ~0.5M | Stage 2+ |
| Feature Projector | 4e-4 | 3.2M | Always |
| DiT + Action Head | 4e-4 | 123M | Always |

**LoRA Configuration:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    layers_to_transform=[-4, -3, -2, -1],  # Last 4 layers only
)

vlm = get_peft_model(vlm, lora_config)
# Trainable: 16 × 2 × 4 × 2048 ≈ 0.5M (0.017% of 3B)
```

**Training Phases and VLM Usage:**

| Phase | VLM Usage | Purpose |
|-------|-----------|---------|
| Phase 0 (Pre-compute) | Run once offline, save features to HDF5 | Fast iteration on DiT |
| Phase 1-2 (Toy/Multi-emb) | Load pre-computed features, VLM not in GPU | Validate architecture |
| Phase 3+ (Scale-up) | Load VLM live with LoRA, co-train | Adapt VLM to robot domain |
| Inference | Run VLM live (~40ms per observation) | Real-time control |

#### 3.2.2 System 2: Drifting DiT Action Generator

The action generation module is a Transformer-based architecture that processes random noise conditioned on VLM features to produce action sequences in a single forward pass.

**Architecture Overview:**

| Layer Type | Count | Hidden Dim | Num Heads | Parameters |
|------------|-------|------------|-----------|------------|
| Noise Tokenizer | 1 MLP (3-layer) | 64→768→768 | — | ~1.2M |
| Timestep Embedding | Learned | 768 | — | 16×768 = 12K |
| Embodiment Embedding | Learned | 768 | — | 3×768 = 2.3K |
| CFG Scale MLP | 2-layer | 1→768→768 | — | ~1.2M |
| VLM Sequence Projector | Linear + LN | 2048→768 | — | 1.6M |
| VLM Pooled Projector | Linear + LN | 2048→768 | — | 1.6M |
| Cross-Attention Layers | 2 | 768 | 12 | ~9.4M |
| DiT Blocks | 12 | 768 | 12 | ~108M |
| Action Head | MLP (2-layer) | 768→768→128 | — | ~0.6M |
| **Total** | | | | **~123.6M** |

**Input Representations:**

**1. Noise Tokenization:**

Random noise $\epsilon \sim \mathcal{N}(0, I_{T \times 64})$ is projected to token space:

$$\text{MLP}_{\text{noise}}(\epsilon) = \text{Linear}_3(\text{SiLU}(\text{Linear}_2(\text{SiLU}(\text{Linear}_1(\epsilon)))))$$

where $\text{Linear}_1: 64 \to 768$, $\text{Linear}_2: 768 \to 768$, $\text{Linear}_3: 768 \to 768$.

Then add learned positional embeddings:

$$\mathbf{x}_0 = \text{MLP}_{\text{noise}}(\epsilon) + \text{Embed}_{\text{pos}}(\text{arange}(T)) \in \mathbb{R}^{T \times 768}$$

**2. Global Conditioning Vector:**

The global conditioning aggregates task context, embodiment type, and CFG scale:

$$\mathbf{c}_{\text{global}} = \mathbf{c}_{\text{pool}} + \mathbf{e}_{\text{emb}} + \mathbf{c}_{\text{cfg}}$$

where:
- $\mathbf{c}_{\text{pool}} = \text{Proj}_{\text{pool}}(\text{mean}(\mathbf{h}_{\text{VLM}})) \in \mathbb{R}^{768}$ (task context)
- $\mathbf{e}_{\text{emb}} = \text{Embed}_e(\text{embodiment\_id}) \in \mathbb{R}^{768}$ (embodiment: 0/1/2)
- $\mathbf{c}_{\text{cfg}} = \text{MLP}_{\text{cfg}}(\alpha) \in \mathbb{R}^{768}$ where $\alpha$ is CFG scale

**Cross-Attention Layers:**

Two cross-attention layers allow noise tokens to attend to fine-grained VLM features:

$$\mathbf{Q} = \mathbf{x}_0 \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{c}_{\text{seq}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{c}_{\text{seq}} \mathbf{W}_V$$

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\mathbf{x}_1 = \mathbf{x}_0 + \text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$

This is repeated for 2 layers, with layer normalization and residual connections.

**DiT Transformer Blocks:**

Each of the 12 DiT blocks follows the adaLN-Zero design from "Scalable Diffusion Models with Transformers":

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \gamma_1 \cdot \text{Attn}(\text{Norm}(\mathbf{x}_i; \mathbf{c}_{\text{global}})) + \gamma_2 \cdot \text{MLP}(\text{Norm}(\mathbf{x}_i; \mathbf{c}_{\text{global}}))$$

where:
- $\text{Norm}(\mathbf{x}; \mathbf{c}) = \text{LayerNorm}(\mathbf{x}) \cdot (1 + \text{scale}(\mathbf{c})) + \text{shift}(\mathbf{c})$ (adaptive normalization)
- $\gamma_1, \gamma_2, \text{scale}, \text{shift}$ are predicted from $\mathbf{c}_{\text{global}}$ via small MLPs
- Self-attention uses RoPE (Rotary Position Embedding) for temporal modeling
- QK-Norm: $\mathbf{Q}' = \text{LayerNorm}(\mathbf{Q})$, $\mathbf{K}' = \text{LayerNorm}(\mathbf{K})$ before attention
- MLP uses SwiGLU activation: $\text{SwiGLU}(x) = (\mathbf{W}_1 x \odot \text{SiLU}(\mathbf{W}_2 x)) \mathbf{W}_3$

**Detailed DiTBlock Specification:**

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_ratio=4.0):
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)  # scale_1, shift_1, gate_1, scale_2, shift_2, gate_2
        )
        
        # Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)  # No affine (replaced by adaLN)
        self.attn = MultiheadAttention(hidden_dim, num_heads, qk_norm=True, rope=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio))
    
    def forward(self, x, c):
        # c: [B, 768] global conditioning
        # Predict modulation parameters
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Attention block
        x_norm = self.norm1(x) * (1 + scale_1.unsqueeze(1)) + shift_1.unsqueeze(1)
        x = x + gate_1.unsqueeze(1) * self.attn(x_norm)
        
        # MLP block
        x_norm = self.norm2(x) * (1 + scale_2.unsqueeze(1)) + shift_2.unsqueeze(1)
        x = x + gate_2.unsqueeze(1) * self.mlp(x_norm)
        
        return x
```

**Action Head:**

Final MLP projects transformer output to 128-dimensional action space:

$$\mathbf{a}_{\text{raw}} = \text{Linear}_2(\text{SiLU}(\text{Linear}_1(\mathbf{x}_L))) \in \mathbb{R}^{T \times 128}$$

where $\text{Linear}_1: 768 \to 768$ and $\text{Linear}_2: 768 \to 128$.

**Action Masking and Normalization:**

$$\mathbf{a}_{\text{masked}} = \mathbf{a}_{\text{raw}} \odot \mathbf{m}_e$$

where $\mathbf{m}_e \in \{0,1\}^{128}$ is the embodiment-specific mask. Inactive dimensions are forced to zero.

For quaternion dimensions (e.g., dims 3-6 for gripper wrist orientation):

$$\mathbf{q}_{\text{norm}}[j] = \frac{\mathbf{a}_{\text{masked}}[j]}{\sqrt{\sum_{k \in \text{quat\_dims}} \mathbf{a}_{\text{masked}}[k]^2} + \epsilon}$$

This ensures valid unit quaternions for end-effector orientations.

**Training efficiency:** VLM features are pre-computed offline and stored on disk. During training, the VLM is never loaded into GPU memory, saving ~6 GB per GPU. The DiT (123M params) + gradients + optimizer states fit comfortably in 192 GB, enabling batch_size=128 per GPU.

### 3.3 Unified Action Space Design

Following RDT-1B's proven approach, we adopt a fixed 128-dimensional action vector:

```
Dimensions:  [0-6]    [7]     [8-14]  [15]    [16-31]      [32-127]
             arm1_pose grip1   arm2    grip2   joints       padding
             
Gripper (8 active):
  [x, y, z, qx, qy, qz, qw, gripper_open, 0, ..., 0]
  mask = [1,1,1,1,1,1,1,1, 0,...,0]

Bimanual (16 active):
  [left_j1..j7, left_grip, right_j1..j7, right_grip, 0, ..., 0]
  mask = [1,...,1 (×16), 0,...,0]

Dexterous Hand (23 active):
  [wrist_x,y,z,qx,qy,qz,qw, 0, 0,...,0, joint1..joint16, 0,...,0]
  mask = [1,...,1 (×7), 0, 0,..., 1,...,1 (×16), 0,...,0]
```

**Rationale:** Zero-padding is computationally wasteful (dexterous hand uses only 18% of 128 dims) but simplifies the architecture significantly. RDT-1B demonstrated this approach scales to 46 datasets. The action mask ensures the loss and gradient only backpropagate through active dimensions, preventing the model from learning spurious correlations with padding.

### 3.4 Loss Function

#### 3.4.1 Pure Drifting Loss (Paper's Formulation)

The canonical drifting loss from [arxiv:2602.04770] is:

$$\mathcal{L}_{\text{drift}} = \mathbb{E}_{\epsilon}[\|f_\theta(\epsilon) - \text{sg}(f_\theta(\epsilon) + V_{p,q}(f_\theta(\epsilon)))\|^2]$$

where the drifting field $V_{p,q}$ is computed from positive samples $\{y^+\} \sim p_{\text{data}}$ and negative samples $\{y^-\} = \{f_\theta(\epsilon_i).detach()\}_{i=1}^B$ (predictions from the current batch, detached).

**Paper's Training Setup:**
- Effective batch size: **8192** (128 GPUs × 64 per GPU)
- Positive samples: 128 per class from sample queue (for conditional generation)
- Negative samples: Current batch predictions (8192 samples)
- The large batch provides sufficient diversity in negatives

**Key Property:** With drift normalization, the loss magnitude is approximately constant ($\mathcal{L} \approx D$ where $D$ is action dimensionality). This is **correct by design** — the loss value itself is not the convergence metric. Instead:
- The *direction* of the drifting field $V$ evolves during training
- Convergence is measured by task performance (success rate, FID, etc.)
- The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ pushes the model toward equilibrium

**Challenge for Single-GPU Training:** With batch size 32-128 (typical for single GPU), using $y^- = x.detach()$ causes the drifting field to have insufficient diversity. The 32 negative samples are too similar to the 32 query samples, leading to noisy field estimates.

#### 3.4.2 Our Modification: Hybrid MSE + Drifting Loss

To enable effective training on smaller batches (512-2048 vs paper's 8192), we propose a hybrid formulation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{drift}} \mathcal{L}_{\text{drift}}$$

**Component 1: Masked MSE Loss (Our Addition)**

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathbf{m}_e|} \sum_{i,j} \mathbf{m}_e[j] \cdot (\mathbf{a}_{\text{pred}}[i,j] - \mathbf{a}_{\text{gt}}[i,j])^2$$

where $|\mathbf{m}_e|$ is the number of active dimensions. This component:
- Provides a direct, reliable gradient signal at ANY batch size
- Ensures the model learns to predict expert actions accurately
- Acts as a "warm start" for the drifting field to build upon

**Component 2: Drifting Field Loss (Paper's Formulation with Negative Queue)**

The drifting field is computed following the paper's formulation on active action dimensions:

$$\mathbf{a}_{\text{active}} = \mathbf{a}_{\text{pred}} \odot \mathbf{m}_e$$

$$V(\mathbf{a}_{\text{active}}) = \mathbb{E}_{\mathbf{a}^+, \mathbf{a}^-}[\tilde{k}(\mathbf{a}_{\text{active}}, \mathbf{a}^+)\tilde{k}(\mathbf{a}_{\text{active}}, \mathbf{a}^-)(mathbf{a}^+ - \mathbf{a}^-)]$$

The normalized kernel uses double softmax normalization (Paper Eq. 13-15):

$$k(\mathbf{a}, \mathbf{a}') = \exp\left(-\frac{\|\mathbf{a} - \mathbf{a}'\|}{\tau}\right)$$

$$\tilde{k}(x_i, y_j) = \sqrt{\text{softmax}_j(s_{ij}) \cdot \text{softmax}_i(s_{ij})}$$

where $s_{ij} = -\|x_i - y_j\| / \tau$ (note: L2 norm, not squared). Multi-temperature kernels $\tau \in \{0.02, 0.05, 0.2\}$ are summed.

After computing $V$, feature normalization (Paper Eq. 18-21) and drift normalization (Paper Eq. 23-25) are applied:

- Feature normalization: scale samples so average pairwise distance ≈ $\sqrt{D}$
- Drift normalization: scale $V$ so that $\mathbb{E}[\|V\|^2 / D] = 1$

The drifting loss is:

$$\mathcal{L}_{\text{drift}} = \|\mathbf{a}_{\text{active}} - \text{sg}(\mathbf{a}_{\text{active}} + V(\mathbf{a}_{\text{active}}))\|^2$$

After normalization, this loss is approximately constant ($\mathcal{L}_{\text{drift}} \approx D$). **This is correct and expected** — the gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ provides the learning signal through the direction of $V$, not its magnitude.

**Our Modifications for Small-Batch Training:**

1. **Negative Sample Queue:** The paper uses $\mathbf{a}^- = \mathbf{a}_{\text{pred}}.detach()$ (current batch) with **batch size 8192**. For single-GPU training (batch 32-512), this provides insufficient diversity. We introduce a `NegativeSampleQueue` storing predictions from previous steps, ensuring negatives differ from queries even with small batches.

2. **Hybrid Loss:** The paper trains with pure drifting loss at large batch (8192). For smaller batches (512-2048), we add an MSE component:
   - MSE provides a reliable gradient when drifting field estimation is noisy
   - Weight $\lambda_{\text{drift}} = 0.1$ ensures drifting still contributes multi-modal structure
   - This is an ablation — we compare pure vs hybrid in Phase 5

**Justification:** The paper's algorithm (x.detach() as negatives, pure drifting loss) is optimal for large-scale multi-GPU training (batch 8192). Our modifications (negative queue, hybrid loss) enable convergence on limited compute (single GPU, batch 32-512) while preserving the drifting paradigm's core benefits. These are **practical engineering contributions**, not fundamental algorithmic changes.

**Critical Understanding: Why Loss ≈ Constant is Correct**

A common misunderstanding is that "loss not decreasing means the model isn't learning." For drifting models, this is WRONG:

From the paper (Section 3.4, Eq. 23-25):
- Drift normalization scales $V$ so that $\mathbb{E}[\|V\|^2 / D] = 1$
- Therefore, $\mathcal{L}_{\text{drift}} = \|V_{\text{normalized}}\|^2 \approx D$ (constant)
- The loss magnitude doesn't change, but the **direction of $V$** evolves
- The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ pushes the model based on V's direction
- As training progresses, $V$ points more accurately toward the data manifold
- At equilibrium, $V \to 0$ (which would make loss → 0, but in practice we stop before perfect convergence)

**Convergence Metrics for Drifting Models:**
- For image generation: FID, IS (not the drifting loss value)
- For robotics: Success rate, MMD (not the drifting loss value)
- Optional: Track $\lambda_V$ (the normalization factor before scaling) — this does decrease as $V_{\text{raw}}$ shrinks

In our development, we initially misinterpreted constant loss as lack of convergence, leading to the hybrid loss modification. Phase 5 ablations will determine if this modification is beneficial (for multi-modality) or unnecessary (for convergence).

#### 3.4.2 Action Normalization

Per-dataset, per-dimension normalization to zero-mean unit-variance:

$$\mathbf{a}_{\text{norm}}[j] = \frac{\mathbf{a}_{\text{raw}}[j] - \mu_{\text{dataset}}[j]}{\sigma_{\text{dataset}}[j] + \epsilon}$$

This is critical because:
- Position dimensions (meters) have magnitudes ~0.1-1.0
- Quaternion dimensions are unit-normalized, magnitudes ~0.1-1.0
- Joint angles (radians) have magnitudes ~0-3.14
- Without normalization, MSE is dominated by position errors

Statistics $\mu, \sigma$ are computed once per dataset from the training split.

---

## 4. Data Pipeline

### 4.1 Dataset Sources

| Dataset | Embodiment | Samples | Action Format | Image Format | URL |
|---------|-----------|---------|--------------|-------------|-----|
| RLBench (18 tasks) | Parallel gripper | ~270K | Absolute EE pose + grip (8-dim) | Multi-view RGB 128×128 | [HuggingFace](https://huggingface.co/datasets/hqfang/rlbench-18-tasks) |
| DexGraspNet 2.0 | Dexterous hand | ~500K | Wrist pose + joint angles (23-dim) | Point cloud → rendered RGB | [HuggingFace](https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0) |
| Bridge V2 | WidowX gripper | 60K | Delta EE (7-dim) | Wrist RGB 256×256 | [Project](https://rail-berkeley.github.io/bridgedata/) |
| ALOHA | Bimanual | 50K | Absolute joint pos (16-dim) | Multi-view RGB | [GitHub](https://github.com/tonyzhaozh/aloha) |
| RT-1 | Everyday Robots | 130K | Delta EE (7-dim) | RGB 320×256 | [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/rt_1) |

**Total:** ~1.01M demonstrations across 5 datasets and 3 embodiment types.

### 4.2 Complete Data Processing Pipeline

The data preprocessing pipeline transforms raw demonstrations from heterogeneous sources into a unified format compatible with our model. This section provides the complete end-to-end processing workflow.

#### Step 1: Image Standardization and Multi-View Handling

All images resized to **448×448** (PaliGemma2's native resolution):

```python
def preprocess_images(images: np.ndarray, dataset_name: str, cameras: List[str]) -> np.ndarray:
    """
    Standardize images across datasets.
    
    Args:
        images: Raw images, format varies by dataset:
                - RLBench: dict[camera_name] → [H, W, 3] uint8
                - DexGraspNet: rendered from point cloud [H, W, 3]
                - Bridge: [H, W, 3] uint8 (wrist camera)
                - ALOHA: dict[cam] → [H, W, 3] (bimanual setup)
        dataset_name: Source dataset identifier
        cameras: List of camera names to use
    
    Returns:
        images: [V, 3, 448, 448] float32 in [0, 1]
                V = len(cameras), standardized across datasets
    """
    images_standardized = []
    
    # Extract cameras in consistent order
    for cam in cameras:
        if isinstance(images, dict):
            img = images.get(cam, None)
        else:
            img = images  # Single-view datasets
        
        if img is None:
            # Missing camera → black frame
            img = np.zeros((448, 448, 3), dtype=np.uint8)
        
        # Resize to 448×448 with anti-aliasing
        if img.shape[0] != 448 or img.shape[1] != 448:
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # CHW format
        img = img.transpose(2, 0, 1)
        
        images_standardized.append(img)
    
    return np.stack(images_standardized, axis=0)  # [V, 3, 448, 448]
```

**Camera Mapping Across Datasets:**

| Dataset | Native Cameras | Mapped to Standard |
|---------|---------------|-------------------|
| RLBench | `front_rgb`, `wrist_rgb`, `left_shoulder_rgb` | `front`, `wrist` |
| DexGraspNet | Rendered from point cloud (4 angles) | `front`, `side`, `top`, `wrist` |
| Bridge V2 | Single wrist camera | `wrist` |
| ALOHA | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` | `front`, `wrist_left`, `wrist_right` |
| RT-1 | Multiple angles per episode | `front` (primary) |

**Standardization:** We use `front` + `wrist` as the canonical 2-view setup. Datasets lacking a camera are filled with black frames (the VLM learns to ignore these).

#### Step 2: Action Conversion to Absolute Representation

Datasets use heterogeneous action representations. We standardize to **absolute end-effector pose** (or absolute joint positions for bimanual/dex hand):

**Action Format by Dataset:**

| Dataset | Native Format | Conversion Needed |
|---------|--------------|-------------------|
| RLBench | Absolute EE pose [7] + grip [1] | None ✓ |
| DexGraspNet | Absolute wrist [7] + joint angles [16] | None ✓ |
| Bridge V2 | **Delta EE [6]** + grip [1] | Integrate deltas → absolute |
| RT-1 | **Delta EE [6]** + grip [1] | Integrate deltas → absolute |
| ALOHA | Absolute joint positions [7+7] + grips [2] | None ✓ |

**Delta to Absolute Integration:**

For datasets storing delta actions (position delta $\Delta \mathbf{p}$, rotation delta as axis-angle or delta quaternion):

```python
def delta_to_absolute(delta_actions: np.ndarray, 
                      initial_pose: np.ndarray,
                      rotation_format: str = 'quat') -> np.ndarray:
    """
    Integrate delta actions to absolute end-effector poses.
    
    Args:
        delta_actions: [T, 7] — [delta_x, delta_y, delta_z, 
                                   delta_qx/rx, delta_qy/ry, delta_qz/rz, (delta_qw), 
                                   gripper]
        initial_pose: [7] — [x0, y0, z0, qx0, qy0, qz0, qw0, grip0]
        rotation_format: 'quat' (quaternion delta) or 'axis_angle'
    
    Returns:
        absolute_actions: [T, 8] — [x, y, z, qx, qy, qz, qw, gripper]
    """
    absolute_actions = []
    current_pose = initial_pose.copy()
    
    for delta in delta_actions:
        # Integrate position
        current_pose[:3] += delta[:3]
        
        # Integrate rotation
        if rotation_format == 'quat':
            # Delta is small quaternion change
            delta_quat = delta[3:7]
            delta_quat = delta_quat / (np.linalg.norm(delta_quat) + 1e-8)
            current_pose[3:7] = quaternion_multiply(current_pose[3:7], delta_quat)
            current_pose[3:7] /= np.linalg.norm(current_pose[3:7])
        elif rotation_format == 'axis_angle':
            # Delta is axis-angle [rx, ry, rz]
            delta_angle = np.linalg.norm(delta[3:6])
            if delta_angle > 1e-6:
                delta_axis = delta[3:6] / delta_angle
                delta_quat = axis_angle_to_quat(delta_axis, delta_angle)
                current_pose[3:7] = quaternion_multiply(current_pose[3:7], delta_quat)
                current_pose[3:7] /= np.linalg.norm(current_pose[3:7])
        
        # Gripper state
        if len(delta) > 6:
            current_pose[7] = delta[6] if len(delta) == 7 else delta[7]
        
        absolute_actions.append(current_pose.copy())
    
    return np.array(absolute_actions)
```

**Initial Pose Extraction:**

For delta-action datasets, the initial pose is extracted from the first observation in the episode:

```python
# Bridge V2 / RT-1: extract from episode metadata
initial_pose = episode.observations[0].gripper_pose  # [x,y,z,qx,qy,qz,qw,grip]

# If not available: use dataset mean pose
initial_pose = DATASET_MEAN_POSES[dataset_name]
```

#### Step 3: Action Mapping to 128-dim Unified Space

Each embodiment's native action is mapped to a fixed 128-dim vector following a standardized layout:

**Unified Action Layout:**

```
Index:    [0-2]   [3-6]    [7]     [8-14]  [15]    [16-31]          [32-127]
Field:    pos1    quat1   grip1    pos2    grip2   joint_angles     padding
Type:     xyz     xyzw    binary   xyz/j   binary  θ1..θ16          zeros
Units:    meters  unit    {0,1}    m/rad   {0,1}   radians          —
```

**Per-Embodiment Mapping Functions:**

```python
def map_to_unified_gripper(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gripper: [x, y, z, qx, qy, qz, qw, gripper_open]
    
    Args:
        action: [8] or [T, 8] — native gripper action
    
    Returns:
        unified: [128] or [T, 128] — unified action
        mask: [128] — active dimension mask
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:8] = action
        mask = np.zeros(128, dtype=bool)
        mask[:8] = True
    else:  # [T, 8]
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :8] = action
        mask = np.zeros(128, dtype=bool)
        mask[:8] = True
    
    return unified, mask


def map_to_unified_bimanual(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bimanual: [left_j1..j7, left_gripper, right_j1..j7, right_gripper]
    
    ALOHA uses joint positions (not EE pose). We map to slots 0-15.
    
    Args:
        action: [16] or [T, 16] — native bimanual action
    
    Returns:
        unified: [128] or [T, 128]
        mask: [128]
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:16] = action
        mask = np.zeros(128, dtype=bool)
        mask[:16] = True
    else:
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :16] = action
        mask = np.zeros(128, dtype=bool)
        mask[:16] = True
    
    return unified, mask


def map_to_unified_dexhand(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dexterous Hand: [wrist_x, y, z, qx, qy, qz, qw, joint1..joint16]
    
    Wrist pose goes to slots 0-6.
    Joint angles go to slots 16-31.
    Slots 7-15 are unused (reserved for grip/bimanual).
    
    Args:
        action: [23] or [T, 23] — [wrist_pose(7), joints(16)]
    
    Returns:
        unified: [128] or [T, 128]
        mask: [128]
    """
    if action.ndim == 1:
        unified = np.zeros(128, dtype=np.float32)
        unified[:7] = action[:7]     # Wrist pose
        unified[16:32] = action[7:]  # Joint angles
        mask = np.zeros(128, dtype=bool)
        mask[:7] = True
        mask[16:32] = True
    else:
        T = action.shape[0]
        unified = np.zeros((T, 128), dtype=np.float32)
        unified[:, :7] = action[:, :7]
        unified[:, 16:32] = action[:, 7:]
        mask = np.zeros(128, dtype=bool)
        mask[:7] = True
        mask[16:32] = True
    
    return unified, mask


# Embodiment-to-function mapping
EMBODIMENT_MAPPERS = {
    0: map_to_unified_gripper,
    1: map_to_unified_bimanual,
    2: map_to_unified_dexhand,
}

# Reverse mapping (for inference)
def extract_from_unified(unified_action: np.ndarray, 
                        embodiment_id: int) -> np.ndarray:
    """Extract embodiment-specific action from 128-dim unified vector."""
    if embodiment_id == 0:  # Gripper
        return unified_action[:8]
    elif embodiment_id == 1:  # Bimanual
        return unified_action[:16]
    elif embodiment_id == 2:  # Dex hand
        wrist = unified_action[:7]
        joints = unified_action[16:32]
        return np.concatenate([wrist, joints])  # [23]
```

#### Step 4: VLM Feature Pre-computation (Offline)

**Motivation:** PaliGemma2-3B requires ~6 GB GPU memory and ~40ms per forward pass. Pre-computing features offline allows:
1. Training without VLM in GPU memory → larger batch sizes
2. Faster data loading (read 4KB features vs process 448×448 images)
3. Easy ablation of different VLMs (re-run pre-computation, no training code change)

**Complete Pre-computation Script:**

```python
# scripts/precompute_vlm_features.py

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from torch.utils.data import DataLoader
import h5py
from tqdm import tqdm

def precompute_vlm_features(
    dataset_name: str,
    model_name: str = "google/paligemma2-3b-mix-448",
    batch_size: int = 32,
    num_gpus: int = 8,
):
    """
    Pre-compute VLM features for a dataset.
    
    Time estimate: ~40ms per sample per GPU
    For 1M samples across 8 GPUs: 1M / (8 × 25 samples/sec) ≈ 83 minutes
    """
    # Load VLM
    processor = AutoProcessor.from_pretrained(model_name)
    vlm = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    vlm.eval()
    
    # Load dataset
    dataset = load_dataset(dataset_name)  # Returns UnifiedDataset
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    # HDF5 file for storing features
    output_file = f'./data/vlm_features/{dataset_name}_paligemma2.h5'
    h5_file = h5py.File(output_file, 'w')
    
    # Process in batches
    for batch_idx, batch in enumerate(tqdm(loader)):
        images = batch['images']        # [B, V, 3, 448, 448]
        languages = batch['language']   # List[str]
        sample_ids = batch['sample_id'] # List[int]
        
        # Flatten multi-view: [B, V, 3, 448, 448] → [B×V, 3, 448, 448]
        B, V = images.shape[:2]
        images_flat = images.reshape(B * V, 3, 448, 448)
        
        # Replicate language for each view
        langs_flat = [lang for lang in languages for _ in range(V)]
        
        # Process through VLM
        inputs = processor(
            text=langs_flat,
            images=images_flat,
            return_tensors="pt",
            padding=True,
        ).to('cuda')
        
        with torch.no_grad():
            outputs = vlm(
                **inputs,
                output_hidden_states=True,
            )
        
        # Extract last hidden state
        hidden = outputs.hidden_states[-1]  # [B×V, L, 2048]
        
        # Reshape: [B×V, L, 2048] → [B, V, L, 2048]
        L, D = hidden.shape[1], hidden.shape[2]
        hidden = hidden.reshape(B, V, L, D)
        
        # Aggregate views (mean across V)
        hidden_agg = hidden.mean(dim=1)  # [B, L, 2048]
        pooled = hidden_agg.mean(dim=1)  # [B, 2048]
        
        # Save to HDF5 (fp16 for storage efficiency)
        for i, sample_id in enumerate(sample_ids):
            grp = h5_file.create_group(f'sample_{sample_id}')
            grp.create_dataset('hidden', data=hidden_agg[i].cpu().half().numpy(), compression='gzip')
            grp.create_dataset('pooled', data=pooled[i].cpu().half().numpy())
            grp.attrs['sequence_length'] = L
            grp.attrs['hidden_dim'] = D
    
    h5_file.close()
    print(f"Saved to {output_file}")
    print(f"Size: {os.path.getsize(output_file) / 1e9:.2f} GB")
```

**HDF5 Storage Format:**

```
vlm_features_{dataset}.h5
├── sample_0/
│   ├── hidden: [L, 2048] float16 (gzip compressed)
│   ├── pooled: [2048] float16
│   └── attrs: {sequence_length: L, hidden_dim: 2048}
├── sample_1/
│   ├── ...
└── ...
```

**Loading During Training:**

```python
class PrecomputedVLMDataset:
    def __init__(self, base_dataset, vlm_features_path):
        self.base_dataset = base_dataset
        self.vlm_h5 = h5py.File(vlm_features_path, 'r')
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_id = sample['sample_id']
        
        # Load pre-computed VLM features
        hidden = torch.from_numpy(self.vlm_h5[f'sample_{sample_id}/hidden'][:]).float()
        pooled = torch.from_numpy(self.vlm_h5[f'sample_{sample_id}/pooled'][:]).float()
        
        sample['vlm_features'] = hidden
        sample['vlm_pooled'] = pooled
        
        return sample
```

**Storage estimate:** ~4KB per sample (L=1050, 2048-dim, fp16) → 1M samples = ~4 GB total.

#### Step 5: DexGraspNet 2.0 Specific Processing

DexGraspNet 2.0 provides point clouds (not images). We render multi-view RGB images:

```python
def render_pointcloud_to_images(
    pointcloud: np.ndarray,  # [N, 3] xyz
    colors: np.ndarray,       # [N, 3] rgb
    camera_poses: List[Tuple[np.ndarray, np.ndarray]],  # [(pos, rot), ...]
) -> np.ndarray:
    """
    Render point cloud from multiple camera views.
    
    Uses Open3D for fast GPU-accelerated rendering.
    
    Args:
        pointcloud: [N, 3] point positions
        colors: [N, 3] RGB colors
        camera_poses: List of (position, rotation) tuples for each view
    
    Returns:
        images: [V, 448, 448, 3] uint8
    """
    import open3d as o3d
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    images = []
    for cam_pos, cam_rot in camera_poses:
        # Set up virtual camera
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=448, height=448)
        vis.add_geometry(pcd)
        
        # Set camera parameters
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        camera_params.extrinsic = create_extrinsic_matrix(cam_pos, cam_rot)
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        
        # Render
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img) * 255).astype(np.uint8)
        
        images.append(img)
        vis.destroy_window()
    
    return np.array(images)  # [V, 448, 448, 3]


# Standard camera poses for DexGraspNet rendering
DEXGRASP_CAMERA_POSES = {
    'front': (pos=[0.5, 0, 0.3], lookat=[0,0,0]),
    'side':  (pos=[0, 0.5, 0.3], lookat=[0,0,0]),
    'top':   (pos=[0, 0, 0.8], lookat=[0,0,0]),
    'wrist': (pos=[0.1, 0, 0.15], lookat=[0,0,0.05]),  # Close-up
}
```

**Grasp Pose to Action:**

DexGraspNet stores grasp poses as:
- Wrist SE(3) pose: 4×4 transformation matrix
- Joint angles: [θ1, ..., θ16] for Allegro hand

```python
def dexgrasp_pose_to_action(grasp_pose_dict):
    """Convert DexGraspNet grasp to our action format."""
    # Extract wrist pose from 4×4 matrix
    T = grasp_pose_dict['wrist_transform']  # [4, 4]
    pos = T[:3, 3]  # Translation
    rot_mat = T[:3, :3]  # Rotation matrix
    quat = rotation_matrix_to_quaternion(rot_mat)  # [qx, qy, qz, qw]
    
    # Joint angles
    joints = grasp_pose_dict['joint_angles']  # [16]
    
    # Concatenate
    action = np.concatenate([pos, quat, joints])  # [23]
    
    return action
```

### 4.3 Data Loading During Training

```python
class UnifiedDataLoader:
    def __init__(self, datasets, sampling_weights):
        self.datasets = {
            'rlbench': RLBenchDataset(...),
            'dexgraspnet': DexGraspNetDataset(...),
            'bridge': BridgeDataset(...),
            'aloha': ALOHADataset(...),
            'rt1': RT1Dataset(...),
        }
        self.weights = sampling_weights  # e.g., [0.3, 0.3, 0.2, 0.1, 0.1]
    
    def __iter__(self):
        while True:
            # Sample dataset according to weights
            dataset_name = np.random.choice(
                list(self.datasets.keys()),
                p=self.weights
            )
            
            # Sample batch from chosen dataset
            batch = self.datasets[dataset_name].sample(batch_size)
            
            # Load pre-computed VLM features from disk
            vlm_features = load_vlm_features(batch['sample_ids'])
            
            # Normalize actions using dataset-specific stats
            actions_norm = normalize_actions(batch['actions'], dataset_name)
            
            yield {
                'vlm_features': vlm_features,
                'vlm_pooled': vlm_pooled,
                'embodiment_id': EMBODIMENT_MAP[dataset_name],
                'action_mask': ACTION_MASK_MAP[dataset_name],
                'actions': actions_norm,
                'dataset_name': dataset_name,
            }
```

---

## 5. Training Procedure

### 5.1 Optimization

**Optimizer:** AdamW with weight decay 0.05, differential LR (Section 3.2.1)

**Learning rate schedule:**
- Warmup: Linear 0 → 4e-4 over 2,000 steps
- Main: Cosine decay 4e-4 → 1e-5 over 98,000 steps
- Total: 100,000 steps
- VLM LoRA unfreezes at step 5,000 with LR 1e-5

**Compute Configurations (system supports both):**

| Setting | 8×B200 (192GB each) | 64×H100 (80GB each) |
|---------|---------------------|---------------------|
| Per-GPU batch | 128 | 64 |
| GPUs | 8 | 64 |
| Grad accumulation | 2 | 2 |
| **Effective batch** | **2,048** | **8,192** |
| Flash Attention | ✅ (bf16 auto-cast) | ✅ (native) |
| FSDP | ✅ | ✅ (ZeRO-3) |
| VLM in GPU (Phase 3+) | ✅ (fits in 192GB) | ✅ (FSDP sharded) |
| Training time (100K steps) | ~40 hrs | ~22 hrs |

**Primary target: 64×H100** (effective batch 8192, matching the Drifting paper exactly).

With batch 8192 on 64×H100:
- Can use **paper's exact formulation** (pure drifting loss, x.detach() as negatives)
- Negative sample queue becomes optional (ablate in Phase 5)
- Hybrid MSE loss becomes optional (ablate in Phase 5)

**Launch command (64×H100, 8 nodes × 8 GPUs):**

```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=$RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  scripts/train.py \
  training.batch_size=64 \
  training.grad_accumulation_steps=2 \
  training.num_workers=8 \
  training.drifting.n_pos_samples=256 \
  training.drifting.n_neg_samples=256
```

**Launch command (8×B200, single node):**

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  training.batch_size=128 \
  training.grad_accumulation_steps=2
```

**Mixed precision:** BF16 with Flash Attention 2 (auto-cast fix in dit.py)

**Gradient clipping:** Max norm 2.0

### 5.2 Negative Sample Queue

To prevent drifting field collapse ($V \approx 0$), we maintain a queue of recent model predictions:

```python
neg_queue = NegativeSampleQueue(size=4096, action_dim=128)

# Each training step:
neg_samples = neg_queue.sample(n=128)  # Sample from previous steps
neg_queue.push(actions_pred.detach())   # Store current predictions

# Drifting loss uses neg_samples (NOT actions_pred.detach())
```

Queue size 4096 stores ~2 batches of predictions, ensuring negatives are distinct from current queries.

### 5.3 Positive Sample Queue

Per-task queues for expert actions:

```python
pos_queue = SampleQueue(queue_size=256, num_tasks=len(all_tasks), action_dim=128)

# Each step:
pos_queue.add(task_id=batch['task_id'], actions=batch['actions'])
pos_samples = pos_queue.sample(n=128, task_ids=batch['task_id'])
```

### 5.4 Training Algorithm

**Algorithm: Drifting-VLA v2 Training (64×H100, Effective Batch 8192)**

At 64×H100 with effective batch 8192, we adopt the **paper's canonical formulation** as default, with our modifications available as configuration flags for ablation.

```
Algorithm: Drifting-VLA v2 Training

Input:  Datasets D = {D_1, ..., D_K}, sampling weights w
        Compute: 64×H100, effective batch B = 8192
Output: Trained model θ

--- OFFLINE (one-time, ~4 hours on 8 GPUs) ---
1: For each dataset D_k:
2:     For each sample (images, language, actions):
3:         For each view v in images:
4:             h_v ← PaliGemma2(view_v, language)  # Independent per view
5:         h_concat ← cat([h_1, ..., h_V], dim=0)   # Variable V per dataset
6:         Save h_concat to HDF5

--- ONLINE (100K steps, ~22 hours on 64×H100) ---
7:  Initialize DiT θ, VLM LoRA (frozen until step 5K)
8:  Initialize optimizer (differential LR), pos_queue, neg_queue
9:  for step = 1 to 100,000 do
10:     Sample dataset D_k ~ Categorical(w)
11:     Sample batch B ~ D_k, size B=8192
12:     Load vlm_features, actions, embodiment_id, action_mask from B
13:     
14:     # Normalize GT actions (per-dataset, per-dim)
15:     actions_gt ← (actions - μ_k) / (σ_k + ε)
16:     
17:     # Forward through DiT
18:     noise ← N(0, I_{B×T×64})
19:     α ← sample_cfg_scale()                    # CFG: α ~ p(α) ∝ α^{-3}
20:     c_seq ← proj(vlm_features)                # [B, L, 768]
21:     c_pool ← proj_pool(mean(vlm_features))    # [B, 768]
22:     actions_pred ← DiT(noise, c_seq, c_pool, embodiment_id, α)  # [B, T, 128]
23:     
24:     # Sample positive/negative for drifting field
25:     y_pos ← pos_queue.sample(n=256)
26:     y_neg ← actions_pred.detach()              # Paper's approach (works at batch 8192)
27:     
28:     # Compute drifting field V (paper's formulation)
29:     V ← compute_drifting_field(
30:         x=actions_pred, y_pos=y_pos, y_neg=y_neg,
31:         τ=[0.02, 0.05, 0.2],
32:         normalize_features=True, normalize_drift=True,
33:         action_mask=action_mask                 # Only compute on active dims
34:     )
35:     
36:     # Loss (default: pure drifting; flag: +MSE hybrid)
37:     L_drift ← ||actions_pred - sg(actions_pred + V)||²
38:     if hybrid_mode:
39:         L_mse ← masked_mse(actions_pred, actions_gt, action_mask)
40:         L ← L_mse + λ_drift × L_drift
41:     else:
42:         L ← L_drift                            # Paper's default
43:     
44:     # Optimization
45:     L.backward()
46:     clip_grad_norm(θ, max_norm=2.0)
47:     optimizer.step()                            # Differential LR
48:     ema.update(θ)
49:     
50:     # Update queues
51:     pos_queue.push(actions_gt)
52:     
53:     # Unfreeze VLM LoRA at step 5K (staged training)
54:     if step == 5000:
55:         unfreeze(vlm_lora, lr=1e-5)
56: end for
```

**Configuration Flags for Ablation:**
- `hybrid_mode`: True adds MSE component (default: False at batch 8192)
- `use_neg_queue`: True uses queue instead of x.detach() (default: False at batch 8192)
- `λ_drift`: Weight for drifting loss in hybrid mode (default: 0.1)
- These are tested systematically in Phase 5

---

## 6. Evaluation Protocol

### 6.1 Benchmarks

#### 6.1.1 RLBench (Parallel Gripper)

**Tasks:** 18 manipulation tasks (close_jar, open_drawer, stack_blocks, ...)

**Metric:** Success rate (%) over 25 episodes per task

**Baselines:** RDT-1B, Diffusion Policy, 3D Diffuser Actor, OpenVLA

**Evaluation:** CoppeliaSim simulation with our RLBench environment wrapper

#### 6.1.2 DexGraspNet 2.0 (Dexterous Hand)

**Tasks:** Grasping diverse objects in cluttered scenes

**Metric:** Grasp success rate (%) in IsaacGym simulation

**Baselines:** DexGraspNet 2.0 baseline, GraspTTA, IsaGrasp

**Evaluation:** IsaacGym with Allegro hand model

#### 6.1.3 ManiSkill (Multi-Task)

**Tasks:** 5 benchmark tasks (PegInsertion, PickCube, StackCube, PlugCharger, PushCube)

**Metric:** Success rate (%) over 250 trials (10 seeds × 25 trials)

**Baselines:** RDT-1B (53.6%), Diffusion Policy (30.2%), OpenVLA (4.8%)

**Evaluation:** ManiSkill simulation environment

### 6.2 Ablation Studies

| Ablation | Variants | Purpose |
|----------|----------|---------|
| VLM backbone | PaliGemma2 vs Qwen3-VL vs DINOv2+CLIP | Validate VLM benefit |
| Loss weight | $\lambda_{\text{drift}} \in \{0, 0.05, 0.1, 0.2\}$ | Tune MSE/drift balance |
| Batch size | 512, 1024, 2048, 4096 | Find minimal for drifting |
| Action space | 128-dim vs 64-dim vs multi-head | Validate padding approach |
| Negative sampling | Queue vs x.detach() vs random | Validate queue necessity |
| Multi-embodiment | Joint vs separate training | Transfer learning analysis |

### 6.3 Metrics

**Primary:** Task success rate (%)

**Secondary:**
- Inference latency (ms per action)
- Action distribution quality (MMD vs expert)
- Multi-modality score (GMM entropy)
- Generalization: zero-shot to new objects/scenes

---

## 7. Implementation Requirements

### 7.1 Hardware

**Development:** 1× NVIDIA A40 (48GB) for single-GPU debugging and simulation evaluation

**Training (supported configurations):**

| Config | GPUs | Effective Batch | Training Time (100K) | Cost |
|--------|------|----------------|---------------------|------|
| 8×B200 (single node) | 8 × 192GB | 2,048 | ~40 hrs | ~$3,000 |
| **64×H100 (8 nodes)** | 64 × 80GB | **8,192** | **~22 hrs** | ~$4,500 |
| 8×H100 (single node) | 8 × 80GB | 1,024 | ~80 hrs | ~$3,000 |

**Primary: 64×H100** — matches the Drifting paper's batch size (8192), enabling faithful reproduction.

**Fallback: 8×B200** — effective batch 2048, requires hybrid loss modification.

**Evaluation:** A40 (dev) with CoppeliaSim + RLBench for gripper eval; H100 node with IsaacGym for DexGraspNet eval

### 7.2 Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| PyTorch | 2.2+ | Deep learning framework |
| Flash Attention | 2.5+ | Efficient attention (now with auto-cast fix) |
| Transformers | 4.45+ | PaliGemma2-3B and Qwen3-VL-2B loading |
| PEFT | 0.10+ | LoRA adapters |
| DeepSpeed | 0.14+ | FSDP / ZeRO optimization |
| WandB | Latest | Experiment tracking |
| CoppeliaSim | 4.1.0 | RLBench simulation |
| IsaacGym | 4.0 | DexGraspNet evaluation |

### 7.3 Docker Images

**Base image:** `drifting-vla:base` (CUDA 12.1, PyTorch 2.2, Flash Attn)

**RLBench image:** `drifting-vla:rlbench` (+ CoppeliaSim 4.1, PyRep, RLBench, Xvfb)

**DexGrasp image:** `drifting-vla:dexgrasp` (+ IsaacGym, Open3D, point cloud tools)

### 7.4 Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **VLM backbone** | PaliGemma2-3B (frozen + LoRA) | Best visual grounding at 448×448; 3B fits in memory; LoRA enables task adaptation with <1% params |
| **Why not end-to-end VLM?** | VLMs output text tokens, not continuous actions | Need separate action generation head |
| **Why pre-compute VLM features?** | Saves 6GB GPU memory during training | Enables larger batch sizes; VLM runs once offline per sample |
| **Action space** | Fixed 128-dim with masking (RDT-style) | Proven to work; simpler than multi-head; easy to add new embodiments |
| **Why not multi-head decoder?** | Adds complexity without clear benefit | RDT showed padding works; multi-head requires embodiment-specific tuning |
| **Loss design** | Hybrid MSE + Drifting | MSE ensures convergence; drifting adds multi-modality; neither alone is sufficient |
| **Negative sampling** | Queue of past predictions | Prevents V≈0 self-cancellation; critical for small batch |
| **Why Drifting not Diffusion?** | 1-step vs 100-step inference | 10× faster; enables real-time control (20-100 Hz) |
| **Action normalization** | Per-dataset, per-dim | Prevents position dominating rotation in MSE |
| **Multi-view** | Front + wrist (2 views) | Matches 3D Diffuser Actor baseline; provides depth cues |

---

## 8. Expected Results

### 8.1 Quantitative Targets

| Benchmark | Metric | RDT-1B (100-NFE) | Ours (1-NFE) Target |
|-----------|--------|------------------|---------------------|
| RLBench (18 tasks) | Success rate | ~85% (estimated) | **>80%** |
| DexGraspNet 2.0 | Grasp success | ~75% (baseline) | **>70%** |
| ManiSkill (5 tasks) | Success rate | 53.6% | **>50%** |
| Inference latency | ms per action | ~500ms | **<50ms** |

### 8.2 Ablation Expectations

**VLM Backbone Ablation:**

| Config | Model | Hidden Dim | Multi-Image | Hypothesis |
|--------|-------|-----------|-------------|-----------|
| A | PaliGemma2-3B | 2048 | Independent per view | Best grounding (default) |
| B | Qwen3-VL-2B | 1536 | Native multi-image | Better multi-view, smaller model |
| C | DINOv2 + CLIP (v1) | 1024 | Independent per view | Baseline (no joint VL reasoning) |

Expected: A ≈ B > C by +5-10% success rate

**Loss Formulation Ablations:**
- **At effective batch 2048 (8×B200):**
  - Pure drifting (paper) vs Hybrid (ours): Hypothesis: similar performance (paper's formulation should work at this scale)
  - Negative queue vs x.detach() (paper): Hypothesis: x.detach() sufficient at large batch
  - $\lambda_{\text{drift}} \in \{0, 0.05, 0.1, 0.2, 1.0\}$: Hypothesis: pure drifting ($\lambda=1.0$) achieves best multi-modality
  
- **At effective batch 512 (1 GPU):**
  - Pure drifting vs Hybrid: Hypothesis: hybrid gives +15-25% (pure drifting noisy at small batch)
  - Negative queue vs x.detach(): Hypothesis: queue gives +10-15% (x.detach() causes field collapse)

**Multi-Embodiment Ablations:**
- Joint training vs separate per-embodiment models: -2-5% per task vs single-task (acceptable trade-off for unified model)
- Pre-train on gripper → fine-tune on dex hand: +5-10% vs train-from-scratch (transfer learning benefit)
- Embodiment token: +3-5% vs no embodiment conditioning (helps the model specialize)

---

## 9. Development Schedule

### Phase 0: Infrastructure Setup (Week 1)

**Days 1-2: Environment & Docker**
- Set up 8×B200 node with CUDA 12.1, PyTorch 2.2
- Build Docker images (base, rlbench, dexgrasp)
- Verify FSDP, Flash Attention, multi-GPU work
- Set up WandB project

**Days 3-4: Data Download**
- Download RLBench (18 tasks) → ~50 GB
- Download DexGraspNet 2.0 → ~100 GB
- Download Bridge V2, ALOHA, RT-1 → ~700 GB
- Verify data integrity

**Days 5-7: VLM Feature Pre-computation**
- Implement `scripts/precompute_vlm_features.py`
- Pre-compute PaliGemma2 features for all datasets
- Verify features load correctly during training
- **Deliverable:** ~4 GB HDF5 files with VLM features

---

### Phase 1: Toy Case Validation (Week 2)

**Days 8: Verify Paper's Formulation (Critical First Step)**

Before any modifications, we verify our drifting field implementation matches the paper:

- Implement 2D toy example from [demo notebook](https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb)
- Reproduce Figure 3 from paper (Gaussian mixture → drifting field evolution)
- Verify:
  - Double softmax normalization produces correct kernel weights
  - Feature normalization scales distances to ~$\sqrt{D}$
  - Drift normalization produces $\mathbb{E}[\|V\|^2/D] = 1$
  - Loss stays ~constant while distribution evolves (as expected)
- **Success criteria:** 2D toy example converges to correct distribution with pure drifting loss

This sanity check ensures our understanding is correct before scaling to robotics.

**Days 9-10: VLM Integration (Single Task)**
- Implement `VLMBackbone` with pre-computed feature loading
- Replace DINOv2+CLIP in current codebase
- Train on RLBench close_jar only (15K samples, 1 task)
- **Try pure drifting first** (paper's approach) at batch 512
- If it doesn't converge, add hybrid loss
- **Target:** Success rate > 70% in 5K steps (~1 hour)
- **Success criteria:** Model learns to grasp (qualitative video check)

**Days 11-12: Action Space Unification**
- Implement 128-dim action mapping
- Implement action mask in loss computation
- Verify masked MSE works correctly
- Test on RLBench (8-dim gripper in 128-dim space)

**Days 13-14: Ablation (VLM Backbone)**
- Compare PaliGemma2 vs Qwen3-VL features
- Measure MSE convergence speed, final success rate
- **Deliverable:** Table showing VLM > DINOv2+CLIP

---

### Phase 2: Multi-Embodiment (Weeks 3-4)

**Days 15-17: DexGraspNet Integration**
- Implement `DexGraspNetDataset` (point cloud → rendered images)
- Implement 23-dim action mapping (wrist + joints)
- Pre-compute VLM features for DexGraspNet
- Verify data loading works

**Days 18-21: Joint Training (2 Embodiments)**
- Train on RLBench (gripper) + DexGraspNet (dex hand) jointly
- Sampling weights: 50% each
- 20K steps (~7 hours on 8×B200)
- **Target:** Both losses decrease; gripper MSE < 0.1, dex MSE < 0.2
- **Success criteria:** No catastrophic interference between embodiments

**Days 22-24: Embodiment Token Ablation**
- Train with vs without embodiment embedding
- Measure per-embodiment success rates
- **Deliverable:** Proof that embodiment conditioning helps

**Days 25-28: Add Bimanual (ALOHA)**
- Implement ALOHA dataset loader
- 16-dim action mapping
- Train on 3 embodiments jointly
- **Target:** All three losses decrease
- **Deliverable:** First unified gripper+bimanual+dex model

---

### Phase 3: Scale-Up & Pre-training (Weeks 5-7)

**Days 29-32: Full Dataset Integration**
- Add Bridge V2 and RT-1 datasets
- Implement delta→absolute action conversion
- Tune sampling weights (start with uniform, adjust based on loss)
- **Target:** All 5 datasets load without errors

**Days 33-42: Large-Scale Pre-training**
- Train on all datasets for 100K steps
- Effective batch 2,048
- Monitor per-dataset losses on WandB
- Checkpoint every 5K steps
- **Time:** ~40 hours (~2 days)
- **Target:** All dataset losses converge; no single dataset dominates

**Days 43-49: Hyperparameter Tuning**
- Tune $\lambda_{\text{drift}} \in \{0.05, 0.1, 0.2\}$
- Tune sampling weights based on loss curves
- Tune n_pos_samples, n_neg_samples
- **Deliverable:** Optimal hyperparameters for final training

---

### Phase 4: Evaluation & Baselines (Weeks 8-9)

**Days 50-52: RLBench Evaluation**
- Evaluate on 18 tasks × 25 episodes
- Compare vs RDT-1B, Diffusion Policy, 3D Diffuser Actor
- Measure success rate and inference latency
- **Target:** >80% success, <50ms latency

**Days 53-55: DexGraspNet Evaluation**
- Set up IsaacGym with Allegro hand
- Evaluate on test scenes
- Compare vs DexGraspNet 2.0 baseline
- **Target:** >70% grasp success

**Days 56-58: ManiSkill Evaluation**
- Set up ManiSkill environment
- Evaluate on 5 tasks × 250 trials
- Compare vs RDT-1B (53.6%), Diffusion Policy (30.2%)
- **Target:** >50% average success

**Days 59-63: Baseline Implementations**
- Implement Diffusion Policy on our data (if needed)
- Run RDT-1B evaluation (use their released checkpoint)
- Ensure fair comparison (same data, same evaluation protocol)

---

### Phase 5: Ablations & Analysis (Weeks 10-11)

**Days 64-66: Architecture Ablations**
- VLM backbone: PaliGemma2 vs Qwen3-VL vs DINOv2+CLIP
- DiT depth: 6L vs 12L vs 24L
- Action space: 64-dim vs 128-dim vs 256-dim

**Days 67-69: Loss Ablations (Critical for Validating Our Modifications)**

**Experiment 1: Pure Drifting (Paper) vs Hybrid (Ours) at Different Batch Sizes**

| Config | Batch Size | Negatives | Loss | Expected Result |
|--------|-----------|-----------|------|----------------|
| Paper baseline | 2048 (8 GPU) | x.detach() | Pure drifting | Baseline performance |
| Our modification | 2048 (8 GPU) | neg_queue | Hybrid MSE+drift | Similar or slightly better |
| Paper at small batch | 512 (2 GPU) | x.detach() | Pure drifting | Poor (noisy field) |
| Our fix | 512 (2 GPU) | neg_queue | Hybrid MSE+drift | Recovers performance |

**Experiment 2: Negative Sampling Strategy**

Fix batch=2048, vary negative source:
- x.detach() (paper): Baseline
- neg_queue (ours): Hypothesis: similar at large batch, better at small batch
- Random noise: Control (should be worst)

**Experiment 3: Loss Component Weights**

Fix batch=2048, vary $\lambda_{\text{drift}}$:
- $\lambda = 0$: MSE only (no drifting)
- $\lambda = 0.1$: Our default
- $\lambda = 1.0$: Pure drifting (paper)
- $\lambda = 0.05, 0.2$: Intermediate

Measure: Success rate + action diversity (GMM entropy)

**Days 70-72: Multi-Embodiment Analysis**
- Joint training vs separate per-embodiment models
- Transfer learning: pre-train on gripper → fine-tune on dex hand
- Embodiment token ablation

**Days 73-77: Visualization & Analysis**
- Drifting field evolution over training
- Action distribution diversity (GMM analysis)
- Failure case analysis per embodiment
- **Deliverable:** 10+ plots for paper

---

### Phase 6: Paper Writing (Weeks 12-14)

**Days 78-82: Drafting**
- Abstract (300 words)
- Introduction (2 pages)
- Related Work (1.5 pages)
- Method (4 pages): VLM backbone, Drifting DiT, unified action space, hybrid loss
- **Deliverable:** Draft sections

**Days 83-87: Experiments Section**
- Main results tables (RLBench, DexGraspNet, ManiSkill)
- Ablation tables
- Inference latency comparison
- Qualitative results (robot execution frames)
- **Deliverable:** Complete results section

**Days 88-91: Figures & Polish**
- Architecture diagram
- Training dynamics plots
- Drifting field visualization
- Success rate bar charts
- **Deliverable:** All figures camera-ready

**Days 92-94: Supplementary Materials**
- Implementation details
- Additional ablations
- Failure case analysis
- Hyperparameter tables
- **Deliverable:** 20+ page appendix

**Days 95-97: Final Review**
- Co-author feedback
- Proofreading
- NeurIPS format compliance
- **Deliverable:** Submission-ready PDF

---

## 9.5 Lessons Learned: Common Pitfalls When Implementing Drifting Models

During our development of Drifting-VLA v1, we encountered several implementation issues that led to misunderstandings of the drifting paradigm. We document these to help future implementers:

### Pitfall 1: "Loss Not Decreasing = Model Not Learning" ❌

**Misconception:** With drift normalization, $\mathcal{L}_{\text{drift}} \approx D$ (constant). We initially thought this meant the model wasn't learning and added MSE loss to "fix" it.

**Reality:** Constant loss is **correct and expected** per the paper. The gradient $\nabla_\theta \mathcal{L} = -2V \cdot \nabla_\theta f$ provides learning signal through V's **direction**, not magnitude. Convergence is measured by task metrics (success rate), not loss value.

**Resolution:** Track task performance (success rate, FID) as the primary metric. Optionally track $\lambda_V$ (the normalization factor before scaling) which does decrease.

### Pitfall 2: Using x.detach() as Negatives at Small Batch ❌

**Misconception:** The paper uses `y_neg = x.detach()`, so we did the same with batch_size=32.

**Reality:** With batch=32, the 32 negatives are nearly identical to the 32 queries → drifting field V ≈ 0. The paper uses batch=8192 where 8192 negatives provide sufficient diversity.

**Resolution:** For batch <1024, use a negative sample queue storing predictions from previous steps. For batch ≥2048, x.detach() likely suffices (ablate in Phase 5).

### Pitfall 3: Incorrect Action Space Alignment ❌

**Misconception:** Dataset stores absolute poses, simulator uses delta actions (or vice versa).

**Reality:** Misalignment causes the robot to not move (path planning fails, or moves to wrong location). Must verify dataset format matches simulator's action_mode.

**Resolution:** Explicitly check: print first action, verify robot moves in simulation, compare GT trajectory vs predicted trajectory in 3D plots.

### Pitfall 4: Forgetting to Normalize Actions Per-Dimension ❌

**Misconception:** Actions are already in a reasonable range, no normalization needed.

**Reality:** Position (meters, ~0.3-1.5) dominates rotation (quaternion, ~0-1) and gripper (binary) in MSE. The model fits position well but rotation poorly.

**Resolution:** Per-dimension normalization to zero-mean unit-variance. This makes all dims equally important in the loss.

### Pitfall 5: Single Camera Provides Insufficient Information ❌

**Misconception:** DINOv2 is so powerful that one camera view is enough.

**Reality:** Manipulation tasks require depth/proximity cues. Single front camera can't infer 3D workspace. Success rates are 20-30% lower than multi-view baselines (3D Diffuser Actor, RDT).

**Resolution:** Use ≥2 cameras (front + wrist). Multi-view is standard in all SOTA manipulation models.

These lessons inform our phased development plan — Phase 1 (toy case) catches these issues early before committing to expensive large-scale training.

---

## 10. Risk Analysis & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| VLM features too generic for manipulation | Medium | High | Add LoRA fine-tuning; ablate against DINOv2 |
| Drifting loss doesn't converge with 128-dim actions | Low | High | Hybrid MSE already works; increase batch if needed |
| Multi-embodiment interference | Medium | Medium | Tune sampling weights; add per-embodiment batch norm |
| DexGraspNet point clouds hard to render | Low | Medium | Use Open3D rendering; or add PointNet branch |
| 100K steps insufficient for convergence | Medium | Medium | Extend to 200K if needed (~4 days) |

### 10.2 Computational Risks

| Risk | Mitigation |
|------|------------|
| B200 node unavailable | Fall back to 8×H100 (2× slower, still ~4 days) |
| OOM with batch 128 per GPU | Reduce to 64, increase grad_accum to 4 |
| VLM feature pre-compute takes too long | Parallelize across 8 GPUs (~2 hours total) |

### 10.3 Timeline Risks

| Risk | Mitigation |
|------|------------|
| Phase 2 takes longer than 2 weeks | Reduce scope: train only gripper + dex hand (skip bimanual) |
| Baseline comparisons delayed | Use RDT's released checkpoints; skip re-training baselines |
| Paper writing takes >3 weeks | Start writing introduction/related work during Phase 4 |

---

## 11. Success Criteria

### 11.1 Minimum Viable Product (MVP)

By end of Week 7 (Phase 3):
- ✅ Model trains on 3+ datasets without errors
- ✅ MSE converges to < 0.15 on all datasets
- ✅ Gripper success > 70% on RLBench
- ✅ Dex hand grasp success > 60% on DexGraspNet
- ✅ Inference latency < 50ms

### 11.2 Target for Paper Submission

By end of Week 11 (Phase 5):
- ✅ RLBench success > 80% (within 5% of RDT-1B)
- ✅ ManiSkill success > 50% (within 5% of RDT-1B)
- ✅ DexGraspNet success > 70%
- ✅ 10× faster inference than RDT (verified)
- ✅ All ablations complete
- ✅ Qualitative videos showing robot execution

### 11.3 Stretch Goals

- Bimanual manipulation benchmark (if time permits)
- Real-world deployment on physical robot
- Open-source model release (HuggingFace)

---

## 12. Budget Estimate

### 12.1 Compute Costs

**Primary configuration: 64×H100 (8 nodes)**

| Phase | GPU-Hours (64×H100) | Wall Time | Cost (@$2/GPU-hr) |
|-------|---------------------|-----------|-------------------|
| Phase 0 (setup + pre-compute) | 64 × 4 hrs = 256 | 4 hrs | $512 |
| Phase 1 (toy case) | 64 × 2 hrs = 128 | 2 hrs | $256 |
| Phase 2 (multi-embodiment) | 64 × 12 hrs = 768 | 12 hrs | $1,536 |
| Phase 3 (pre-training, 100K steps) | 64 × 22 hrs = 1,408 | 22 hrs | $2,816 |
| Phase 4 (evaluation) | 8 × 20 hrs = 160 | 20 hrs | $320 |
| Phase 5 (ablations, 5 runs) | 64 × 50 hrs = 3,200 | 50 hrs | $6,400 |
| **Total** | **5,920** | **~5 days active** | **$11,840** |

**Fallback: 8×B200 (single node)**

| Phase | GPU-Hours | Wall Time | Cost (@$7.50/hr) |
|-------|-----------|-----------|-------------------|
| Phase 0-5 combined | 8 × 200 hrs = 1,600 | ~8 days | $12,000 |

### 12.2 Personnel

- 1 PhD student (full-time, 14 weeks)
- 1 Advisor (guidance, 2 hours/week)
- Optional: 1 Undergraduate for data preprocessing

### 12.3 Total Budget

- Compute (64×H100): ~$12,000
- Storage (datasets + features): ~$200 (1 TB cloud)
- Personnel: Covered by lab funding
- **Total: ~$12,200**

---

## 13. Conclusion

Drifting-VLA v2 represents a principled approach to unifying multi-embodiment robotic manipulation through one-step action generation. By combining frozen VLM backbones (for rich visual-language understanding) with Drifting-based action generation (for fast, multi-modal inference), we address key limitations of existing foundation models:

1. **Speed:** 10× faster than diffusion-based policies (RDT-1B, Diffusion Policy)
2. **Generality:** First model to handle gripper, bimanual, and dexterous hand in one architecture
3. **Efficiency:** 127M trainable parameters vs RDT's 1.2B (9× fewer)
4. **Scalability:** Proven training pipeline from toy case (1 task, 1 hour) to full scale (5 datasets, 2 days)

The phased development plan ensures early validation at each stage, minimizing risk of wasted compute on a broken pipeline. With **64×H100 GPUs** (effective batch 8192, matching the Drifting paper exactly) and **14 weeks**, this project is **feasible and ambitious**, targeting a high-quality NeurIPS 2026 submission that advances the state-of-the-art in robotic foundation models.

The codebase supports both 64×H100 (8 nodes) and 8×B200 (1 node) configurations, with automatic adjustment of batch size, gradient accumulation, and loss formulation. The 64×H100 configuration enables faithful reproduction of the Drifting paper's training regime while extending to multi-embodiment robotics.

---

## References

1. Drifting: One-Step Generation via Training-Time Distribution Evolution. arXiv:2602.04770, 2025.
2. RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. arXiv:2410.07864, 2024. [GitHub](https://github.com/thu-ml/RoboticsDiffusionTransformer)
3. DexGraspNet 2.0: Learning Generative Dexterous Grasping. CoRL 2024. [GitHub](https://github.com/PKU-EPIC/DexGraspNet2)
4. PaliGemma 2: A Family of Versatile VLMs for Transfer. arXiv:2412.03555, 2024.
5. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864, 2023.
6. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023.
7. 3D Diffuser Actor: Policy Diffusion with 3D Scene Representations. arXiv:2402.10885, 2024.
