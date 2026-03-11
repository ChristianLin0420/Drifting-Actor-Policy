"""
Drifting Field Computation (Algorithm 2)
========================================

Implements the drifting field V_{p,q}(x) from:
  "Generative Modeling via Drifting" (Deng et al., 2025)
  https://arxiv.org/abs/2602.04770

Aligned with the paper's Algorithm 2 (Appendix A.1) and normalization
scheme (Appendix A.6, Eq. 18-25):

  V(x) = E_{p,q}[ k̃(x,y+) k̃(x,y-) (y+ - y-) ]        — Eq. (11)

where k̃ is the doubly-stochastic normalized kernel (Alg. 2) and the
implementation uses cross-weighting W_pos = A_pos * Σ(A_neg) per Alg. 2.

Feature normalization: scale features so E[‖x-y‖] ≈ √D, then use
τ̃ = τ·√D for the kernel logits (Eq. 22).  This makes τ values
(default {0.02, 0.05, 0.2}) dimension-independent.

Drift normalization: λ = √(E[‖V‖²/D]), Ṽ = V/λ  (Eq. 23-25).
λ→0 at equilibrium, serving as the convergence metric.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional


def compute_drifting_field(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: list[float] = [0.02, 0.05, 0.2],
    normalize_features: bool = True,
    normalize_drift: bool = True,
    eps: float = 1e-8,
    return_raw_norm: bool = False,
):
    """
    Compute the drifting field V_{p,q}(x) for generated samples.

    Follows Algorithm 2 and Appendix A.6 of the paper exactly:
      1. Feature normalization so E[‖x̃-ỹ‖] ≈ √D  (Eq. 18-21)
      2. Kernel logits: -dist / (τ·√D)              (Eq. 22)
      3. Doubly-stochastic normalization (Alg. 2)
      4. Cross-weighting: W_pos *= Σ(A_neg)          (Alg. 2)
      5. V = W_pos @ ỹ_pos - W_neg @ ỹ_neg          (Eq. 11)
      6. Drift normalization: Ṽ = V/λ               (Eq. 23-25)

    Args:
        x: Generated samples [N, D].
        y_pos: Positive samples (expert actions) [N_pos, D].
        y_neg: Negative samples (generated) [N_neg, D].
        temperatures: Temperature values for multi-scale kernels.
        normalize_features: Normalize features so E[‖x-y‖] ≈ √D.
        normalize_drift: Normalize drift field (E[‖V‖²/D] ≈ 1).
        eps: Numerical stability constant.
        return_raw_norm: If True, also return (V_raw_norm, lambda_V).

    Returns:
        V: Drifting field [N, D] (normalized if normalize_drift=True).
        If return_raw_norm=True, returns (V, V_raw_norm, lambda_V).
    """
    N, D = x.shape

    if normalize_features:
        scale = compute_feature_normalization_scale(x, y_pos, y_neg, D, eps)
        x_n = x / (scale + eps)
        y_pos_n = y_pos / (scale + eps)
        y_neg_n = y_neg / (scale + eps)
    else:
        x_n = x
        y_pos_n = y_pos
        y_neg_n = y_neg

    V_total = torch.zeros_like(x)

    for tau in temperatures:
        V_tau = _compute_drifting_field_single_temp(
            x_n, y_pos_n, y_neg_n, tau, D, eps,
        )
        V_total = V_total + V_tau

    V_raw_norm = (V_total ** 2).sum(dim=1).mean().detach()
    lambda_V = torch.sqrt(V_raw_norm / D + eps).detach()

    if normalize_drift:
        V_total = normalize_drift_field(V_total, D, eps)

    if return_raw_norm:
        return V_total, V_raw_norm, lambda_V
    return V_total


def _compute_drifting_field_single_temp(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    tau: float,
    D: int,
    eps: float,
) -> torch.Tensor:
    """
    Drifting field for one temperature — Algorithm 2 of the paper.

    All inputs are in the same (optionally normalized) feature space.
    Kernel logits use τ̃ = τ·√D so that τ values are dimension-independent
    (Appendix A.6, Eq. 22).

    Drift is computed as V = W_pos @ y_pos - W_neg @ y_neg (Eq. 11),
    with NO center-subtraction (unlike mean-shift Eq. 8/10 decomposition).
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    dist_pos = torch.cdist(x, y_pos, p=2)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg, p=2)  # [N, N_neg]

    if N == N_neg:
        self_mask = torch.eye(N, device=x.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(self_mask, float('inf'))

    # τ̃ = τ · √D  (Eq. 22 — makes τ dimension-independent)
    tau_scaled = tau * math.sqrt(D)

    logit_pos = -dist_pos / tau_scaled  # [N, N_pos]
    logit_neg = -dist_neg / tau_scaled  # [N, N_neg]

    logits = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # Doubly-stochastic normalization (Alg. 2)
    A_row = F.softmax(logits, dim=1)
    A_col = F.softmax(logits, dim=0)
    A = torch.sqrt(A_row * A_col + eps)

    A_pos = A[:, :N_pos]  # [N, N_pos]
    A_neg = A[:, N_pos:]  # [N, N_neg]

    # Cross-weighting (Alg. 2): W_pos *= Σ(A_neg), W_neg *= Σ(A_pos)
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)  # [N, N_neg]

    # V = W_pos @ y_pos - W_neg @ y_neg  (Eq. 11, no center-subtraction)
    V = W_pos @ y_pos - W_neg @ y_neg  # [N, D]

    return V


def compute_feature_normalization_scale(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    D: int,
    eps: float = 1e-8,
    max_samples: int = 1000,
) -> torch.Tensor:
    """
    Compute normalization scale such that E[||x - y||] ≈ sqrt(D).
    
    This provides scale invariance by normalizing features based on
    the average pairwise distance in the batch. This is important
    for ensuring consistent kernel behavior across different feature
    scales and dimensions.
    
    Args:
        x: Generated samples [N, D]
        y_pos: Positive samples [N_pos, D]
        y_neg: Negative samples [N_neg, D]
        D: Feature dimension (for target scale)
        eps: Numerical stability constant
        max_samples: Maximum samples for distance computation
            (subsampling for memory efficiency)
    
    Returns:
        scale: Scalar normalization factor
    
    Note:
        The scale is computed with stop-gradient to prevent it from
        affecting the optimization of the model parameters.
    """
    # Concatenate all samples
    all_samples = torch.cat([x, y_pos, y_neg], dim=0)  # [N + N_pos + N_neg, D]
    N_total = all_samples.shape[0]
    
    # Subsample if too many samples (memory efficiency)
    if N_total > max_samples:
        idx = torch.randperm(N_total, device=all_samples.device)[:max_samples]
        all_samples = all_samples[idx]
    
    # Compute pairwise distances
    dist = torch.cdist(all_samples, all_samples, p=2)  # [N_sub, N_sub]
    
    # Compute average distance (excluding diagonal)
    mask = ~torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
    avg_dist = dist[mask].mean()
    
    # Scale such that avg_dist ≈ sqrt(D)
    target_dist = torch.sqrt(torch.tensor(D, dtype=avg_dist.dtype, device=avg_dist.device))
    scale = avg_dist / (target_dist + eps)
    
    return scale.detach()  # Stop gradient


def normalize_drift_field(
    V: torch.Tensor,
    D: int,
    eps: float = 1e-8,
    return_lambda: bool = False,
) -> torch.Tensor:
    """
    Normalize drift field such that E[||V||^2 / D] ≈ 1.
    
    This normalization ensures stable training by keeping the drift
    magnitude consistent regardless of the feature dimension or
    the number of samples.
    
    The normalization factor lambda_V is the KEY CONVERGENCE METRIC:
    - Large lambda_V = predictions far from data (early training)
    - Small lambda_V = predictions close to data (converged)
    - lambda_V → 0 at equilibrium (p_theta = p_data)
    
    Args:
        V: Drifting field [N, D]
        D: Feature dimension
        eps: Numerical stability constant
        return_lambda: If True, return (V_normalized, lambda_V)
    
    Returns:
        V_normalized: Normalized drifting field [N, D]
        lambda_V: (optional) Normalization factor (convergence metric)
    
    Note:
        The normalization factor is computed with stop-gradient.
    """
    # Compute normalization factor
    norm_squared = (V ** 2).sum(dim=1).mean()  # E[||V||^2]
    lambda_V = torch.sqrt(norm_squared / D + eps)
    
    # Normalize
    V_normalized = V / (lambda_V.detach() + eps)
    
    if return_lambda:
        return V_normalized, lambda_V.detach()
    return V_normalized


def compute_kernel_similarities(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Compute kernel similarities between x and y samples.
    
    Uses the exponential kernel: k(x, y) = exp(-||x - y|| / tau)
    
    Args:
        x: Query samples [N, D]
        y: Key samples [M, D]
        tau: Temperature parameter
    
    Returns:
        K: Kernel similarity matrix [N, M]
    
    Note:
        Lower temperature makes the kernel more selective (focused on
        nearest neighbors), while higher temperature provides more
        global information.
    """
    dist = torch.cdist(x, y, p=2)  # [N, M]
    K = torch.exp(-dist / tau)  # [N, M]
    return K


