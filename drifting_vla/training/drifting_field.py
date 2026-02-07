"""
Drifting Field Computation (Algorithm 2)
========================================

This module implements the core drifting field computation from the Drifting
model paper. The drifting field V_{p,q}(x) provides the direction to evolve
generated samples toward the data distribution.

Mathematical Formulation:
    V_{p,q}(x) = V_p^+(x) - V_q^-(x)
    
    where:
    - V_p^+(x) = E_{y+ ~ p}[k_tilde(x, y+)(y+ - x)]  (attraction to data)
    - V_q^-(x) = E_{y- ~ q}[k_tilde(x, y-)(y- - x)]  (repulsion from generated)
    - k_tilde(x, y) = k(x, y) / Z(x)  (normalized kernel)
    - k(x, y) = exp(-||x - y|| / tau)  (base kernel)

Key Features:
    - Multi-temperature kernels (tau = 0.02, 0.05, 0.2)
    - Double softmax normalization (row and column)
    - Feature normalization for scale invariance
    - Drift normalization for stable training
"""

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
    
    The drifting field provides the direction to evolve generated samples
    toward the data distribution through attraction to positive samples
    and repulsion from negative (generated) samples.
    
    Args:
        x: Generated samples [N, D].
        y_pos: Positive samples (expert actions) [N_pos, D].
        y_neg: Negative samples (generated) [N_neg, D].
        temperatures: Temperature values for multi-scale kernels.
        normalize_features: Normalize features for scale invariance.
        normalize_drift: Normalize drift field (E[||V||^2/D] ≈ 1).
        eps: Numerical stability constant.
        return_raw_norm: If True, also return (V_raw_norm, lambda_V).
    
    Returns:
        V: Drifting field [N, D] (normalized if normalize_drift=True).
        If return_raw_norm=True, returns (V, V_raw_norm, lambda_V):
            V_raw_norm: E[||V_raw||^2] before normalization (convergence metric)
            lambda_V: Normalization factor sqrt(E[||V||^2]/D) → 0 at convergence
    """
    N, D = x.shape
    
    # Feature normalization (scale invariance)
    if normalize_features:
        scale = compute_feature_normalization_scale(x, y_pos, y_neg, D, eps)
        x_normalized = x / (scale + eps)
        y_pos_normalized = y_pos / (scale + eps)
        y_neg_normalized = y_neg / (scale + eps)
    else:
        x_normalized = x
        y_pos_normalized = y_pos
        y_neg_normalized = y_neg
    
    # Accumulate drifting field across temperatures
    V_total = torch.zeros_like(x)
    
    for tau in temperatures:
        V_tau = _compute_drifting_field_single_temp(
            x_normalized, y_pos_normalized, y_neg_normalized,
            x, y_pos, y_neg,
            tau, eps
        )
        V_total = V_total + V_tau
    
    # Compute raw drift magnitude BEFORE normalization (true convergence metric)
    V_raw_norm = (V_total ** 2).sum(dim=1).mean().detach()
    lambda_V = torch.sqrt(V_raw_norm / D + eps).detach()
    
    # Drift normalization (stable training)
    if normalize_drift:
        V_total = normalize_drift_field(V_total, D, eps)
    
    if return_raw_norm:
        return V_total, V_raw_norm, lambda_V
    return V_total


def _compute_drifting_field_single_temp(
    x_norm: torch.Tensor,
    y_pos_norm: torch.Tensor, 
    y_neg_norm: torch.Tensor,
    x_orig: torch.Tensor,
    y_pos_orig: torch.Tensor,
    y_neg_orig: torch.Tensor,
    tau: float,
    eps: float,
) -> torch.Tensor:
    """
    Compute drifting field for a single temperature value.
    
    This implements the double softmax normalization from the paper:
    1. Row-wise softmax: normalize over y-axis (samples)
    2. Column-wise softmax: normalize over x-axis (queries)
    3. Geometric mean of both normalizations
    
    Args:
        x_norm: Normalized generated samples [N, D]
        y_pos_norm: Normalized positive samples [N_pos, D]
        y_neg_norm: Normalized negative samples [N_neg, D]
        x_orig: Original generated samples (for drift direction) [N, D]
        y_pos_orig: Original positive samples [N_pos, D]
        y_neg_orig: Original negative samples [N_neg, D]
        tau: Temperature value for the kernel
        eps: Numerical stability constant
    
    Returns:
        V: Drifting field at this temperature [N, D]
    """
    N = x_norm.shape[0]
    N_pos = y_pos_norm.shape[0]
    N_neg = y_neg_norm.shape[0]
    
    # Compute pairwise L2 distances
    dist_pos = torch.cdist(x_norm, y_pos_norm, p=2)  # [N, N_pos]
    dist_neg = torch.cdist(x_norm, y_neg_norm, p=2)  # [N, N_neg]
    
    # Handle self-comparison in negatives (if y_neg is x)
    # Check if shapes match and content is similar
    if N == N_neg:
        # Create mask for self-comparisons
        self_mask = torch.eye(N, device=x_norm.device, dtype=torch.bool)
        dist_neg = dist_neg.masked_fill(self_mask, float('inf'))
    
    # Compute kernel logits (negative distance / temperature)
    logit_pos = -dist_pos / tau  # [N, N_pos]
    logit_neg = -dist_neg / tau  # [N, N_neg]
    
    # Concatenate for joint normalization
    logits = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]
    
    # Double softmax normalization
    # Row-wise: normalize over all samples (positive + negative)
    A_row = F.softmax(logits, dim=1)  # [N, N_pos + N_neg]
    
    # Column-wise: normalize over all queries
    A_col = F.softmax(logits, dim=0)  # [N, N_pos + N_neg]
    
    # Geometric mean of normalizations
    A = torch.sqrt(A_row * A_col + eps)  # [N, N_pos + N_neg]
    
    # Split attention weights
    A_pos = A[:, :N_pos]  # [N, N_pos]
    A_neg = A[:, N_pos:]  # [N, N_neg]
    
    # Compute cross-attention weights
    # W_pos weights positive samples by negative attention
    # W_neg weights negative samples by positive attention
    neg_sum = A_neg.sum(dim=1, keepdim=True)  # [N, 1]
    pos_sum = A_pos.sum(dim=1, keepdim=True)  # [N, 1]
    
    W_pos = A_pos * neg_sum  # [N, N_pos]
    W_neg = A_neg * pos_sum  # [N, N_neg]
    
    # Compute drift directions (use original scale)
    # Attraction toward positive samples
    drift_pos = W_pos @ y_pos_orig  # [N, D]
    center_pos = W_pos.sum(dim=1, keepdim=True) * x_orig  # Weighted center
    V_pos = drift_pos - center_pos  # [N, D]
    
    # Repulsion from negative samples  
    drift_neg = W_neg @ y_neg_orig  # [N, D]
    center_neg = W_neg.sum(dim=1, keepdim=True) * x_orig
    V_neg = drift_neg - center_neg  # [N, D]
    
    # Final drifting field: attraction - repulsion
    V = V_pos - V_neg  # [N, D]
    
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


