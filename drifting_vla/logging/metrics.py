"""
Evaluation Metrics for Drifting-VLA
===================================

Metrics for evaluating VLA policy performance:
- Success rate
- Action distribution quality (MMD)
- Multi-modality analysis
"""

import torch
import numpy as np
from typing import Optional, Union
from sklearn.metrics.pairwise import rbf_kernel
import logging

logger = logging.getLogger(__name__)


def compute_success_rate(
    results: list[dict],
    per_task: bool = False,
) -> Union[float, dict[str, float]]:
    """
    Compute success rate from rollout results.
    
    Args:
        results: List of rollout result dicts with 'success' and 'task' keys.
        per_task: If True, return per-task success rates.
    
    Returns:
        Overall success rate or dict of per-task rates.
    
    Example:
        >>> results = [{'success': True, 'task': 'reach'}, ...]
        >>> rate = compute_success_rate(results)
        >>> print(f"Success rate: {rate:.2%}")
    """
    if not results:
        return 0.0 if not per_task else {}
    
    if per_task:
        task_results = {}
        for r in results:
            task = r.get('task', 'unknown')
            if task not in task_results:
                task_results[task] = []
            task_results[task].append(r.get('success', False))
        
        return {task: np.mean(successes) for task, successes in task_results.items()}
    else:
        return np.mean([r.get('success', False) for r in results])


def compute_action_mmd(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
    bandwidth: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy between predicted and GT actions.
    
    MMD measures the distance between two distributions in a kernel
    feature space. Lower values indicate more similar distributions.
    
    Args:
        actions_pred: Predicted actions [N, D].
        actions_gt: Ground truth actions [M, D].
        bandwidth: RBF kernel bandwidth.
    
    Returns:
        MMD value (lower is better, 0 = identical distributions).
    
    Example:
        >>> pred = model.generate(obs)
        >>> gt = expert_actions
        >>> mmd = compute_action_mmd(pred, gt)
        >>> print(f"MMD: {mmd:.4f}")
    """
    # Flatten if needed
    if actions_pred.ndim > 2:
        actions_pred = actions_pred.reshape(actions_pred.shape[0], -1)
    if actions_gt.ndim > 2:
        actions_gt = actions_gt.reshape(actions_gt.shape[0], -1)
    
    gamma = 1.0 / (2 * bandwidth ** 2)
    
    # Compute kernel matrices
    XX = rbf_kernel(actions_pred, actions_pred, gamma=gamma)
    YY = rbf_kernel(actions_gt, actions_gt, gamma=gamma)
    XY = rbf_kernel(actions_pred, actions_gt, gamma=gamma)
    
    # Compute MMD^2
    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    
    return np.sqrt(max(mmd_squared, 0))


def compute_multimodality_score(
    actions: np.ndarray,
    n_modes: int = 5,
    method: str = 'gmm',
) -> dict[str, float]:
    """
    Compute multimodality metrics using GMM.
    
    Measures how well the predicted actions cover multiple modes
    of the action distribution.
    
    Args:
        actions: Action samples [N, D].
        n_modes: Number of modes to fit.
        method: Clustering method ('gmm', 'kmeans').
    
    Returns:
        Dict with:
        - 'bic': Bayesian Information Criterion (lower = better fit)
        - 'entropy': Entropy of mode assignment (higher = more diverse)
        - 'effective_modes': Effective number of modes
        - 'mode_balance': Balance metric (1.0 = perfectly balanced)
    
    Example:
        >>> scores = compute_multimodality_score(predicted_actions, n_modes=5)
        >>> print(f"Effective modes: {scores['effective_modes']:.2f}")
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    
    # Flatten if needed
    if actions.ndim > 2:
        actions = actions.reshape(actions.shape[0], -1)
    
    N = len(actions)
    
    if method == 'gmm':
        model = GaussianMixture(n_components=n_modes, covariance_type='full')
        model.fit(actions)
        
        # Get assignment probabilities
        probs = model.predict_proba(actions)
        
        # Compute metrics
        bic = model.bic(actions)
        
        # Entropy of mode assignment
        entropy_per_sample = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        entropy = np.mean(entropy_per_sample)
        
        # Effective number of modes
        effective_modes = np.exp(entropy)
        
        # Mode balance (how evenly samples are distributed)
        mode_counts = np.sum(probs, axis=0)
        mode_probs = mode_counts / N
        ideal_prob = 1.0 / n_modes
        mode_balance = 1.0 - np.std(mode_probs) / ideal_prob
        
    else:  # kmeans
        model = KMeans(n_clusters=n_modes)
        labels = model.fit_predict(actions)
        
        bic = -model.inertia_  # Use negative inertia as BIC proxy
        
        # Mode distribution
        mode_counts = np.bincount(labels, minlength=n_modes)
        mode_probs = mode_counts / N
        
        # Entropy
        entropy = -np.sum(mode_probs * np.log(mode_probs + 1e-10))
        effective_modes = np.exp(entropy)
        
        # Balance
        ideal_prob = 1.0 / n_modes
        mode_balance = 1.0 - np.std(mode_probs) / ideal_prob
    
    return {
        'bic': bic,
        'entropy': entropy,
        'effective_modes': effective_modes,
        'mode_balance': max(0, mode_balance),
    }


def compute_action_error(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
) -> dict[str, float]:
    """
    Compute action prediction errors.
    
    Args:
        actions_pred: Predicted actions [N, T, D] or [N, D].
        actions_gt: Ground truth actions [N, T, D] or [N, D].
    
    Returns:
        Dict with error metrics:
        - 'mse': Mean squared error
        - 'mae': Mean absolute error
        - 'rmse': Root mean squared error
        - 'max_error': Maximum absolute error
    """
    diff = actions_pred - actions_gt
    
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(diff))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
    }


def compute_trajectory_metrics(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> dict[str, float]:
    """
    Compute trajectory-level metrics.
    
    Args:
        pred_positions: Predicted positions [T, 3].
        gt_positions: Ground truth positions [T, 3].
    
    Returns:
        Dict with trajectory metrics:
        - 'final_distance': Distance at final timestep
        - 'average_distance': Average distance over trajectory
        - 'max_deviation': Maximum deviation from GT
        - 'path_length_ratio': Ratio of pred/gt path lengths
    """
    # Per-timestep distances
    distances = np.linalg.norm(pred_positions - gt_positions, axis=-1)
    
    final_distance = distances[-1]
    average_distance = np.mean(distances)
    max_deviation = np.max(distances)
    
    # Path lengths
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=-1))
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=-1))
    path_length_ratio = pred_length / (gt_length + 1e-8)
    
    return {
        'final_distance': final_distance,
        'average_distance': average_distance,
        'max_deviation': max_deviation,
        'path_length_ratio': path_length_ratio,
    }


def compute_drift_equilibrium(
    drift_norms: list[float],
    window: int = 100,
) -> dict[str, float]:
    """
    Analyze drift norm convergence (equilibrium indicator).
    
    As training progresses, drift norm should decrease toward zero,
    indicating the generated distribution is approaching the data.
    
    Args:
        drift_norms: List of ||V||^2 values over training.
        window: Smoothing window size.
    
    Returns:
        Dict with:
        - 'final_norm': Last drift norm value
        - 'convergence_rate': Rate of decrease
        - 'smoothed_final': Smoothed final value
        - 'is_converging': Whether norm is decreasing
    """
    if len(drift_norms) < window:
        window = len(drift_norms)
    
    norms = np.array(drift_norms)
    
    # Smoothed values
    smoothed = np.convolve(norms, np.ones(window)/window, mode='valid')
    
    # Final values
    final_norm = norms[-1]
    smoothed_final = smoothed[-1] if len(smoothed) > 0 else final_norm
    
    # Convergence rate (negative = decreasing)
    if len(smoothed) > 1:
        convergence_rate = (smoothed[-1] - smoothed[0]) / len(smoothed)
    else:
        convergence_rate = 0.0
    
    is_converging = convergence_rate < 0
    
    return {
        'final_norm': final_norm,
        'smoothed_final': smoothed_final,
        'convergence_rate': convergence_rate,
        'is_converging': is_converging,
    }


