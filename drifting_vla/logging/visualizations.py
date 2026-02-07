"""
Visualization Functions for Drifting-VLA
=========================================

Paper-ready visualizations for understanding and debugging Drifting-VLA training.

Visualization Suite:
--------------------
**Category A — Training Convergence** (periodic during training)
  A1. Drifting Field: 2D arrow plot showing drift vectors from predictions → data
  A2. Drift Magnitude Trend: ||V||² over recent steps (convergence signal)
  A3. Prediction vs GT Scatter: Calibration per action dimension
  A4. Per-Dim Error Radar: Spider chart of MAE per action dimension
  A5. Temperature Loss Breakdown: Which scale dominates learning

**Category B — Action Quality** (at eval)
  B1. Action Distribution: Per-dim histograms pred vs GT
  B2. Action Error Heatmap: [T × D] error map showing where predictions fail
  B3. Eval Trajectory Video: Animated pred vs GT (handled in trainer)
  B4. 3D Trajectory: End-effector path

**Category C — Drifting-Specific**
  C1. Sample Transport: Before/after drifting overlay
  C2. Mode Coverage: GMM clustering + proportions

Interpretation Guide:
---------------------
- Drifting Field: Arrows shrink as model converges (predictions near data).
- Drift Magnitude: Should decrease over training. Plateau = stuck.
- Prediction Scatter: Points on y=x diagonal = perfect. Color = error.
- Error Radar: Spikes = hard dimensions (rotation is common).
- Temperature Breakdown: Low-τ = precision, High-τ = mode discovery.
- Action Error Heatmap: Vertical stripe = one dim always wrong. 
  Errors growing with timestep = compounding.
- Sample Transport: Orange dots near green = good convergence.
- Mode Coverage: Even bars = no mode collapse.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Union
import logging

from drifting_vla.logging.themes import ThemeManager

logger = logging.getLogger(__name__)


# =============================================================================
# Color Palette
# =============================================================================
DRIFT_CMAP = LinearSegmentedColormap.from_list(
    'drift', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
)
TRAJECTORY_CMAP = LinearSegmentedColormap.from_list(
    'trajectory', ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8', '#FFB703', '#FB8500', '#E63946']
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    'error', ['#FFFFFF', '#FFD166', '#EF476F', '#9B2226']
)

ACTION_PRED_COLOR = '#3A86FF'
ACTION_GT_COLOR = '#06D6A0'
POSITIVE_COLOR = '#06D6A0'
NEGATIVE_COLOR = '#EF476F'
QUERY_COLOR = '#118AB2'

# Consistent action dimension names
DEFAULT_DIM_NAMES = ['pos_x', 'pos_y', 'pos_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'grip']


# =============================================================================
# A1. Drifting Field
# =============================================================================
def plot_drifting_field(
    x: np.ndarray,
    V: np.ndarray,
    y_pos: Optional[np.ndarray] = None,
    y_neg: Optional[np.ndarray] = None,
    title: str = 'Drifting Field',
    theme: str = 'light',
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D drifting field with gradient-colored arrows.

    **How to Interpret:**
    - Green circles = Expert data (positive samples) — the target
    - Colored squares = Model predictions — current state
    - Arrows = Drifting field V, pointing predictions toward data
    - Arrow color/width = drift magnitude (blue→red = weak→strong)
    - At convergence: arrows become small, predictions overlap data

    Args:
        x: Generated samples [N, 2] (2D projection).
        V: Drifting field vectors [N, 2].
        y_pos: Positive samples [N_pos, 2] (optional).
        y_neg: Negative samples [N_neg, 2] (optional).
        title: Plot title.
        theme: Color theme ('light', 'dark', 'paper').
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    if theme == 'dark':
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#16213e')

    # Drift magnitudes for coloring
    magnitudes = np.linalg.norm(V, axis=1)
    mag_normalized = magnitudes / (magnitudes.max() + 1e-8)

    # Expert data (with glow)
    if y_pos is not None:
        ax.scatter(y_pos[:, 0], y_pos[:, 1], c=POSITIVE_COLOR, s=200, alpha=0.15,
                   marker='o', edgecolors='none')
        ax.scatter(y_pos[:, 0], y_pos[:, 1], c=POSITIVE_COLOR, s=80, alpha=0.8,
                   marker='o', edgecolors='white', linewidths=1.5,
                   label='Expert Data', zorder=5)

    # Negative samples
    if y_neg is not None:
        ax.scatter(y_neg[:, 0], y_neg[:, 1], c=NEGATIVE_COLOR, s=40, alpha=0.5,
                   marker='x', linewidths=1.5, label='Negative Samples', zorder=4)

    # Model predictions colored by drift magnitude
    scatter = ax.scatter(x[:, 0], x[:, 1], c=mag_normalized, cmap=DRIFT_CMAP,
                         s=60, alpha=0.9, marker='s', edgecolors='white',
                         linewidths=0.8, label='Model Predictions', zorder=6)

    # Gradient-colored drift arrows
    V_norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V_direction = V / V_norm
    arrow_scale = 0.3 * (magnitudes / (magnitudes.max() + 1e-8))[:, None]
    V_scaled = V_direction * np.maximum(arrow_scale, 0.05)

    for i in range(len(x)):
        color = DRIFT_CMAP(mag_normalized[i])
        ax.annotate('', xy=(x[i, 0] + V_scaled[i, 0], x[i, 1] + V_scaled[i, 1]),
                    xytext=(x[i, 0], x[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5 + mag_normalized[i] * 2,
                                    alpha=0.7 + mag_normalized[i] * 0.3),
                    zorder=3)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Drift Magnitude (normalized)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('Principal Component 1', fontsize=12, fontweight='medium')
    ax.set_ylabel('Principal Component 2', fontsize=12, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')

    ax.text(0.02, 0.02,
            'Arrows: drift direction & strength\nSmaller arrows = closer to equilibrium',
            transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='bottom',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
    return fig


# =============================================================================
# A2. Drift Magnitude Trend
# =============================================================================
def plot_drift_magnitude_trend(
    drift_norms: list[float],
    steps: list[int],
    title: str = 'Drift Magnitude Trend',
    theme: str = 'light',
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot ||V||² over training steps — the core convergence signal.

    **How to Interpret:**
    - Decreasing curve = model is converging (predictions approaching data)
    - Plateau at high value = model stuck, not learning
    - Sudden spike = instability (LR too high or bad batch)
    - Near-zero = model has converged

    Args:
        drift_norms: List of drift magnitudes ||V||².
        steps: Corresponding training steps.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    color = '#3A86FF'
    ax.plot(steps, drift_norms, color=color, linewidth=1.5, alpha=0.35, label='Raw')

    # Smoothed trend
    if len(drift_norms) > 5:
        window = max(len(drift_norms) // 10, 3)
        kernel = np.ones(window) / window
        smoothed = np.convolve(drift_norms, kernel, mode='valid')
        offset = window // 2
        ax.plot(steps[offset:offset + len(smoothed)], smoothed,
                color=color, linewidth=2.5, label='Smoothed')

    ax.fill_between(steps, drift_norms, alpha=0.08, color=color)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('‖V‖² (Drift Magnitude)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.98, 0.95, '↓ Lower = Better Convergence',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color='gray', style='italic')

    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# A3. Prediction vs GT Scatter (Calibration)
# =============================================================================
def plot_prediction_scatter(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
    dim_names: Optional[list[str]] = None,
    title: str = 'Prediction vs Ground Truth',
    theme: str = 'light',
    figsize: tuple = (16, 4),
) -> plt.Figure:
    """
    Scatter plot of predicted vs ground truth for each action dimension.

    **How to Interpret:**
    - Red dashed line = perfect prediction (y = x)
    - Points on diagonal = accurate predictions
    - Points above diagonal = model over-predicts
    - Points below diagonal = model under-predicts
    - Tight cluster = low variance. Wide spread = high uncertainty.
    - Color = error magnitude (green=low, red=high)

    Args:
        actions_pred: Predicted actions [N, D].
        actions_gt: Ground truth actions [N, D].
        dim_names: Names for each action dimension.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    D = min(actions_pred.shape[1], 8)
    dim_names = dim_names or DEFAULT_DIM_NAMES[:D]

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), layout='constrained')
    axes = np.atleast_2d(axes).flatten()

    for i in range(D):
        ax = axes[i]
        gt = actions_gt[:, i]
        pred = actions_pred[:, i]

        errors = np.abs(pred - gt)
        err_norm = errors / (errors.max() + 1e-8)

        ax.scatter(gt, pred, c=err_norm, cmap='RdYlGn_r', s=20, alpha=0.7,
                   edgecolors='white', linewidths=0.3)

        # Perfect prediction line
        lims = [min(gt.min(), pred.min()), max(gt.max(), pred.max())]
        margin = (lims[1] - lims[0]) * 0.05 + 1e-6
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'r--', linewidth=1.5, alpha=0.6, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f'GT {dim_names[i]}', fontsize=9)
        ax.set_ylabel(f'Pred {dim_names[i]}', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, linestyle='--')

        mae = np.mean(errors)
        ax.text(0.05, 0.92, f'MAE={mae:.4f}', transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for ax in axes[D:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# A4. Per-Dimension Error Radar
# =============================================================================
def plot_per_dim_error_radar(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
    dim_names: Optional[list[str]] = None,
    title: str = 'Per-Dimension Error',
    theme: str = 'light',
    figsize: tuple = (7, 7),
) -> plt.Figure:
    """
    Radar/spider chart of MAE per action dimension.

    **How to Interpret:**
    - Each spoke = one action dimension
    - Radius = Mean Absolute Error for that dimension
    - Larger spike = model struggles with this dim
    - Symmetric shape = uniform difficulty
    - Spikes on rotation dims are common (harder to predict)

    Args:
        actions_pred: Predicted actions [N, D].
        actions_gt: Ground truth actions [N, D].
        dim_names: Names for each dimension.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    D = min(actions_pred.shape[1], 8)
    dim_names = dim_names or DEFAULT_DIM_NAMES[:D]

    mae_per_dim = np.mean(np.abs(actions_pred[:, :D] - actions_gt[:, :D]), axis=0)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, D, endpoint=False).tolist()
    values = mae_per_dim.tolist()
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), layout='constrained')

    ax.fill(angles, values, color='#3A86FF', alpha=0.15)
    ax.plot(angles, values, color='#3A86FF', linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='#3A86FF', markeredgecolor='white',
            markeredgewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names, fontsize=10, fontweight='medium')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=25)

    # Value annotations
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.annotate(f'{value:.4f}', xy=(angle, value), fontsize=8,
                    ha='center', va='bottom', color='#333333')

    ax.grid(True, alpha=0.3)
    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# A5. Temperature Loss Breakdown
# =============================================================================
def plot_temperature_loss_breakdown(
    per_temp_losses: dict[str, float],
    title: str = 'Loss by Temperature',
    theme: str = 'light',
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Bar chart showing loss contribution from each temperature scale.

    **How to Interpret:**
    - Low τ (0.02): Fine-grained local structure. High loss = model
      hasn't learned precise positioning.
    - Mid τ (0.05): Medium-scale structure.
    - High τ (0.2): Global mode coverage. High loss = model still
      discovering the main action modes.
    - As training progresses, all bars should decrease.
    - If low-τ stays high while high-τ is low: model found modes
      but lacks precision (common early in training).

    Args:
        per_temp_losses: Dict mapping 'tau_X.XX' to loss value.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    temps = sorted(per_temp_losses.keys(), key=lambda t: float(t.replace('tau_', '')))
    values = [per_temp_losses[t] for t in temps]
    labels = [t.replace('tau_', 'τ=') for t in temps]

    # Gradient colors from cool (fine-grained) to warm (global)
    n = len(temps)
    colors = [plt.cm.coolwarm(i / max(n - 1, 1)) for i in range(n)]

    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2, alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='medium')

    ax.set_xlabel('Temperature', fontsize=11, fontweight='medium')
    ax.set_ylabel('Loss', fontsize=11, fontweight='medium')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(0.02, 0.95, 'Low τ → precision  |  High τ → mode coverage',
            transform=ax.transAxes, fontsize=8, color='gray', style='italic', va='top')

    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# B1. Action Distribution
# =============================================================================
def plot_action_distribution(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
    dim_names: Optional[list[str]] = None,
    title: str = 'Action Distribution',
    theme: str = 'light',
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted vs ground truth action distributions with gradient styling.

    **How to Interpret:**
    - Green = Expert/Ground Truth distribution
    - Blue = Model predictions
    - Overlap = Good! Model learned the action distribution
    - Mismatch = Model hasn't captured certain action modes
    - Multi-modal GT with uni-modal pred = Mode collapse

    Args:
        actions_pred: Predicted actions [N, D].
        actions_gt: Ground truth actions [M, D].
        dim_names: Names for each dimension.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    D = actions_pred.shape[1]
    dim_names = dim_names or DEFAULT_DIM_NAMES[:D] if D <= len(DEFAULT_DIM_NAMES) else [f'dim_{i}' for i in range(D)]

    ncols = min(D, 4)
    nrows = (D + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), layout='constrained')
    axes = np.atleast_2d(axes).flatten()

    for i, (ax, name) in enumerate(zip(axes[:D], dim_names)):
        gt_hist, gt_bins = np.histogram(actions_gt[:, i], bins=40, density=True)
        pred_hist, pred_bins = np.histogram(actions_pred[:, i], bins=40, density=True)

        gt_centers = (gt_bins[:-1] + gt_bins[1:]) / 2
        pred_centers = (pred_bins[:-1] + pred_bins[1:]) / 2

        ax.fill_between(gt_centers, gt_hist, alpha=0.4, color=ACTION_GT_COLOR, label='Expert')
        ax.plot(gt_centers, gt_hist, color=ACTION_GT_COLOR, linewidth=2, alpha=0.9)
        ax.fill_between(pred_centers, pred_hist, alpha=0.4, color=ACTION_PRED_COLOR, label='Predicted')
        ax.plot(pred_centers, pred_hist, color=ACTION_PRED_COLOR, linewidth=2, alpha=0.9)

        ax.set_xlabel(name, fontsize=10, fontweight='medium')
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    for ax in axes[D:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    # (layout='constrained' set at figure creation ensures fixed size)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# =============================================================================
# B2. Action Error Heatmap
# =============================================================================
def plot_action_error_heatmap(
    actions_pred: np.ndarray,
    actions_gt: np.ndarray,
    dim_names: Optional[list[str]] = None,
    title: str = 'Action Error Heatmap',
    theme: str = 'light',
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Heatmap of |pred - GT| over [timestep × action_dim].

    **How to Interpret:**
    - X-axis: action dimensions (pos, rot, gripper)
    - Y-axis: timesteps in the action horizon
    - Color: absolute error (white=low, red=high)
    - Errors growing with timestep = compounding errors
    - Vertical stripe = one dimension is consistently wrong
    - Horizontal stripe = one timestep is consistently wrong

    Args:
        actions_pred: Predicted actions [T, D] (single sample).
        actions_gt: Ground truth actions [T, D] (single sample).
        dim_names: Names for action dimensions.
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    T, D = actions_pred.shape
    D = min(D, 8)
    dim_names = dim_names or DEFAULT_DIM_NAMES[:D]

    errors = np.abs(actions_pred[:, :D] - actions_gt[:, :D])

    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    im = ax.imshow(errors, cmap=ERROR_CMAP, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(D))
    ax.set_xticklabels(dim_names, fontsize=10, fontweight='medium')
    ax.set_xlabel('Action Dimension', fontsize=11)
    ax.set_ylabel('Timestep', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|Predicted − Ground Truth|', fontsize=10)

    # Annotate cells if small enough
    if T <= 20 and D <= 10:
        for t in range(T):
            for d in range(D):
                val = errors[t, d]
                color = 'white' if val > errors.max() * 0.6 else 'black'
                ax.text(d, t, f'{val:.3f}', ha='center', va='center',
                        fontsize=7, color=color)

    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# B4. 3D Trajectory
# =============================================================================
def render_trajectory_3d(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = '3D Trajectory',
    theme: str = 'light',
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render 3D robot trajectory with gradient coloring.

    **How to Interpret:**
    - Color gradient: Blue (start) → Cyan → Yellow → Red (end)
    - Smooth trajectory = coherent action predictions
    - Jagged trajectory = noisy/inconsistent predictions
    - Green marker = start position
    - Red star = end position

    Args:
        positions: Trajectory positions [T, 3].
        colors: Optional per-timestep colors [T].
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    fig = plt.figure(figsize=figsize, layout='constrained')
    ax = fig.add_subplot(111, projection='3d')

    T = len(positions)
    t_normalized = np.linspace(0, 1, T)
    trajectory_colors = TRAJECTORY_CMAP(t_normalized)

    for i in range(T - 1):
        ax.plot3D(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                  color=trajectory_colors[i], linewidth=3, alpha=0.9)

    sizes = np.linspace(30, 80, T)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=t_normalized, cmap=TRAJECTORY_CMAP, s=sizes, alpha=0.7,
               edgecolors='white', linewidths=0.5)

    # Start / End markers
    ax.scatter(*positions[0], color='#06D6A0', s=250, marker='o', alpha=0.3, zorder=10)
    ax.scatter(*positions[0], color='#06D6A0', s=150, marker='o',
               edgecolors='white', linewidths=2, label='Start', zorder=11)
    ax.scatter(*positions[-1], color='#E63946', s=300, marker='*', alpha=0.3, zorder=10)
    ax.scatter(*positions[-1], color='#E63946', s=200, marker='*',
               edgecolors='white', linewidths=1.5, label='End', zorder=11)

    ax.set_xlabel('X', fontsize=11, fontweight='medium')
    ax.set_ylabel('Y', fontsize=11, fontweight='medium')
    ax.set_zlabel('Z', fontsize=11, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.view_init(elev=20, azim=45)

    sm = plt.cm.ScalarMappable(cmap=TRAJECTORY_CMAP, norm=plt.Normalize(0, T-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Timestep', fontsize=10)

    # (layout='constrained' set at figure creation ensures fixed size)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# =============================================================================
# C1. Sample Transport (Before/After Drifting)
# =============================================================================
def plot_sample_transport(
    x_before: np.ndarray,
    x_after: np.ndarray,
    y_gt: np.ndarray,
    title: str = 'Sample Transport (Drifting)',
    theme: str = 'light',
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Show how drifting moves samples from noise toward data.

    **How to Interpret:**
    - Gray squares = initial samples (noise/pre-drift)
    - Orange circles = samples after one-step drifting correction
    - Green diamonds = ground truth data
    - Arrows show the transport direction
    - Good training: orange dots cluster near green dots
    - Bad training: orange dots scattered, far from green

    Args:
        x_before: Samples before drifting [N, 2].
        x_after: Samples after drifting [N, 2].
        y_gt: Ground truth samples [M, 2].
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')

    # Ground truth
    ax.scatter(y_gt[:, 0], y_gt[:, 1], c=POSITIVE_COLOR, s=80, alpha=0.5,
               marker='D', edgecolors='white', linewidths=1, label='Ground Truth', zorder=3)

    # Before drifting
    ax.scatter(x_before[:, 0], x_before[:, 1], c='#ADB5BD', s=50, alpha=0.6,
               marker='s', edgecolors='white', linewidths=0.5,
               label='Before Drift (noise)', zorder=4)

    # After drifting
    ax.scatter(x_after[:, 0], x_after[:, 1], c='#FB8500', s=60, alpha=0.8,
               marker='o', edgecolors='white', linewidths=1,
               label='After Drift', zorder=5)

    # Transport arrows
    for i in range(len(x_before)):
        ax.annotate('', xy=(x_after[i, 0], x_after[i, 1]),
                    xytext=(x_before[i, 0], x_before[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='#6C757D',
                                    lw=1.0, alpha=0.4),
                    zorder=2)

    ax.set_xlabel('PC 1', fontsize=11)
    ax.set_ylabel('PC 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')

    ax.text(0.02, 0.02,
            'Arrows: noise → drifted position\nOrange near green = good convergence',
            transform=ax.transAxes, fontsize=8, color='gray', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (layout='constrained' set at figure creation ensures fixed size)
    return fig


# =============================================================================
# C2. Mode Coverage
# =============================================================================
def plot_mode_coverage(
    actions: np.ndarray,
    n_modes: int = 5,
    method: str = 'gmm',
    title: str = 'Mode Coverage',
    theme: str = 'light',
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Clustering analysis to detect mode collapse.

    **How to Interpret:**
    - Left: 2D PCA projection with cluster (mode) assignments
    - Right: Bar chart showing fraction of samples per mode
    - Even bars = good mode coverage (all modes represented)
    - One dominant bar = potential mode collapse
    - Empty modes = model hasn't discovered all modes

    Args:
        actions: Action samples [N, D].
        n_modes: Number of modes to fit.
        method: Clustering method ('gmm', 'kmeans').
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans

    ThemeManager.apply_theme(theme)

    if actions.shape[1] > 2:
        pca = PCA(n_components=2)
        actions_2d = pca.fit_transform(actions)
    else:
        actions_2d = actions

    if method == 'gmm':
        model = GaussianMixture(n_components=n_modes, random_state=42)
        labels = model.fit_predict(actions_2d)
    else:
        model = KMeans(n_clusters=n_modes, random_state=42)
        labels = model.fit_predict(actions_2d)

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout='constrained')

    cluster_colors = ['#3A86FF', '#8338EC', '#FF006E', '#FB5607', '#FFBE0B',
                      '#06D6A0', '#118AB2', '#073B4C']

    ax1 = axes[0]
    for i in range(n_modes):
        mask = labels == i
        ax1.scatter(actions_2d[mask, 0], actions_2d[mask, 1],
                    c=cluster_colors[i % len(cluster_colors)],
                    alpha=0.6, s=40, label=f'Mode {i}',
                    edgecolors='white', linewidths=0.5)
    ax1.set_xlabel('PC 1', fontsize=11)
    ax1.set_ylabel('PC 2', fontsize=11)
    ax1.set_title('Mode Assignment', fontsize=12, fontweight='medium')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='--')

    ax2 = axes[1]
    mode_counts = np.bincount(labels, minlength=n_modes)
    mode_fracs = mode_counts / len(labels)
    bars = ax2.bar(range(n_modes), mode_fracs,
                   color=[cluster_colors[i % len(cluster_colors)] for i in range(n_modes)],
                   edgecolor='white', linewidth=1.5, alpha=0.8)
    for bar, frac in zip(bars, mode_fracs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{frac:.1%}', ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel('Mode', fontsize=11)
    ax2.set_ylabel('Fraction', fontsize=11)
    ax2.set_title('Mode Distribution', fontsize=12, fontweight='medium')
    ax2.set_ylim(0, max(mode_fracs) * 1.2)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    # (layout='constrained' set at figure creation ensures fixed size)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


# =============================================================================
# Utility: Training Curves
# =============================================================================
def plot_training_curves(
    metrics: dict[str, list[float]],
    steps: Optional[list[int]] = None,
    title: str = 'Training Progress',
    theme: str = 'light',
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training metrics over time.

    Args:
        metrics: Dict mapping metric names to value lists.
        steps: X-axis steps (indices if None).
        title: Plot title.
        theme: Color theme.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    ThemeManager.apply_theme(theme)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, layout='constrained')
    if n_metrics == 1:
        axes = [axes]

    gradient_colors = ['#3A86FF', '#8338EC', '#FF006E', '#FB5607', '#FFBE0B']

    for idx, (ax, (name, values)) in enumerate(zip(axes, metrics.items())):
        color = gradient_colors[idx % len(gradient_colors)]
        x = steps if steps else list(range(len(values)))
        ax.plot(x, values, color=color, linewidth=2.5, alpha=0.9, label=name)
        ax.fill_between(x, values, alpha=0.15, color=color)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='medium')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(values) > 10:
            window = max(len(values) // 20, 5)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(x[window-1:], smoothed, color=color, linewidth=3, alpha=0.4)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    # (layout='constrained' set at figure creation ensures fixed size)
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
