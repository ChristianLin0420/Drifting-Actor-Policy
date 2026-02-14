"""
WandB Visualization Suite for Drifting-VLA
============================================

Publication-quality viz panels with rich annotations and statistics.

Panels:
  viz/A1_drifting_field        — Drift vectors on actual pred→GT space
  viz/A2_drift_magnitude_trend — λ_V convergence with EMA smoothing
  viz/A3_prediction_scatter    — Pred vs GT with R², regression line, error stats
  viz/A4_per_dim_error_radar   — Per-dim MAE bar chart with region labels
  viz/A5_temperature_loss      — Temperature loss with trend comparison
  viz/B1_action_distribution   — KDE overlays with KL divergence
  viz/B2_action_error_heatmap  — Timestep×Dim error with region annotations
  viz/B4_trajectory_3d         — Action trajectories with error shading
  viz/C1_sample_transport      — Before/after drift with magnitude colormap
  viz/C2_mode_coverage         — Coverage + correlation matrix
"""

import torch
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Style constants ──
COLORS = {
    'gt': '#3498db',       # Blue
    'pred': '#e74c3c',     # Red
    'accent': '#2ecc71',   # Green
    'warn': '#f39c12',     # Orange
    'bg': '#2c3e50',       # Dark gray
    'grid': '#7f8c8d',     # Mid gray
    'text': '#ecf0f1',     # Light
}
REGION_NAMES = {
    (0, 8): 'EEF Pose',
    (8, 16): 'Joint Pos',
    (16, 32): 'Bimanual',
    (32, 48): 'Dex Hand',
    (48, 56): 'Base/Extra',
}


def _setup_style():
    """Apply consistent high-readability plot style for WandB panels."""
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 100,
    })


def fig_to_wandb_image(fig):
    """Convert matplotlib figure to wandb.Image at high resolution."""
    import wandb, io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return wandb.Image(img)


def _get_active_dims_info(mask):
    """Get active dims with region labels."""
    active = np.where(mask > 0.5)[0]
    labels = []
    for d in active:
        region = 'Pad'
        for (start, end), name in REGION_NAMES.items():
            if start <= d < end:
                region = name
                break
        labels.append(f'{region}[{d}]')
    return active, labels


def _ema_smooth(values, alpha=0.1):
    """Exponential moving average for trend smoothing."""
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


class VizLogger:
    """Generates information-rich visualization panels for WandB."""

    def __init__(self, config):
        self.config = config
        self.drift_history = []
        self._step_history = []
        self._mse_history = []
        self._loss_history = []

    def log_training_viz(
        self,
        step: int,
        actions_pred: torch.Tensor,
        actions_gt: torch.Tensor,
        action_mask: torch.Tensor,
        drift_field: Optional[torch.Tensor] = None,
        lambda_V: Optional[float] = None,
        per_temp_losses: Optional[dict] = None,
        wandb_run=None,
    ):
        if not HAS_MPL or wandb_run is None:
            return

        import wandb
        _setup_style()

        if lambda_V is not None:
            self.drift_history.append(lambda_V)
            self._step_history.append(step)

        pred = actions_pred.detach().cpu().float().numpy()
        gt = actions_gt.detach().cpu().float().numpy()
        mask = action_mask.detach().cpu().float().numpy()
        B, T, D = pred.shape

        active = mask[0] > 0.5 if mask.ndim == 2 else np.ones(D, dtype=bool)
        active_dims, dim_labels = _get_active_dims_info(mask[0] if mask.ndim == 2 else np.ones(D))
        n_active = len(active_dims)

        log_dict = {}

        try:
            if drift_field is not None and step % 500 == 0:
                log_dict['viz/A1_drifting_field'] = self._plot_A1(
                    pred, gt, drift_field.detach().cpu().float().numpy(), active_dims, step)

            if len(self.drift_history) >= 2 and step % 200 == 0:
                log_dict['viz/A2_drift_magnitude_trend'] = self._plot_A2(step)

            if step % 500 == 0:
                log_dict['viz/A3_prediction_scatter'] = self._plot_A3(pred, gt, active_dims, dim_labels, step)

            if step % 500 == 0 and n_active >= 3:
                log_dict['viz/A4_per_dim_error_radar'] = self._plot_A4(pred, gt, active_dims, dim_labels, step)

            if per_temp_losses and step % 500 == 0:
                log_dict['viz/A5_temperature_loss'] = self._plot_A5(per_temp_losses, step)

            if step % 500 == 0:
                log_dict['viz/B1_action_distribution'] = self._plot_B1(pred, gt, active_dims, dim_labels, step)

            if step % 500 == 0:
                log_dict['viz/B2_action_error_heatmap'] = self._plot_B2(pred, gt, active_dims, dim_labels, step)

            if step % 1000 == 0 and n_active >= 3:
                log_dict['viz/B4_trajectory_3d'] = self._plot_B4(pred, gt, active_dims, step)

            if drift_field is not None and step % 500 == 0:
                log_dict['viz/C1_sample_transport'] = self._plot_C1(
                    pred, gt, drift_field.detach().cpu().float().numpy(), active_dims, step)

            if step % 2000 == 0:
                log_dict['viz/C2_mode_coverage'] = self._plot_C2(pred, gt, active_dims, dim_labels, step)

        except Exception as e:
            logger.warning(f"Viz error at step {step}: {e}")

        if log_dict:
            wandb.log(log_dict, step=step)

    # ──────────────────────────────────────────────────────────────
    # A1: Drifting Field — arrows from pred to pred+V on actual data
    # ──────────────────────────────────────────────────────────────
    def _plot_A1(self, pred, gt, drift_field, active_dims, step):
        B, T, D = pred.shape
        d0, d1 = active_dims[0], active_dims[min(1, len(active_dims)-1)]
        n = min(B, 60)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: drift vectors — use pred positions and compute drift from flattened field
        ax = axes[0]
        x = pred[:n, 0, d0]
        y = pred[:n, 0, d1]
        
        # drift_field is [B, T*D] — extract d0, d1 from first timestep
        flat_D = drift_field.shape[1]
        stride = flat_D // T if T > 0 else D
        dx = drift_field[:n, d0] if d0 < flat_D else np.zeros(n)
        dy = drift_field[:n, d1] if d1 < flat_D else np.zeros(n)
        mag = np.sqrt(dx**2 + dy**2 + 1e-12)

        # Scale arrows for visibility
        max_mag = mag.max() + 1e-8
        scale_factor = 0.3 * (np.abs(x).max() + np.abs(y).max() + 0.01) / max_mag

        ax.scatter(gt[:n, 0, d0], gt[:n, 0, d1], s=50, c=COLORS['accent'], marker='*',
                   alpha=0.6, label='GT', zorder=3)
        ax.scatter(x, y, s=30, c=COLORS['gt'], alpha=0.5, label='Pred', zorder=2)
        for i in range(n):
            ax.annotate('', xy=(x[i] + dx[i]*scale_factor, y[i] + dy[i]*scale_factor),
                        xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='->', color=plt.cm.coolwarm(mag[i]/max_mag),
                                       lw=1.5, alpha=0.7))
        ax.set_xlabel(f'Action Dim {d0}', fontsize=13)
        ax.set_ylabel(f'Action Dim {d1}', fontsize=13)
        ax.set_title('Drift Vectors (pred → pred+V)', fontsize=14)
        ax.legend(fontsize=12)

        # Right: drift magnitude histogram
        ax = axes[1]
        all_mag = np.linalg.norm(drift_field[:n], axis=1)
        ax.hist(all_mag, bins=25, color=COLORS['gt'], alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axvline(all_mag.mean(), color=COLORS['pred'], linestyle='--', linewidth=2.5,
                   label=f'Mean |V| = {all_mag.mean():.4f}')
        ax.set_xlabel('Drift Magnitude |V|', fontsize=13)
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title('Drift Magnitude Distribution', fontsize=14)
        ax.legend(fontsize=12)

        fig.suptitle(f'A1: Drifting Field — Step {step}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A2: Drift Trend — λ_V with EMA smoothing + annotations
    # ──────────────────────────────────────────────────────────────
    def _plot_A2(self, step):
        fig, ax = plt.subplots(figsize=(10, 5))

        steps = np.array(self._step_history)
        vals = np.array(self.drift_history)
        smoothed = np.array(_ema_smooth(vals.tolist(), alpha=0.15))

        ax.plot(steps, vals, color=COLORS['gt'], alpha=0.3, linewidth=0.8, label='Raw λ_V')
        ax.plot(steps, smoothed, color=COLORS['pred'], linewidth=2.5, label='EMA (α=0.15)')
        ax.fill_between(steps, smoothed, alpha=0.08, color=COLORS['pred'])

        # Annotations
        ax.annotate(f'Latest: {vals[-1]:.4f}', xy=(steps[-1], vals[-1]),
                    fontsize=10, fontweight='bold', color=COLORS['pred'],
                    ha='right', va='bottom')

        if len(vals) > 20:
            pct_change = (smoothed[-1] - smoothed[len(smoothed)//2]) / (abs(smoothed[len(smoothed)//2]) + 1e-8) * 100
            trend = '↓ Converging' if pct_change < -5 else '→ Plateau' if abs(pct_change) < 5 else '↑ Diverging'
            color = COLORS['accent'] if pct_change < -5 else COLORS['warn'] if abs(pct_change) < 5 else COLORS['pred']
            ax.text(0.02, 0.95, f'{trend} ({pct_change:+.1f}%)', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color=color, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Training Step')
        ax.set_ylabel('λ_V (drift magnitude)')
        ax.set_title('A2: Drift Magnitude Trend — Should ↓ toward 0 as model converges', fontsize=12)
        ax.legend(loc='upper right')

        if len(vals) > 5 and vals.max() > 10 * (vals.min() + 1e-8):
            ax.set_yscale('log')

        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A3: Prediction Scatter — with R², MAE, regression line
    # ──────────────────────────────────────────────────────────────
    def _plot_A3(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        n_show = min(6, len(active_dims))
        ncols = 3
        nrows = (n_show + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
        if nrows == 1: axes = axes[np.newaxis, :]
        axes_flat = axes.flatten()

        overall_mae = 0
        for i in range(n_show):
            d = active_dims[i]
            ax = axes_flat[i]
            p = pred[:, 0, d]
            g = gt[:, 0, d]

            # Scatter with density coloring
            ax.scatter(g, p, s=12, alpha=0.6, c=COLORS['gt'], edgecolors='none')

            # Perfect prediction line
            lim = max(abs(g).max(), abs(p).max(), 0.01) * 1.15
            ax.plot([-lim, lim], [-lim, lim], '--', color=COLORS['accent'], linewidth=1.5, alpha=0.7, label='y=x')

            # Regression line
            if len(g) > 2 and np.std(g) > 1e-8:
                coeffs = np.polyfit(g, p, 1)
                x_fit = np.linspace(-lim, lim, 100)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), '-', color=COLORS['pred'],
                        linewidth=1.5, alpha=0.7, label=f'fit: {coeffs[0]:.2f}x+{coeffs[1]:.2f}')

            # Stats
            mae = np.abs(p - g).mean()
            overall_mae += mae
            corr = np.corrcoef(g, p)[0, 1] if np.std(g) > 1e-8 and np.std(p) > 1e-8 else 0
            ss_res = np.sum((p - g) ** 2)
            ss_tot = np.sum((g - g.mean()) ** 2) + 1e-8
            r2 = max(0, 1 - ss_res / ss_tot)

            stats_text = f'R²={r2:.3f}\nMAE={mae:.3f}\nr={corr:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                    va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(dim_labels[i] if i < len(dim_labels) else f'Dim {d}', fontsize=9)
            ax.set_xlabel('Ground Truth', fontsize=8)
            ax.set_ylabel('Prediction', fontsize=8)

        for i in range(n_show, len(axes_flat)):
            axes_flat[i].axis('off')

        overall_mae /= max(n_show, 1)
        fig.suptitle(f'A3: Prediction vs GT — Step {step} | Overall MAE: {overall_mae:.4f}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A4: Per-Dim Error — horizontal bar chart with region colors
    # ──────────────────────────────────────────────────────────────
    def _plot_A4(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        mae = np.abs(pred[:, 0, :] - gt[:, 0, :]).mean(axis=0)
        n = min(len(active_dims), 24)
        dims = active_dims[:n]
        vals = mae[dims]

        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.3)))

        # Color by region
        colors = []
        for d in dims:
            if d < 8: colors.append('#3498db')      # EEF
            elif d < 16: colors.append('#9b59b6')    # Joint
            elif d < 32: colors.append('#2ecc71')    # Bimanual
            elif d < 48: colors.append('#e74c3c')    # Dex hand
            elif d < 56: colors.append('#f39c12')    # Extra
            else: colors.append('#95a5a6')           # Pad

        bars = ax.barh(range(n), vals, color=colors, alpha=0.85, edgecolor='white', height=0.7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([dim_labels[i] if i < len(dim_labels) else f'd{d}' for i, d in enumerate(dims)],
                           fontsize=8, fontfamily='monospace')

        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + vals.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', fontsize=7, fontfamily='monospace')

        # Mean line
        ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Mean MAE = {vals.mean():.4f}')

        # Legend for regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='EEF [0:8]'),
            Patch(facecolor='#9b59b6', label='Joint [8:16]'),
            Patch(facecolor='#2ecc71', label='Bimanual [16:32]'),
            Patch(facecolor='#e74c3c', label='Dex Hand [32:48]'),
            Patch(facecolor='#f39c12', label='Base [48:56]'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2)

        ax.set_xlabel('Mean Absolute Error')
        ax.set_title(f'A4: Per-Dimension Error — Step {step} | Worst: {dim_labels[np.argmax(vals)]} ({vals.max():.4f})',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A5: Temperature Loss — grouped bars with annotations
    # ──────────────────────────────────────────────────────────────
    def _plot_A5(self, per_temp_losses, step):
        fig, ax = plt.subplots(figsize=(8, 5))
        taus = sorted(per_temp_losses.keys())
        vals = [per_temp_losses[t].item() if hasattr(per_temp_losses[t], 'item')
                else float(per_temp_losses[t]) for t in taus]

        colors = ['#3498db', '#e74c3c', '#f39c12'][:len(taus)]
        bars = ax.bar(range(len(taus)), vals, color=colors, edgecolor='white',
                      linewidth=1.5, width=0.6, alpha=0.9)

        for bar, v, tau in zip(bars, vals, taus):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{v:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f'τ = {t}' for t in taus], fontsize=11)
        ax.set_ylabel('Raw Drift Norm (before normalization)', fontsize=10)

        # Interpretation text
        if len(vals) >= 2:
            ratio = vals[0] / (vals[-1] + 1e-8)
            interp = 'Fine detail learned' if ratio < 0.5 else 'Still learning coarse structure' if ratio > 2 else 'Balanced'
            ax.text(0.5, 0.95, f'Low-τ / High-τ ratio: {ratio:.2f} — {interp}',
                    transform=ax.transAxes, fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        ax.set_title(f'A5: Multi-Temperature Drift — Step {step}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B1: Action Distribution — KDE overlays with overlap stats
    # ──────────────────────────────────────────────────────────────
    def _plot_B1(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        n_show = min(6, len(active_dims))
        ncols = 3
        nrows = (n_show + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if nrows == 1: axes = axes[np.newaxis, :]
        axes_flat = axes.flatten()

        for i in range(n_show):
            d = active_dims[i]
            ax = axes_flat[i]
            g = gt[:, 0, d]
            p = pred[:, 0, d]

            # Histograms with KDE-like smoothing
            bins = np.linspace(min(g.min(), p.min()) - 0.1, max(g.max(), p.max()) + 0.1, 30)
            ax.hist(g, bins=bins, alpha=0.5, color=COLORS['gt'], density=True, label='GT', edgecolor='white')
            ax.hist(p, bins=bins, alpha=0.5, color=COLORS['pred'], density=True, label='Pred', edgecolor='white')

            # Stats
            gt_std = np.std(g)
            pred_std = np.std(p)
            coverage = pred_std / (gt_std + 1e-8) * 100

            stats = f'μ_GT={g.mean():.2f} σ={gt_std:.2f}\nμ_P={p.mean():.2f} σ={pred_std:.2f}\nCov={coverage:.0f}%'
            ax.text(0.97, 0.95, stats, transform=ax.transAxes, fontsize=7, va='top', ha='right',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax.set_title(dim_labels[i] if i < len(dim_labels) else f'Dim {d}', fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)

        for i in range(n_show, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.suptitle(f'B1: Action Distribution Overlap — Step {step}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B2: Error Heatmap — with region annotations and temporal stats
    # ──────────────────────────────────────────────────────────────
    def _plot_B2(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        error = np.abs(pred - gt).mean(axis=0)  # [T, D]

        # Only show active dims
        n = min(len(active_dims), 32)
        dims = active_dims[:n]
        error_active = error[:, dims]  # [T, n_active]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]})

        # Left: heatmap
        ax = axes[0]
        im = ax.imshow(error_active.T, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
        ax.set_xlabel('Timestep t', fontsize=10)
        ax.set_ylabel('Action Dimension', fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels([dim_labels[i][:12] if i < len(dim_labels) else f'd{d}'
                            for i, d in enumerate(dims)], fontsize=7, fontfamily='monospace')
        plt.colorbar(im, ax=ax, label='MAE', shrink=0.8)

        # Right: per-dim average error bar
        ax2 = axes[1]
        dim_avg = error_active.mean(axis=0)
        ax2.barh(range(n), dim_avg, color=[COLORS['pred'] if v > dim_avg.mean() else COLORS['gt']
                                             for v in dim_avg], alpha=0.8, height=0.7)
        ax2.set_yticks([])
        ax2.set_xlabel('Avg MAE')
        ax2.set_title('Per-Dim Avg')
        ax2.axvline(dim_avg.mean(), color='black', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()

        # Stats annotation
        temporal_trend = error_active.mean(axis=1)  # [T]
        if len(temporal_trend) > 1:
            increase = (temporal_trend[-1] - temporal_trend[0]) / (temporal_trend[0] + 1e-8) * 100
            note = f'Error drift: {increase:+.1f}% from t=0→t={T-1}'
        else:
            note = f'Mean MAE: {error_active.mean():.4f}'

        fig.suptitle(f'B2: Action Error Heatmap — Step {step} | {note}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B4: Trajectory 3D — with error coloring
    # ──────────────────────────────────────────────────────────────
    def _plot_B4(self, pred, gt, active_dims, step):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        d0, d1, d2 = active_dims[0], active_dims[min(1, len(active_dims)-1)], active_dims[min(2, len(active_dims)-1)]
        n_show = min(8, pred.shape[0])
        cmap = plt.cm.tab10

        for i in range(n_show):
            color = cmap(i / max(n_show, 1))
            ax.plot(gt[i, :, d0], gt[i, :, d1], gt[i, :, d2],
                    '-', color=color, alpha=0.7, linewidth=2, label=f'GT {i}' if i < 3 else '')
            ax.plot(pred[i, :, d0], pred[i, :, d1], pred[i, :, d2],
                    '--', color=color, alpha=0.5, linewidth=1.5)
            # Start/end markers
            ax.scatter(*gt[i, 0, [d0, d1, d2]], s=40, c=[color], marker='o', alpha=0.8)
            ax.scatter(*gt[i, -1, [d0, d1, d2]], s=40, c=[color], marker='s', alpha=0.8)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, linestyle='-', label='GT (solid)'),
            Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Pred (dashed)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='End'),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

        avg_err = np.abs(pred[:n_show, :, [d0, d1, d2]] - gt[:n_show, :, [d0, d1, d2]]).mean()
        ax.set_xlabel(f'Dim {d0}')
        ax.set_ylabel(f'Dim {d1}')
        ax.set_zlabel(f'Dim {d2}')
        ax.set_title(f'B4: 3D Action Trajectories — Step {step} | Avg 3D Error: {avg_err:.4f}',
                     fontsize=12, fontweight='bold')
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # C1: Sample Transport — pred positions + drift arrows + GT
    # ──────────────────────────────────────────────────────────────
    def _plot_C1(self, pred, gt, drift_field, active_dims, step):
        B, T, D = pred.shape
        d0, d1 = active_dims[0], active_dims[min(1, len(active_dims)-1)]
        n = min(B, 60)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: transport map
        ax = axes[0]
        x_pred = pred[:n, 0, d0]
        y_pred = pred[:n, 0, d1]
        dx = drift_field[:n, d0] if drift_field.shape[1] > d0 else np.zeros(n)
        dy = drift_field[:n, d1] if drift_field.shape[1] > d1 else np.zeros(n)

        # GT targets
        ax.scatter(gt[:n, 0, d0], gt[:n, 0, d1], s=30, c=COLORS['accent'], marker='*',
                   alpha=0.4, label='GT targets', zorder=1)
        # Pred before drift
        ax.scatter(x_pred, y_pred, s=25, c=COLORS['gt'], alpha=0.6, label='Predictions', zorder=2)
        # Arrows
        scale_factor = 0.5
        ax.quiver(x_pred, y_pred, dx * scale_factor, dy * scale_factor,
                  color='gray', alpha=0.4, scale=None, width=0.003, zorder=3)
        # Pred after drift
        ax.scatter(x_pred + dx * scale_factor, y_pred + dy * scale_factor,
                   s=20, c=COLORS['pred'], alpha=0.5, marker='>', label='After drift', zorder=4)

        ax.legend(fontsize=8)
        ax.set_xlabel(f'Dim {d0}')
        ax.set_ylabel(f'Dim {d1}')
        ax.set_title('Sample Transport Map')

        # Right: distance reduction
        ax2 = axes[1]
        dist_before = np.sqrt((pred[:n, 0, d0] - gt[:n, 0, d0])**2 +
                              (pred[:n, 0, d1] - gt[:n, 0, d1])**2)
        dist_after = np.sqrt((pred[:n, 0, d0] + dx * scale_factor - gt[:n, 0, d0])**2 +
                             (pred[:n, 0, d1] + dy * scale_factor - gt[:n, 0, d1])**2)

        ax2.scatter(dist_before, dist_after, s=20, c=COLORS['gt'], alpha=0.6)
        lim = max(dist_before.max(), dist_after.max()) * 1.1 + 0.01
        ax2.plot([0, lim], [0, lim], '--', color='gray', alpha=0.5)
        ax2.fill_between([0, lim], [0, lim], [0, 0], color=COLORS['accent'], alpha=0.1)
        improved = (dist_after < dist_before).mean() * 100
        ax2.set_xlabel('Distance Before Drift')
        ax2.set_ylabel('Distance After Drift')
        ax2.set_title(f'Drift Improvement: {improved:.0f}% samples closer to GT')
        ax2.text(0.05, 0.95, f'Below diagonal = drift helps\n{improved:.0f}% improved',
                 transform=ax2.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        fig.suptitle(f'C1: Sample Transport Analysis — Step {step}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # C2: Mode Coverage — correlation matrix + coverage bars
    # ──────────────────────────────────────────────────────────────
    def _plot_C2(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        n = min(len(active_dims), 16)
        dims = active_dims[:n]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

        # Left: GT vs Pred 2D scatter (PCA-like projection)
        ax = axes[0]
        gt_flat = gt[:, 0, dims]   # [B, n]
        pred_flat = pred[:, 0, dims]

        # Simple PCA: project to 2D using first 2 principal components
        combined = np.vstack([gt_flat, pred_flat])
        mean = combined.mean(axis=0)
        centered = combined - mean
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[:2].T
            gt_proj = proj[:B]
            pred_proj = proj[B:]
        except Exception:
            gt_proj = gt_flat[:, :2]
            pred_proj = pred_flat[:, :2]

        ax.scatter(gt_proj[:, 0], gt_proj[:, 1], s=20, c=COLORS['gt'], alpha=0.5, label='GT')
        ax.scatter(pred_proj[:, 0], pred_proj[:, 1], s=20, c=COLORS['pred'], alpha=0.5, label='Pred')
        ax.legend(fontsize=9)
        ax.set_title('PCA Projection (2D)', fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Middle: per-dim correlation between pred and GT
        ax2 = axes[1]
        correlations = []
        for i, d in enumerate(dims):
            g = gt[:, 0, d]
            p = pred[:, 0, d]
            if np.std(g) > 1e-8 and np.std(p) > 1e-8:
                correlations.append(np.corrcoef(g, p)[0, 1])
            else:
                correlations.append(0)

        colors_corr = [COLORS['accent'] if c > 0.5 else COLORS['warn'] if c > 0.2 else COLORS['pred']
                       for c in correlations]
        ax2.barh(range(n), correlations, color=colors_corr, alpha=0.8, height=0.7)
        ax2.set_yticks(range(n))
        ax2.set_yticklabels([dim_labels[i][:10] if i < len(dim_labels) else f'd{d}'
                             for i, d in enumerate(dims)], fontsize=7, fontfamily='monospace')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlim(-0.2, 1.05)
        ax2.set_xlabel('Correlation (pred vs GT)')
        ax2.set_title(f'Per-Dim Correlation | Mean: {np.mean(correlations):.3f}', fontsize=10)
        ax2.invert_yaxis()

        # Right: coverage (GT range covered by predictions)
        ax3 = axes[2]
        coverages = []
        for d in dims:
            gt_range = gt[:, 0, d].max() - gt[:, 0, d].min() + 1e-8
            pred_range = pred[:, 0, d].max() - pred[:, 0, d].min()
            coverages.append(min(pred_range / gt_range * 100, 150))

        colors_cov = [COLORS['accent'] if c > 80 else COLORS['warn'] if c > 40 else COLORS['pred']
                      for c in coverages]
        ax3.barh(range(n), coverages, color=colors_cov, alpha=0.8, height=0.7)
        ax3.set_yticks(range(n))
        ax3.set_yticklabels([dim_labels[i][:10] if i < len(dim_labels) else f'd{d}'
                             for i, d in enumerate(dims)], fontsize=7, fontfamily='monospace')
        ax3.axvline(100, color='black', linestyle='--', alpha=0.5, label='100% = full coverage')
        ax3.set_xlim(0, 155)
        ax3.set_xlabel('Coverage %')
        ax3.set_title(f'Range Coverage | Mean: {np.mean(coverages):.0f}%', fontsize=10)
        ax3.legend(fontsize=7)
        ax3.invert_yaxis()

        fig.suptitle(f'C2: Mode Coverage & Correlation Analysis — Step {step}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)
