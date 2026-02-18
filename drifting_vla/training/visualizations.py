"""
WandB Visualization Suite for Drifting-VLA
============================================

Publication-quality viz panels with rich annotations and statistics.

Panels (Training):
  viz/A1_drifting_field        — Drift vectors on actual pred→GT space
  viz/A2_drift_magnitude_trend — λ_V convergence with EMA smoothing
  viz/A3_prediction_scatter    — Pred vs GT ALL dims, ALL timesteps (dense scatter)
  viz/A4_per_dim_error_bar     — ALL active dims (0-55) error bar chart
  viz/A5_temperature_loss      — Temperature loss with trend comparison
  viz/B1_action_distribution   — ALL active dims violin/histogram grid
  viz/B2_action_error_heatmap  — ALL active dims × Timestep heatmap
  viz/B3_per_region_summary    — Per-region aggregated error radar
  viz/B4_trajectory_2d         — Clear 2D multi-panel trajectories (GT vs Pred)
  viz/B5_temporal_error_curve  — Error evolution across action horizon
  viz/C1_sample_transport      — PCA-projected transport with magnitude colormap
  viz/C2_mode_coverage         — Coverage + correlation matrix
  viz/C3_action_smoothness     — Frequency spectrum / jerk analysis

Panels (Evaluation):
  viz/D1_eval_dashboard        — Per-dataset eval bar charts
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
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
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

REGION_DEFS = [
    (0, 8,   'EEF',      '#3498db'),
    (8, 16,  'Joint',    '#9b59b6'),
    (16, 32, 'Bimanual', '#2ecc71'),
    (32, 48, 'DexHand',  '#e74c3c'),
    (48, 56, 'Base',     '#f39c12'),
]

REGION_NAMES = {(s, e): n for s, e, n, _ in REGION_DEFS}


def _setup_style():
    """Apply consistent high-readability plot style for WandB panels."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 120,
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


def _dim_color(d):
    """Get color for a given dimension index based on region."""
    for start, end, _, color in REGION_DEFS:
        if start <= d < end:
            return color
    return '#95a5a6'


def _dim_label_short(d):
    """Short label like 'E0' for EEF[0], 'J3' for Joint[11], etc."""
    for start, end, name, _ in REGION_DEFS:
        if start <= d < end:
            return f'{name[0]}{d - start}'
    return f'P{d}'


def _ema_smooth(values, alpha=0.1):
    """Exponential moving average for trend smoothing."""
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def _region_legend():
    """Create shared region legend elements."""
    return [Patch(facecolor=c, label=f'{n} [{s}:{e}]')
            for s, e, n, c in REGION_DEFS]


class VizLogger:
    """Generates information-rich visualization panels for WandB."""

    def __init__(self, config):
        self.config = config
        self.drift_history = []
        self._step_history = []
        self._mse_history = []
        self._loss_history = []
        # For D2: training curve history
        self._train_loss_steps = []
        self._train_loss_vals = []
        self._train_mse_vals = []
        self._train_drift_vals = []
        # For D3: per-dataset eval history
        self._eval_history = {}  # {ds_name: {'steps': [], 'mse': [], 'mae': [], 'corr': []}}
        # For D4: data balance tracking
        self._dataset_batch_counts = {}  # {ds_name: count}

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

        # Collect ALL active dims across the batch (union of masks)
        if mask.ndim == 2:
            union_mask = (mask.max(axis=0) > 0.5)
        else:
            union_mask = np.ones(D, dtype=bool)
        active_dims = np.where(union_mask)[0]
        dim_labels = [f'{_dim_label_short(d)}' for d in active_dims]
        n_active = len(active_dims)

        log_dict = {}

        # Each panel is independently try/excepted
        panels = []

        if drift_field is not None and step % 500 == 0:
            panels.append(('viz/A1_drifting_field', lambda: self._plot_A1(
                pred, gt, drift_field.detach().cpu().float().numpy(), active_dims, step)))

        if len(self.drift_history) >= 2 and step % 200 == 0:
            panels.append(('viz/A2_drift_magnitude_trend', lambda: self._plot_A2(step)))

        if step % 500 == 0:
            panels.append(('viz/A3_prediction_scatter', lambda: self._plot_A3(
                pred, gt, active_dims, dim_labels, step)))

        if step % 500 == 0 and n_active >= 1:
            panels.append(('viz/A4_per_dim_error_bar', lambda: self._plot_A4(
                pred, gt, active_dims, dim_labels, step)))

        if per_temp_losses and step % 500 == 0:
            panels.append(('viz/A5_temperature_loss', lambda: self._plot_A5(
                per_temp_losses, step)))

        if step % 500 == 0:
            panels.append(('viz/B1_action_distribution', lambda: self._plot_B1(
                pred, gt, active_dims, dim_labels, step)))

        if step % 500 == 0:
            panels.append(('viz/B2_action_error_heatmap', lambda: self._plot_B2(
                pred, gt, active_dims, dim_labels, step)))

        if step % 1000 == 0 and n_active >= 1:
            panels.append(('viz/B3_per_region_summary', lambda: self._plot_B3(
                pred, gt, active_dims, step)))

        if step % 1000 == 0 and n_active >= 2 and B >= 2:
            panels.append(('viz/B4_trajectory_2d', lambda: self._plot_B4(
                pred, gt, active_dims, step)))

        if step % 500 == 0 and T >= 2:
            panels.append(('viz/B5_temporal_error_curve', lambda: self._plot_B5(
                pred, gt, active_dims, dim_labels, step)))

        if drift_field is not None and step % 500 == 0 and B >= 2:
            panels.append(('viz/C1_sample_transport', lambda: self._plot_C1(
                pred, gt, drift_field.detach().cpu().float().numpy(), active_dims, step)))

        if step % 2000 == 0 and B >= 4:
            panels.append(('viz/C2_mode_coverage', lambda: self._plot_C2(
                pred, gt, active_dims, dim_labels, step)))

        if step % 1000 == 0 and T >= 4:
            panels.append(('viz/C3_action_smoothness', lambda: self._plot_C3(
                pred, gt, active_dims, step)))

        for panel_key, panel_fn in panels:
            try:
                result = panel_fn()
                if result is not None:
                    log_dict[panel_key] = result
            except Exception as e:
                logger.warning(f"Viz {panel_key} error at step {step}: {e}")

        if log_dict:
            wandb.log(log_dict, step=step)

    # ──────────────────────────────────────────────────────────────
    # A1: Drifting Field — arrows from pred to pred+V on actual data
    # ──────────────────────────────────────────────────────────────
    def _plot_A1(self, pred, gt, drift_field, active_dims, step):
        B, T, D = pred.shape
        d0, d1 = active_dims[0], active_dims[min(1, len(active_dims)-1)]

        # drift_field is [N_emb, T*D] (per-embodiment subset, flattened)
        # Clip n to the smaller of batch size and drift_field rows
        n_drift = drift_field.shape[0]
        n = min(B, n_drift, 60)
        if n < 2:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        x = pred[:n, 0, d0]
        y = pred[:n, 0, d1]

        # drift_field is flattened [N, T*D]; extract dim d0 at timestep 0
        flat_D = drift_field.shape[1]
        dx = drift_field[:n, d0] if d0 < flat_D else np.zeros(n)
        dy = drift_field[:n, d1] if d1 < flat_D else np.zeros(n)
        mag = np.sqrt(dx**2 + dy**2 + 1e-12)
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
        ax.set_xlabel(f'Dim {d0} ({_dim_label_short(d0)})')
        ax.set_ylabel(f'Dim {d1} ({_dim_label_short(d1)})')
        ax.set_title(f'Drift Vectors (pred → pred+V) | {n} samples')
        ax.legend()

        ax = axes[1]
        all_mag = np.linalg.norm(drift_field[:n], axis=1)
        ax.hist(all_mag, bins=25, color=COLORS['gt'], alpha=0.8, edgecolor='white')
        ax.axvline(all_mag.mean(), color=COLORS['pred'], linestyle='--', linewidth=2.5,
                   label=f'Mean |V| = {all_mag.mean():.4f}')
        ax.set_xlabel('Drift Magnitude |V|')
        ax.set_ylabel('Count')
        ax.set_title('Drift Magnitude Distribution')
        ax.legend()

        fig.suptitle(f'A1: Drifting Field — Step {step}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A2: Drift Trend — λ_V with EMA smoothing
    # ──────────────────────────────────────────────────────────────
    def _plot_A2(self, step):
        fig, ax = plt.subplots(figsize=(10, 5))

        steps = np.array(self._step_history)
        vals = np.array(self.drift_history)
        smoothed = np.array(_ema_smooth(vals.tolist(), alpha=0.15))

        ax.plot(steps, vals, color=COLORS['gt'], alpha=0.3, linewidth=0.8, label='Raw λ_V')
        ax.plot(steps, smoothed, color=COLORS['pred'], linewidth=2.5, label='EMA (α=0.15)')
        ax.fill_between(steps, smoothed, alpha=0.08, color=COLORS['pred'])

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
        ax.set_title('A2: Drift Magnitude Trend — Should ↓ toward 0', fontsize=12)
        ax.legend(loc='upper right')

        if len(vals) > 5 and vals.max() > 10 * (vals.min() + 1e-8):
            ax.set_yscale('log')

        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A3: Prediction Scatter — ALL dims, ALL timesteps for dense dots
    # ──────────────────────────────────────────────────────────────
    def _plot_A3(self, pred, gt, active_dims, dim_labels, step):
        """Dense scatter: pool ALL batch samples × ALL timesteps for each dim."""
        B, T, D = pred.shape
        n_active = len(active_dims)
        n_show = min(n_active, 12)

        if n_active <= n_show:
            show_indices = list(range(n_active))
        else:
            show_indices = np.linspace(0, n_active - 1, n_show, dtype=int).tolist()

        ncols = min(4, n_show)
        nrows = (n_show + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]
        axes_flat = axes.flatten()

        overall_r2 = []
        for plot_i, dim_i in enumerate(show_indices):
            d = active_dims[dim_i]
            ax = axes_flat[plot_i]

            p = pred[:, :, d].flatten()
            g = gt[:, :, d].flatten()

            max_pts = 3000
            if len(p) > max_pts:
                idx = np.random.choice(len(p), max_pts, replace=False)
                p_plot, g_plot = p[idx], g[idx]
            else:
                p_plot, g_plot = p, g

            ax.scatter(g_plot, p_plot, s=8, alpha=0.4, c=_dim_color(d), edgecolors='none')

            lim = max(np.abs(g).max(), np.abs(p).max(), 0.01) * 1.15
            ax.plot([-lim, lim], [-lim, lim], '--', color='gray', linewidth=1, alpha=0.6)

            if np.std(g) > 1e-8:
                coeffs = np.polyfit(g, p, 1)
                x_fit = np.linspace(-lim, lim, 100)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), '-', color=COLORS['pred'],
                        linewidth=1.5, alpha=0.7)

            mae = np.abs(p - g).mean()
            corr = np.corrcoef(g, p)[0, 1] if np.std(g) > 1e-8 and np.std(p) > 1e-8 else 0
            ss_res = np.sum((p - g) ** 2)
            ss_tot = np.sum((g - g.mean()) ** 2) + 1e-8
            r2 = max(0, 1 - ss_res / ss_tot)
            overall_r2.append(r2)

            ax.text(0.04, 0.96, f'R²={r2:.3f}\nMAE={mae:.3f}\nr={corr:.3f}\nN={len(p)}',
                    transform=ax.transAxes, fontsize=7, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_title(f'{dim_labels[dim_i]} (d{d})', fontsize=9, fontweight='bold',
                         color=_dim_color(d))
            ax.set_xlabel('GT', fontsize=8)
            ax.set_ylabel('Pred', fontsize=8)

        for i in range(len(show_indices), len(axes_flat)):
            axes_flat[i].axis('off')

        avg_r2 = np.mean(overall_r2) if overall_r2 else 0
        fig.suptitle(f'A3: Pred vs GT (all timesteps pooled, {B*T} pts/dim) — '
                     f'Step {step} | Mean R²={avg_r2:.3f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A4: Per-Dim Error — ALL active dims, full bar chart 0-55
    # ──────────────────────────────────────────────────────────────
    def _plot_A4(self, pred, gt, active_dims, dim_labels, step):
        """Show MAE for EVERY active dimension — no truncation."""
        B, T, D = pred.shape
        mae_all = np.abs(pred - gt).mean(axis=(0, 1))  # [D]
        n = len(active_dims)
        vals = mae_all[active_dims]

        fig_h = max(5, n * 0.28)
        fig, ax = plt.subplots(figsize=(12, fig_h))

        colors = [_dim_color(d) for d in active_dims]
        y_pos = np.arange(n)

        bars = ax.barh(y_pos, vals, color=colors, alpha=0.85, edgecolor='white', height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'd{d:2d} {dim_labels[i]}' for i, d in enumerate(active_dims)],
                           fontsize=7, fontfamily='monospace')

        max_v = vals.max() if len(vals) > 0 else 1
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                    f'{v:.4f}', va='center', fontsize=6, fontfamily='monospace')

        ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.5, alpha=0.5,
                   label=f'Mean = {vals.mean():.4f}')
        ax.axvline(np.median(vals), color=COLORS['accent'], linestyle=':', linewidth=1.5, alpha=0.5,
                   label=f'Median = {np.median(vals):.4f}')

        for start, end, name, color in REGION_DEFS:
            in_region = [i for i, d in enumerate(active_dims) if start <= d < end]
            if in_region:
                mid = (in_region[0] + in_region[-1]) / 2
                ax.text(max_v * 1.12, mid, name, fontsize=8, fontweight='bold',
                        color=color, va='center', ha='left')

        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlabel('Mean Absolute Error (all timesteps)', fontsize=10)
        worst_i = np.argmax(vals)
        ax.set_title(f'A4: Per-Dimension Error (all {n} active dims) — Step {step}\n'
                     f'Worst: d{active_dims[worst_i]} {dim_labels[worst_i]} = {vals[worst_i]:.4f}',
                     fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, max_v * 1.25)
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # A5: Temperature Loss
    # ──────────────────────────────────────────────────────────────
    def _plot_A5(self, per_temp_losses, step):
        fig, ax = plt.subplots(figsize=(8, 5))
        taus = sorted(per_temp_losses.keys())
        vals = [per_temp_losses[t].item() if hasattr(per_temp_losses[t], 'item')
                else float(per_temp_losses[t]) for t in taus]

        colors = ['#3498db', '#e74c3c', '#f39c12'][:len(taus)]
        bars = ax.bar(range(len(taus)), vals, color=colors, edgecolor='white',
                      linewidth=1.5, width=0.6, alpha=0.9)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{v:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f'τ = {t}' for t in taus], fontsize=11)
        ax.set_ylabel('Raw Drift Norm')

        if len(vals) >= 2:
            ratio = vals[0] / (vals[-1] + 1e-8)
            interp = 'Fine detail learned' if ratio < 0.5 else 'Still learning coarse' if ratio > 2 else 'Balanced'
            ax.text(0.5, 0.95, f'Low-τ / High-τ = {ratio:.2f} — {interp}',
                    transform=ax.transAxes, fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        ax.set_title(f'A5: Multi-Temperature Drift — Step {step}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B1: Action Distribution — ALL active dims in compact grid
    # ──────────────────────────────────────────────────────────────
    def _plot_B1(self, pred, gt, active_dims, dim_labels, step):
        """Compact distribution view: ALL active dims as mini histograms in a grid."""
        B, T, D = pred.shape
        n = len(active_dims)
        if n == 0:
            return fig_to_wandb_image(plt.figure())

        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 1.8 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]
        axes_flat = axes.flatten()

        for i in range(n):
            d = active_dims[i]
            ax = axes_flat[i]

            g = gt[:, :, d].flatten()
            p = pred[:, :, d].flatten()

            lo = min(g.min(), p.min())
            hi = max(g.max(), p.max())
            margin = (hi - lo) * 0.1 + 0.01
            bins = np.linspace(lo - margin, hi + margin, 25)

            ax.hist(g, bins=bins, alpha=0.55, color=COLORS['gt'], density=True, edgecolor='none')
            ax.hist(p, bins=bins, alpha=0.55, color=COLORS['pred'], density=True, edgecolor='none')

            gt_std = np.std(g) + 1e-8
            pred_std = np.std(p)
            cov = pred_std / gt_std * 100

            ax.set_title(f'd{d} {dim_labels[i]}', fontsize=7, fontweight='bold',
                         color=_dim_color(d), pad=2)
            ax.tick_params(labelsize=5, length=2)
            ax.set_yticks([])

            ax.text(0.97, 0.95, f'{cov:.0f}%', transform=ax.transAxes, fontsize=6,
                    ha='right', va='top', color=COLORS['accent'] if cov > 60 else COLORS['pred'],
                    fontweight='bold')

        for i in range(n, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.legend([Patch(color=COLORS['gt'], alpha=0.55), Patch(color=COLORS['pred'], alpha=0.55)],
                   ['Ground Truth', 'Prediction'], loc='upper right', fontsize=8, ncol=2,
                   framealpha=0.9)

        fig.suptitle(f'B1: Action Distribution — ALL {n} dims — Step {step}\n'
                     f'(% = pred coverage of GT range)',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B2: Error Heatmap — ALL active dims × ALL timesteps
    # ──────────────────────────────────────────────────────────────
    def _plot_B2(self, pred, gt, active_dims, dim_labels, step):
        """Full heatmap: ALL active dims on y-axis, timesteps on x-axis."""
        B, T, D = pred.shape
        error = np.abs(pred - gt).mean(axis=0)  # [T, D]
        n = len(active_dims)
        error_active = error[:, active_dims]  # [T, n]

        fig_h = max(5, n * 0.22)
        fig, axes = plt.subplots(1, 2, figsize=(16, fig_h),
                                 gridspec_kw={'width_ratios': [3, 1]})

        ax = axes[0]
        im = ax.imshow(error_active.T, aspect='auto', cmap='YlOrRd',
                       interpolation='nearest', origin='upper')
        ax.set_xlabel('Timestep t', fontsize=10)
        ax.set_ylabel('Action Dimension', fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f'd{d:2d} {dim_labels[i]}' for i, d in enumerate(active_dims)],
                           fontsize=6, fontfamily='monospace')

        if T <= 16:
            ax.set_xticks(range(T))
        else:
            ax.set_xticks(np.linspace(0, T-1, min(16, T), dtype=int))

        plt.colorbar(im, ax=ax, label='MAE', shrink=0.8, pad=0.02)

        for start, end, name, color in REGION_DEFS:
            in_region = [i for i, d in enumerate(active_dims) if start <= d < end]
            if in_region:
                for idx in in_region:
                    ax.add_patch(plt.Rectangle((-0.8, idx - 0.4), 0.5, 0.8,
                                               color=color, alpha=0.6, clip_on=False))

        ax2 = axes[1]
        dim_avg = error_active.mean(axis=0)  # [n]

        bar_colors = [COLORS['pred'] if v > dim_avg.mean() * 1.5
                      else COLORS['warn'] if v > dim_avg.mean()
                      else COLORS['gt'] for v in dim_avg]
        ax2.barh(range(n), dim_avg, color=bar_colors, alpha=0.8, height=0.7)
        ax2.set_yticks([])
        ax2.set_xlabel('Avg MAE', fontsize=9)
        ax2.set_title('Per-Dim\nAvg Error', fontsize=9)
        ax2.axvline(dim_avg.mean(), color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.invert_yaxis()

        temporal_trend = error_active.mean(axis=1)  # [T]
        if len(temporal_trend) > 1:
            increase = (temporal_trend[-1] - temporal_trend[0]) / (temporal_trend[0] + 1e-8) * 100
            note = f'Error drift: {increase:+.1f}% from t=0→t={T-1} | Mean MAE={error_active.mean():.4f}'
        else:
            note = f'Mean MAE: {error_active.mean():.4f}'

        fig.suptitle(f'B2: Error Heatmap — ALL {n} dims × {T} timesteps — Step {step}\n{note}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B3: Per-Region Summary — aggregated error by action region
    # ──────────────────────────────────────────────────────────────
    def _plot_B3(self, pred, gt, active_dims, step):
        """Grouped bar chart showing MAE, correlation, coverage per action region."""
        B, T, D = pred.shape

        region_stats = {}
        for start, end, name, color in REGION_DEFS:
            dims_in = [d for d in active_dims if start <= d < end]
            if not dims_in:
                continue

            mae_vals = []
            corr_vals = []
            cov_vals = []
            for d in dims_in:
                g = gt[:, :, d].flatten()
                p = pred[:, :, d].flatten()
                mae_vals.append(np.abs(p - g).mean())
                if np.std(g) > 1e-8 and np.std(p) > 1e-8:
                    corr_vals.append(np.corrcoef(g, p)[0, 1])
                else:
                    corr_vals.append(0)
                gt_range = g.max() - g.min() + 1e-8
                pred_range = p.max() - p.min()
                cov_vals.append(min(pred_range / gt_range * 100, 200))

            region_stats[name] = {
                'mae': np.mean(mae_vals),
                'corr': np.mean(corr_vals),
                'coverage': np.mean(cov_vals),
                'n_dims': len(dims_in),
                'color': color,
            }

        if not region_stats:
            return fig_to_wandb_image(plt.figure())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        names = list(region_stats.keys())
        x = np.arange(len(names))
        colors = [region_stats[n]['color'] for n in names]

        # MAE
        ax = axes[0]
        vals = [region_stats[n]['mae'] for n in names]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', width=0.6)
        for bar, v, n in zip(bars, vals, names):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'({region_stats[n]["n_dims"]}d)', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error ↓', fontsize=11, fontweight='bold')

        # Correlation
        ax = axes[1]
        vals = [region_stats[n]['corr'] for n in names]
        corr_colors = [COLORS['accent'] if c > 0.5 else COLORS['warn'] if c > 0.2 else COLORS['pred']
                       for c in vals]
        bars = ax.bar(x, vals, color=corr_colors, alpha=0.85, edgecolor='white', width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Correlation')
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Pred↔GT Correlation ↑', fontsize=11, fontweight='bold')

        # Coverage
        ax = axes[2]
        vals = [region_stats[n]['coverage'] for n in names]
        cov_colors = [COLORS['accent'] if c > 80 else COLORS['warn'] if c > 40 else COLORS['pred']
                      for c in vals]
        bars = ax.bar(x, vals, color=cov_colors, alpha=0.85, edgecolor='white', width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 2,
                    f'{v:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('Coverage %')
        ax.axhline(100, color='black', linestyle='--', alpha=0.5, label='100%')
        ax.set_ylim(0, max(vals + [100]) * 1.2)
        ax.legend(fontsize=8)
        ax.set_title('Range Coverage ↑', fontsize=11, fontweight='bold')

        fig.suptitle(f'B3: Per-Region Action Quality — Step {step}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B4: Trajectory 2D — Clear multi-panel with error ribbons
    # (FIXED: replaced confusing 3D with clean 2D per-sample view)
    # ──────────────────────────────────────────────────────────────
    def _plot_B4(self, pred, gt, active_dims, step):
        """2D multi-panel trajectory: XY + XZ views per sample with error lines."""
        B, T, D = pred.shape
        if len(active_dims) < 2:
            return fig_to_wandb_image(plt.figure())

        d0 = active_dims[0]
        d1 = active_dims[min(1, len(active_dims) - 1)]
        d2 = active_dims[min(2, len(active_dims) - 1)] if len(active_dims) >= 3 else d0
        n_show = min(4, B)

        fig, axes = plt.subplots(2, n_show, figsize=(4.5 * n_show, 8),
                                 squeeze=False)

        for i in range(n_show):
            # ── Top row: d0 vs d1 (e.g. X vs Y) ──
            ax = axes[0, i]
            ax.plot(gt[i, :, d0], gt[i, :, d1], '-o', color=COLORS['gt'],
                    markersize=3, linewidth=2, alpha=0.8, label='GT', zorder=3)
            ax.plot(pred[i, :, d0], pred[i, :, d1], '--s', color=COLORS['pred'],
                    markersize=3, linewidth=1.5, alpha=0.7, label='Pred', zorder=2)
            # Error lines connecting GT↔Pred at each timestep
            for t in range(T):
                ax.plot([gt[i, t, d0], pred[i, t, d0]],
                        [gt[i, t, d1], pred[i, t, d1]],
                        '-', color='gray', alpha=0.3, linewidth=0.8, zorder=1)
            # Start/end markers
            ax.scatter(gt[i, 0, d0], gt[i, 0, d1], s=80, c=COLORS['accent'],
                       marker='^', zorder=5, label='Start')
            ax.scatter(gt[i, -1, d0], gt[i, -1, d1], s=80, c=COLORS['warn'],
                       marker='v', zorder=5, label='End')

            err = np.abs(pred[i, :, [d0, d1]] - gt[i, :, [d0, d1]]).mean()
            ax.set_title(f'Sample {i} | MAE={err:.4f}', fontsize=9, fontweight='bold')
            ax.set_xlabel(f'd{d0} ({_dim_label_short(d0)})', fontsize=8)
            ax.set_ylabel(f'd{d1} ({_dim_label_short(d1)})', fontsize=8)
            if i == 0:
                ax.legend(fontsize=7, loc='best')

            # ── Bottom row: d0 vs d2 (e.g. X vs Z) ──
            ax2 = axes[1, i]
            ax2.plot(gt[i, :, d0], gt[i, :, d2], '-o', color=COLORS['gt'],
                     markersize=3, linewidth=2, alpha=0.8, zorder=3)
            ax2.plot(pred[i, :, d0], pred[i, :, d2], '--s', color=COLORS['pred'],
                     markersize=3, linewidth=1.5, alpha=0.7, zorder=2)
            for t in range(T):
                ax2.plot([gt[i, t, d0], pred[i, t, d0]],
                         [gt[i, t, d2], pred[i, t, d2]],
                         '-', color='gray', alpha=0.3, linewidth=0.8, zorder=1)
            ax2.scatter(gt[i, 0, d0], gt[i, 0, d2], s=80, c=COLORS['accent'],
                        marker='^', zorder=5)
            ax2.scatter(gt[i, -1, d0], gt[i, -1, d2], s=80, c=COLORS['warn'],
                        marker='v', zorder=5)
            ax2.set_xlabel(f'd{d0} ({_dim_label_short(d0)})', fontsize=8)
            ax2.set_ylabel(f'd{d2} ({_dim_label_short(d2)})', fontsize=8)

        avg_err = np.abs(pred[:n_show] - gt[:n_show]).mean()
        fig.suptitle(f'B4: Action Trajectories (2D views, gray=error) — Step {step} | '
                     f'Avg Error: {avg_err:.4f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # B5: Temporal Error Evolution — how error grows over horizon
    # (NEW: critical for understanding near-future vs far-future)
    # ──────────────────────────────────────────────────────────────
    def _plot_B5(self, pred, gt, active_dims, dim_labels, step):
        """Line plot: error at each timestep across the action horizon.

        Shows whether the model is good at near-future (small t) but degrades
        at far-future (large t), which is typical for action prediction.
        """
        B, T, D = pred.shape

        # Per-timestep MAE averaged across batch and active dims
        error = np.abs(pred - gt)  # [B, T, D]
        overall_per_t = error[:, :, active_dims].mean(axis=(0, 2))  # [T]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: overall temporal error curve with confidence band
        ax = axes[0]
        # Per-sample curves for std band
        per_sample_t = error[:, :, active_dims].mean(axis=2)  # [B, T]
        mean_t = per_sample_t.mean(axis=0)
        std_t = per_sample_t.std(axis=0)

        timesteps = np.arange(T)
        ax.fill_between(timesteps, mean_t - std_t, mean_t + std_t,
                        alpha=0.15, color=COLORS['pred'])
        ax.plot(timesteps, mean_t, '-o', color=COLORS['pred'], linewidth=2.5,
                markersize=5, label='Mean MAE', zorder=3)

        # Annotate start/end
        ax.annotate(f't=0: {mean_t[0]:.4f}', xy=(0, mean_t[0]),
                    fontsize=9, fontweight='bold', color=COLORS['accent'])
        ax.annotate(f't={T-1}: {mean_t[-1]:.4f}', xy=(T-1, mean_t[-1]),
                    fontsize=9, fontweight='bold', color=COLORS['pred'],
                    ha='right')

        # Growth rate
        if T > 1:
            growth = (mean_t[-1] - mean_t[0]) / (mean_t[0] + 1e-8) * 100
            color = COLORS['accent'] if growth < 20 else COLORS['warn'] if growth < 50 else COLORS['pred']
            ax.text(0.5, 0.95, f'Error growth: {growth:+.1f}% over horizon',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    ha='center', va='top', color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Timestep t (action horizon)', fontsize=10)
        ax.set_ylabel('Mean Absolute Error', fontsize=10)
        ax.set_title('Overall Error vs Timestep', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        if T <= 16:
            ax.set_xticks(timesteps)

        # Right: per-region temporal curves
        ax2 = axes[1]
        for start, end, name, color in REGION_DEFS:
            dims_in = [d for d in active_dims if start <= d < end]
            if not dims_in:
                continue
            region_err = error[:, :, dims_in].mean(axis=(0, 2))  # [T]
            ax2.plot(timesteps, region_err, '-', color=color, linewidth=2,
                     label=f'{name} ({len(dims_in)}d)', alpha=0.8)

        ax2.set_xlabel('Timestep t', fontsize=10)
        ax2.set_ylabel('MAE', fontsize=10)
        ax2.set_title('Per-Region Error vs Timestep', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        if T <= 16:
            ax2.set_xticks(timesteps)

        fig.suptitle(f'B5: Temporal Error Evolution — Step {step} | T={T} horizon',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # C1: Sample Transport — PCA-projected with magnitude colormap
    # (FIXED: quiver divide-by-zero, PCA projection, bigger arrows)
    # ──────────────────────────────────────────────────────────────
    def _plot_C1(self, pred, gt, drift_field, active_dims, step):
        B, T, D = pred.shape

        # drift_field is [N_emb, T*D] (per-embodiment subset, flattened)
        # Clip n to the smaller of batch size and drift_field rows
        n_drift = drift_field.shape[0]
        n = min(B, n_drift, 200)
        if n < 2:
            return None

        # PCA for 2D projection (much better than raw 2 dims)
        dims = active_dims[:min(24, len(active_dims))]
        pred_flat = pred[:n, 0, dims]
        gt_flat = gt[:n, 0, dims]
        combined = np.vstack([pred_flat, gt_flat])
        mean_c = combined.mean(axis=0)
        centered = combined - mean_c

        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj_mat = Vt[:2].T
            pred_2d = (pred_flat - mean_c) @ proj_mat
            gt_2d = (gt_flat - mean_c) @ proj_mat
        except Exception:
            d0 = active_dims[0]
            d1 = active_dims[min(1, len(active_dims) - 1)]
            pred_2d = pred[:n, 0, [d0, d1]]
            gt_2d = gt[:n, 0, [d0, d1]]
            proj_mat = None

        # Project drift field to 2D
        flat_D = drift_field.shape[1]
        drift_proj = np.zeros((n, len(dims)))
        for i, d in enumerate(dims):
            if d < flat_D:
                drift_proj[:, i] = drift_field[:n, d]

        if proj_mat is not None:
            try:
                drift_2d = drift_proj @ proj_mat
            except Exception:
                drift_2d = np.zeros((n, 2))
        else:
            drift_2d = np.zeros((n, 2))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: PCA transport map
        ax = axes[0]
        mag = np.linalg.norm(drift_2d, axis=1)
        max_mag = mag.max() + 1e-8

        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=50, c=COLORS['accent'], marker='*',
                   alpha=0.4, label='GT', zorder=1)
        sc = ax.scatter(pred_2d[:, 0], pred_2d[:, 1], s=35, c=mag, cmap='YlOrRd',
                        alpha=0.7, edgecolors='white', linewidth=0.3, zorder=2)
        plt.colorbar(sc, ax=ax, label='Drift |V|', shrink=0.8)

        # Arrows with EXPLICIT scale (fixes divide-by-zero RuntimeWarning)
        scale_val = max_mag / 0.25 if max_mag > 1e-8 else 1.0
        ax.quiver(pred_2d[:, 0], pred_2d[:, 1],
                  drift_2d[:, 0], drift_2d[:, 1],
                  mag, cmap='YlOrRd', alpha=0.5,
                  scale=scale_val, width=0.004, zorder=3)

        ax.legend(fontsize=9)
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_title(f'Transport Map (PCA) | {n} samples', fontsize=11)

        # Right: before/after distance scatter
        ax2 = axes[1]
        dist_before = np.linalg.norm(pred_2d - gt_2d, axis=1)
        dist_after = np.linalg.norm(pred_2d + drift_2d - gt_2d, axis=1)

        ax2.scatter(dist_before, dist_after, s=35, c=mag, cmap='YlOrRd',
                    alpha=0.7, edgecolors='white', linewidth=0.3)
        lim = max(dist_before.max(), dist_after.max(), 0.01) * 1.15
        ax2.plot([0, lim], [0, lim], '--', color='gray', alpha=0.5, linewidth=1.5)
        ax2.fill_between([0, lim], [0, lim], [0, 0], color=COLORS['accent'], alpha=0.08)

        improved = (dist_after < dist_before).mean() * 100
        ax2.set_xlabel('Distance Before Drift', fontsize=10)
        ax2.set_ylabel('Distance After Drift', fontsize=10)
        ax2.set_title(f'Drift Improvement: {improved:.0f}% closer to GT',
                      fontsize=11, fontweight='bold')
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.set_aspect('equal')

        fig.suptitle(f'C1: Sample Transport (PCA) — Step {step} | Mean |V|={mag.mean():.4f}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # C2: Mode Coverage
    # ──────────────────────────────────────────────────────────────
    def _plot_C2(self, pred, gt, active_dims, dim_labels, step):
        B, T, D = pred.shape
        n = min(len(active_dims), 24)
        dims = active_dims[:n]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

        # PCA projection
        ax = axes[0]
        gt_flat = gt[:, 0, dims]
        pred_flat = pred[:, 0, dims]
        combined = np.vstack([gt_flat, pred_flat])
        mean = combined.mean(axis=0)
        centered = combined - mean
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[:2].T
            gt_proj, pred_proj = proj[:B], proj[B:]
        except Exception:
            gt_proj, pred_proj = gt_flat[:, :2], pred_flat[:, :2]

        ax.scatter(gt_proj[:, 0], gt_proj[:, 1], s=20, c=COLORS['gt'], alpha=0.5, label='GT')
        ax.scatter(pred_proj[:, 0], pred_proj[:, 1], s=20, c=COLORS['pred'], alpha=0.5, label='Pred')
        ax.legend(fontsize=9)
        ax.set_title('PCA Projection (2D)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        # Per-dim correlation
        ax2 = axes[1]
        correlations = []
        for d in dims:
            g, p = gt[:, 0, d], pred[:, 0, d]
            if np.std(g) > 1e-8 and np.std(p) > 1e-8:
                correlations.append(np.corrcoef(g, p)[0, 1])
            else:
                correlations.append(0)

        colors_corr = [COLORS['accent'] if c > 0.5 else COLORS['warn'] if c > 0.2 else COLORS['pred']
                       for c in correlations]
        ax2.barh(range(n), correlations, color=colors_corr, alpha=0.8, height=0.7)
        ax2.set_yticks(range(n))
        ax2.set_yticklabels([f'd{d} {dim_labels[i][:6]}' for i, d in enumerate(dims)],
                            fontsize=7, fontfamily='monospace')
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlim(-0.2, 1.05)
        ax2.set_xlabel('Correlation')
        ax2.set_title(f'Per-Dim Corr | Mean: {np.mean(correlations):.3f}')
        ax2.invert_yaxis()

        # Coverage
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
        ax3.set_yticklabels([f'd{d} {dim_labels[i][:6]}' for i, d in enumerate(dims)],
                            fontsize=7, fontfamily='monospace')
        ax3.axvline(100, color='black', linestyle='--', alpha=0.5, label='100%')
        ax3.set_xlim(0, 155)
        ax3.set_xlabel('Coverage %')
        ax3.set_title(f'Range Coverage | Mean: {np.mean(coverages):.0f}%')
        ax3.legend(fontsize=7)
        ax3.invert_yaxis()

        fig.suptitle(f'C2: Mode Coverage & Correlation — Step {step}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)

    # ──────────────────────────────────────────────────────────────
    # C3: Action Smoothness — frequency spectrum / jerk analysis
    # (NEW: does the model predict smooth or jerky actions?)
    # ──────────────────────────────────────────────────────────────
    def _plot_C3(self, pred, gt, active_dims, step):
        """Compare action smoothness: GT vs Pred via velocity/jerk and spectrum.

        - Left: velocity magnitude over time (pred should match GT dynamics)
        - Center: jerk (3rd derivative) — lower = smoother
        - Right: frequency spectrum (FFT) — pred should match GT spectrum shape
        """
        B, T, D = pred.shape
        dims = active_dims[:min(8, len(active_dims))]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Compute velocity (first difference) across timesteps
        gt_vel = np.diff(gt[:, :, dims], axis=1)   # [B, T-1, n_dims]
        pred_vel = np.diff(pred[:, :, dims], axis=1)

        gt_vel_mag = np.linalg.norm(gt_vel, axis=2).mean(axis=0)    # [T-1]
        pred_vel_mag = np.linalg.norm(pred_vel, axis=2).mean(axis=0)

        # Left: velocity magnitude
        ax = axes[0]
        t_vel = np.arange(T - 1) + 0.5
        ax.plot(t_vel, gt_vel_mag, '-o', color=COLORS['gt'], linewidth=2,
                markersize=4, label='GT velocity', alpha=0.8)
        ax.plot(t_vel, pred_vel_mag, '--s', color=COLORS['pred'], linewidth=2,
                markersize=4, label='Pred velocity', alpha=0.8)
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('|Δaction|', fontsize=10)
        ax.set_title('Velocity Magnitude', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)

        # Center: jerk (second difference of velocity = third diff of position)
        ax2 = axes[1]
        if T >= 4:
            gt_jerk = np.diff(gt_vel, axis=1)   # [B, T-2, n_dims]
            pred_jerk = np.diff(pred_vel, axis=1)
            gt_jerk_mag = np.linalg.norm(gt_jerk, axis=2).mean(axis=0)    # [T-2]
            pred_jerk_mag = np.linalg.norm(pred_jerk, axis=2).mean(axis=0)

            t_jerk = np.arange(T - 2) + 1
            ax2.plot(t_jerk, gt_jerk_mag, '-o', color=COLORS['gt'], linewidth=2,
                     markersize=4, label=f'GT jerk (μ={gt_jerk_mag.mean():.4f})', alpha=0.8)
            ax2.plot(t_jerk, pred_jerk_mag, '--s', color=COLORS['pred'], linewidth=2,
                     markersize=4, label=f'Pred jerk (μ={pred_jerk_mag.mean():.4f})', alpha=0.8)
            jerk_ratio = pred_jerk_mag.mean() / (gt_jerk_mag.mean() + 1e-8)
            smooth_label = 'Smoother' if jerk_ratio < 0.8 else 'Jerkier' if jerk_ratio > 1.2 else 'Similar'
            ax2.text(0.5, 0.95, f'Pred/GT jerk: {jerk_ratio:.2f}× ({smooth_label})',
                     transform=ax2.transAxes, fontsize=9, ha='center', va='top',
                     fontweight='bold',
                     color=COLORS['accent'] if jerk_ratio < 1.2 else COLORS['pred'],
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax2.set_xlabel('Timestep', fontsize=10)
        ax2.set_ylabel('|Δ²action| (jerk)', fontsize=10)
        ax2.set_title('Jerk (smoothness) ↓ = better', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)

        # Right: frequency spectrum (FFT)
        ax3 = axes[2]
        for d_idx, d in enumerate(dims[:4]):  # Show top 4 dims
            gt_fft = np.abs(np.fft.rfft(gt[:, :, d].mean(axis=0)))
            pred_fft = np.abs(np.fft.rfft(pred[:, :, d].mean(axis=0)))
            freqs = np.fft.rfftfreq(T)

            if d_idx == 0:
                ax3.plot(freqs[1:], gt_fft[1:], '-', color=COLORS['gt'],
                         alpha=0.6, linewidth=1.5, label='GT')
                ax3.plot(freqs[1:], pred_fft[1:], '--', color=COLORS['pred'],
                         alpha=0.6, linewidth=1.5, label='Pred')
            else:
                ax3.plot(freqs[1:], gt_fft[1:], '-', color=COLORS['gt'], alpha=0.3, linewidth=1)
                ax3.plot(freqs[1:], pred_fft[1:], '--', color=COLORS['pred'], alpha=0.3, linewidth=1)

        ax3.set_xlabel('Frequency', fontsize=10)
        ax3.set_ylabel('Amplitude', fontsize=10)
        ax3.set_title('Frequency Spectrum (FFT)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)

        fig.suptitle(f'C3: Action Smoothness Analysis — Step {step}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_to_wandb_image(fig)


# ══════════════════════════════════════════════════════════════
# D1: Eval Dashboard — visual bar charts (replaces WandB table)
# (FIXED: tight_layout warning replaced with subplots_adjust)
# ══════════════════════════════════════════════════════════════

def create_eval_dashboard(per_dataset: dict, per_embodiment: dict, step: int):
    """Create a visual eval dashboard image to replace the hard-to-read WandB table.

    Args:
        per_dataset: {ds_name: {'mse': float, 'mae': float, 'corr': float, 'coverage': float, 'emb_name': str}}
        per_embodiment: {emb_name: {'mse': [float], 'mae': [float], ...}}
        step: training step

    Returns:
        wandb.Image of the dashboard
    """
    if not HAS_MPL:
        return None

    _setup_style()

    n_ds = len(per_dataset)
    if n_ds == 0:
        return None

    # Sort datasets by embodiment for visual grouping
    sorted_ds = sorted(per_dataset.items(), key=lambda x: x[1].get('emb_name', ''))

    fig = plt.figure(figsize=(18, max(6, n_ds * 0.55 + 3)))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1.5, 1, 1], wspace=0.35)

    ds_names = [d[0] for d in sorted_ds]
    emb_names = [d[1].get('emb_name', '?') for d in sorted_ds]
    y_pos = np.arange(n_ds)

    # Assign colors by embodiment
    emb_color_map = {}
    for _, _, name, color in REGION_DEFS:
        emb_color_map[name.lower()] = color
    emb_color_map.update({
        'gripper_eef': '#3498db', 'gripper_joint': '#9b59b6', 'bimanual': '#2ecc71',
        'dex_hand': '#e74c3c', 'gripper_delta_eef': '#f39c12', 'bimanual_mobile': '#1abc9c',
    })
    bar_colors = [emb_color_map.get(e, '#95a5a6') for e in emb_names]

    # ── Panel 1: MSE ──
    ax1 = fig.add_subplot(gs[0])
    mse_vals = [d[1].get('mse', 0) for d in sorted_ds]
    ax1.barh(y_pos, mse_vals, color=bar_colors, alpha=0.85, height=0.65, edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f'{ds_names[i]}\n({emb_names[i][:8]})' for i in range(n_ds)],
                        fontsize=7, fontfamily='monospace')
    for i, v in enumerate(mse_vals):
        ax1.text(v + max(mse_vals) * 0.02, i, f'{v:.5f}', va='center', fontsize=6, fontfamily='monospace')
    ax1.set_xlabel('MSE', fontsize=9)
    ax1.set_title('MSE ↓', fontsize=10, fontweight='bold')
    ax1.invert_yaxis()

    # ── Panel 2: MAE ──
    ax2 = fig.add_subplot(gs[1])
    mae_vals = [d[1].get('mae', 0) for d in sorted_ds]
    ax2.barh(y_pos, mae_vals, color=bar_colors, alpha=0.85, height=0.65, edgecolor='white')
    ax2.set_yticks([])
    for i, v in enumerate(mae_vals):
        ax2.text(v + max(mae_vals) * 0.02, i, f'{v:.4f}', va='center', fontsize=6, fontfamily='monospace')
    ax2.set_xlabel('MAE', fontsize=9)
    ax2.set_title('MAE ↓', fontsize=10, fontweight='bold')
    ax2.invert_yaxis()

    # ── Panel 3: Correlation ──
    ax3 = fig.add_subplot(gs[2])
    corr_vals = [d[1].get('corr', 0) for d in sorted_ds]
    corr_colors = [COLORS['accent'] if c > 0.5 else COLORS['warn'] if c > 0.2 else COLORS['pred']
                   for c in corr_vals]
    ax3.barh(y_pos, corr_vals, color=corr_colors, alpha=0.85, height=0.65, edgecolor='white')
    ax3.set_yticks([])
    ax3.axvline(0, color='black', linewidth=0.5)
    for i, v in enumerate(corr_vals):
        ax3.text(max(v, 0) + 0.02, i, f'{v:.3f}', va='center', fontsize=6, fontfamily='monospace')
    ax3.set_xlabel('Correlation', fontsize=9)
    ax3.set_xlim(-0.1, 1.05)
    ax3.set_title('Corr ↑', fontsize=10, fontweight='bold')
    ax3.invert_yaxis()

    # ── Panel 4: Coverage ──
    ax4 = fig.add_subplot(gs[3])
    cov_vals = [d[1].get('coverage', 0) for d in sorted_ds]
    cov_colors = [COLORS['accent'] if c > 80 else COLORS['warn'] if c > 40 else COLORS['pred']
                  for c in cov_vals]
    ax4.barh(y_pos, cov_vals, color=cov_colors, alpha=0.85, height=0.65, edgecolor='white')
    ax4.set_yticks([])
    ax4.axvline(100, color='black', linestyle='--', alpha=0.5)
    for i, v in enumerate(cov_vals):
        ax4.text(min(v, 145) + 2, i, f'{v:.0f}%', va='center', fontsize=6, fontfamily='monospace')
    ax4.set_xlabel('Coverage %', fontsize=9)
    ax4.set_xlim(0, 160)
    ax4.set_title('Coverage ↑', fontsize=10, fontweight='bold')
    ax4.invert_yaxis()

    # ── Legend: embodiment colors ──
    unique_embs = sorted(set(emb_names))
    legend_handles = [Patch(facecolor=emb_color_map.get(e, '#95a5a6'), label=e) for e in unique_embs]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(6, len(unique_embs)),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'D1: Evaluation Dashboard — Step {step} | {n_ds} datasets',
                 fontsize=13, fontweight='bold')
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.92, wspace=0.35)
    return fig_to_wandb_image(fig)


# ══════════════════════════════════════════════════════════════
# D2: Training Curve Dashboard
# ══════════════════════════════════════════════════════════════

def create_training_curve_dashboard(
    loss_history: list,
    mse_history: list,
    drift_history: list,
    steps: list,
    step: int,
):
    """3-panel training curve: loss, MSE, drift norm over all steps.

    Called from train.py at log_every intervals.
    """
    if not HAS_MPL or len(steps) < 2:
        return None

    _setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    steps_arr = np.array(steps)

    # Loss
    ax = axes[0]
    vals = np.array(loss_history)
    ax.plot(steps_arr, vals, color=COLORS['gt'], alpha=0.3, linewidth=0.8)
    if len(vals) >= 5:
        smoothed = np.array(_ema_smooth(vals.tolist(), alpha=0.1))
        ax.plot(steps_arr, smoothed, color=COLORS['pred'], linewidth=2.5, label='EMA')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss ↓', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # MSE
    ax = axes[1]
    if mse_history:
        vals = np.array(mse_history)
        ax.plot(steps_arr[:len(vals)], vals, color=COLORS['gt'], alpha=0.3, linewidth=0.8)
        if len(vals) >= 5:
            smoothed = np.array(_ema_smooth(vals.tolist(), alpha=0.1))
            ax.plot(steps_arr[:len(vals)], smoothed, color=COLORS['accent'], linewidth=2.5, label='EMA')
        ax.legend(fontsize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE')
    ax.set_title('MSE Loss ↓', fontsize=11, fontweight='bold')

    # Drift norm
    ax = axes[2]
    if drift_history:
        vals = np.array(drift_history)
        ax.plot(steps_arr[:len(vals)], vals, color=COLORS['gt'], alpha=0.3, linewidth=0.8)
        if len(vals) >= 5:
            smoothed = np.array(_ema_smooth(vals.tolist(), alpha=0.1))
            ax.plot(steps_arr[:len(vals)], smoothed, color=COLORS['warn'], linewidth=2.5, label='EMA')
        ax.legend(fontsize=8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Drift Norm')
    ax.set_title('Drift Norm ↓', fontsize=11, fontweight='bold')

    fig.suptitle(f'D2: Training Curves — Step {step}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_wandb_image(fig)


# ══════════════════════════════════════════════════════════════
# D3: Per-Dataset Learning Curve (across evals)
# ══════════════════════════════════════════════════════════════

def create_per_dataset_learning_curve(
    eval_history: dict,
    step: int,
):
    """Line plot: MSE and Correlation per dataset across eval steps.

    eval_history: {ds_name: {'steps': [int], 'mse': [float], 'corr': [float]}}
    """
    if not HAS_MPL or not eval_history:
        return None

    _setup_style()

    # Filter datasets with at least 2 eval points
    valid = {k: v for k, v in eval_history.items() if len(v.get('steps', [])) >= 2}
    if not valid:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.tab20

    # Left: MSE learning curves
    ax = axes[0]
    for i, (ds_name, data) in enumerate(sorted(valid.items())):
        color = cmap(i / max(len(valid), 1))
        ax.plot(data['steps'], data['mse'], '-o', color=color, markersize=4,
                linewidth=1.8, label=ds_name, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title('Per-Dataset MSE ↓', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)

    # Right: Correlation learning curves
    ax2 = axes[1]
    for i, (ds_name, data) in enumerate(sorted(valid.items())):
        color = cmap(i / max(len(valid), 1))
        corr_vals = data.get('corr', [])
        if corr_vals:
            ax2.plot(data['steps'][:len(corr_vals)], corr_vals, '-o', color=color,
                     markersize=4, linewidth=1.8, label=ds_name, alpha=0.8)
    ax2.set_xlabel('Training Step', fontsize=10)
    ax2.set_ylabel('Correlation', fontsize=10)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax2.set_title('Per-Dataset Correlation ↑', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=7, loc='best', ncol=2)

    fig.suptitle(f'D3: Per-Dataset Learning Curves — Step {step}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_wandb_image(fig)


# ══════════════════════════════════════════════════════════════
# D4: Data Balance Monitor
# ══════════════════════════════════════════════════════════════

def create_data_balance_monitor(
    dataset_batch_counts: dict,
    dataset_sizes: dict,
    step: int,
):
    """Pie + bar: actual sampled batches vs dataset sizes.

    dataset_batch_counts: {ds_name: int} — how many times each dataset appeared in batches
    dataset_sizes: {ds_name: int} — configured dataset sizes
    """
    if not HAS_MPL or not dataset_batch_counts:
        return None

    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = sorted(dataset_batch_counts.keys())
    counts = [dataset_batch_counts[n] for n in names]
    total = sum(counts) + 1e-8
    pcts = [c / total * 100 for c in counts]

    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(names), 1)) for i in range(len(names))]

    # Left: pie chart of actual sampling
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={'fontsize': 7},
    )
    ax.legend(wedges, [f'{n} ({c})' for n, c in zip(names, counts)],
              loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=7)
    ax.set_title(f'Actual Sampling Distribution\n({sum(counts)} total batch samples)', fontsize=10)

    # Right: configured vs actual bar chart
    ax2 = axes[1]
    sizes = [dataset_sizes.get(n, 0) for n in names]
    total_size = sum(sizes) + 1e-8
    configured_pct = [s / total_size * 100 for s in sizes]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax2.bar(x - w/2, configured_pct, w, color=colors, alpha=0.5,
                    edgecolor='white', label='Configured %')
    bars2 = ax2.bar(x + w/2, pcts, w, color=colors, alpha=0.9,
                    edgecolor='white', label='Actual %')

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('Percentage %', fontsize=9)
    ax2.set_title('Configured vs Actual Sampling', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)

    fig.suptitle(f'D4: Data Balance Monitor — Step {step}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig_to_wandb_image(fig)
