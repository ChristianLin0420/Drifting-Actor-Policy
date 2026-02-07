"""
Drifting-VLA Logging Module
===========================

WandB integration and visualization utilities:
- WandB logger with organized dashboards
- Visualization functions for debugging
- Metrics computation
- Theme system for paper-ready figures
"""

from drifting_vla.logging.wandb_logger import WandBLogger, LoggerConfig, generate_run_name
from drifting_vla.logging.visualizations import (
    plot_drifting_field,
    plot_drift_magnitude_trend,
    plot_prediction_scatter,
    plot_per_dim_error_radar,
    plot_temperature_loss_breakdown,
    plot_action_distribution,
    plot_action_error_heatmap,
    plot_training_curves,
    render_trajectory_3d,
    plot_sample_transport,
    plot_mode_coverage,
)
from drifting_vla.logging.metrics import (
    compute_success_rate,
    compute_action_mmd,
    compute_multimodality_score,
)
from drifting_vla.logging.themes import ThemeManager, THEMES

__all__ = [
    "WandBLogger",
    "LoggerConfig",
    "generate_run_name",
    # Visualization functions
    "plot_drifting_field",
    "plot_drift_magnitude_trend",
    "plot_prediction_scatter",
    "plot_per_dim_error_radar",
    "plot_temperature_loss_breakdown",
    "plot_action_distribution",
    "plot_action_error_heatmap",
    "plot_training_curves",
    "render_trajectory_3d",
    "plot_sample_transport",
    "plot_mode_coverage",
    # Metrics
    "compute_success_rate",
    "compute_action_mmd",
    "compute_multimodality_score",
    "ThemeManager",
    "THEMES",
]


