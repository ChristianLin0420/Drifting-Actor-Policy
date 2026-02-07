"""
Theme System for Paper-Ready Figures
====================================

Configurable themes for consistent, publication-quality visualizations.
Supports light, dark, and custom color schemes.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThemeColors:
    """
    Color palette for a theme.
    
    Attributes:
        primary: Primary accent color
        secondary: Secondary accent color
        background: Background color
        text: Text color
        grid: Grid line color
        positive: Color for positive/success
        negative: Color for negative/failure
        neutral: Neutral color
    """
    primary: str = '#2E86AB'
    secondary: str = '#A23B72'
    background: str = '#FFFFFF'
    text: str = '#1A1A1A'
    grid: str = '#E0E0E0'
    positive: str = '#2ECC71'
    negative: str = '#E74C3C'
    neutral: str = '#95A5A6'
    palette: list[str] = None
    
    def __post_init__(self):
        if self.palette is None:
            self.palette = [
                self.primary, self.secondary,
                '#F18F01', '#C73E1D', '#3A7CA5',
                '#81B29A', '#F2CC8F', '#E07A5F'
            ]


# Predefined themes
THEMES = {
    'light': ThemeColors(
        primary='#2E86AB',
        secondary='#A23B72',
        background='#FFFFFF',
        text='#1A1A1A',
        grid='#E0E0E0',
        positive='#2ECC71',
        negative='#E74C3C',
        neutral='#95A5A6',
    ),
    'dark': ThemeColors(
        primary='#00D4FF',
        secondary='#FF6B6B',
        background='#1E1E2E',
        text='#CDD6F4',
        grid='#45475A',
        positive='#A6E3A1',
        negative='#F38BA8',
        neutral='#6C7086',
    ),
    'paper': ThemeColors(
        primary='#0072B2',
        secondary='#D55E00',
        background='#FFFFFF',
        text='#000000',
        grid='#CCCCCC',
        positive='#009E73',
        negative='#CC79A7',
        neutral='#999999',
        palette=['#0072B2', '#D55E00', '#009E73', '#CC79A7', 
                 '#F0E442', '#56B4E9', '#E69F00', '#000000'],
    ),
    'presentation': ThemeColors(
        primary='#FF6B35',
        secondary='#004E89',
        background='#1A1A2E',
        text='#EAEAEA',
        grid='#3A3A5C',
        positive='#7ED957',
        negative='#FF4757',
        neutral='#747D8C',
    ),
}


class ThemeManager:
    """
    Manager for applying themes to matplotlib figures.
    
    Provides consistent styling for all visualizations with
    support for light/dark modes and custom themes.
    
    Example:
        >>> ThemeManager.apply_theme('dark')
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)  # Uses dark theme colors
        >>> plt.show()
    """
    
    current_theme: str = 'light'
    _custom_themes: Dict[str, ThemeColors] = {}
    
    @classmethod
    def register_theme(cls, name: str, colors: ThemeColors) -> None:
        """Register a custom theme."""
        cls._custom_themes[name] = colors
        logger.info(f"Registered custom theme: {name}")
    
    @classmethod
    def get_theme(cls, name: str) -> ThemeColors:
        """Get theme colors by name."""
        if name in cls._custom_themes:
            return cls._custom_themes[name]
        if name in THEMES:
            return THEMES[name]
        logger.warning(f"Unknown theme {name}, using 'light'")
        return THEMES['light']
    
    @classmethod
    def apply_theme(cls, name: str) -> None:
        """
        Apply theme to matplotlib.
        
        Args:
            name: Theme name ('light', 'dark', 'paper', etc.)
        """
        colors = cls.get_theme(name)
        cls.current_theme = name
        
        # Base style
        if 'dark' in name:
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # Custom rcParams
        params = {
            # Colors
            'figure.facecolor': colors.background,
            'axes.facecolor': colors.background,
            'axes.edgecolor': colors.grid,
            'axes.labelcolor': colors.text,
            'text.color': colors.text,
            'xtick.color': colors.text,
            'ytick.color': colors.text,
            'grid.color': colors.grid,
            'axes.prop_cycle': plt.cycler('color', colors.palette),
            
            # Typography
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            
            # Figure
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'savefig.facecolor': colors.background,
            
            # Lines and markers
            'lines.linewidth': 2,
            'lines.markersize': 6,
            
            # Legend
            'legend.framealpha': 0.9,
            'legend.edgecolor': colors.grid,
            
            # Grid
            'grid.alpha': 0.3,
            'grid.linestyle': '-',
        }
        
        mpl.rcParams.update(params)
        logger.debug(f"Applied theme: {name}")
    
    @classmethod
    def get_colors(cls) -> ThemeColors:
        """Get current theme colors."""
        return cls.get_theme(cls.current_theme)
    
    @classmethod
    def reset(cls) -> None:
        """Reset to default matplotlib style."""
        mpl.rcdefaults()
        cls.current_theme = 'light'
    
    @classmethod
    def create_colormap(
        cls,
        n_colors: int = 10,
        cmap_name: str = 'viridis',
    ) -> list[str]:
        """
        Create a list of colors from a colormap.
        
        Args:
            n_colors: Number of colors.
            cmap_name: Matplotlib colormap name.
        
        Returns:
            List of hex color strings.
        """
        import numpy as np
        
        cmap = plt.cm.get_cmap(cmap_name)
        colors = [mpl.colors.to_hex(cmap(i / (n_colors - 1))) 
                  for i in range(n_colors)]
        return colors


def setup_paper_style() -> None:
    """
    Configure matplotlib for publication-quality figures.
    
    Sets up:
    - Computer Modern fonts (LaTeX-compatible)
    - High DPI output
    - Tight layouts
    - Colorblind-friendly palette
    """
    ThemeManager.apply_theme('paper')
    
    # Additional paper-specific settings
    params = {
        # Use LaTeX-compatible fonts
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        
        # High quality output
        'savefig.dpi': 300,
        'figure.dpi': 150,
        
        # Tight layouts
        'figure.constrained_layout.use': True,
        
        # Line widths for print
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
    }
    
    mpl.rcParams.update(params)
    logger.info("Configured paper style")


def setup_presentation_style() -> None:
    """
    Configure matplotlib for presentation slides.
    
    Sets up:
    - Large fonts
    - Bold colors
    - Dark-friendly palette
    """
    ThemeManager.apply_theme('presentation')
    
    params = {
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        
        'lines.linewidth': 3,
        'lines.markersize': 10,
    }
    
    mpl.rcParams.update(params)
    logger.info("Configured presentation style")


