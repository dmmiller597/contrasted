"""Shared plotting style for ContrasTED publication figures.

Usage:
    from benchmarking._style import setup, COLORS, C
    
    setup()  # Call once at script start
    
    ax.plot(x, y, color=C['contrasted'])
    ax.bar(x, heights, color=COLORS[0])
"""
import matplotlib.pyplot as plt
from typing import Tuple

# =============================================================================
# VanGogh2 Color Palette
# =============================================================================
# Deep, rich, natural tones inspired by Van Gogh's later works

COLORS: Tuple[str, ...] = (
    "#bd3106",  # 0: Deep red
    "#d9700e",  # 1: Orange-red
    "#e9a00e",  # 2: Golden yellow
    "#eebe04",  # 3: Bright yellow
    "#5b7314",  # 4: Forest green
    "#c3d6ce",  # 5: Light blue-green / mint
    "#89a6bb",  # 6: Sky blue
    "#454b87",  # 7: Deep blue / navy
    "#9085a4",  # 8: Lavender
    "#7d4a36",  # 9: Earth brown
)

# Semantic color shortcuts for common use cases
C = {
    # Methods
    'contrasted': COLORS[7],   # navy - our method (prominent)
    'baseline': COLORS[2],     # gold - ProstT5 baseline
    'prostt5': COLORS[2],      # gold - alias
    'prott5': COLORS[1],       # orange-red - ProtT5 baseline
    'prottucker': COLORS[4],   # forest green - ProtTucker
    'mmseqs2': COLORS[0],      # red
    'hhsuite': COLORS[8],      # lavender - HH-suite3
    'foldseek': COLORS[6],     # sky blue
    'foldclass': COLORS[5],    # mint
    
    # Distance distributions
    'intra': COLORS[6],        # sky blue - same superfamily
    'inter': COLORS[7],        # navy - different superfamily
    
    # Clustering / holdouts
    'train': COLORS[6],        # sky blue
    'test': COLORS[2],         # gold
    'holdout': COLORS[0],      # red
    'novel': COLORS[4],        # green - novel clusters
    'noise': COLORS[0],        # red - noise points
    'small': COLORS[2],        # gold - small clusters
    
    # General purpose
    'primary': COLORS[7],      # navy
    'secondary': COLORS[6],    # sky blue
    'accent': COLORS[0],       # red
    'neutral': COLORS[9],      # brown
}

# =============================================================================
# Figure Dimensions (inches)
# =============================================================================
# Common sizes for publication layouts

FIGSIZE = {
    'single': (4.0, 3.0),       # Single column
    'double': (8.0, 4.0),       # Double column / full width
    'square': (4.0, 4.0),       # Square
    'wide': (10.0, 4.0),        # Extra wide (3 panels)
    'tall': (4.0, 6.0),         # Tall single column
    'full': (8.0, 8.0),         # Full page square
    
    # Specific layouts
    'side_by_side': (10.0, 5.0),    # Two panels
    'three_panel': (15.0, 4.5),     # Three panels horizontal
    'two_by_two': (10.0, 8.0),      # 2x2 grid
    'two_by_three': (16.0, 10.0),   # 2x3 grid
    'dashboard': (18.0, 12.0),      # Large dashboard
}


# =============================================================================
# Style Setup
# =============================================================================

def setup():
    """Apply Tufte-style publication-ready aesthetics.
    
    Call this once at the start of any plotting script.
    
    Features:
    - Clean white background
    - Minimal spines (no top/right)
    - Subtle grid
    - Professional fonts
    - 300 DPI output
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'font.weight': 'light',  # Light weight for all text by default
        'axes.titlesize': 12,
        'axes.titleweight': 'light',  # Lighter weight for titles
        'axes.labelweight': 'light',  # Light weight for axis labels
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Spines (Tufte-style: minimal)
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 0.8,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '-',
        'axes.axisbelow': True,
        
        # Lines and markers
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'patch.linewidth': 1.0,
        
        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
    })


def setup_minimal():
    """Even more minimal style - no grid, thinner lines.
    
    Good for scatter plots and t-SNE visualizations.
    """
    setup()
    plt.rcParams.update({
        'axes.grid': False,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.5,
    })


# =============================================================================
# Utility Functions
# =============================================================================

def get_method_style(method: str) -> dict:
    """Get consistent style kwargs for a method.
    
    Args:
        method: One of 'contrasted', 'prostt5', 'mmseqs2', 'hhsuite', 
                'foldseek', 'foldclass', 'baseline'
    
    Returns:
        Dict with 'color' and 'label' keys for use in plotting
    
    Example:
        ax.plot(x, y, **get_method_style('contrasted'))
    """
    labels = {
        'contrasted': 'ContrasTED',
        'baseline': 'ProstT5-AA',
        'prostt5': 'ProstT5-AA',
        'prott5': 'ProtT5',
        'prottucker': 'ProtTucker',
        'mmseqs2': 'MMseqs2',
        'hhsuite': 'HH-suite3',
        'foldseek': 'Foldseek',
        'foldclass': 'Foldclass',
    }
    return {
        'color': C.get(method, COLORS[0]),
        'label': labels.get(method, method.title()),
    }


def categorical_colors(n: int) -> list:
    """Get n distinct colors from the palette.
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of hex color strings
    """
    if n <= len(COLORS):
        return list(COLORS[:n])
    # Cycle if more colors needed
    return [COLORS[i % len(COLORS)] for i in range(n)]


def despine(ax=None, left: bool = False, bottom: bool = False):
    """Remove spines from axes (Tufte-style).
    
    Args:
        ax: Matplotlib axes (uses current axes if None)
        left: Also remove left spine
        bottom: Also remove bottom spine
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)


# =============================================================================
# For backwards compatibility with existing code
# =============================================================================

# Direct alias for easier import (used by plotting scripts)
VANGOGH2 = COLORS

# Dict format used in some existing scripts
VanGogh2 = {'colors': COLORS}

def setup_tufte_style():
    """Alias for setup() - backwards compatibility."""
    setup()
