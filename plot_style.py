"""Shared figure style so every paper figure uses one font, palette, and chrome.
Import and call `apply()` before plotting.
"""
import matplotlib.pyplot as plt

# Validated categorical slots (dataviz reference palette, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
# Ink / chrome
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f0efec"  # neutral band (e.g. training-age range)
# Diverging poles for R² heatmap (red = worse than mean, blue = better)
DIV_NEG = "#d03b3b"
DIV_MID = "#f0efec"
DIV_POS = "#1c5cab"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "lines.linewidth": 2,
        "lines.markersize": 5,
    })
