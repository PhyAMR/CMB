"""
Global matplotlib style for the CMB analysis figures.

Import this module once (e.g. at the top of ``plots.py``) to apply a
consistent publication-quality look to every figure produced in the
pipeline.  No other module should set ``rcParams`` directly.

Usage
-----
::

    from .plot_style import apply_style, COLORS

    apply_style()          # call once; idempotent
"""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# rcParams dict
# ---------------------------------------------------------------------------

_STYLE = {
    # Typography
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica",
                           "Liberation Sans"],
    "text.usetex":        False,
    "font.size":          15,
    "axes.labelsize":     15,
    "axes.titlesize":     15,
    "xtick.labelsize":    15,
    "ytick.labelsize":    15,
    "legend.fontsize":    15,

    # Figure / saving
    "figure.dpi":         150,
    "figure.facecolor":   "white",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",

    # Axes
    "axes.facecolor":     "#FAFAFA",
    "axes.edgecolor":     "#7F8C8D",
    "axes.linewidth":     0.9,
    "axes.grid":          True,
    "axes.axisbelow":     True,

    # Grid
    "grid.color":         "#E5E8E8",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.7,
    "grid.alpha":         0.6,

    # Ticks
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",

    # Lines / markers
    "lines.linewidth":    1.8,
    "lines.markersize":   4,

    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#BDC3C7",
    "legend.borderpad":   0.8,
    "legend.labelspacing": 0.55,
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS: dict = {
    "theory":       "#2C3E50",   # deep slate — histogram bars & median line
    "theory_fill":  "#34495E",   # histogram bar fill
    "ci_band":      "#BDC3C7",   # 68 % CI shading
    "experimental": "#E74C3C",   # crimson — observed value line
    "grid":         "#E5E8E8",
    "spine":        "#7F8C8D",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """
    Apply ``_STYLE`` to ``matplotlib.rcParams``.

    Idempotent — safe to call multiple times.
    """
    mpl.rcParams.update(_STYLE)