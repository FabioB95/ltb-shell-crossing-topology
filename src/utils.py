"""
Shared utility functions for the LTB multiverse project.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from config import FIGURES_PNG, FIGURES_PDF, FIGURE_DPI, FIGURE_SIZE


def save_figure(fig: Figure, name: str, formats: Tuple[str, ...] = ("png", "pdf")) -> None:
    """
    Save a matplotlib figure to both PNG and PDF directories.
    
    Parameters
    ----------
    fig : Figure
        The figure to save.
    name : str
        Base filename without extension (e.g., "fig01_shell_trajectories").
    formats : tuple
        Output formats to generate.
    """
    FIGURES_PNG.mkdir(parents=True, exist_ok=True)
    FIGURES_PDF.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt == "png":
            path = FIGURES_PNG / f"{name}.png"
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        elif fmt == "pdf":
            path = FIGURES_PDF / f"{name}.pdf"
            fig.savefig(path, bbox_inches="tight", format="pdf")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    print(f"[SAVE] {name} -> {formats}")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path