"""
Global configuration and physical constants for LTB multiverse simulations.
"""

import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_PNG = PROJECT_ROOT / "figures" / "png"
FIGURES_PDF = PROJECT_ROOT / "figures" / "pdf"
DATA_DIR = PROJECT_ROOT / "data"

# Physical constants (geometric units: G = c = 1)
G_NEWTON = 1.0
C_LIGHT = 1.0

# Default numerical parameters
DEFAULT_R_MIN = 0.01      # Inner boundary (avoid r=0 singularity)
DEFAULT_R_MAX = 10.0      # Outer boundary
DEFAULT_NR = 256          # Radial lattice points
DEFAULT_T_MAX = 20.0      # Max evolution time
DEFAULT_NT = 1000         # Temporal snapshots

# Optimization parameters
JAX_LEARNING_RATE = 1e-3
JAX_MAX_ITER = 10000
JAX_TOL = 1e-6

# Figure style
FIGURE_DPI = 300
FIGURE_SIZE = (8, 6)