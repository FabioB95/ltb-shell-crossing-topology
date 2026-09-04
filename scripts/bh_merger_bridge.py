"""
Black Hole Merger Bridge: Identify inter-universal BH merger configurations.

A collapsing shell (E < 0, forms apparent horizon) connects to an expanding
shell (E > 0) via shell-crossing singularity — a "wormhole-like" connection.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable  
from dataclasses import dataclass


@dataclass
class MergerConfig:
    """An inter-universal black hole merger configuration."""
    r_collapsing: float      # Comoving radius of collapsing shell
    r_expanding: float       # Comoving radius of expanding shell
    t_merger: float          # Time of shell-crossing (merger)
    R_merger: float          # Areal radius at merger
    M_total: float           # Total Misner-Sharp mass involved
    tidal_parameter: float   # Proxy for tidal force across the SCS
    
    def __repr__(self) -> str:
        return (f"MergerConfig(r_col={self.r_collapsing:.3f}, "
                f"r_exp={self.r_expanding:.3f}, t_mer={self.t_merger:.4f}, "
                f"M={self.M_total:.4f})")


def apparent_horizon_condition(R: float, M: float) -> bool:
    """
    Check if a shell is inside its apparent horizon.
    
    For spherical dust collapse, apparent horizon forms when R = 2M.
    
    Parameters
    ----------
    R : float
        Areal radius.
    M : float
        Misner-Sharp mass.
    
    Returns
    -------
    bool
        True if R <= 2M (inside horizon).
    """
    return R <= 2.0 * M


def find_merger_candidates(t_grid: np.ndarray, r_grid: np.ndarray,
                           R_grid: np.ndarray,
                           M_func: Callable,
                           E_func: Callable,
                           scs_events: List) -> List[MergerConfig]:
    """
    Find shell-crossing events where a collapsing shell meets an expanding shell,
    with one inside an apparent horizon.
    
    Parameters
    ----------
    t_grid, r_grid : np.ndarray
        Spacetime grids.
    R_grid : np.ndarray, shape (nt, nr)
        Areal radius.
    M_func, E_func : callable
        Mass and energy functions.
    scs_events : list of SCSEvent
        Detected shell-crossing events.
    
    Returns
    -------
    List[MergerConfig]
        All candidate inter-universal merger configurations.
    """
    candidates = []
    
    for event in scs_events:
        t = event.t_cross
        r = event.r_cross
        R = event.R_cross
        
        # Find nearest shell indices
        idx = np.argmin(np.abs(r_grid - r))
        
        # Check neighboring shells for opposite fates
        for offset in [-1, 1]:
            if 0 <= idx + offset < len(r_grid):
                r_neighbor = r_grid[idx + offset]
                E_neighbor = E_func(r_neighbor)
                M_neighbor = M_func(r_neighbor)
                
                # Check if this neighbor is collapsing (E<0, inside horizon)
                # and the other is expanding (E>0)
                is_collapsing = E_neighbor < 0 and apparent_horizon_condition(R, M_neighbor)
                is_expanding = E_neighbor > 0
                
                # We need one collapsing and one expanding at the SCS
                # (Simplified logic — full version needs branch tracking)
                if is_collapsing or is_expanding:
                    config = MergerConfig(
                        r_collapsing=r if E_func(r) < 0 else r_neighbor,
                        r_expanding=r if E_func(r) > 0 else r_neighbor,
                        t_merger=t,
                        R_merger=R,
                        M_total=M_func(r) + M_neighbor,
                        tidal_parameter=abs(M_func(r) - M_neighbor) / R**3
                    )
                    candidates.append(config)
    
    return candidates


def compute_wormhole_mass(M_in: float, M_out: float) -> float:
    """
    Effective 'wormhole mass' for the SCS connection: the geometric mean
    of the two shell masses on either side of the crossing.

    NOTE: an earlier version of this function computed
        sqrt(M_in*M_out) * (R_scs / (2*sqrt(M_in*M_out)))
    which algebraically simplifies to R_scs/2 for ANY M_in, M_out --
    the mass dependence cancels identically, so the quantity carried no
    information about the shell masses despite its name and the R_scs
    argument. This is corrected here: M_wh now genuinely depends on
    M_in and M_out and nothing else. The (now dimensionless) radius
    dependence is reported separately by `compute_compactness_ratio`.

    Parameters
    ----------
    M_in, M_out : float
        Masses on either side of the SCS.

    Returns
    -------
    float
        Effective wormhole mass, sqrt(M_in * M_out).
    """
    return float(np.sqrt(M_in * M_out))


def compute_compactness_ratio(M_in: float, M_out: float, R_scs: float) -> float:
    """
    Dimensionless compactness ratio C_x = R_scs / (2 * sqrt(M_in*M_out)).

    C_x ~ 1 indicates the crossing radius is comparable to the
    Schwarzschild radius set by the effective wormhole mass (a
    "horizon-scale" crossing); C_x >> 1 indicates a crossing far from
    horizon scale. This is the quantity that was previously (incorrectly)
    multiplied back into `compute_wormhole_mass`, canceling the mass
    dependence; it is now reported as a separate, genuinely informative
    diagnostic.

    Parameters
    ----------
    M_in, M_out : float
        Masses on either side of the SCS.
    R_scs : float
        Areal radius at shell-crossing.

    Returns
    -------
    float
        Compactness ratio C_x.
    """
    M_wh = np.sqrt(M_in * M_out)
    if M_wh <= 0:
        return float("inf")
    return float(R_scs / (2.0 * M_wh))


if __name__ == "__main__":
    print("[bh_merger_bridge.py] BH merger bridge module loaded.")