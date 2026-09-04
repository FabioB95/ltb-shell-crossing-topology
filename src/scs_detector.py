"""
Shell-Crossing Singularity (SCS) Detector.

Detects R'(t,r) = 0 crossings (Eq. A.3), records (t_cross, r_cross, R_cross),
classifies severity, and computes density at crossing.

Key equation:
    ρ = M'(r) / (4π R² R')  →  ∞  as  R' → 0  with  M' ≠ 0
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.signal import argrelextrema


@dataclass
class SCSEvent:
    """
    A single shell-crossing event.

    Attributes
    ----------
    t_cross : float
        Time of crossing.
    r_cross : float
        Comoving radius where R' = 0.
    R_cross : float
        Areal radius at crossing.
    shell_indices : Tuple[int, ...]
        Shell indices involved in the crossing.
    crossing_type : str
        'simple' : two shells cross
        'multiple' : more than two shells at same R
        'tangential' : R' touches zero without sign change
    severity : str
        'mild', 'strong', or 'coordinate_only'
    density_estimate : float
        Estimated ρ at crossing (may be inf).
    """
    t_cross: float
    r_cross: float
    R_cross: float
    shell_indices: Tuple[int, ...]
    crossing_type: str
    severity: str
    density_estimate: float

    def __repr__(self) -> str:
        return (f"SCSEvent(t={self.t_cross:.4f}, r={self.r_cross:.4f}, "
                f"R={self.R_cross:.4f}, type={self.crossing_type}, "
                f"severity={self.severity}, ρ_est={self.density_estimate:.2e})")


def compute_R_prime(R_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """
    Compute ∂R/∂r on the lattice using 2nd-order central differences.

    Parameters
    ----------
    R_grid : np.ndarray, shape (nt, nr)
    r_grid : np.ndarray, shape (nr,)

    Returns
    -------
    R_prime : np.ndarray, shape (nt, nr)
    """
    dr = float(r_grid[1] - r_grid[0])
    return np.gradient(R_grid, dr, axis=1)


def detect_shell_crossings(
    t_grid: np.ndarray,
    r_grid: np.ndarray,
    R_grid: np.ndarray,
    M: Callable,
    dM_dr: Optional[Callable] = None,
    R_prime_threshold: float = 1e-3,
    density_threshold: float = 1e3,
) -> List[SCSEvent]:
    """
    Detect all shell-crossing singularities in the spacetime.

    Algorithm:
    1. Compute R'(t,r) across the lattice.
    2. At each time slice, find where |R'| < threshold or sign changes.
    3. Classify by M'(r) and estimate density.

    Parameters
    ----------
    t_grid, r_grid : np.ndarray
        Time and radial grids.
    R_grid : np.ndarray, shape (nt, nr)
        Areal radius.
    M : callable
        Mass function M(r).
    dM_dr : callable, optional
        Derivative dM/dr. If None, computed numerically.
    R_prime_threshold : float
        |R'| below this flags a potential crossing.
    density_threshold : float
        ρ above this flags a 'strong' crossing.

    Returns
    -------
    List[SCSEvent]
        All detected events, sorted by time.
    """
    R_prime = compute_R_prime(R_grid, r_grid)
    nt, nr = R_grid.shape
    events: List[SCSEvent] = []

    # Precompute M'(r)
    if dM_dr is None:
        dr = float(r_grid[1] - r_grid[0])
        M_vals = np.array([M(float(r)) for r in r_grid])
        dM_vals = np.gradient(M_vals, dr)
    else:
        dM_vals = np.array([dM_dr(float(r)) for r in r_grid])

    for i in range(nt):
        rp_slice = R_prime[i, :]

        abs_rp = np.abs(rp_slice)

        # Sign changes
        sign_rp = np.sign(rp_slice)
        sign_changes = np.where(np.diff(sign_rp) != 0)[0]

        candidates = set()

        for j in sign_changes:
            candidates.add(int(j))
            candidates.add(int(j) + 1)

        # Local minima of |R'|
        local_mins = argrelextrema(abs_rp, np.less, order=2)[0]
        for j in local_mins:
            if abs_rp[j] < R_prime_threshold:
                candidates.add(int(j))

        processed = set()
        for j in candidates:
            if j < 1 or j >= nr - 1 or j in processed:
                continue

            rp_j = float(rp_slice[j])
            R_j = float(R_grid[i, j])
            r_j = float(r_grid[j])

            rp_left = float(rp_slice[j - 1])
            rp_right = float(rp_slice[j + 1])

            sign_change = (rp_left * rp_j < 0) or (rp_j * rp_right < 0)
            near_zero = abs(rp_j) < R_prime_threshold

            if sign_change:
                ctype = "simple"
            elif near_zero:
                ctype = "tangential"
            else:
                continue

            # Severity classification, matching paper Sec. V.B.1:
            #   Strong:   |R'| < 1e-2  AND  M'(r_x) > 1e-1
            #   Weak:     |R'| < 1e-2  but  M'(r_x) < 1e-1
            #   Marginal: 1e-2 < |R'| < 1e-1  (excluded from strong events;
            #             not separately classified here, falls to "mild")
            # NOTE: an earlier version of this code used |R'| < 1e-6 for
            # "strong", four orders of magnitude stricter than what the
            # paper states. On a discretized (t,r) lattice, R' essentially
            # never gets that close to exactly zero even at a genuine
            # shell-crossing, which made "strong" events pathologically
            # rare and — more importantly — made the graph-connectivity
            # statistics in Experiments 1-2 measure something other than
            # what the paper describes. This threshold now matches the
            # paper text exactly.
            SCS_STRONG_RPRIME = 1e-2
            SCS_STRONG_MPRIME = 1e-1

            Mp = float(dM_vals[j])
            if abs(Mp) < 1e-10:
                severity = "coordinate_only"
            elif abs(rp_j) < SCS_STRONG_RPRIME and abs(Mp) > SCS_STRONG_MPRIME:
                severity = "strong"
            elif abs(rp_j) < SCS_STRONG_RPRIME:
                severity = "weak"
            else:
                severity = "mild"

            # Density estimate
            denom = 4.0 * np.pi * R_j**2 * rp_j
            if abs(denom) > 1e-15:
                rho_est = float(Mp / denom)
            else:
                rho_est = float("inf")

            if abs(rho_est) > density_threshold and severity != "coordinate_only":
                severity = "strong"

            # Multiple crossing
            R_tol = 0.01 * R_j if R_j > 1e-6 else 1e-6
            same_R = np.where(np.abs(R_grid[i, :] - R_j) < R_tol)[0]

            if len(same_R) > 2:
                ctype = "multiple"
                shell_idx = tuple(int(x) for x in same_R)
            else:
                shell_idx = (max(0, j - 1), min(nr - 1, j + 1))

            event = SCSEvent(
                t_cross=float(t_grid[i]),
                r_cross=r_j,
                R_cross=R_j,
                shell_indices=shell_idx,
                crossing_type=ctype,
                severity=severity,
                density_estimate=rho_est,
            )
            events.append(event)
            processed.update(int(x) for x in same_R)

    events.sort(key=lambda e: e.t_cross)
    return events


def compute_scs_density_profile(events: List[SCSEvent], t_grid: np.ndarray) -> np.ndarray:
    """
    Build a time series of maximum estimated density at SCS events.
    """
    rho_max = np.zeros_like(t_grid, dtype=float)
    for ev in events:
        idx = int(np.argmin(np.abs(t_grid - ev.t_cross)))
        val = ev.density_estimate if np.isfinite(ev.density_estimate) else 1e20
        rho_max[idx] = max(float(rho_max[idx]), float(val))
    return rho_max


def print_scs_summary(events: List[SCSEvent]) -> None:
    """Pretty-print a summary of detected shell-crossing events."""
    print(f"\n{'='*60}")
    print("SHELL-CROSSING SINGULARITY SUMMARY")
    print(f"{'='*60}")
    print(f"Total events detected: {len(events)}")
    print(f"{'='*60}")

    if not events:
        print("No shell-crossing singularities detected.")
        return

    severe = [e for e in events if e.severity == "strong"]
    mild = [e for e in events if e.severity == "mild"]
    coord = [e for e in events if e.severity == "coordinate_only"]

    print(f"\n  Strong (ρ → ∞) : {len(severe)}")
    print(f"  Mild             : {len(mild)}")
    print(f"  Coordinate-only  : {len(coord)}")

    print(f"\n  First SCS at t = {events[0].t_cross:.4f}")
    print(f"  Last SCS at t  = {events[-1].t_cross:.4f}")

    if severe:
        print("\n  --- Strongest events ---")
        for ev in severe[:5]:
            print(f"    {ev}")

    print(f"{'='*60}")


if __name__ == "__main__":
    print("[scs_detector.py] Standalone smoke test")

    nt, nr = 100, 64
    t_grid = np.linspace(0, 2.0, nt)
    r_grid = np.linspace(0.1, 2.0, nr)

    R_grid = np.zeros((nt, nr))
    for i, t in enumerate(t_grid):
        for j, r in enumerate(r_grid):
            R1 = 1.5 * np.exp(-0.5 * ((r - (1.0 - 0.3 * t)) / 0.2) ** 2)
            R2 = 1.2 * np.exp(-0.5 * ((r - (1.5 - 0.5 * t)) / 0.2) ** 2)
            R_grid[i, j] = R1 + R2 + 0.1

    M = lambda r: r**3
    events = detect_shell_crossings(t_grid, r_grid, R_grid, M)

    print_scs_summary(events)
    print("[scs_detector.py] Smoke test complete.")