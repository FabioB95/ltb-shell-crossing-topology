"""
LTB Solver: Robust numerical integration of the Lemaître–Tolman–Bondi metric.

Key equation (Eq. A.1):
    Ṙ² = 2M(r)/R + 2E(r)

Branches:
    E > 0 : hyperbolic  (unbound, expands forever or collapses from infinity)
    E = 0 : parabolic   (marginally bound, R ~ (t_c - t)^(2/3))
    E < 0 : elliptic    (bound, recollapses after turnaround at R = M/|E|)

We use ODE integration with event detection for:
    (i)   Singularity: R → 0
    (ii)  Turnaround: Ṙ = 0 (only for E < 0 expansion branch)
"""

from __future__ import annotations

import numpy as np
import math
from typing import cast
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from typing import Callable, Tuple, Optional, Union, Any

try:
    from config import DEFAULT_R_MIN, DEFAULT_R_MAX, DEFAULT_NR, DEFAULT_T_MAX
except ImportError:
    from config import DEFAULT_R_MIN, DEFAULT_R_MAX, DEFAULT_NR, DEFAULT_T_MAX


# --------------------------------------------------------------------------
# Single-shell ODE solver
# --------------------------------------------------------------------------

def ltb_rhs(
    t: float,
    y: np.ndarray,
    r: float,
    M: Callable[[float], float],
    E: Callable[[float], float],
    sign: float,
) -> np.ndarray:
    """
    RHS of the LTB evolution equation.

    dy/dt = [Ṙ] where y = [R]
    Ṙ = sign * sqrt(2M(r)/R + 2E(r))

    Parameters
    ----------
    t : float
        Time.
    y : np.ndarray, shape (1,)
        State vector [R].
    r : float
        Comoving radius of this shell.
    M, E : callable
        Mass and energy functions.
    sign : float
        +1.0 for expansion, -1.0 for collapse.

    Returns
    -------
    np.ndarray
        [dR/dt].
    """
    R = float(y[0])
    if R <= 1e-12:
        return np.array([0.0])

    term = 2.0 * M(r) / R + 2.0 * E(r)

    if term < 0.0:
        # Classically forbidden — should not happen for valid initial data
        return np.array([0.0])

    return np.array([sign * np.sqrt(term)])


def solve_shell(
    r: float,
    M: Callable,
    E: Callable,
    R0: float,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    sign: float = -1.0,
    detect_singularity: bool = True,
) -> Any:
    """
    Solve the LTB equation for a single shell.

    Parameters
    ----------
    r : float
        Comoving radial coordinate.
    M, E : callable
        Mass and energy functions.
    R0 : float
        Initial areal radius at t = t_span[0].
    t_span : tuple
        (t_start, t_end).
    t_eval : np.ndarray, optional
        Dense output time points.
    sign : float
        +1.0 for expansion, -1.0 for collapse (default).
    detect_singularity : bool
        Stop integration if R → 0.

    Returns
    -------
    OdeResult
        scipy.integrate.solve_ivp result with extra fields:
        - result.singularity_time : float or None
        - result.turnaround_time : float or None
    """
    y0 = np.array([float(R0)])

    # Validate initial data
    term0 = 2.0 * M(r) / R0 + 2.0 * E(r)
    if term0 < 0.0:
        raise ValueError(
            f"Inconsistent initial data at r={r}: 2M/R0 + 2E = {term0:.4e} < 0. "
            f"Need R0 <= M/|E| for E<0. Got R0={R0:.4e}, M={M(r):.4e}, E={E(r):.4e}"
        )

    events = []

    # Event 1: Singularity R → 0
    if detect_singularity:
        def hit_singularity(t: float, y: np.ndarray) -> float:
            return float(y[0] - 1e-6)

        hit_singularity.terminal = True  # type: ignore[attr-defined]
        hit_singularity.direction = -1   # type: ignore[attr-defined]
        events.append(hit_singularity)

    # Event 2: Turnaround (Ṙ = 0) for bound shells in expansion
    if E(r) < 0 and sign > 0:
        def hit_turnaround(t: float, y: np.ndarray) -> float:
            # Ṙ = 0 when 2M/R + 2E = 0  =>  R = -M/E
            return float(2.0 * M(r) / y[0] + 2.0 * E(r))

        hit_turnaround.terminal = True  # type: ignore[attr-defined]
        hit_turnaround.direction = -1   # type: ignore[attr-defined]
        events.append(hit_turnaround)

    sol = solve_ivp(
        fun=lambda t, y: ltb_rhs(t, y, r, M, E, sign),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
        max_step=0.005,
        events=events if events else None,
        rtol=1e-9,
        atol=1e-12,
    )

    # Attach metadata
    sol.r_shell = r          # type: ignore[attr-defined]
    sol.E_shell = E(r)       # type: ignore[attr-defined]
    sol.M_shell = M(r)       # type: ignore[attr-defined]
    sol.sign = sign          # type: ignore[attr-defined]

    if detect_singularity and sol.t_events is not None and len(sol.t_events) > 0:
        sol.singularity_time = (  # type: ignore[attr-defined]
            float(sol.t_events[0][0]) if sol.t_events[0].size > 0 else None
        )
    else:
        sol.singularity_time = None  # type: ignore[attr-defined]

    if E(r) < 0 and sign > 0 and sol.t_events is not None and len(sol.t_events) > 1:
        sol.turnaround_time = (  # type: ignore[attr-defined]
            float(sol.t_events[1][0]) if sol.t_events[1].size > 0 else None
        )
    else:
        sol.turnaround_time = None  # type: ignore[attr-defined]

    return sol


# --------------------------------------------------------------------------
# Full lattice solver
# --------------------------------------------------------------------------

def solve_ltb_lattice(
    M: Callable,
    E: Callable,
    r_min: float = DEFAULT_R_MIN,
    r_max: float = DEFAULT_R_MAX,
    nr: int = DEFAULT_NR,
    t_max: float = DEFAULT_T_MAX,
    nt: int = 500,
    sign: Union[float, Callable] = -1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve LTB on a radial lattice.

    Parameters
    ----------
    M, E : callable
        Mass and energy functions.
    r_min, r_max : float
        Radial domain.
    nr : int
        Number of radial shells.
    t_max : float
        Maximum evolution time.
    nt : int
        Number of temporal snapshots.
    sign : float or callable
        +1 for expansion, -1 for collapse (default). Can be a function sign(r).

    Returns
    -------
    t_grid : np.ndarray, shape (nt,)
    r_grid : np.ndarray, shape (nr,)
    R_grid : np.ndarray, shape (nt, nr)
        Areal radius R(t, r). NaN after singularity.
    """
    r_grid = np.linspace(r_min, r_max, nr)
    t_grid = np.linspace(0.0, t_max, nt)

    # Initial condition: R(0, r) = r  (standard LTB gauge)
    R0_grid = r_grid.copy()

    R_grid = np.full((nt, nr), np.nan)

    for i, r in enumerate(r_grid):
        s = sign(r) if callable(sign) else sign

        try:
            sol = solve_shell(r, M, E, R0_grid[i], (0.0, t_max), t_eval=t_grid, sign=s)

            # Handle early termination: sol.t may be shorter than t_grid
            if sol.t.size != t_grid.size:
                # Interpolate onto the full t_grid using dense_output
                # For times beyond termination, return NaN
                R_interp = np.full(t_grid.size, np.nan)
                for j, tt in enumerate(t_grid):
                    if tt <= sol.t[-1]:
                        R_interp[j] = (
                            sol.sol(tt)[0]
                            if hasattr(sol, "sol") and sol.sol is not None
                            else np.nan
                        )
                    else:
                        R_interp[j] = np.nan
                R_grid[:, i] = R_interp
            else:
                R_grid[:, i] = sol.y[0, :]

        except ValueError as e:
            print(f"[WARN] Skipping shell r={r:.4f}: {e}")
            continue

    return t_grid, r_grid, R_grid


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------

def compute_R_prime(R_grid: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """
    Compute ∂R/∂r on the lattice using central differences.

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


def compute_density(
    R_grid: np.ndarray,
    R_prime_grid: np.ndarray,
    r_grid: np.ndarray,
    M: Callable,
    dM_dr: Optional[Callable] = None,
) -> np.ndarray:
    """
    Compute the dust energy density ρ(t,r) from Eq. (A.2):
        ρ = M'(r) / (4π R² R')

    Parameters
    ----------
    R_grid : np.ndarray, shape (nt, nr)
    R_prime_grid : np.ndarray, shape (nt, nr)
    r_grid : np.ndarray, shape (nr,)
    M : callable
        Mass function.
    dM_dr : callable, optional
        Derivative of mass function. If None, computed numerically.

    Returns
    -------
    rho : np.ndarray, shape (nt, nr)
        Energy density. NaN where R' = 0 (SCS) or R = 0.
    """
    nt, nr = R_grid.shape

    if dM_dr is None:
        dr = r_grid[1] - r_grid[0]
        M_vals = np.array([M(r) for r in r_grid])
        dM = np.gradient(M_vals, dr)
    else:
        dM = np.array([dM_dr(r) for r in r_grid])

    # Broadcast dM to match R_grid shape: (nt, nr)
    dM_grid = np.broadcast_to(dM[np.newaxis, :], (nt, nr))

    # Eq. (A.2): ρ = M' / (4π R² R')
    denominator = 4.0 * np.pi * R_grid**2 * R_prime_grid

    rho = np.full_like(R_grid, np.nan)
    mask = (np.abs(denominator) > 1e-15) & (R_grid > 1e-12)
    rho[mask] = dM_grid[mask] / denominator[mask]

    return rho


def compute_expansion_scalar(
    R_grid: np.ndarray,
    R_prime_grid: np.ndarray,
    r_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute the volume expansion scalar θ = 3 Ṙ / R + Ṙ' / R'.
    For LTB dust, this measures the rate of change of proper volume.
    """
    # Compute Ṙ by finite differences in time
    R_dot = np.gradient(R_grid, axis=0)

    # Ṙ' / R'
    dr = float(r_grid[1] - r_grid[0])
    R_dot_prime = np.gradient(R_dot, dr, axis=1)

    theta = 3.0 * R_dot / R_grid + R_dot_prime / R_prime_grid
    return theta


# --------------------------------------------------------------------------
# Parametric exact solutions (for validation)
# --------------------------------------------------------------------------

def solve_shell_parametric(
    r: float,
    M: Callable,
    E: Callable,
    t_eval: np.ndarray,
) -> np.ndarray:
    """
    Exact parametric solution for LTB (for validation only).

    For E < 0 (elliptic):
        R(η) = M/(2|E|) (1 - cos η)
        t(η) = M/(2|E|)^(3/2) (η - sin η) + t_B

    For E = 0 (parabolic):
        R(t) = [(9/2) M (t - t_B)^2]^(1/3)

    For E > 0 (hyperbolic):
        R(η) = M/(2E) (cosh η - 1)
        t(η) = M/(2E)^(3/2) (sinh η - η) + t_B

    The bang time t_B is fixed by R(0,r) = r.
    """
    m = float(M(r))
    e = float(E(r))

    if abs(e) < 1e-12:
        # Parabolic
        t_B = -np.sqrt(2.0 * r**3 / (9.0 * m))
        R = ((9.0 / 2.0) * m * (t_eval - t_B) ** 2) ** (1.0 / 3.0)
        return R

    elif e < 0:
        # Elliptic
        val = 1.0 - 2.0 * abs(e) * r / m
        eta_0 = float(np.arccos(np.clip(val, -1.0, 1.0)))
        t_B = -m / (2.0 * abs(e)) ** 1.5 * (eta_0 - np.sin(eta_0))

        def t_of_eta(eta: float) -> float:
            return m / (2.0 * abs(e)) ** 1.5 * (eta - math.sin(eta)) + t_B

        R_out = np.full_like(t_eval, np.nan, dtype=float)
        for i, t in enumerate(t_eval):
            if t < t_B:
                continue
            try:
                eta = cast(float, brentq(lambda ee: t_of_eta(float(ee)) - t, 0.0, 2.0 * math.pi))
                R_out[i] = m / (2.0 * abs(e)) * (1.0 - math.cos(eta))
            except ValueError:
                pass
        return R_out    
    
    else:
        # Hyperbolic
        val = 1.0 + 2.0 * e * r / m
        eta_0 = float(np.arccosh(np.maximum(val, 1.0)))
        t_B = -m / (2.0 * e) ** 1.5 * (math.sinh(eta_0) - eta_0)

        def t_of_eta(eta: float) -> float:
            return m / (2.0 * e) ** 1.5 * (math.sinh(eta) - eta) + t_B

        R_out = np.full_like(t_eval, np.nan, dtype=float)
        for i, t in enumerate(t_eval):
            if t < t_B:
                continue
            try:
                eta = cast(float, brentq(lambda ee: t_of_eta(float(ee)) - t, 0.0, 10.0))
                R_out[i] = m / (2.0 * e) * (math.cosh(eta) - 1.0)
            except ValueError:
                pass
        return R_out
    

# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("[ltb_solver.py] Comprehensive smoke test")
    print("=" * 60)

    # Test 1: Marginally bound (E=0) — Oppenheimer-Snyder-like
    print("\n--- Test 1: Parabolic collapse (E=0) ---")
    M = lambda r: r**3
    E = lambda r: 0.0
    t, r, R = solve_ltb_lattice(M, E, r_min=0.5, r_max=2.0, nr=32, t_max=0.8, nt=100)
    print(f"  Grid: t={t.shape}, r={r.shape}, R={R.shape}")
    idx_r1 = np.argmin(np.abs(r - 1.0))
    print(f"  R(0, r=1) = {R[0, idx_r1]:.6f} (should be 1.0)")
    print(f"  R(t={t[-1]:.2f}, r=1) = {R[-1, idx_r1]:.6f}")

    # Test 2: Elliptic collapse (E < 0) — use larger r where M/|E| > r
    print("\n--- Test 2: Elliptic collapse (E<0) ---")
    M2 = lambda r: 2.0 * r**2  # M ~ r^2 so M/|E| = 2r^2/0.5 = 4r^2 > r for r > 0.25
    E2 = lambda r: -0.5
    # Check: M/|E| = 2r^2 / 0.5 = 4r^2. Need R0=r <= 4r^2 => r >= 0.25. OK.
    t2, r2, R2 = solve_ltb_lattice(M2, E2, r_min=0.3, r_max=2.0, nr=32, t_max=2.0, nt=200)
    print(f"  Grid: t={t2.shape}, r={r2.shape}, R2={R2.shape}")
    idx_r05 = np.argmin(np.abs(r2 - 0.5))
    print(f"  R(0, r=0.5) = {R2[0, idx_r05]:.6f} (should be 0.5)")
    # Check that some shells hit singularity (NaN at late times)
    n_nan = np.sum(np.isnan(R2[-1, :]))
    print(f"  Shells that hit singularity by t={t2[-1]:.2f}: {n_nan}/{r2.size}")

    # Test 3: Hyperbolic expansion (E > 0)
    print("\n--- Test 3: Hyperbolic expansion (E>0) ---")
    M3 = lambda r: r**3
    E3 = lambda r: 0.5 * r**2
    t3, r3, R3 = solve_ltb_lattice(M3, E3, r_min=0.5, r_max=1.5, nr=32, t_max=1.0, nt=100, sign=+1.0)
    print(f"  Grid: t={t3.shape}, r={r3.shape}, R3={R3.shape}")
    idx_r05_3 = np.argmin(np.abs(r3 - 0.5))
    print(f"  R(0, r=0.5) = {R3[0, idx_r05_3]:.6f} (should be 0.5)")
    print(f"  R(t=1.0, r=0.5) = {R3[-1, idx_r05_3]:.6f} (should be > 0.5)")

    # Test 4: Compute R' and density
    print("\n--- Test 4: Derived quantities ---")
    Rp = compute_R_prime(R, r)
    rho = compute_density(R, Rp, r, M)
    print(f"  R' shape: {Rp.shape}")
    print(f"  ρ shape: {rho.shape}")
    idx_tmid = len(t) // 2
    print(f"  ρ(t={t[idx_tmid]:.2f}, r=1) = {rho[idx_tmid, idx_r1]:.6f}")

    # Test 5: Parametric validation (E=0) ---
    print("\n--- Test 5: Parametric vs ODE (E=0) ---")
    t_test = np.linspace(0, 0.5, 50)
    sol_ode = solve_shell(1.0, M, E, 1.0, (0.0, 0.5), t_eval=t_test, sign=-1.0)
    
    # ODE may terminate early due to singularity — use dense output for fair comparison
    if hasattr(sol_ode, 'sol') and sol_ode.sol is not None:
        R_ode = np.array([sol_ode.sol(tt)[0] for tt in t_test])
    else:
        R_ode = sol_ode.y[0, :]
        # Pad with NaN if needed
        if len(R_ode) < len(t_test):
            R_ode = np.pad(R_ode, (0, len(t_test) - len(R_ode)), constant_values=np.nan)
    
    R_par = solve_shell_parametric(1.0, M, E, t_test)
    
    # Compare only where both are valid
    valid = np.isfinite(R_ode) & np.isfinite(R_par)
    if np.any(valid):
        max_diff = np.nanmax(np.abs(R_ode[valid] - R_par[valid]))
        print(f"  Max |R_ode - R_parametric| = {max_diff:.2e}")
        if max_diff < 1e-4:
            print("  ✓ Excellent agreement")
        elif max_diff < 1e-2:
            print("  ✓ Good agreement")
        else:
            print("  ⚠ Check solver tolerance")
    else:
        print("  No valid overlap for comparison")