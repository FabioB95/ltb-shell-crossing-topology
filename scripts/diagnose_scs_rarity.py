#!/usr/bin/env python3
"""
Diagnostic: why are "strong" SCS events so rare in Experiment 2's
parameter regime?

Findings so far (from a run in the dev sandbox): across sigma in
[0.01, 0.5], mean number of strong SCS per profile stays close to 0 for
most of the range, meaning the "percolation transition" measured in
Experiment 2 is almost entirely driven by branch *count* (how many
sign-changes E(r) has), not by actual SCS connectivity. That is NOT the
mechanism the paper wants to describe (SCS-mediated percolation), so we
need to understand why strong events are rare before trusting/reframing
Experiment 2's results.

This script instruments a few representative profiles at different sigma
and prints:
  - total SCS events detected (all severities)
  - breakdown by severity (strong / mild / coordinate_only)
  - the actual |R'| and M' values at each detected event, so we can see
    how close they come to the "strong" thresholds in scs_detector.py
    (|R'| < 1e-2 in the paper's Sec. V.B.1 description vs the actual
    R_prime_threshold=1e-3 hardcoded in scs_detector.detect_shell_crossings,
    and the "strong" classification requiring |R'| < 1e-6 in the code,
    which is much stricter than the 1e-2 described in the paper text!)

Run this FIRST before re-running Experiment 2 at scale, to decide whether
the threshold mismatch (paper says 1e-2, code says 1e-6) is the root
cause, and whether to fix scs_detector.py to match the paper's stated
thresholds or fix the paper text to match the code.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from ltb_solver import solve_ltb_lattice, compute_R_prime
from scs_detector import detect_shell_crossings, compute_R_prime as scs_compute_R_prime


def M_func(r):
    # Updated to match the fix in experiment1/2 (see parameter_scan.py):
    # M_scale=0.5 instead of the original 2.0.
    return 0.5 * r ** 2


def dM_dr(r):
    return 1.0 * r


def make_profile(r_grid, sigma, seed=1):
    rng = np.random.default_rng(seed)
    n_modes = 6
    r_max_local = r_grid[-1]
    E0 = rng.uniform(-0.1, 0.1)
    E_vals = np.full_like(r_grid, E0)
    for n in range(1, n_modes + 1):
        a_n = rng.normal(0.0, sigma / n)
        b_n = rng.normal(0.0, sigma / n)
        E_vals = E_vals + a_n * np.sin(n * np.pi * r_grid / r_max_local) \
                         + b_n * np.cos(n * np.pi * r_grid / r_max_local)
    E_vals = np.clip(E_vals, -0.8, 0.8)
    phys = 2.0 * M_func(r_grid) / r_grid + 2.0 * E_vals
    if np.any(phys <= 0):
        s = 1.0
        for _ in range(30):
            trial = E0 + s * (E_vals - E0)
            phys_trial = 2.0 * M_func(r_grid) / r_grid + 2.0 * trial
            if np.all(phys_trial > 1e-6):
                break
            s *= 0.7
        E_vals = E0 + s * (E_vals - E0)
    return E_vals


def diagnose(sigma, seed, nr=64, nt=100):
    r_grid_template = np.linspace(0.2, 2.0, nr)
    E_vals = make_profile(r_grid_template, sigma, seed=seed)

    def E_callable(r, _v=E_vals, _g=r_grid_template):
        return float(np.interp(r, _g, _v))

    t_grid, r_grid, R_grid = solve_ltb_lattice(
        M_func, E_callable, r_min=0.2, r_max=2.0, nr=nr, t_max=1.5, nt=nt
    )

    R_prime = compute_R_prime(R_grid, r_grid)
    events = detect_shell_crossings(t_grid, r_grid, R_grid, M_func, dM_dr)

    n_sign_changes = int(np.sum(np.abs(np.diff(np.sign(E_vals))) > 0))
    print(f"\n--- sigma={sigma:.3f} seed={seed} n_sign_changes(E)={n_sign_changes} ---")
    print(f"total SCS events detected: {len(events)}")

    by_sev = {}
    for ev in events:
        by_sev.setdefault(ev.severity, []).append(ev)
    for sev, evs in by_sev.items():
        print(f"  severity={sev:15s} count={len(evs)}")

    if events:
        print("  Sample events (up to 5), with |R'| at crossing and M'(r_cross):")
        for ev in events[:5]:
            i = int(np.argmin(np.abs(t_grid - ev.t_cross)))
            j = int(np.argmin(np.abs(r_grid - ev.r_cross)))
            Rp = float(R_prime[i, j])
            Mp = float(dM_dr(ev.r_cross))
            print(f"    t={ev.t_cross:.4f} r={ev.r_cross:.4f} |R'|={abs(Rp):.3e} "
                  f"M'={Mp:.4f} severity={ev.severity} rho_est={ev.density_estimate:.3e}")
    else:
        print("  No events at all detected for this profile.")

    # Also report global min |R'| over the whole (t,r) lattice, ignoring
    # boundaries, to see how close the evolution ever gets to a true
    # shell-crossing (R'=0) even if the detector's candidate-finding logic
    # missed it.
    interior = np.abs(R_prime[:, 1:-1])
    print(f"  global min |R'| over lattice interior: {np.nanmin(interior):.3e}")


if __name__ == "__main__":
    print("=" * 70)
    print("SCS RARITY DIAGNOSTIC")
    print("=" * 70)
    print("""
Known thresholds in scs_detector.py (as currently written):
    R_prime_threshold (candidate detection) = 1e-3   [function default]
    'strong' classification requires        |R'| < 1e-6   AND  M' > 1e-10
                                             (see detect_shell_crossings body)
    density_threshold (also triggers strong) = 1e3

Paper text (Sec. V.B.1) states:
    Strong:   |R'| < 1e-2  AND  M'(r_x) > 1e-1
    Weak:     |R'| < 1e-2  but  M'(r_x) < 1e-1
    Marginal: 1e-2 < |R'| < 1e-1

=> The code's 'strong' threshold (|R'|<1e-6) is FOUR ORDERS OF MAGNITUDE
   stricter than the paper's stated 1e-2. This is a prime suspect for why
   strong events are so rare: on a coarse t,r lattice, R' rarely gets
   numerically as small as 1e-6 even when a genuine shell-crossing is
   occurring nearby in a continuum sense.
""")

    for sigma in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
        for seed in [1, 2, 3]:
            diagnose(sigma, seed)
