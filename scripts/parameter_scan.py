#!/usr/bin/env python3
"""
Parameter scan: find an (M(r) scale, t_max) regime where genuine shell-
crossings occur with reasonable frequency across random E(r) profiles,
BEFORE re-running Experiment 1/2 at scale.

Root-cause context (see diagnose_scs_rarity.py output): with M(r)=2r^2 and
t_max=1.5, most randomly generated E(r) profiles never produce any R'->0
event at all -- the global minimum of |R'| over the whole (t,r) lattice
stays around 0.5-0.8 for most profiles. A back-of-envelope estimate
suggests collapsing (E<0) inner shells reach R=0 on a timescale much
shorter than t_max for this M(r), so they vanish before ever encountering
neighboring shells -- i.e. this may be a genuine property of the chosen
(M(r), t_max, r-range) combination, not just a severity-threshold bug.

This script tries a small grid of (M_scale, t_max) combinations and
reports, for a fixed set of test profiles, the fraction of profiles that
produce at least one SCS event of any severity, and the fraction that
produce at least one "strong" event under the corrected threshold
(|R'|<1e-2, M'>1e-1, see the scs_detector.py fix applied alongside this
script).

Once a good regime is identified, hardcode it into
experiment1_phase_diagram.py and experiment2_percolation.py (replacing
M_func / T_MAX there) and re-run.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from ltb_solver import solve_ltb_lattice, compute_R_prime
from scs_detector import detect_shell_crossings


def make_profile(r_grid, sigma, seed):
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
    return np.clip(E_vals, -0.8, 0.8)


def run_regime(M_scale, t_max, nr=48, nt=80, n_profiles=15, r_min=0.2, r_max=2.0):
    """
    Test a single (M_scale, t_max) regime against n_profiles random E(r)
    profiles spanning a range of disorder strengths, and report SCS
    statistics. Uses a reduced lattice (nr=48, nt=80) for scan speed;
    the final chosen regime should be re-validated at the resolution
    used for the actual experiments (nr=64, nt=100 or higher).
    """
    def M_func(r):
        return M_scale * r ** 2

    def dM_dr(r):
        return 2.0 * M_scale * r

    r_grid_template = np.linspace(r_min, r_max, nr)
    sigmas = np.linspace(0.05, 0.5, n_profiles)

    n_any_event = 0
    n_strong_event = 0
    global_min_rprime = []

    for i, sigma in enumerate(sigmas):
        E_vals = make_profile(r_grid_template, sigma, seed=i)

        # physical validity: 2M(r)/r + 2E(r) > 0  =>  E(r) > -M(r)/r.
        # Clip elementwise (per-shell floor) rather than rescaling toward
        # the profile mean, which does not always guarantee validity for
        # every shell simultaneously.
        E_floor = -M_func(r_grid_template) / r_grid_template + 1e-3
        E_vals = np.maximum(E_vals, E_floor)

        def E_callable(r, _v=E_vals, _g=r_grid_template):
            return float(np.interp(r, _g, _v))

        try:
            t_grid, r_grid, R_grid = solve_ltb_lattice(
                M_func, E_callable, r_min=r_min, r_max=r_max, nr=nr, t_max=t_max, nt=nt
            )
        except Exception:
            continue

        R_prime = compute_R_prime(R_grid, r_grid)
        interior = np.abs(R_prime[:, 1:-1])
        global_min_rprime.append(float(np.nanmin(interior)))

        events = detect_shell_crossings(t_grid, r_grid, R_grid, M_func, dM_dr)
        strong = [e for e in events if e.severity == "strong"]

        if events:
            n_any_event += 1
        if strong:
            n_strong_event += 1

    return {
        "M_scale": M_scale,
        "t_max": t_max,
        "frac_any_event": n_any_event / n_profiles,
        "frac_strong_event": n_strong_event / n_profiles,
        "median_global_min_rprime": float(np.median(global_min_rprime)) if global_min_rprime else float("nan"),
    }


def main():
    M_scales = [0.5, 1.0, 2.0]
    t_maxs = [1.5, 3.0, 6.0, 10.0]

    print(f"{'M_scale':>8} {'t_max':>7} {'frac_any':>9} {'frac_strong':>12} {'median_min|R\'|':>16}")
    t0 = time.time()
    results = []
    for M_scale in M_scales:
        for t_max in t_maxs:
            r = run_regime(M_scale, t_max)
            results.append(r)
            print(f"{r['M_scale']:>8.2f} {r['t_max']:>7.1f} "
                  f"{r['frac_any_event']:>9.2f} {r['frac_strong_event']:>12.2f} "
                  f"{r['median_global_min_rprime']:>16.3e}")
    print(f"\nTotal scan time: {time.time()-t0:.1f}s")

    best = max(results, key=lambda r: r["frac_strong_event"])
    print(f"\nBest regime by frac_strong_event: M_scale={best['M_scale']}, "
          f"t_max={best['t_max']} -> frac_strong={best['frac_strong_event']:.2f}")


if __name__ == "__main__":
    main()
