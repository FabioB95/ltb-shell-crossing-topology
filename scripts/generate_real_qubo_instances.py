#!/usr/bin/env python3
"""
Generate REAL QUBO instances from the branch-connectivity graph, for
Experiment 3 (Sec. VI.C of the paper).

Unlike the QUBO instances used in an earlier version of this experiment
(random Gaussian weights, later a random 3-regular MaxCut graph -- see
Sec. VII.D "Implementation status and open gaps"), this generator builds
the connection-strength matrix W_ij (Eq. 16) directly from real strong
SCS transitions detected by the LTB solver + SCS detector pipeline used
throughout the rest of the paper (Sec. V.A, V.C).

Method
------
For a target branch count N, we construct a piecewise E(r) profile with
exactly N contiguous zones of alternating sign (so that Algorithm 3's
branch grouping produces exactly N branches), with randomized zone
widths and E-magnitudes per zone (seeded, reproducible). We then run the
real LTB solver and SCS detector, and build W_ij as the number of strong
SCS transitions detected between each ordered pair of branches (i,j),
symmetrized: W_ij = W_ji = (# transitions i->j) + (# transitions j->i).

If an instance produces an all-zero W (no strong SCS events at all), it
is discarded and regenerated with a new seed, since a trivial instance
(H_QUBO with no quadratic term at all) is not a meaningful benchmark
instance.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "src"))

import numpy as np
import json

from ltb_solver import solve_ltb_lattice
from scs_detector import detect_shell_crossings

R_MIN, R_MAX = 0.2, 2.0
NR, NT = 150, 250
T_MAX = 3.0


def M_func(r):
    return 0.5 * r ** 2


def dM_dr(r):
    return 1.0 * r


def make_zone_profile(n_zones, seed, r_min=R_MIN, r_max=R_MAX):
    """
    Piecewise-constant E(r) with n_zones contiguous zones of alternating
    sign and randomized (seeded) widths/magnitudes.
    """
    rng = np.random.default_rng(seed)

    # Randomized zone boundaries: n_zones-1 interior cut points, sorted
    cuts = np.sort(rng.uniform(r_min + 0.05, r_max - 0.05, size=n_zones - 1))
    edges = np.concatenate([[r_min], cuts, [r_max]])

    signs = rng.choice([-1, 1], size=n_zones)
    for i in range(1, n_zones):
        if signs[i] == signs[i - 1]:
            signs[i] *= -1  # force alternation so zones don't merge

    mags = rng.uniform(0.15, 0.45, size=n_zones)

    def E_callable(r, _edges=edges, _signs=signs, _mags=mags, _n=n_zones):
        idx = np.searchsorted(_edges, r, side="right") - 1
        idx = min(max(idx, 0), _n - 1)
        return float(_signs[idx] * _mags[idx])

    return E_callable, edges, signs, mags


def build_branch_graph_W(n_zones, seed):
    """
    Run the real LTB -> SCS pipeline for an n_zones profile and return
    the symmetrized strong-SCS connection-strength matrix W (n_zones x
    n_zones), matching Eq. (16)'s W_ij ("derived from the SCS density
    and proximity in r").
    """
    E_callable, edges, signs, mags = make_zone_profile(n_zones, seed)
    r_grid_template = np.linspace(R_MIN, R_MAX, NR)

    # Physical-validity floor (see experiment1_phase_diagram.py for the
    # same fix and its justification).
    E_floor = -M_func(r_grid_template) / r_grid_template + 1e-3
    E_vals_on_grid = np.array([E_callable(r) for r in r_grid_template])
    E_vals_on_grid = np.maximum(E_vals_on_grid, E_floor)

    def E_safe(r, _grid=r_grid_template, _vals=E_vals_on_grid):
        return float(np.interp(r, _grid, _vals))

    t_grid, r_grid, R_grid = solve_ltb_lattice(
        M_func, E_safe, r_min=R_MIN, r_max=R_MAX, nr=NR, t_max=T_MAX, nt=NT
    )
    events = detect_shell_crossings(t_grid, r_grid, R_grid, M_func, dM_dr)
    strong_events = [e for e in events if e.severity == "strong"]

    # Assign each shell index to its zone/branch
    def zone_of_r(r):
        idx = np.searchsorted(edges, r, side="right") - 1
        return int(min(max(idx, 0), n_zones - 1))

    W = np.zeros((n_zones, n_zones))
    for ev in strong_events:
        bi = zone_of_r(ev.r_cross)
        # nearest *different* zone by radial distance to its boundary
        # (consistent with Algorithm 3's "nearest branch of opposite/any sign")
        best_bj, best_dist = None, np.inf
        for bj in range(n_zones):
            if bj == bi:
                continue
            zone_lo, zone_hi = edges[bj], edges[bj + 1]
            dist = min(abs(ev.r_cross - zone_lo), abs(ev.r_cross - zone_hi))
            if dist < best_dist:
                best_dist, best_bj = dist, bj
        if best_bj is not None:
            W[bi, best_bj] += 1.0
            W[best_bj, bi] += 1.0  # symmetrize

    return W, len(strong_events)


def generate_instances(n_zones, n_instances=5, base_seed=7000, max_tries=60):
    """
    Generate n_instances non-trivial (W not all-zero) QUBO instances for
    the given branch count, retrying with new seeds as needed.
    """
    instances = []
    seed = base_seed
    tries = 0
    while len(instances) < n_instances and tries < max_tries:
        W, n_strong = build_branch_graph_W(n_zones, seed)
        tries += 1
        seed += 1
        if np.any(W > 0):
            instances.append({"seed": seed - 1, "W": W.tolist(), "n_strong_scs": n_strong})
        else:
            continue
    return instances


def main():
    all_instances = {}
    for n in [4, 6, 8, 10]:
        print(f"Generating instances for N={n}...")
        insts = generate_instances(n, n_instances=5)
        print(f"  got {len(insts)} instances "
              f"(mean n_strong_scs = {np.mean([i['n_strong_scs'] for i in insts]):.1f})")
        all_instances[str(n)] = insts

    out_path = PROJECT_ROOT.parent / "data" / "qubo_instances_real.json"
    with open(out_path, "w") as f:
        json.dump(all_instances, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
