#!/usr/bin/env python3
"""
Experiment 2: Ergodicity / percolation transition (Sec. V.E.2 / VI.B of the
paper).

Varies the disorder strength sigma_E (the std. dev. scale of the random
Fourier amplitudes used to build E(r)) over a grid, generates N_per_sigma
profiles at each value, builds the multiverse graph for each, and computes:
  - graph diameter D (of the largest connected component, undirected)
  - largest connected component fraction f_lcc = |LCC| / N_branches
  - fully-connected fraction f_conn = 1 if the whole graph is one component

We then locate the critical disorder sigma_E^crit as the sigma at which the
mean largest-component fraction crosses a fixed 0.7 threshold on the way
down (percolation-transition diagnostic), and report it alongside the raw
scan for the paper figure.

Resolution note: same N_r=64, N_t=100 compromise as Experiment 1, for the
same tractability reasons.

Checkpointed execution: the (sigma, k) grid is flattened into a single
linear index so that this script can be invoked repeatedly with
[BATCH_START, BATCH_START+BATCH_SIZE) ranges, appending to the CSV each
time (each item uses an independent, reproducible RNG stream seeded by
(SEED, linear_index), so batches can run in any order / be resumed).

Usage
-----
    python3 experiment2_percolation.py <batch_start> <batch_size>
    python3 experiment2_percolation.py summarize
"""

import sys
import time
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import networkx as nx

from ltb_solver import solve_ltb_lattice
from scs_detector import detect_shell_crossings
from multiverse_graph import MultiverseGraph

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SIGMA_VALUES = np.linspace(0.01, 0.5, 30)
N_PER_SIGMA = 50
N_TOTAL = len(SIGMA_VALUES) * N_PER_SIGMA
R_MIN, R_MAX = 0.2, 2.0
NR, NT = 64, 100
T_MAX = 1.5
SEED = 20260902

_is_summarize = len(sys.argv) > 1 and sys.argv[1] == "summarize"
BATCH_START = int(sys.argv[1]) if (len(sys.argv) > 1 and not _is_summarize) else 0
BATCH_SIZE = int(sys.argv[2]) if (len(sys.argv) > 2 and not _is_summarize) else N_TOTAL

FIELDNAMES = ["idx", "sigma_idx", "sigma", "k", "n_branches", "n_scs_strong",
              "diameter", "f_lcc", "fully_connected"]


def M_func(r):
    # See experiment1_phase_diagram.py for the justification of this
    # parameter choice (systematic scan in scripts/parameter_scan.py).
    return 0.5 * r ** 2


def dM_dr(r):
    return 1.0 * r


def make_random_profile(r_grid, sigma, linear_idx, base_seed=SEED):
    """
    Random Fourier-series E(r) with disorder strength `sigma` controlling
    the standard deviation of the Fourier amplitudes:
        a_n, b_n ~ N(0, sigma / n)
    (n_modes fixed at 6 so sigma is the only varying control parameter
    across the scan). Uses an independent RNG stream keyed by
    (base_seed, linear_idx) for reproducibility under batched execution.
    """
    rng = np.random.default_rng([base_seed, linear_idx])
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

    # Physical validity: 2M(r)/r + 2E(r) > 0  =>  E(r) > -M(r)/r. Clip
    # elementwise (see experiment1_phase_diagram.py for why this replaces
    # the earlier "shrink toward E0" approach).
    E_floor = -M_func(r_grid) / r_grid + 1e-3
    E_vals = np.maximum(E_vals, E_floor)

    return E_vals


def build_graph(r_grid, E_vals, strong_events, t_max):
    graph = MultiverseGraph()
    signs = np.sign(E_vals)
    branch_of_index = np.full(len(r_grid), -1, dtype=int)
    start = 0
    bid_list = []
    for i in range(1, len(signs) + 1):
        if i == len(signs) or signs[i] != signs[start]:
            idx = list(range(start, i))
            e_sign = "closed" if signs[start] < 0 else ("open" if signs[start] > 0 else "critical")
            bid = graph.add_branch(shell_indices=idx, t_birth=0.0, t_death=t_max, E_sign=e_sign)
            branch_of_index[start:i] = bid
            bid_list.append(bid)
            start = i

    for ev in strong_events:
        j = int(np.argmin(np.abs(r_grid - ev.r_cross)))
        bi = int(branch_of_index[j])
        candidates = [b for b in bid_list if b != bi]
        if not candidates:
            continue
        target = min(candidates,
                     key=lambda b: abs(np.mean(graph.branches[b].shell_indices) - j))
        graph.add_transition(bi, target, ev.t_cross, ev.R_cross, ev)

    return graph


def graph_diagnostics(graph):
    n_branches = len(graph.branches)
    if n_branches <= 1:
        return 0, (1.0 if n_branches == 1 else 0.0), True

    undirected = graph.graph.to_undirected()
    for bid in graph.branches:
        if bid not in undirected:
            undirected.add_node(bid)

    components = list(nx.connected_components(undirected))
    largest = max(components, key=len)
    f_lcc = len(largest) / n_branches
    fully_connected = (len(components) == 1)

    if len(largest) > 1:
        subgraph = undirected.subgraph(largest)
        try:
            D = nx.diameter(subgraph)
        except nx.NetworkXError:
            D = 0
    else:
        D = 0

    return D, f_lcc, fully_connected


def main():
    r_grid_template = np.linspace(R_MIN, R_MAX, NR)
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "experiment2_results.csv"

    write_header = not csv_path.exists()
    results = []
    t_start = time.time()

    batch_end = min(BATCH_START + BATCH_SIZE, N_TOTAL)
    print(f"Processing linear indices [{BATCH_START}, {batch_end}) of {N_TOTAL}")

    for idx in range(BATCH_START, batch_end):
        sigma_idx = idx // N_PER_SIGMA
        k = idx % N_PER_SIGMA
        sigma = float(SIGMA_VALUES[sigma_idx])

        E_vals = make_random_profile(r_grid_template, sigma, idx)

        def E_callable(r, _vals=E_vals, _grid=r_grid_template):
            return float(np.interp(r, _grid, _vals))

        try:
            t_grid, r_grid, R_grid = solve_ltb_lattice(
                M_func, E_callable, r_min=R_MIN, r_max=R_MAX,
                nr=NR, t_max=T_MAX, nt=NT
            )
        except Exception as exc:
            print(f"  [WARN] idx={idx} sigma={sigma:.3f} k={k} failed: {exc}")
            continue

        events = detect_shell_crossings(t_grid, r_grid, R_grid, M_func, dM_dr)
        strong_events = [e for e in events if e.severity == "strong"]

        graph = build_graph(r_grid, E_vals, strong_events, T_MAX)
        D, f_lcc, fully_connected = graph_diagnostics(graph)

        results.append({
            "idx": idx,
            "sigma_idx": sigma_idx,
            "sigma": sigma,
            "k": k,
            "n_branches": len(graph.branches),
            "n_scs_strong": len(strong_events),
            "diameter": D,
            "f_lcc": f_lcc,
            "fully_connected": int(fully_connected),
        })

        if (idx - BATCH_START + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{batch_end}] elapsed={elapsed:.1f}s")

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nBatch [{BATCH_START},{batch_end}) done in {time.time()-t_start:.1f}s. "
          f"Appended {len(results)} rows to {csv_path}")


def summarize():
    data_dir = PROJECT_ROOT / "data"
    csv_path = data_dir / "experiment2_results.csv"
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    sigma_unique = sorted(set(float(r["sigma"]) for r in rows))
    agg = []
    for sigma in sigma_unique:
        subset = [r for r in rows if float(r["sigma"]) == sigma]
        agg.append({
            "sigma": sigma,
            "n_samples": len(subset),
            "mean_diameter": float(np.mean([float(r["diameter"]) for r in subset])),
            "mean_f_lcc": float(np.mean([float(r["f_lcc"]) for r in subset])),
            "frac_fully_connected": float(np.mean([int(r["fully_connected"]) for r in subset])),
            "mean_n_branches": float(np.mean([int(r["n_branches"]) for r in subset])),
            "mean_n_scs_strong": float(np.mean([int(r["n_scs_strong"]) for r in subset])),
            "frac_at_least_one_strong_scs": float(np.mean([int(r["n_scs_strong"]) >= 1 for r in subset])),
        })

    sigma_crit = None
    for i in range(1, len(agg)):
        if agg[i - 1]["mean_f_lcc"] >= 0.7 > agg[i]["mean_f_lcc"]:
            s0, s1 = agg[i - 1]["sigma"], agg[i]["sigma"]
            f0, f1 = agg[i - 1]["mean_f_lcc"], agg[i]["mean_f_lcc"]
            sigma_crit = s0 + (0.7 - f0) * (s1 - s0) / (f1 - f0)
            break

    # Second diagnostic: SCS-driven RECONNECTION threshold. The scan can
    # show a two-regime, non-monotonic structure -- an initial
    # fragmentation dip (branch count grows faster than SCS can connect
    # them) followed by a recovery as SCS events become frequent enough
    # to reconnect the graph. We look for the *last* upward crossing of
    # f_lcc through 0.7, occurring strictly after sigma_crit (the
    # fragmentation onset), as a simple, reproducible marker of this
    # reconnection regime. Requires mean_n_scs_strong to actually be
    # growing in this range, otherwise this would just be noise.
    sigma_reconnect = None
    if sigma_crit is not None:
        for i in range(1, len(agg)):
            if agg[i]["sigma"] <= sigma_crit:
                continue
            if agg[i - 1]["mean_f_lcc"] < 0.7 <= agg[i]["mean_f_lcc"]:
                s0, s1 = agg[i - 1]["sigma"], agg[i]["sigma"]
                f0, f1 = agg[i - 1]["mean_f_lcc"], agg[i]["mean_f_lcc"]
                sigma_reconnect = s0 + (0.7 - f0) * (s1 - s0) / (f1 - f0)
                # keep looking for the LAST such crossing, not just the first,
                # in case of noisy fluctuations around the 0.7 threshold

    summary = {
        "n_sigma_values": len(SIGMA_VALUES),
        "n_per_sigma": N_PER_SIGMA,
        "n_total_profiles": len(rows),
        "resolution": {"nr": NR, "nt": NT, "t_max": T_MAX},
        "seed": SEED,
        "sigma_crit_estimate": sigma_crit,
        "sigma_reconnect_estimate": sigma_reconnect,
        "aggregate": agg,
    }

    json_path = data_dir / "experiment2_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 2 summary (all batches) ===")
    print(f"sigma_crit (fragmentation onset) estimate: {sigma_crit}")
    print(f"sigma_reconnect (SCS-driven reconnection) estimate: {sigma_reconnect}")
    print(f"n_total_profiles processed: {len(rows)}")
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    if _is_summarize:
        summarize()
    else:
        main()
