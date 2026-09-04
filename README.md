# LTB Branch-Connectivity: Shell-Crossing Singularities as a Graph-Theoretic and Optimization Problem

Companion code and data for the paper:

> **A Graph-Theoretic Framework for Shell-Crossing Singularities in LTB Collapse: Percolation, Topology, and the Multiverse Interpretation**
> Fabio Buffoli, University of Brescia
> *submitted to Classical and Quantum Gravity* — preprint on arXiv (gr-qc, cross-listed quant-ph)

---

## What this repository contains

This repository implements the full numerical pipeline behind the paper: a Lemaître–Tolman–Bondi (LTB) dust-collapse solver, a shell-crossing singularity (SCS) detector, a branch-connectivity graph builder, and three experiments —

1. **Statistical topology** of the branch-connectivity graph across 1000 random initial-data profiles (Sec. VI.A of the paper).
2. **Connectivity vs. disorder**: a two-regime, non-monotonic connectivity structure (fragmentation followed by SCS-driven reconnection) across 1500 profiles (Sec. VI.B).
3. **A QUBO solver benchmark** (JAX gradient descent, classical simulated annealing, and QAOA) on branch-connectivity instances derived directly from real LTB data (Sec. VI.C).

All source code, the figure-generation scripts, and the raw per-profile results are included, so every number and figure in the paper is reproducible from this repository.

---

## Repository structure

```
src/
  ltb_solver.py            LTB lattice solver (elliptic/parabolic/hyperbolic branches)
  scs_detector.py          Shell-crossing detection and severity classification
  multiverse_graph.py      Branch-connectivity graph: entropy S, connectivity K (Eq. 7-8)
  cost_functional.py       Variational cost functional L (Eq. 10-13)
  bh_merger_bridge.py      Merger observables: wormhole mass, compactness ratio, tidal parameter (Eq. 18-20)
  optimize_jax.py          JAX-based continuous relaxation (see note below)
  qubo_solver.py           Classical QUBO utilities

scripts/
  experiment1_phase_diagram.py     Experiment 1: statistical topology (1000 profiles)
  experiment2_percolation.py       Experiment 2: connectivity vs. disorder (1500 profiles)
  generate_real_qubo_instances.py  Builds real branch-connectivity QUBO instances (W_ij)
  run_experiment3_real.py          Experiment 3: JAX / SA / QAOA benchmark on real instances
  parameter_scan.py                Mass-profile parameter scan (Table III of the paper)
  diagnose_scs_rarity.py           Diagnostic tool for SCS detection sensitivity
  make_fig_*.py                    Figure-generation scripts (Fig. 1, 2, 4, 5, 6)
  figure_style.py                  Shared plotting style used across all figures

data/
  experiment1_results.csv / _summary.json    Full per-profile results, Experiment 1
  experiment2_results.csv / _summary.json    Full per-profile results, Experiment 2
  qubo_instances_real.json                   The 20 real QUBO instances (N=4,6,8,10)
  experiment3_real_results.json              Solver benchmark results, Experiment 3
  qubo_instances_scan.json                   Mass-profile scan raw data (Table III)

figures/
  All figures in the paper (PDF + PNG), regenerable from data/ via scripts/make_fig_*.py
```

---

## Reproducing the results

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
pip install numpy scipy networkx matplotlib jax optax qiskit qiskit-algorithms
```

**Experiment 1** (statistical topology, ~15–20 min on a single core):
```bash
cd scripts
python experiment1_phase_diagram.py 0 1000
python experiment1_phase_diagram.py summarize
```

**Experiment 2** (connectivity vs. disorder, ~20–30 min):
```bash
python experiment2_percolation.py 0 1500
python experiment2_percolation.py summarize
```

**Experiment 3** (QUBO benchmark; QAOA simulation dominates runtime, N=10 can take ~30–40 min):
```bash
python generate_real_qubo_instances.py
python run_experiment3_real.py 4
python run_experiment3_real.py 6
python run_experiment3_real.py 8
python run_experiment3_real.py 10
```

**Figures** (after the corresponding data exists):
```bash
python make_fig_exp1.py
python make_fig_exp2.py
python make_fig_exp3.py
python make_fig_branch_graph.py
python make_fig_bh_merger.py
```

All random generation is seeded (`np.random.default_rng` with a fixed base seed), so every run is bit-for-bit reproducible.

---

## A note on scientific honesty

This project underwent substantial revision after several code–text discrepancies were found during preparation — including a shell-crossing severity threshold four orders of magnitude stricter than what the paper described, graph entropy/connectivity computed from the wrong quantities, an algebraic error that silently canceled all mass-dependence out of the "wormhole mass" observable, and — most significantly — a QUBO benchmark (Experiment 3) that was originally built from synthetic random graphs unrelated to the physics it claimed to represent.

All of these are described explicitly in **Sec. VII.D ("Implementation status and open gaps")** of the paper, and are reflected in the current state of this code. We believe an honest account of what has and has not been verified is more valuable than a repository that reads as more complete than it is; the git history reflects this process rather than hiding it.

**Known open gap**: the JAX automatic-differentiation optimizer (`optimize_jax.py`) does not yet backpropagate through the full LTB → SCS → graph pipeline as described in Sec. V.D.1 of the paper; it currently optimizes a simplified proxy objective. This does not affect the QUBO relaxation or Experiment 3, which operate on precomputed connection-strength matrices.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Buffoli2026ltbmultiverse,
  author  = {Fabio Buffoli},
  title   = {A Graph-Theoretic Framework for Shell-Crossing Singularities in {LTB} Collapse: Percolation, Topology, and the Multiverse Interpretation},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

Code released under the MIT License. See `LICENSE` for details.

## Contact

Fabio Buffoli — University of Brescia
