"""
Multiverse Cost Functional L[M, E; sigma].

Implements the three terms of Eq. (10)-(13) of the paper:

    L = L_junction + L_disconnect + L_curv

  (i)   L_junction   [Eq. (11)]: regularized penalty for the violation of
        the Israel-Darmois extrinsic-curvature matching condition across
        each detected shell-crossing singularity (SCS).
  (ii)  L_disconnect [Eq. (12)]: penalizes pairs of branches that are not
        linked by any SCS transition.
  (iii) L_curv       [Eq. (13)]: penalizes rapid oscillation (high second
        derivative) of the energy profile E(r).

See Sec. IV.D of the paper ("Status of the cost functional") for the
explicit caveat that L_junction is a regularized heuristic proxy for the
extrinsic-curvature jump [K_ab], not a covariant thin-shell (Israel-
Darmois) construction.
"""

import numpy as np
from typing import Callable, List, Dict
from multiverse_graph import MultiverseGraph
from scs_detector import SCSEvent


# ---------------------------------------------------------------------
# (i) Junction penalty, Eq. (11)
# ---------------------------------------------------------------------

def junction_penalty_term(R_prime_at_scs: float, M_prime_at_scs: float) -> float:
    """
    Single-SCS regularized junction penalty,

        L_junction^(sigma) = |[K_ab]|^2 / (1 + delta^-2),   delta = |R'|_min at sigma.

    We use [K_ab] ~ M'(r_x) / R'(t_x, r_x) as a scalar proxy for the
    extrinsic-curvature jump across the SCS (see module docstring and
    Sec. IV.D of the paper for the physical status of this proxy).
    Taking delta = |R'| itself at the event means the term saturates to
    a finite value (rather than diverging) as R' -> 0, which is the
    point of the regularization: it keeps the optimizer well-posed
    without pretending to resolve the true (ill-defined) matching
    problem at R' = 0.
    """
    delta = max(abs(R_prime_at_scs), 1e-12)
    K_proxy = M_prime_at_scs / delta
    return float(K_proxy ** 2 / (1.0 + delta ** -2))


def junction_penalty(scs_events: List[SCSEvent],
                      R_prime_grid: np.ndarray,
                      t_grid: np.ndarray, r_grid: np.ndarray,
                      dM_dr: Callable) -> float:
    """
    Total junction penalty L_junction, Eq. (11), summed over all supplied
    SCS events (typically the "strong" events used to build the
    multiverse graph, Algorithm 3).
    """
    total = 0.0
    for ev in scs_events:
        i = int(np.argmin(np.abs(t_grid - ev.t_cross)))
        j = int(np.argmin(np.abs(r_grid - ev.r_cross)))
        Rp = float(R_prime_grid[i, j])
        Mp = float(dM_dr(ev.r_cross))
        total += junction_penalty_term(Rp, Mp)
    return total


# ---------------------------------------------------------------------
# (ii) Disconnectedness penalty, Eq. (12)
# ---------------------------------------------------------------------

def disconnect_penalty(graph: MultiverseGraph, lambda_disc: float = 1.0) -> float:
    """
    Disconnectedness penalty,

        L_disconnect = lambda_disc * sum_{i<j} (1 - A_ij)

    where A_ij is the binarized, undirected adjacency of the multiverse
    graph: A_ij = 1 if branches i and j are linked by at least one SCS
    transition (in either direction), else 0. Isolated branch pairs each
    contribute a full unit of penalty.
    """
    nodes = sorted(graph.branches.keys())
    n = len(nodes)
    if n <= 1:
        return 0.0
    undirected = graph.graph.to_undirected()
    total = 0.0
    for a in range(n):
        for b in range(a + 1, n):
            i, j = nodes[a], nodes[b]
            connected = undirected.has_edge(i, j)
            total += 0.0 if connected else 1.0
    return float(lambda_disc * total)


# ---------------------------------------------------------------------
# (iii) Curvature regularization, Eq. (13)
# ---------------------------------------------------------------------

def curvature_penalty(E_profile: np.ndarray, r_grid: np.ndarray,
                       lambda_curv: float = 1e-3) -> float:
    """
    Curvature regularization,

        L_curv = lambda_curv * integral dr |E''(r)|^2,

    computed on the radial lattice via a second-order finite-difference
    second derivative of E(r) and trapezoidal integration. Note this
    penalizes oscillation in the *energy profile* E(r), not in the areal
    radius R(t,r); an earlier version of this module incorrectly
    penalized second derivatives of R(t,r), which is a different
    (unphysical, coordinate-dependent) quantity not appearing in Eq. (13).
    """
    dr = float(r_grid[1] - r_grid[0])
    E_pp = np.gradient(np.gradient(E_profile, dr), dr)
    integrand = np.abs(E_pp) ** 2
    integral = float(np.trapezoid(integrand, r_grid))
    return float(lambda_curv * integral)


# ---------------------------------------------------------------------
# Total cost functional, Eq. (10)
# ---------------------------------------------------------------------

def multiverse_cost(scs_events: List[SCSEvent],
                     R_prime_grid: np.ndarray,
                     t_grid: np.ndarray, r_grid: np.ndarray,
                     E_profile: np.ndarray,
                     dM_dr: Callable,
                     graph: MultiverseGraph,
                     lambda_disc: float = 1.0,
                     lambda_curv: float = 1e-3) -> Dict[str, float]:
    """
    Total cost functional L[M, E; sigma], Eq. (10)-(13):

        L = L_junction + L_disconnect + L_curv

    Returns
    -------
    dict
        Each term and the total, for transparency and diagnostics.
    """
    L_junction = junction_penalty(scs_events, R_prime_grid, t_grid, r_grid, dM_dr)
    L_disconnect = disconnect_penalty(graph, lambda_disc=lambda_disc)
    L_curv = curvature_penalty(E_profile, r_grid, lambda_curv=lambda_curv)
    total = L_junction + L_disconnect + L_curv
    return {
        "L_junction": L_junction,
        "L_disconnect": L_disconnect,
        "L_curv": L_curv,
        "L_total": total,
    }


if __name__ == "__main__":
    print("[cost_functional.py] Module loaded. Implements Eq. (10)-(13).")
