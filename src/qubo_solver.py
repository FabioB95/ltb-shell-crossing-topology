"""
QUBO Solver: Map branch-connectivity optimization to a Quadratic Unconstrained
Binary Optimization problem. Solve classically via simulated annealing.

Optional: Connect to D-Wave Ocean SDK if quantum hardware is available.
"""

import numpy as np
from typing import Dict, Tuple, Optional

try:
    import dimod
    HAS_DIMOD = True
except ImportError:
    HAS_DIMOD = False
    print("[qubo_solver.py] dimod not installed. Using custom SA.")


def connectivity_to_qubo(n_branches: int, 
                         transition_matrix: np.ndarray,
                         penalty_unconnected: float = 2.0) -> Dict[Tuple[int, int], float]:
    """
    Convert multiverse branch connectivity to QUBO formulation.
    
    Variables: x_i ∈ {0,1} indicating whether branch i is 'active' in the
    optimal multiverse configuration.
    
    Objective: Maximize connectivity between active branches.
    
    Parameters
    ----------
    n_branches : int
        Number of potential branches.
    transition_matrix : np.ndarray, shape (n_branches, n_branches)
        Weighted adjacency matrix of possible transitions.
    penalty_unconnected : float
        Penalty for selecting disconnected branches.
    
    Returns
    -------
    Dict[Tuple[int, int], float]
        QUBO dictionary: {(i,j): Q_ij}.
    """
    Q = {}
    
    for i in range(n_branches):
        for j in range(i, n_branches):
            if i == j:
                # Linear term: bias toward branches with high self-connectivity
                Q[(i, i)] = -transition_matrix[i, i]
            else:
                # Quadratic term: reward transitions between branches
                Q[(i, j)] = -transition_matrix[i, j] + penalty_unconnected
    
    return Q


def solve_simulated_annealing(Q: Dict[Tuple[int, int], float],
                              n_reads: int = 1000,
                              temperature_schedule: Optional[np.ndarray] = None) -> Dict:
    """
    Solve QUBO using simple simulated annealing.
    
    Parameters
    ----------
    Q : dict
        QUBO coefficients.
    n_reads : int
        Number of annealing runs.
    
    Returns
    -------
    dict
        Best solution found: {'solution': array, 'energy': float}.
    """
    n_vars = max(max(i, j) for i, j in Q.keys()) + 1
    
    # Simple greedy SA
    best_solution = np.random.randint(0, 2, n_vars)
    
    def energy(x):
        return sum(Q[(i, j)] * x[i] * x[j] for (i, j), coeff in Q.items() if coeff != 0)
    
    best_energy = energy(best_solution)
    
    if temperature_schedule is None:
        T_schedule = np.linspace(10.0, 0.01, n_reads)
    else:
        T_schedule = temperature_schedule
    
    current = best_solution.copy()
    current_energy = best_energy
    
    for T in T_schedule:
        # Flip one random bit
        flip_idx = np.random.randint(n_vars)
        candidate = current.copy()
        candidate[flip_idx] = 1 - candidate[flip_idx]
        
        candidate_energy = energy(candidate)
        delta = candidate_energy - current_energy
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current = candidate
            current_energy = candidate_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current.copy()
    
    return {
        'solution': best_solution,
        'energy': best_energy,
        'active_branches': np.where(best_solution == 1)[0].tolist()
    }


if __name__ == "__main__":
    print("[qubo_solver.py] QUBO module loaded.")
    # Demo
    n = 4
    T = np.random.rand(n, n)
    T = (T + T.T) / 2  # Symmetric
    Q = connectivity_to_qubo(n, T)
    result = solve_simulated_annealing(Q, n_reads=5000)
    print(f"  Best energy: {result['energy']:.4f}")
    print(f"  Active branches: {result['active_branches']}")