"""
Multiverse Graph Builder.

Constructs a graph where:
    Nodes = (branch_id, t, R)  — distinct spacetime branches
    Edges = SCS transitions     — shell-crossing connections between branches

Computes graph entropy and other information-theoretic measures.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from dataclasses import dataclass
from scs_detector import SCSEvent


@dataclass
class Branch:
    """A single branch (sheet) of the multiverse."""
    branch_id: int
    shell_indices: List[int]  # Which shells belong to this branch
    t_birth: float
    t_death: float  # When it crosses into another branch or hits singularity
    E_sign: str  # 'open', 'critical', 'closed'


class MultiverseGraph:
    """
    Graph representation of the LTB multiverse topology.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.branches: Dict[int, Branch] = {}
        self.branch_counter = 0
    
    def add_branch(self, shell_indices: List[int], t_birth: float, 
                   t_death: float, E_sign: str) -> int:
        """
        Add a new branch node to the multiverse.
        
        Returns
        -------
        int
            The branch_id assigned.
        """
        bid = self.branch_counter
        self.branch_counter += 1
        
        branch = Branch(bid, shell_indices, t_birth, t_death, E_sign)
        self.branches[bid] = branch
        
        self.graph.add_node(bid, 
                           t_birth=t_birth,
                           t_death=t_death,
                           E_sign=E_sign,
                           n_shells=len(shell_indices))
        return bid
    
    def add_transition(self, from_branch: int, to_branch: int, 
                       t_cross: float, R_cross: float,
                       scs_event: SCSEvent) -> None:
        """
        Add a directed edge representing a shell-crossing transition.
        """
        self.graph.add_edge(from_branch, to_branch,
                           t_cross=t_cross,
                           R_cross=R_cross,
                           r_cross=scs_event.r_cross,
                           weight=1.0)  # Can be modified by junction cost
    
    def compute_graph_entropy(self) -> float:
        """
        Compute the graph (branch-population) entropy, Eq. (7) of the paper:

            S[G(M,E)] = - sum_i p_i log(p_i),   p_i = N(b_i) / sum_j N(b_j)

        where N(b_i) is the number of shells belonging to branch b_i.
        This is a Shannon entropy over how the N_r radial shells are
        distributed among the dynamical branches, NOT the entropy of the
        node-degree distribution (which is a different, purely topological
        quantity previously computed here in error).
        """
        if len(self.branches) == 0:
            return 0.0

        counts = np.array([len(b.shell_indices) for b in self.branches.values()],
                           dtype=float)
        total = counts.sum()
        if total <= 0:
            return 0.0

        p = counts / total
        p = p[p > 0]
        if len(p) <= 1:
            # Single populated branch (or none): S = 0 by definition.
            return 0.0
        entropy = float(-np.sum(p * np.log(p)))
        return entropy

    def compute_connectivity(self, alpha: float = 0.1,
                              n_scs: "int | None" = None) -> float:
        """
        Compute the connectivity order parameter, Eq. (8) of the paper:

            K[G(M,E)] = S + alpha * N_SCS

        Parameters
        ----------
        alpha : float
            Graph-entropy weight (paper default: alpha = 0.1).
        n_scs : int, optional
            Number of strong shell-crossing transitions. If not supplied,
            defaults to the number of directed edges currently stored in
            the graph (i.e. every transition added via `add_transition`
            is assumed to be a strong SCS, consistent with the graph
            builder in Algorithm 3, which only ever registers strong
            events as edges).
        """
        S = self.compute_graph_entropy()
        if n_scs is None:
            n_scs = self.graph.number_of_edges()
        return float(S + alpha * n_scs)

    def compute_degree_entropy(self) -> float:
        """
        Auxiliary (non-paper) quantity: Shannon entropy of the branch
        degree distribution. Kept for diagnostic / topological-structure
        purposes only; NOT used in Eq. (7)-(8) and not reported as S or K
        in the paper. Distinct from `compute_graph_entropy`.
        """
        if len(self.graph) == 0:
            return 0.0
        degrees = np.array([d for _, d in self.graph.degree()])
        if degrees.sum() == 0:
            return 0.0
        p = degrees / degrees.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def compute_edge_density(self) -> float:
        """
        Auxiliary (non-paper) quantity: ratio of actual to maximum
        possible directed edges. Purely topological; distinct from the
        connectivity order parameter K of Eq. (8).
        """
        n = len(self.graph)
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)
        actual_edges = self.graph.number_of_edges()
        return actual_edges / max_edges
    
    def find_richest_multiverse(self, metric: str = 'entropy', alpha: float = 0.1) -> float:
        """
        Return the 'richness' score of this multiverse.

        Parameters
        ----------
        metric : str
            'entropy'      -> graph entropy S, Eq. (7)
            'connectivity' -> connectivity K = S + alpha*N_SCS, Eq. (8)
            'combined'     -> equal-weight mix of S and edge density (diagnostic only)
        """
        if metric == 'entropy':
            return self.compute_graph_entropy()
        elif metric == 'connectivity':
            return self.compute_connectivity(alpha=alpha)
        elif metric == 'combined':
            return 0.5 * self.compute_graph_entropy() + 0.5 * self.compute_edge_density()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def to_igraph(self):
        """Convert to igraph for advanced analysis."""
        import igraph as ig
        return ig.Graph.from_networkx(self.graph)


if __name__ == "__main__":
    print("[multiverse_graph.py] Module loaded. Build graph from SCS events.")