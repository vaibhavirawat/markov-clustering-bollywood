"""Markov Clustering (MCL), implemented from scratch with NumPy.

MCL (van Dongen, 2000) simulates random-walk flow on a graph. Two operators are
alternated until the flow matrix stops changing:

  expansion  M <- M^e          lets flow spread (walks of length e)
  inflation  M <- M .^ r       boosts strong flow, starves weak flow (r > 1)

Dense regions keep their flow, sparse links between them dry up, and the
converged matrix has a few "attractor" rows that identify the clusters.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MCLResult:
    clusters: list            # list of lists of node indices
    labels: np.ndarray        # cluster id per node
    iterations: int
    converged: bool
    history: list = field(default_factory=list)   # flow matrices (if requested)
    chaos: list = field(default_factory=list)     # chaos per iteration


def _normalize(M):
    """Make columns sum to 1 (column-stochastic transition matrix)."""
    s = M.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    return M / s


def _chaos(M):
    """van Dongen's convergence measure: 0 when M is idempotent."""
    return float((M.max(axis=0) - (M ** 2).sum(axis=0)).max())


def mcl(adj, expansion=2, inflation=2.0, self_loops="max", prune=1e-5,
        max_iter=200, tol=1e-6, keep_history=False):
    A = np.array(adj, dtype=float)
    n = A.shape[0]
    if self_loops == "max":          # loop weight = strongest edge of the node
        np.fill_diagonal(A, A.max(axis=0))
    elif self_loops:
        A = A + float(self_loops) * np.eye(n)

    M = _normalize(A)
    history = [M.copy()] if keep_history else []
    chaos = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        M = np.linalg.matrix_power(M, expansion)   # expansion
        M = M ** inflation                         # inflation
        M[M < prune] = 0.0                         # pruning keeps M sparse
        M = _normalize(M)
        c = _chaos(M)
        chaos.append(c)
        if keep_history:
            history.append(M.copy())
        if c < tol:
            converged = True
            break

    # Each column j (a node) flows to its strongest attractor row.
    labels_raw = M.argmax(axis=0)
    ids = {a: k for k, a in enumerate(sorted(set(labels_raw)))}
    labels = np.array([ids[a] for a in labels_raw])
    clusters = [[] for _ in ids]
    for node, lab in enumerate(labels):
        clusters[lab].append(node)
    return MCLResult(clusters, labels, it, converged, history, chaos)


def mcl_graph(G, weight="weight", **kw):
    """Run MCL on a networkx graph; returns clusters as lists of node names."""
    import networkx as nx
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight)
    res = mcl(A, **kw)
    named = [[nodes[i] for i in c] for c in res.clusters]
    return named, res, nodes
