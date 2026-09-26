"""Interaction graphs for multi-agent formation control.

Graphs are represented by their (weighted) adjacency matrix ``A`` with
``A[i, j] > 0`` meaning agent ``i`` receives information from agent ``j``.
All builders below return undirected (symmetric) graphs; :func:`laplacian`
also works for directed graphs (it uses in-degree, i.e. row sums).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np


def _check_n(n: int, minimum: int = 1) -> None:
    if n < minimum:
        raise ValueError(f"need at least {minimum} agents, got {n}")


def from_edges(n: int, edges: Iterable[tuple[int, int]], weight: float = 1.0) -> np.ndarray:
    """Undirected adjacency matrix from an edge list."""
    _check_n(n)
    A = np.zeros((n, n))
    for i, j in edges:
        if i == j:
            raise ValueError(f"self-loop ({i}, {j}) not allowed")
        A[i, j] = A[j, i] = weight
    return A


def complete_graph(n: int) -> np.ndarray:
    _check_n(n)
    return np.ones((n, n)) - np.eye(n)


def path_graph(n: int) -> np.ndarray:
    """Chain 0 - 1 - ... - (n-1). Minimally connected: any interior node is a cut vertex."""
    _check_n(n)
    return from_edges(n, [(i, i + 1) for i in range(n - 1)])


def cycle_graph(n: int) -> np.ndarray:
    _check_n(n, 3)
    return circulant_graph(n, 1)


def star_graph(n: int, center: int = 0) -> np.ndarray:
    _check_n(n, 2)
    return from_edges(n, [(center, j) for j in range(n) if j != center])


def circulant_graph(n: int, k: int) -> np.ndarray:
    """Each agent linked to its ``k`` nearest neighbours on each side of a ring.

    ``k = 1`` is the cycle; ``k >= 2`` adds chords, making the graph
    ``2k``-connected (for ``n > 2k``) and therefore robust to agent loss.
    """
    _check_n(n, 3)
    if not 1 <= k <= n // 2:
        raise ValueError(f"k={k} invalid for n={n}")
    edges = {(i, (i + d) % n) for i in range(n) for d in range(1, k + 1)}
    return from_edges(n, edges)


def degree_matrix(A: np.ndarray) -> np.ndarray:
    return np.diag(A.sum(axis=1))


def laplacian(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian ``L = D - A`` (row sums are zero)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    if np.any(np.diag(A) != 0):
        raise ValueError("adjacency must have a zero diagonal")
    return degree_matrix(A) - A


def algebraic_connectivity(A: np.ndarray) -> float:
    """Second-smallest Laplacian eigenvalue (Fiedler value) of an undirected graph."""
    if not np.allclose(A, A.T):
        raise ValueError("algebraic connectivity is defined here for undirected graphs only")
    if A.shape[0] < 2:
        return 0.0
    return float(np.sort(np.linalg.eigvalsh(laplacian(A)))[1])


def connected_components(A: np.ndarray, nodes: Sequence[int] | None = None) -> list[list[int]]:
    """Connected components (treating any nonzero entry as an undirected edge).

    ``nodes`` restricts the search to a subset of agents (e.g. the ones still active).
    """
    n = A.shape[0]
    nodes = list(range(n)) if nodes is None else list(nodes)
    allowed = set(nodes)
    undirected = (A != 0) | (A.T != 0)
    seen: set[int] = set()
    components = []
    for start in nodes:
        if start in seen:
            continue
        comp, queue = [], deque([start])
        seen.add(start)
        while queue:
            i = queue.popleft()
            comp.append(i)
            for j in np.flatnonzero(undirected[i]):
                j = int(j)
                if j in allowed and j not in seen:
                    seen.add(j)
                    queue.append(j)
        components.append(sorted(comp))
    return components


def is_connected(A: np.ndarray, nodes: Sequence[int] | None = None) -> bool:
    return len(connected_components(A, nodes)) == 1


def mask_agents(A: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Copy of ``A`` with every edge touching an inactive agent removed."""
    active = np.asarray(active, dtype=bool)
    keep = np.outer(active, active)
    return np.where(keep, A, 0.0)


def pinned_laplacian(A: np.ndarray, pinning: np.ndarray) -> np.ndarray:
    """``L + diag(b)`` -- the matrix governing leader-follower error dynamics.

    For an undirected connected graph with at least one ``b_i > 0`` it is
    symmetric positive definite, which is what makes the leader-follower law
    exponentially stable.
    """
    return laplacian(A) + np.diag(np.asarray(pinning, dtype=float))
