r"""Displacement-based formation control laws over an interaction graph.

Notation (2-D, stacked per agent): ``p_i`` position of the controlled point,
``r_i(t)`` desired offset, ``xi_i = p_i - r_i`` the "formation coordinate".
The formation is achieved when all ``xi_i`` agree; with a virtual leader
``p_L(t)`` it is achieved *at the right place* when all ``xi_i = p_L``.

First-order law (single integrator / unicycle hand point)::

    u_i = r_dot_i + b_i p_L_dot
          - k [ sum_j a_ij (xi_i - xi_j) + b_i (xi_i - p_L) ]

* Leaderless (``b = 0``): ``xi`` reaches consensus at the average of the
  initial ``xi`` (undirected graph), so the shape forms around the swarm's
  own centroid.
* Leader-follower (virtual leader pinned to agents with ``b_i > 0``): the
  tracking error ``e = xi - 1 p_L`` obeys ``e_dot = -k (L + B) e`` whenever
  every agent knows ``p_L_dot`` (the feed-forward assumption, stated
  explicitly -- set ``feedforward_to_all=False`` to give it only to pinned
  agents). ``L + B`` is positive definite for a connected undirected graph with
  at least one pinned agent, so ``||e(t)|| <= exp(-k lambda_min t) ||e(0)||``.

Second-order law (double integrator), with ``zeta_i = v_i - r_dot_i``::

    u_i = r_ddot_i + p_L_ddot
          - sum_j a_ij [kp (xi_i - xi_j) + kv (zeta_i - zeta_j)]
          - b_i [kp (xi_i - p_L) + kv (zeta_i - p_L_dot)]

Its error dynamics decouple along the eigenvectors of ``M = L + B`` into
``s^2 + kv lambda s + kp lambda``, which is Hurwitz for every
``lambda > 0`` and ``kp, kv > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .formations import FormationReference, Trajectory
from .graph_topology import laplacian, mask_agents


def _pinning_vector(pinning, n: int) -> np.ndarray:
    if pinning is None:
        return np.zeros(n)
    b = np.asarray(pinning, dtype=float)
    if b.shape != (n,) or np.any(b < 0):
        raise ValueError("pinning must be a non-negative vector of length N")
    return b


@dataclass
class _FormationLawBase:
    adjacency: np.ndarray
    formation: FormationReference
    leader: Trajectory | None = None
    pinning: np.ndarray | None = None
    feedforward_to_all: bool = True

    def __post_init__(self) -> None:
        self.adjacency = np.asarray(self.adjacency, dtype=float)
        n = self.adjacency.shape[0]
        self.b = _pinning_vector(self.pinning, n)
        if self.leader is None and np.any(self.b > 0):
            raise ValueError("pinning gains given but no leader trajectory")
        if self.leader is not None and not np.any(self.b > 0):
            raise ValueError("a leader needs at least one pinned agent (pinning > 0)")

    @property
    def n_agents(self) -> int:
        return self.adjacency.shape[0]

    def _active_terms(self, active: np.ndarray):
        L = laplacian(mask_agents(self.adjacency, active))
        b = np.where(active, self.b, 0.0)
        return L, b

    def _leader_ff_mask(self, b: np.ndarray, active: np.ndarray) -> np.ndarray:
        return active.copy() if self.feedforward_to_all else (b > 0)

    def desired_positions(self, t: float, active: np.ndarray) -> np.ndarray | None:
        """Absolute desired positions ``p_L + r_i`` (None for leaderless control)."""
        if self.leader is None:
            return None
        return self.leader.position(t) + self.formation.offsets(t, active)


@dataclass
class FirstOrderFormationControl(_FormationLawBase):
    """Consensus (leaderless) or leader-follower law for first-order agents."""

    gain: float = 1.0

    order = 1

    def __call__(self, t: float, p: np.ndarray, v: np.ndarray | None, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=bool)
        L, b = self._active_terms(active)
        r = self.formation.offsets(t, active)
        xi = p - r
        u = self.formation.offsets_dot(t, active) - self.gain * (L @ xi)
        if self.leader is not None:
            pL = self.leader.position(t)
            u -= self.gain * b[:, None] * (xi - pL)
            u += self._leader_ff_mask(b, active)[:, None] * self.leader.velocity(t)
        u[~active] = 0.0
        return u


@dataclass
class SecondOrderFormationControl(_FormationLawBase):
    """Consensus (leaderless) or leader-follower law for double integrators."""

    kp: float = 1.0
    kv: float = 2.0

    order = 2

    def __call__(self, t: float, p: np.ndarray, v: np.ndarray, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=bool)
        L, b = self._active_terms(active)
        xi = p - self.formation.offsets(t, active)
        zeta = v - self.formation.offsets_dot(t, active)
        u = self.formation.offsets_ddot(t, active) - L @ (self.kp * xi + self.kv * zeta)
        if self.leader is not None:
            pL, vL = self.leader.position(t), self.leader.velocity(t)
            u -= b[:, None] * (self.kp * (xi - pL) + self.kv * (zeta - vL))
            u += self._leader_ff_mask(b, active)[:, None] * self.leader.acceleration(t)
        u[~active] = 0.0
        return u


def closed_loop_matrix_first_order(adjacency: np.ndarray, pinning, gain: float) -> np.ndarray:
    """``-k (L + B)``: the (per-axis) tracking-error system matrix of the first-order law."""
    return -gain * (laplacian(adjacency) + np.diag(_pinning_vector(pinning, adjacency.shape[0])))


def closed_loop_matrix_second_order(adjacency: np.ndarray, pinning, kp: float, kv: float) -> np.ndarray:
    """``[[0, I], [-kp M, -kv M]]`` with ``M = L + B`` (per-axis error dynamics)."""
    n = adjacency.shape[0]
    M = laplacian(adjacency) + np.diag(_pinning_vector(pinning, n))
    return np.block([[np.zeros((n, n)), np.eye(n)], [-kp * M, -kv * M]])


def saturate(u: np.ndarray, max_norm: float | None) -> np.ndarray:
    """Scale each row of ``u`` so its Euclidean norm is at most ``max_norm``."""
    if max_norm is None:
        return u
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return u * scale
