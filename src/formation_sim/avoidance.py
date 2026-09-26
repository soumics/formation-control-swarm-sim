"""Collision and obstacle avoidance.

Two interchangeable mechanisms act on the controlled point of each agent:

* :class:`PotentialFieldAvoidance` -- additive repulsive term (Khatib-style
  artificial potential field). Works with any dynamics order, but offers no
  formal guarantee and can create local minima.
* :class:`CBFSafetyFilter` -- decentralised control-barrier-function filter
  for first-order agents: each agent minimally modifies its nominal command
  subject to ``h_dot >= -gamma h`` for every nearby agent/obstacle. This
  keeps the safe set forward-invariant as long as the per-agent QPs stay
  feasible (Wang, Ames & Egerstedt, IEEE T-RO 2017).

Agents that have dropped out are treated as static obstacles: they no longer
move, so the survivors must steer around them on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CircleObstacle:
    center: tuple[float, float]
    radius: float


def obstacle_arrays(obstacles: Sequence[CircleObstacle]) -> tuple[np.ndarray, np.ndarray]:
    if not obstacles:
        return np.zeros((0, 2)), np.zeros(0)
    return (
        np.array([o.center for o in obstacles], dtype=float),
        np.array([o.radius for o in obstacles], dtype=float),
    )


# ------------------------------------------------------------ potential field


@dataclass
class PotentialFieldAvoidance:
    """Repulsive potential ``U = eta/2 (1/rho - 1/rho0)^2`` for ``rho < rho0``.

    ``rho`` is the clearance: distance to the obstacle surface (or to the other
    agent) minus ``agent_radius`` (plus ``margin`` for points offset from the
    robot body). The returned term ``-grad U`` is added to the nominal command.

    Pure gradient repulsion has local minima (e.g. an agent whose goal lies
    straight behind an obstacle). ``swirl > 0`` adds a counter-clockwise
    tangential component of ``swirl * |F|`` around obstacles -- a standard
    vortex-field remedy that lets agents slide around instead of stalling.
    """

    obstacles: Sequence[CircleObstacle] = ()
    agent_radius: float = 0.2
    influence: float = 0.8
    gain: float = 0.5
    inter_agent: bool = True
    margin: float = 0.0
    min_clearance: float = 0.02
    max_norm: float | None = 5.0
    swirl: float = 0.0

    def _repulse(self, diff: np.ndarray, clearance: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        direction = diff / np.maximum(dist, 1e-12)
        rho = np.maximum(clearance, self.min_clearance)[..., None]
        mag = np.where(
            rho < self.influence,
            self.gain * (1.0 / rho - 1.0 / self.influence) / rho**2,
            0.0,
        )
        return mag * direction

    def __call__(self, points: np.ndarray, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=bool)
        out = np.zeros_like(points)
        centers, radii = obstacle_arrays(self.obstacles)
        if len(radii):
            diff = points[:, None, :] - centers[None, :, :]
            clear = np.linalg.norm(diff, axis=-1) - radii[None, :] - self.agent_radius - self.margin
            f = self._repulse(diff, clear)
            if self.swirl:
                f = f + self.swirl * np.stack([-f[..., 1], f[..., 0]], axis=-1)
            out += f.sum(axis=1)
        if self.inter_agent and len(points) > 1:
            diff = points[:, None, :] - points[None, :, :]
            clear = np.linalg.norm(diff, axis=-1) - 2 * (self.agent_radius + self.margin)
            np.fill_diagonal(clear, np.inf)
            out += self._repulse(diff, clear).sum(axis=1)
        out[~active] = 0.0
        if self.max_norm is not None:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out *= np.minimum(1.0, self.max_norm / np.maximum(norms, 1e-12))
        return out


# ------------------------------------------------------------------ CBF filter


def solve_halfspace_qp_2d(
    u_nom: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-9
) -> tuple[np.ndarray, bool]:
    """Exactly solve ``min ||u - u_nom||^2  s.t.  A u >= b`` for ``u`` in R^2.

    In 2-D the optimum of this strictly convex QP has at most two linearly
    independent active constraints, so it is one of: ``u_nom`` itself, the
    projection of ``u_nom`` onto one constraint line, or the intersection of
    two constraint lines. We enumerate those candidates and return the
    closest feasible one.

    Returns ``(u, feasible)``. If no candidate is feasible (the constraint set
    is empty) the candidate with the smallest worst-case violation is returned
    with ``feasible=False``.
    """
    u_nom = np.asarray(u_nom, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1, 2)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.shape[0] == 0 or np.all(A @ u_nom >= b - tol):
        return u_nom.copy(), True

    cands = [u_nom[None, :]]
    nrm2 = np.einsum("ij,ij->i", A, A)
    ok = nrm2 > 1e-14
    As, bs, ns = A[ok], b[ok], nrm2[ok]
    cands.append(u_nom + ((bs - As @ u_nom) / ns)[:, None] * As)

    i, j = np.triu_indices(len(As), k=1)
    if len(i):
        a1, a2 = As[i], As[j]
        det = a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0]
        good = np.abs(det) > 1e-12
        a1, a2, det = a1[good], a2[good], det[good]
        b1, b2 = bs[i][good], bs[j][good]
        x = (b1 * a2[:, 1] - b2 * a1[:, 1]) / det
        y = (a1[:, 0] * b2 - a2[:, 0] * b1) / det
        cands.append(np.column_stack([x, y]))

    C = np.vstack(cands)
    slack = C @ A.T - b[None, :]
    worst = slack.min(axis=1)
    dist = np.linalg.norm(C - u_nom, axis=1)
    feasible = worst >= -tol * (1 + np.abs(b).max())
    if np.any(feasible):
        k = np.flatnonzero(feasible)[np.argmin(dist[feasible])]
        return C[k], True
    return C[np.argmax(worst)], False


@dataclass
class CBFSafetyFilter:
    """Decentralised CBF safety filter for single-integrator (point) dynamics.

    Barrier functions (on the controlled point):

    * agent-agent: ``h_ij = ||p_i - p_j||^2 - d_s^2`` with
      ``d_s = 2 (agent_radius + margin)``. The constraint
      ``2 (p_i - p_j)^T (u_i - u_j) >= -gamma h_ij`` is split equally, each agent
      enforcing ``2 (p_i - p_j)^T u_i >= -gamma h_ij / 2``. A dropped (static)
      agent's constraint is taken in full by the surviving agent.
    * agent-obstacle: ``h_io = ||p_i - c_o||^2 - (R_o + agent_radius + margin)^2``,
      constraint ``2 (p_i - c_o)^T u_i >= -gamma h_io``.

    ``margin`` inflates the safety radius for models whose controlled point
    is offset from the body (use ``dynamics.point_offset`` for the unicycle),
    so safety of the point implies safety of the physical robot.
    Only neighbours within ``sensing_radius`` generate constraints.

    The equal split is conservative: an agent directly behind another is
    speed-limited even when both move forward together. A larger ``gamma``
    reduces this without weakening the (continuous-time) safety guarantee;
    keep ``gamma * dt`` well below 1 for the sampled-data implementation.
    """

    obstacles: Sequence[CircleObstacle] = ()
    agent_radius: float = 0.2
    gamma: float = 4.0
    margin: float = 0.0
    sensing_radius: float = 2.5
    infeasible_count: int = field(default=0, init=False)
    intervention_count: int = field(default=0, init=False)

    def constraints(self, i: int, points: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows, rhs = [], []
        p = points[i]
        r_agent = self.agent_radius + self.margin
        for j in range(len(points)):
            if j == i:
                continue
            d = p - points[j]
            dist2 = d @ d
            if dist2 > self.sensing_radius**2:
                continue
            h = dist2 - (2 * r_agent) ** 2
            share = 0.5 if active[j] else 1.0
            rows.append(2 * d)
            rhs.append(-self.gamma * share * h)
        centers, radii = obstacle_arrays(self.obstacles)
        for c, R in zip(centers, radii):
            d = p - c
            if np.linalg.norm(d) - R > self.sensing_radius:
                continue
            h = d @ d - (R + r_agent) ** 2
            rows.append(2 * d)
            rhs.append(-self.gamma * h)
        return np.array(rows).reshape(-1, 2), np.array(rhs)

    def __call__(self, points: np.ndarray, u_nom: np.ndarray, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=bool)
        u = u_nom.copy()
        for i in np.flatnonzero(active):
            A, b = self.constraints(i, points, active)
            u_i, feasible = solve_halfspace_qp_2d(u_nom[i], A, b)
            if not feasible:
                self.infeasible_count += 1
            if not np.allclose(u_i, u_nom[i]):
                self.intervention_count += 1
            u[i] = u_i
        return u
