"""Formation shapes, virtual-leader trajectories and time-varying formation references.

Shapes return offsets ``r_i`` centred on their centroid, with "forward" along
+x. A :class:`FormationReference` turns a schedule of shapes into smooth,
differentiable offsets ``r(t)`` (optionally rotated to the leader's heading),
which is what the controllers consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- shapes


class Shape(Protocol):
    def __call__(self, n: int) -> np.ndarray: ...


def _centered(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0)


@dataclass(frozen=True)
class Line:
    """Agents abreast (side by side, perpendicular to the direction of travel)."""

    spacing: float = 1.0

    def __call__(self, n: int) -> np.ndarray:
        y = (np.arange(n) - (n - 1) / 2) * self.spacing
        return np.column_stack([np.zeros(n), -y])


@dataclass(frozen=True)
class Column:
    """Agents in single file along the direction of travel."""

    spacing: float = 1.0

    def __call__(self, n: int) -> np.ndarray:
        return _centered(np.column_stack([-np.arange(n) * self.spacing, np.zeros(n)]))


@dataclass(frozen=True)
class Wedge:
    """V / wedge: agent 0 at the apex, the rest alternating left/right behind it."""

    spacing: float = 1.0
    half_angle_deg: float = 35.0

    def __call__(self, n: int) -> np.ndarray:
        a = np.deg2rad(self.half_angle_deg)
        pts = [np.zeros(2)]
        for i in range(1, n):
            rank = (i + 1) // 2
            side = 1.0 if i % 2 else -1.0
            pts.append(rank * self.spacing * np.array([-np.cos(a), side * np.sin(a)]))
        return _centered(np.array(pts))


@dataclass(frozen=True)
class Circle:
    radius: float = 1.5

    def __call__(self, n: int) -> np.ndarray:
        ang = 2 * np.pi * np.arange(n) / n
        return self.radius * np.column_stack([np.cos(ang), np.sin(ang)])


# ------------------------------------------------------ leader trajectories


class Trajectory(Protocol):
    def position(self, t: float) -> np.ndarray: ...
    def velocity(self, t: float) -> np.ndarray: ...
    def acceleration(self, t: float) -> np.ndarray: ...


@dataclass
class Stationary:
    point: Sequence[float]

    def position(self, t: float) -> np.ndarray:
        return np.asarray(self.point, dtype=float)

    def velocity(self, t: float) -> np.ndarray:
        return np.zeros(2)

    def acceleration(self, t: float) -> np.ndarray:
        return np.zeros(2)


@dataclass
class StraightLine:
    start: Sequence[float]
    velocity_vec: Sequence[float]

    def position(self, t: float) -> np.ndarray:
        return np.asarray(self.start, float) + t * np.asarray(self.velocity_vec, float)

    def velocity(self, t: float) -> np.ndarray:
        return np.asarray(self.velocity_vec, dtype=float)

    def acceleration(self, t: float) -> np.ndarray:
        return np.zeros(2)


@dataclass
class Sinusoid:
    """``p(t) = start + [speed * t, amplitude * sin(omega * t)]``."""

    start: Sequence[float]
    speed: float
    amplitude: float
    omega: float

    def position(self, t: float) -> np.ndarray:
        return np.asarray(self.start, float) + np.array(
            [self.speed * t, self.amplitude * np.sin(self.omega * t)]
        )

    def velocity(self, t: float) -> np.ndarray:
        return np.array([self.speed, self.amplitude * self.omega * np.cos(self.omega * t)])

    def acceleration(self, t: float) -> np.ndarray:
        return np.array([0.0, -self.amplitude * self.omega**2 * np.sin(self.omega * t)])


# --------------------------------------------------- formation reference


def smoothstep(s: np.ndarray | float) -> np.ndarray | float:
    """C1 blend 0 -> 1 on s in [0, 1] (``3 s^2 - 2 s^3``)."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def rotation(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def assign_slots(current: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Reorder ``slots`` so row ``i`` is the slot for agent ``i`` (Hungarian algorithm).

    Minimises the total squared distance between ``current`` and the assigned
    slots, after removing both centroids (only the shape matters).
    """
    a = current - current.mean(axis=0)
    b = slots - slots.mean(axis=0)
    cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty_like(slots)
    out[rows] = slots[cols]
    return out


@dataclass
class FormationReference:
    """Time-varying desired offsets ``r_i(t)`` for the active agents.

    Parameters
    ----------
    schedule:
        ``[(t_start, shape), ...]`` sorted by time; the first entry should start at 0.
        When a new shape starts, offsets blend smoothly from the previous shape
        over ``transition_time`` seconds.
    align_with:
        If given, offsets are rotated by the heading of this trajectory's
        velocity so the formation "faces" its direction of travel.
    heading:
        Fixed rotation (rad) used when ``align_with`` is None or it is not moving.
    initial_positions:
        If given, the first shape's slots are assigned to agents so as to
        minimise total travel from these positions, and every later shape's
        slots are assigned to minimise travel from the previous shape. This
        avoids agents having to cross through each other (which is what makes
        decentralised collision avoidance deadlock).

    Offsets are recomputed for the *number of active agents*, so after an
    agent drops out the survivors re-space themselves into a full shape.
    Derivatives are central finite differences of the (smooth) offset map.
    """

    schedule: list[tuple[float, Shape]]
    transition_time: float = 0.0
    align_with: Trajectory | None = None
    heading: float = 0.0
    initial_positions: np.ndarray | None = None
    fd_step: float = 1e-4
    _slots: dict = field(default_factory=dict, init=False, repr=False)

    def _shape_rows(self, idx: int, n: int) -> np.ndarray:
        """Body-frame slots of schedule entry ``idx`` for ``n`` agents (cached)."""
        key = (idx, n)
        if key not in self._slots:
            rows = self.schedule[idx][1](n)
            if idx > 0:
                rows = assign_slots(self._shape_rows(idx - 1, n), rows)
            elif self.initial_positions is not None and len(self.initial_positions) == n:
                world = rows @ rotation(self._heading(0.0)).T
                rows = assign_slots(np.asarray(self.initial_positions, float), world) @ rotation(
                    self._heading(0.0)
                )
            self._slots[key] = rows
        return self._slots[key]

    def _shape_offsets(self, t: float, n: int) -> np.ndarray:
        idx = 0
        for k, (t0, _) in enumerate(self.schedule):
            if t >= t0:
                idx = k
        t0 = self.schedule[idx][0]
        cur = self._shape_rows(idx, n)
        if idx == 0 or self.transition_time <= 0:
            return cur
        prev = self._shape_rows(idx - 1, n)
        w = smoothstep((t - t0) / self.transition_time)
        return (1 - w) * prev + w * cur

    def _heading(self, t: float) -> float:
        if self.align_with is None:
            return self.heading
        v = self.align_with.velocity(t)
        if np.linalg.norm(v) < 1e-9:
            return self.heading
        return float(np.arctan2(v[1], v[0]))

    def offsets(self, t: float, active: np.ndarray) -> np.ndarray:
        """``(N, 2)`` offsets; rows of inactive agents are zero."""
        active = np.asarray(active, dtype=bool)
        out = np.zeros((active.size, 2))
        n = int(active.sum())
        if n:
            body = self._shape_offsets(t, n)
            out[active] = body @ rotation(self._heading(t)).T
        return out

    def offsets_dot(self, t: float, active: np.ndarray) -> np.ndarray:
        h = self.fd_step
        return (self.offsets(t + h, active) - self.offsets(t - h, active)) / (2 * h)

    def offsets_ddot(self, t: float, active: np.ndarray) -> np.ndarray:
        h = self.fd_step * 10  # larger step keeps the second difference well-conditioned
        return (
            self.offsets(t + h, active) - 2 * self.offsets(t, active) + self.offsets(t - h, active)
        ) / h**2
