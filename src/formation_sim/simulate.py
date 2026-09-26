"""Fixed-step closed-loop simulator.

Each step (sampled-data, zero-order hold):

1. evaluate the formation law at the controlled points,
2. add the potential-field term (if any) and saturate the nominal command,
3. pass it through the CBF safety filter (if any),
4. map it to the model's native input and integrate one RK4 step.

Agents listed in ``dropouts`` fail at the given time: they stop moving
(their state is frozen), drop out of the interaction graph and the
formation, and remain in the arena as static obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .avoidance import CBFSafetyFilter, CircleObstacle, PotentialFieldAvoidance
from .control_laws import saturate
from .dynamics import rk4_step
from .metrics import (
    formation_error,
    min_obstacle_clearance,
    min_pairwise_distance,
    tracking_error,
)


@dataclass
class SimResult:
    t: np.ndarray                 # (T,)
    states: np.ndarray            # (T, N, state_dim)
    points: np.ndarray            # (T, N, 2) controlled points
    body: np.ndarray              # (T, N, 2) physical robot positions
    offsets: np.ndarray           # (T, N, 2) desired offsets r_i(t)
    desired: np.ndarray | None    # (T, N, 2) absolute desired positions (leader-follower only)
    leader: np.ndarray | None     # (T, 2) virtual-leader position
    controls: np.ndarray          # (T, N, 2) point-level command actually applied
    active: np.ndarray            # (T, N) bool
    formation_error: np.ndarray   # (T,)
    tracking_error: np.ndarray | None
    min_distance: np.ndarray      # (T,) min inter-agent body distance (active agents)
    min_clearance: np.ndarray     # (T,) min body-to-obstacle-surface distance (active agents)
    obstacles: Sequence[CircleObstacle] = ()
    adjacency: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_agents(self) -> int:
        return self.states.shape[1]


@dataclass
class Simulation:
    dynamics: object
    controller: object
    x0: np.ndarray
    t_final: float
    dt: float = 0.01
    obstacles: Sequence[CircleObstacle] = ()
    potential_field: PotentialFieldAvoidance | None = None
    cbf: CBFSafetyFilter | None = None
    max_command: float | None = None
    dropouts: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=float)
        if self.x0.ndim != 2 or self.x0.shape[1] != self.dynamics.state_dim:
            raise ValueError(f"x0 must be (N, {self.dynamics.state_dim})")
        if self.controller.order != self.dynamics.order:
            raise ValueError("controller order does not match the dynamics order")
        if self.cbf is not None and self.dynamics.order != 1:
            raise ValueError("the CBF filter is implemented for first-order (velocity-commanded) points only")

    def active_mask(self, t: float) -> np.ndarray:
        n = self.x0.shape[0]
        return np.array([not (i in self.dropouts and t >= self.dropouts[i]) for i in range(n)])

    def run(self) -> SimResult:
        n_steps = int(round(self.t_final / self.dt))
        N = self.x0.shape[0]
        T = n_steps + 1
        t = np.arange(T) * self.dt
        states = np.empty((T, N, self.dynamics.state_dim))
        controls = np.zeros((T, N, 2))
        actives = np.empty((T, N), dtype=bool)
        offsets = np.empty((T, N, 2))
        has_leader = getattr(self.controller, "leader", None) is not None
        desired = np.empty((T, N, 2)) if has_leader else None
        leader = np.empty((T, 2)) if has_leader else None

        x = self.x0.copy()
        for k in range(T):
            tk = t[k]
            active = self.active_mask(tk)
            states[k], actives[k] = x, active
            offsets[k] = self.controller.formation.offsets(tk, active)
            if has_leader:
                leader[k] = self.controller.leader.position(tk)
                desired[k] = leader[k] + offsets[k]
            if k == n_steps:
                break

            pts = self.dynamics.point(x)
            u = self.controller(tk, pts, self.dynamics.point_velocity(x), active)
            if self.potential_field is not None:
                u = u + self.potential_field(pts, active)
            u = saturate(u, self.max_command)
            if self.cbf is not None:
                u = self.cbf(pts, u, active)
            u[~active] = 0.0
            controls[k] = u

            x_next = rk4_step(self.dynamics, x, self.dynamics.map_input(x, u), self.dt)
            x = np.where(active[:, None], x_next, x)  # failed agents stay frozen

        points = np.stack([self.dynamics.point(s) for s in states])
        body = np.stack([self.dynamics.body_position(s) for s in states])
        f_err = np.array([formation_error(points[k], offsets[k], actives[k]) for k in range(T)])
        t_err = (
            np.array([tracking_error(points[k], desired[k], actives[k]) for k in range(T)])
            if has_leader
            else None
        )
        min_d = np.array([min_pairwise_distance(body[k], actives[k]) for k in range(T)])
        min_c = np.array([min_obstacle_clearance(body[k], self.obstacles, actives[k]) for k in range(T)])

        info = {}
        if self.cbf is not None:
            info["cbf_interventions"] = self.cbf.intervention_count
            info["cbf_infeasible"] = self.cbf.infeasible_count

        return SimResult(
            t=t,
            states=states,
            points=points,
            body=body,
            offsets=offsets,
            desired=desired,
            leader=leader,
            controls=controls,
            active=actives,
            formation_error=f_err,
            tracking_error=t_err,
            min_distance=min_d,
            min_clearance=min_c,
            obstacles=tuple(self.obstacles),
            adjacency=getattr(self.controller, "adjacency", None),
            info=info,
        )
