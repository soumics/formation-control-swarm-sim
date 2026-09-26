"""Agent dynamics models, vectorised over ``N`` homogeneous agents.

Every model exposes the same small interface so the simulator and the
controllers do not need to know which one is in use:

* ``state_dim``        -- dimension of one agent's state
* ``order``            -- 1 if the controller commands the velocity of the
                          controlled point, 2 if it commands its acceleration
* ``deriv(x, u)``      -- state derivative, ``x`` is ``(N, state_dim)``
* ``point(x)``         -- ``(N, 2)`` position of the *controlled point*
* ``point_velocity(x)``-- ``(N, 2)`` its velocity (second-order models only)
* ``body_position(x)`` -- ``(N, 2)`` physical robot position (used for safety metrics)
* ``map_input(x, u)``  -- converts a point-level command into the model's native input
* ``point_offset``     -- max distance between controlled point and body (0 for integrators)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SingleIntegrator:
    """``p_dot = u``. State ``[x, y]``."""

    state_dim = 2
    order = 1
    point_offset = 0.0

    def deriv(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return u

    def point(self, x: np.ndarray) -> np.ndarray:
        return x[:, :2]

    def point_velocity(self, x: np.ndarray) -> None:
        return None

    def body_position(self, x: np.ndarray) -> np.ndarray:
        return x[:, :2]

    def map_input(self, x: np.ndarray, u_point: np.ndarray) -> np.ndarray:
        return u_point


class DoubleIntegrator:
    """``p_ddot = u``. State ``[x, y, vx, vy]``."""

    state_dim = 4
    order = 2
    point_offset = 0.0

    def deriv(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.hstack([x[:, 2:4], u])

    def point(self, x: np.ndarray) -> np.ndarray:
        return x[:, :2]

    def point_velocity(self, x: np.ndarray) -> np.ndarray:
        return x[:, 2:4]

    def body_position(self, x: np.ndarray) -> np.ndarray:
        return x[:, :2]

    def map_input(self, x: np.ndarray, u_point: np.ndarray) -> np.ndarray:
        return u_point


@dataclass
class Unicycle:
    """Nonholonomic unicycle ``x_dot = v cos(th), y_dot = v sin(th), th_dot = w``.

    State ``[x, y, theta]``, native input ``[v, w]``.

    Control uses the standard near-identity (hand-position) transformation:
    the point ``h = p + l [cos th, sin th]`` a distance ``l`` ahead of the axle
    has *fully actuated* single-integrator dynamics ``h_dot = u`` under

        [v, w]^T = [[cos th, sin th], [-sin th / l, cos th / l]] u,

    so any single-integrator formation law applies to ``h`` exactly
    (Lawton, Beard & Young, IEEE T-RA 2003). The body stays within ``l`` of ``h``.
    """

    hand_offset: float = 0.2

    state_dim = 3
    order = 1

    def __post_init__(self) -> None:
        if self.hand_offset <= 0:
            raise ValueError("hand_offset must be positive")

    @property
    def point_offset(self) -> float:
        return self.hand_offset

    def deriv(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        th = x[:, 2]
        v, w = u[:, 0], u[:, 1]
        return np.column_stack([v * np.cos(th), v * np.sin(th), w])

    def point(self, x: np.ndarray) -> np.ndarray:
        th = x[:, 2]
        return x[:, :2] + self.hand_offset * np.column_stack([np.cos(th), np.sin(th)])

    def point_velocity(self, x: np.ndarray) -> None:
        return None

    def body_position(self, x: np.ndarray) -> np.ndarray:
        return x[:, :2]

    def map_input(self, x: np.ndarray, u_point: np.ndarray) -> np.ndarray:
        c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
        ux, uy = u_point[:, 0], u_point[:, 1]
        v = c * ux + s * uy
        w = (-s * ux + c * uy) / self.hand_offset
        return np.column_stack([v, w])


def rk4_step(dynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One Runge-Kutta-4 step with the input held constant (zero-order hold)."""
    k1 = dynamics.deriv(x, u)
    k2 = dynamics.deriv(x + 0.5 * dt * k1, u)
    k3 = dynamics.deriv(x + 0.5 * dt * k2, u)
    k4 = dynamics.deriv(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
