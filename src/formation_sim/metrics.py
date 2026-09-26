"""Performance metrics computed from simulation histories."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .avoidance import CircleObstacle, obstacle_arrays


def formation_error(p: np.ndarray, r: np.ndarray, active: np.ndarray) -> float:
    """Translation-invariant shape error.

    RMS over active agents of ``(p_i - p_bar) - (r_i - r_bar)``, i.e. how far
    the swarm is from the desired *shape*, wherever it currently is.
    """
    active = np.asarray(active, dtype=bool)
    if active.sum() == 0:
        return float("nan")
    pa, ra = p[active], r[active]
    dev = (pa - pa.mean(axis=0)) - (ra - ra.mean(axis=0))
    return float(np.sqrt(np.mean(np.sum(dev**2, axis=1))))


def tracking_error(p: np.ndarray, p_des: np.ndarray, active: np.ndarray) -> float:
    """RMS distance between each active agent and its absolute desired position."""
    active = np.asarray(active, dtype=bool)
    if active.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((p[active] - p_des[active]) ** 2, axis=1))))


def settling_time(t: np.ndarray, e: np.ndarray, threshold: float, after: float = 0.0) -> float | None:
    """First time ``>= after`` from which ``e`` stays at or below ``threshold`` for good.

    Returns ``None`` if the signal is above the threshold at the final sample.
    """
    t, e = np.asarray(t), np.asarray(e)
    mask = t >= after
    t, e = t[mask], e[mask]
    if len(e) == 0 or e[-1] > threshold:
        return None
    above = np.flatnonzero(e > threshold)
    if len(above) == 0:
        return float(t[0])
    return float(t[above[-1] + 1])


def min_pairwise_distance(p: np.ndarray, active: np.ndarray | None = None) -> float:
    """Smallest distance between any two agents (active ones only, if a mask is given)."""
    if active is not None:
        p = p[np.asarray(active, dtype=bool)]
    if len(p) < 2:
        return float("inf")
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    iu = np.triu_indices(len(p), k=1)
    return float(d[iu].min())


def min_obstacle_clearance(
    p: np.ndarray, obstacles: Sequence[CircleObstacle], active: np.ndarray | None = None
) -> float:
    """Smallest centre-to-surface distance from any agent to any obstacle."""
    if active is not None:
        p = p[np.asarray(active, dtype=bool)]
    centers, radii = obstacle_arrays(obstacles)
    if len(radii) == 0 or len(p) == 0:
        return float("inf")
    d = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) - radii[None, :]
    return float(d.min())


def first_crossing(t: np.ndarray, e: np.ndarray, threshold: float, after: float = 0.0) -> float | None:
    """First time ``>= after`` at which ``e`` drops to or below ``threshold`` (None if never)."""
    t, e = np.asarray(t), np.asarray(e)
    idx = np.flatnonzero((t >= after) & (e <= threshold))
    return float(t[idx[0]]) if len(idx) else None


def summarize(res, threshold: float = 0.05, agent_radius: float | None = None) -> dict:
    """Headline numbers for a run (all distances in metres, times in seconds)."""
    fe = res.formation_error
    out = {
        "agents": int(res.n_agents),
        "duration_s": float(res.t[-1]),
        "threshold_m": threshold,
        "formation_error_initial_m": float(fe[0]),
        "formation_error_final_m": float(fe[-1]),
        "formation_first_below_threshold_s": first_crossing(res.t, fe, threshold),
        "formation_settled_s": settling_time(res.t, fe, threshold),
        "min_inter_agent_distance_m": float(np.min(res.min_distance)),
    }
    if res.tracking_error is not None:
        te = res.tracking_error
        out.update(
            tracking_error_initial_m=float(te[0]),
            tracking_error_final_m=float(te[-1]),
            tracking_first_below_threshold_s=first_crossing(res.t, te, threshold),
            tracking_settled_s=settling_time(res.t, te, threshold),
        )
    if np.isfinite(res.min_clearance).any():
        out["min_obstacle_clearance_m"] = float(np.min(res.min_clearance))
    if agent_radius is not None:
        out["collision_distance_m"] = 2 * agent_radius
        out["collisions"] = bool(np.min(res.min_distance) < 2 * agent_radius - 1e-9) or bool(
            np.isfinite(res.min_clearance).any() and np.min(res.min_clearance) < agent_radius - 1e-9
        )
    out.update({k: int(v) for k, v in res.info.items()})
    return out
