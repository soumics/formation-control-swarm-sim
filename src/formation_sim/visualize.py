"""Static plots and animations of simulation results (Matplotlib).

Callers choose the backend; scripts that only save files should call
``matplotlib.use("Agg")`` before importing this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle as CirclePatch

from .simulate import SimResult

# Palette: one hue for "the swarm", categorical slots only for <= 3 compared series.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OBSTACLE = "#b9b7b0"
EDGE = "#c9c7c0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
AGENT = SERIES[0]
LEADER = SERIES[1]

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "font.size": 9,
}


@contextmanager
def _style():
    with plt.rc_context(STYLE):
        yield


def _bounds(res: SimResult, pad: float = 1.0):
    pts = [res.body.reshape(-1, 2)]
    if res.desired is not None:
        pts.append(res.desired.reshape(-1, 2))
    for o in res.obstacles:
        c = np.asarray(o.center)
        pts.append(np.array([c - o.radius, c + o.radius]))
    allp = np.vstack(pts)
    lo, hi = allp.min(axis=0) - pad, allp.max(axis=0) + pad
    return lo, hi


def _map_figsize(lo, hi, width: float = 8.0, legend_in: float = 1.9) -> tuple[float, float]:
    """Figure size whose equal-aspect map area has no dead vertical bands."""
    span = hi - lo
    axes_w = width - legend_in - 0.8
    height = axes_w * span[1] / span[0] + 1.0
    return width, float(np.clip(height, 2.6, 8.0))


def _draw_obstacles(ax, res: SimResult) -> None:
    for o in res.obstacles:
        ax.add_patch(CirclePatch(o.center, o.radius, facecolor=OBSTACLE, edgecolor="none", zorder=1))


def _draw_edges(ax, p: np.ndarray, A: np.ndarray | None, active: np.ndarray) -> list:
    lines = []
    if A is None:
        return lines
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] and active[i] and active[j]:
                (ln,) = ax.plot(*zip(p[i], p[j]), color=EDGE, lw=0.8, zorder=2)
                lines.append(ln)
    return lines


def plot_trajectories(res: SimResult, title: str = "", ax=None, show_graph: bool = True):
    """Paths of every agent, start (hollow) and final (filled) positions, final graph."""
    with _style():
        lo, hi = _bounds(res)
        if ax is None:
            fig, ax = plt.subplots(figsize=_map_figsize(lo, hi))
        else:
            fig = ax.figure
        _draw_obstacles(ax, res)
        k_end = len(res.t) - 1
        if show_graph:
            _draw_edges(ax, res.body[k_end], res.adjacency, res.active[k_end])
        if res.leader is not None:
            ax.plot(*res.leader.T, color=LEADER, lw=1.2, ls="--", label="virtual leader path", zorder=3)
        for i in range(res.n_agents):
            ax.plot(*res.body[:, i].T, color=AGENT, lw=1.2, alpha=0.75, zorder=3,
                    label="agent paths" if i == 0 else None)
        ax.scatter(*res.body[0].T, s=36, facecolors=SURFACE, edgecolors=AGENT, lw=1.5, zorder=4, label="start")
        alive = res.active[k_end]
        ax.scatter(*res.body[k_end, alive].T, s=48, color=AGENT, edgecolors=SURFACE, lw=1.5, zorder=5, label="final")
        if (~alive).any():
            ax.scatter(*res.body[k_end, ~alive].T, s=60, marker="x", color=INK_2, zorder=5, label="failed agent")
        for i in range(res.n_agents):
            ax.annotate(str(i), res.body[k_end, i], xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if title:
            ax.set_title(title, loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        return fig


def plot_errors(res: SimResult, title: str = "", safety_distance: float | None = None,
                agent_radius: float | None = None, markers: Sequence[tuple[float, str]] = ()):
    """Two stacked panels sharing time: (1) formation/tracking error, (2) safety distances."""
    with _style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.2), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        ax1.semilogy(res.t, np.maximum(res.formation_error, 1e-6), color=SERIES[0], label="formation (shape) error")
        if res.tracking_error is not None:
            ax1.semilogy(res.t, np.maximum(res.tracking_error, 1e-6), color=SERIES[1], label="tracking error vs. leader")
        ax1.set_ylabel("RMS error [m]")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        if title:
            ax1.set_title(title, loc="left")

        ax2.plot(res.t, res.min_distance, color=SERIES[0], label="min inter-agent distance")
        if np.isfinite(res.min_clearance).any():
            ax2.plot(res.t, res.min_clearance, color=SERIES[2], label="min obstacle clearance")
        if safety_distance is not None:
            ax2.axhline(safety_distance, color=INK_2, lw=1, ls=":", label=f"collision distance ({safety_distance:.1f} m)")
        if agent_radius is not None and np.isfinite(res.min_clearance).any():
            ax2.axhline(agent_radius, color=INK_2, lw=1, ls="--", label=f"agent radius ({agent_radius:.1f} m)")
        ax2.set_ylim(bottom=0)
        top = np.nanpercentile(np.minimum(res.min_distance, 10), 99)
        ax2.set_ylim(0, max(top * 1.15, (safety_distance or 0) * 2))
        ax2.set_ylabel("distance [m]")
        ax2.set_xlabel("time [s]")
        ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        for ax in (ax1, ax2):
            for tm, label in markers:
                ax.axvline(tm, color=INK_2, lw=0.8, ls="--")
        for tm, label in markers:
            ax1.annotate(label, (tm, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                         textcoords="offset points", fontsize=7, color=INK_2)
        fig.tight_layout()
        return fig


def plot_comparison(results: dict[str, SimResult], title: str = "", metric: str = "tracking_error",
                    markers: Sequence[tuple[float, str]] = ()):
    """Overlay one error metric for several runs (<= 3) on a log axis."""
    if len(results) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} runs can be compared on one axis")
    with _style():
        fig, ax = plt.subplots(figsize=(9.0, 3.8))
        for color, (label, res) in zip(SERIES, results.items()):
            ax.semilogy(res.t, np.maximum(getattr(res, metric), 1e-6), color=color, label=label)
        for tm, label in markers:
            ax.axvline(tm, color=INK_2, lw=0.8, ls="--")
            ax.annotate(label, (tm, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                        textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"RMS {metric.replace('_', ' ')} [m]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        if title:
            ax.set_title(title, loc="left")
        fig.tight_layout()
        return fig


def animate(res: SimResult, path: str | Path, title: str = "", fps: int = 20, frame_dt: float = 0.2,
            trail: float = 6.0, dpi: int = 100, headings: bool = False, window: float | None = 12.0) -> Path:
    """Save an animated GIF of the run (Pillow writer, no ffmpeg needed).

    If the arena is wider than ``window`` metres, the view pans horizontally to
    follow the swarm's centroid so the agents stay readable.
    """
    path = Path(path)
    stride = max(1, int(round(frame_dt / (res.t[1] - res.t[0]))))
    frames = list(range(0, len(res.t), stride))
    trail_steps = int(round(trail / (res.t[1] - res.t[0])))
    with _style():
        lo, hi = _bounds(res)
        follow = window is not None and hi[0] - lo[0] > window
        view_hi = np.array([lo[0] + window, hi[1]]) if follow else hi
        fig, ax = plt.subplots(figsize=_map_figsize(lo, view_hi))
        ax.set_xlim(lo[0], view_hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        if title:
            ax.set_title(title, loc="left")
        _draw_obstacles(ax, res)
        trails = [ax.plot([], [], color=AGENT, lw=1.0, alpha=0.35, zorder=3)[0] for _ in range(res.n_agents)]
        desired = ax.scatter([], [], s=70, facecolors="none", edgecolors=AGENT, lw=1.0, alpha=0.6, zorder=4,
                             label="desired slot")
        agents = ax.scatter([], [], s=55, color=AGENT, edgecolors=SURFACE, lw=1.2, zorder=6, label="agent")
        failed = ax.scatter([], [], s=60, marker="x", color=INK_2, zorder=6,
                            label="failed agent" if (~res.active).any() else None)
        leader = ax.scatter([], [], s=110, marker="*", color=LEADER, zorder=7, label="virtual leader") \
            if res.leader is not None else None
        quiver = None
        if headings:
            quiver = ax.quiver(res.body[0, :, 0], res.body[0, :, 1], np.ones(res.n_agents), np.zeros(res.n_agents),
                               color=INK, scale=22, width=0.004, zorder=7)
        clock = ax.text(0.99, 0.02, "", transform=ax.transAxes, ha="right", va="bottom", color=INK_2, fontsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        edge_lines: list = []

        def update(k):
            nonlocal edge_lines
            for ln in edge_lines:
                ln.remove()
            act = res.active[k]
            edge_lines = _draw_edges(ax, res.body[k], res.adjacency, act)
            k0 = max(0, k - trail_steps)
            for i, ln in enumerate(trails):
                ln.set_data(res.body[k0:k + 1, i, 0], res.body[k0:k + 1, i, 1])
            agents.set_offsets(res.body[k, act])
            failed.set_offsets(res.body[k, ~act] if (~act).any() else np.empty((0, 2)))
            if res.desired is not None:
                desired.set_offsets(res.desired[k, act])
            else:  # leaderless: show the shape around the swarm's current centroid
                c = res.points[k, act].mean(axis=0)
                r = res.offsets[k, act]
                desired.set_offsets(c + r - r.mean(axis=0))
            if leader is not None:
                leader.set_offsets(res.leader[k][None, :])
            if quiver is not None:
                th = res.states[k, :, 2]
                quiver.set_offsets(res.body[k])
                quiver.set_UVC(np.cos(th), np.sin(th))
            if follow:
                cx = res.body[k, act, 0].mean()
                x0 = float(np.clip(cx - window / 2, lo[0], hi[0] - window))
                ax.set_xlim(x0, x0 + window)
            clock.set_text(f"t = {res.t[k]:5.1f} s")
            return []

        anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    return path
