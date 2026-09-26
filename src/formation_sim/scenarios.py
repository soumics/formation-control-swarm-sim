"""Ready-made demonstration scenarios.

Each builder returns a :class:`Scenario` holding a configured
:class:`~formation_sim.simulate.Simulation` plus a description. All random
initial conditions are seeded, so every run is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import graph_topology as gt
from .avoidance import CBFSafetyFilter, CircleObstacle, PotentialFieldAvoidance
from .control_laws import FirstOrderFormationControl, SecondOrderFormationControl
from .dynamics import DoubleIntegrator, SingleIntegrator, Unicycle
from .formations import (
    Circle,
    FormationReference,
    Line,
    Sinusoid,
    StraightLine,
    Wedge,
)
from .simulate import Simulation

AGENT_RADIUS = 0.2  # m, physical radius used for all safety checks


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    sim: Simulation
    settle_threshold: float = 0.05  # m, RMS error used for convergence time


def _scatter(rng: np.random.Generator, n: int, center, half_width: float, min_sep: float) -> np.ndarray:
    """Random, non-overlapping initial positions in a square."""
    pts: list[np.ndarray] = []
    while len(pts) < n:
        cand = np.asarray(center, float) + rng.uniform(-half_width, half_width, size=2)
        if all(np.linalg.norm(cand - q) >= min_sep for q in pts):
            pts.append(cand)
    return np.array(pts)


def circle_consensus(seed: int = 1) -> Scenario:
    """Leaderless consensus: 8 single integrators self-organise into a circle."""
    n = 8
    rng = np.random.default_rng(seed)
    x0 = _scatter(rng, n, (0.0, 0.0), 4.0, 3 * AGENT_RADIUS)
    A = gt.circulant_graph(n, 2)
    ctrl = FirstOrderFormationControl(
        adjacency=A, formation=FormationReference([(0.0, Circle(2.0))], initial_positions=x0),
        gain=0.5
    )
    sim = Simulation(
        dynamics=SingleIntegrator(),
        controller=ctrl,
        x0=x0,
        t_final=15.0,
        max_command=1.5,
        cbf=CBFSafetyFilter(agent_radius=AGENT_RADIUS),
    )
    return Scenario(
        "circle",
        "Leaderless consensus -> circle (8 single integrators)",
        "No leader: agents agree on a common formation centre and form a circle "
        "around it. Circulant graph (each agent talks to 2 neighbours per side); "
        "inter-agent collisions prevented by a CBF safety filter.",
        sim,
    )


def line_double_integrator(seed: int = 2) -> Scenario:
    """Leader-follower line formation with double-integrator agents and potential fields."""
    n = 6
    rng = np.random.default_rng(seed)
    x0 = np.hstack([_scatter(rng, n, (-2.0, 0.0), 2.5, 3 * AGENT_RADIUS), np.zeros((n, 2))])
    A = gt.cycle_graph(n)
    b = np.zeros(n)
    b[[0, 3]] = 1.0
    leader = StraightLine(start=(0.0, 0.0), velocity_vec=(0.4, 0.0))
    obstacles = [CircleObstacle((6.0, 1.3), 0.5), CircleObstacle((9.0, -1.2), 0.5)]
    ctrl = SecondOrderFormationControl(
        adjacency=A,
        formation=FormationReference([(0.0, Line(1.0))], initial_positions=x0[:, :2]),
        leader=leader,
        pinning=b,
        kp=0.8,
        kv=1.8,
    )
    sim = Simulation(
        dynamics=DoubleIntegrator(),
        controller=ctrl,
        x0=x0,
        t_final=45.0,
        obstacles=obstacles,
        max_command=1.5,
        potential_field=PotentialFieldAvoidance(
            obstacles=obstacles, agent_radius=AGENT_RADIUS, influence=0.5, gain=0.4
        ),
    )
    return Scenario(
        "line",
        "Leader-follower line abreast (6 double integrators, potential fields)",
        "A virtual leader moves at 0.4 m/s; agents 0 and 3 are pinned to it. "
        "Second-order consensus on a cycle graph; artificial potential fields push "
        "agents around two obstacles, after which the line re-forms. Shows the "
        "classic potential-field weakness: an agent whose slot lies straight behind "
        "an obstacle stalls in a local minimum for a few seconds.",
        sim,
    )


def wedge_obstacles(seed: int = 3) -> Scenario:
    """Moving V formation through an obstacle field with a CBF safety filter."""
    n = 7
    rng = np.random.default_rng(seed)
    x0 = _scatter(rng, n, (-3.0, 0.0), 2.5, 3 * AGENT_RADIUS)
    A = gt.circulant_graph(n, 2)
    b = np.zeros(n)
    b[[0, 5, 6]] = 1.0
    leader = StraightLine(start=(0.0, 0.0), velocity_vec=(0.5, 0.0))
    obstacles = [
        CircleObstacle((6.0, 0.9), 0.7),
        CircleObstacle((11.0, -1.4), 0.8),
        CircleObstacle((15.5, 1.6), 0.6),
    ]
    ctrl = FirstOrderFormationControl(
        adjacency=A,
        formation=FormationReference(
            [(0.0, Wedge(1.2, 35.0))], align_with=leader, initial_positions=x0
        ),
        leader=leader,
        pinning=b,
        gain=2.0,
    )
    sim = Simulation(
        dynamics=SingleIntegrator(),
        controller=ctrl,
        x0=x0,
        t_final=45.0,
        obstacles=obstacles,
        max_command=1.5,
        cbf=CBFSafetyFilter(obstacles=obstacles, agent_radius=AGENT_RADIUS),
    )
    return Scenario(
        "wedge",
        "Leader-follower V formation through obstacles (7 agents, CBF)",
        "Three agents are pinned to a virtual leader moving at 0.5 m/s. A "
        "decentralised control-barrier-function filter keeps every agent clear of "
        "obstacles and of each other; the V deforms around obstacles and recovers.",
        sim,
    )


def reconfigure_unicycles(seed: int = 4) -> Scenario:
    """Unicycle robots on a curved path, reconfiguring line -> wedge -> circle."""
    n = 6
    rng = np.random.default_rng(seed)
    pos = _scatter(rng, n, (-2.0, 0.0), 2.0, 3 * AGENT_RADIUS)
    x0 = np.hstack([pos, rng.uniform(-np.pi, np.pi, size=(n, 1))])
    dyn = Unicycle(hand_offset=0.15)
    A = gt.circulant_graph(n, 2)
    b = np.zeros(n)
    b[[0, 3]] = 1.0
    leader = Sinusoid(start=(0.0, 0.0), speed=0.5, amplitude=2.0, omega=0.2)
    formation = FormationReference(
        [(0.0, Line(1.2)), (15.0, Wedge(1.2, 35.0)), (30.0, Circle(1.5))],
        transition_time=5.0,
        align_with=leader,
        initial_positions=dyn.point(x0),
    )
    ctrl = FirstOrderFormationControl(
        adjacency=A, formation=formation, leader=leader, pinning=b, gain=1.5
    )
    sim = Simulation(
        dynamics=dyn,
        controller=ctrl,
        x0=x0,
        t_final=45.0,
        max_command=1.5,
        cbf=CBFSafetyFilter(agent_radius=AGENT_RADIUS, margin=dyn.point_offset),
    )
    return Scenario(
        "reconfigure",
        "Reconfiguring formation of unicycle robots: line -> wedge -> circle",
        "Nonholonomic unicycles controlled through the hand-position transform. "
        "The virtual leader follows a sinusoid; the formation rotates with the "
        "leader's heading and smoothly morphs line -> wedge (t=15 s) -> circle (t=30 s).",
        sim,
    )


def dropout(topology: str = "circulant", seed: int = 5) -> Scenario:
    """Agent 2 fails at t = 40 s; the survivors re-space into a 5-agent circle.

    ``topology`` is ``"path"`` (chain, agent 2 is a cut vertex) or
    ``"circulant"`` (each agent linked to 2 neighbours per side). Everything
    else -- initial state, gains, pinning (agent 0 only) -- is identical.
    """
    n = 6
    rng = np.random.default_rng(seed)
    x0 = _scatter(rng, n, (-1.0, 0.0), 3.0, 3 * AGENT_RADIUS)
    if topology == "path":
        A = gt.path_graph(n)
    elif topology == "circulant":
        A = gt.circulant_graph(n, 2)
    else:
        raise ValueError(f"unknown topology {topology!r}")
    b = np.zeros(n)
    b[0] = 1.0
    leader = StraightLine(start=(0.0, 0.0), velocity_vec=(0.3, 0.0))
    ctrl = FirstOrderFormationControl(
        adjacency=A,
        formation=FormationReference([(0.0, Circle(2.0))], initial_positions=x0),
        leader=leader,
        pinning=b,
        gain=2.0,
    )
    sim = Simulation(
        dynamics=SingleIntegrator(),
        controller=ctrl,
        x0=x0,
        t_final=70.0,
        max_command=2.0,
        cbf=CBFSafetyFilter(agent_radius=AGENT_RADIUS),
        dropouts={2: 40.0},
    )
    return Scenario(
        f"dropout_{topology}",
        f"Agent dropout robustness - {topology} graph",
        "Agent 2 fails at t = 40 s and becomes a static obstacle; survivors re-space "
        "into a 5-agent circle. Only agent 0 is pinned to the leader.",
        sim,
    )


SCENARIOS = {
    "circle": circle_consensus,
    "line": line_double_integrator,
    "wedge": wedge_obstacles,
    "reconfigure": reconfigure_unicycles,
}
