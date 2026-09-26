"""Collision/obstacle avoidance: triggers when it should, stays silent when it shouldn't."""

import numpy as np
import pytest

from formation_sim.avoidance import (
    CBFSafetyFilter,
    CircleObstacle,
    PotentialFieldAvoidance,
    solve_halfspace_qp_2d,
)
from formation_sim.control_laws import FirstOrderFormationControl
from formation_sim.dynamics import SingleIntegrator
from formation_sim.formations import FormationReference, Line, Stationary
from formation_sim.simulate import Simulation

OBSTACLE = CircleObstacle((3.0, 0.0), 1.0)


def test_potential_field_zero_outside_influence_and_repulsive_inside():
    apf = PotentialFieldAvoidance(obstacles=[OBSTACLE], agent_radius=0.2, influence=0.5, inter_agent=False)
    far = apf(np.array([[0.0, 0.0]]), np.array([True]))  # clearance 1.8 m
    np.testing.assert_array_equal(far, 0.0)
    near = apf(np.array([[1.5, 0.0]]), np.array([True]))  # clearance 0.3 m
    assert near[0, 0] < 0 and abs(near[0, 1]) < 1e-12  # pushes straight away (-x)


def test_potential_field_inter_agent_symmetric():
    apf = PotentialFieldAvoidance(agent_radius=0.2, influence=0.5)
    f = apf(np.array([[0.0, 0.0], [0.6, 0.0]]), np.array([True, True]))
    np.testing.assert_allclose(f[0], -f[1])
    assert f[0, 0] < 0


def test_qp_matches_brute_force():
    rng = np.random.default_rng(42)
    grid = np.stack(np.meshgrid(np.linspace(-4, 4, 801), np.linspace(-4, 4, 801)), -1).reshape(-1, 2)
    for _ in range(15):
        A = rng.normal(size=(4, 2))
        b = rng.normal(size=4) - 1.0  # origin-ish region is typically feasible
        u_nom = rng.normal(scale=2.0, size=2)
        u, feasible = solve_halfspace_qp_2d(u_nom, A, b)
        ok = np.all(grid @ A.T >= b, axis=1)
        if not ok.any():
            continue
        assert feasible
        assert np.all(A @ u >= b - 1e-9)
        brute = np.min(np.linalg.norm(grid[ok] - u_nom, axis=1))
        assert np.linalg.norm(u - u_nom) <= brute + 1e-9
        assert np.linalg.norm(u - u_nom) >= brute - 0.02  # grid spacing is 0.01


def test_qp_passthrough_and_infeasible_flag():
    u, ok = solve_halfspace_qp_2d(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), np.array([0.0]))
    assert ok and np.allclose(u, [1.0, 0.0])
    # x >= 1 and x <= -1 cannot both hold
    _, ok = solve_halfspace_qp_2d(np.zeros(2), np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([1.0, 1.0]))
    assert not ok


def test_cbf_leaves_safe_command_untouched_and_modifies_unsafe_one():
    cbf = CBFSafetyFilter(obstacles=[OBSTACLE], agent_radius=0.2, gamma=1.0)
    p = np.array([[1.5, 0.0]])  # 0.3 m from the obstacle surface, 0.1 m from the safety boundary
    away = np.array([[-1.0, 0.0]])
    np.testing.assert_allclose(cbf(p, away, np.array([True])), away)
    toward = np.array([[2.0, 0.0]])
    out = cbf(p, toward, np.array([True]))
    assert out[0, 0] < toward[0, 0]
    h = np.sum((p[0] - OBSTACLE.center) ** 2) - 1.2**2
    assert 2 * (p[0] - OBSTACLE.center) @ out[0] >= -cbf.gamma * h - 1e-9


def _drive_through_obstacle(cbf):
    """One agent commanded straight through an obstacle towards a goal behind it."""
    ctrl = FirstOrderFormationControl(
        adjacency=np.zeros((1, 1)), formation=FormationReference([(0.0, Line(1.0))]),
        leader=Stationary((6.0, 0.05)), pinning=[1.0], gain=1.0,
    )
    return Simulation(SingleIntegrator(), ctrl, np.array([[0.0, 0.0]]), t_final=12.0,
                      obstacles=[OBSTACLE], max_command=1.0, cbf=cbf).run()


def test_cbf_prevents_collision_that_otherwise_happens():
    unsafe = _drive_through_obstacle(None)
    assert unsafe.min_clearance.min() < 0.2  # without a filter the agent enters the obstacle
    safe = _drive_through_obstacle(CBFSafetyFilter(obstacles=[OBSTACLE], agent_radius=0.2))
    assert safe.min_clearance.min() >= 0.2 - 1e-3  # 1 mm tolerance for the sampled-data loop


def test_cbf_inter_agent_head_on():
    """Two agents swapping places head-on never come closer than 2 * radius."""

    class Swap:
        """Each agent drives proportionally to the other's start (slight offset breaks symmetry)."""

        order = 1
        adjacency = np.zeros((2, 2))
        leader = None
        formation = FormationReference([(0.0, lambda n: np.zeros((n, 2)))])

        def __call__(self, t, p, v, active):
            return np.array([[2.0, 0.02], [-2.0, 0.0]]) - p

    x0 = np.array([[-2.0, 0.0], [2.0, 0.0]])
    res = Simulation(SingleIntegrator(), Swap(), x0, t_final=15.0, max_command=1.0,
                     cbf=CBFSafetyFilter(agent_radius=0.2)).run()
    assert res.min_distance.min() >= 0.4 - 1e-3
    assert res.states[-1, 0, 0] > 1.0 and res.states[-1, 1, 0] < -1.0  # they actually got past each other


@pytest.mark.parametrize("gamma", [0.5, 4.0])
def test_dropped_agent_is_avoided_as_obstacle(gamma):
    cbf = CBFSafetyFilter(agent_radius=0.2, gamma=gamma)
    p = np.array([[0.0, 0.0], [0.5, 0.0]])
    u = cbf(p, np.array([[5.0, 0.0], [0.0, 0.0]]), np.array([True, False]))
    h = 0.25 - 0.16
    # full (unsplit) constraint against the static, failed agent
    assert 2 * (p[0] - p[1]) @ u[0] >= -gamma * h - 1e-9
    np.testing.assert_array_equal(u[1], 0.0)
