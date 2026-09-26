"""Stability of the formation laws on small fixed cases, checked against theory."""

import numpy as np

from formation_sim import graph_topology as gt
from formation_sim.control_laws import (
    FirstOrderFormationControl,
    SecondOrderFormationControl,
    closed_loop_matrix_first_order,
    closed_loop_matrix_second_order,
)
from formation_sim.dynamics import DoubleIntegrator, SingleIntegrator
from formation_sim.formations import Circle, FormationReference, Line, Stationary, StraightLine
from formation_sim.simulate import Simulation

X0 = np.array([[0.0, 0.0], [3.0, -1.0], [-2.0, 2.0], [1.0, 4.0]])


def _leader_follower(k=1.0, leader=None, t_final=40.0):
    A = gt.path_graph(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    leader = leader or Stationary((5.0, 1.0))
    ctrl = FirstOrderFormationControl(
        adjacency=A, formation=FormationReference([(0.0, Line(1.0))]), leader=leader, pinning=b, gain=k
    )
    return Simulation(SingleIntegrator(), ctrl, X0, t_final=t_final, dt=0.01), A, b


def test_first_order_leader_follower_meets_exponential_bound():
    """||e(t)|| <= exp(-k lambda_min(L+B) t) ||e(0)|| (and decreases monotonically)."""
    k = 1.0
    sim, A, b = _leader_follower(k)
    res = sim.run()
    e = np.linalg.norm((res.points - res.desired).reshape(len(res.t), -1), axis=1)
    lam = np.linalg.eigvalsh(gt.pinned_laplacian(A, b)).min()
    bound = np.exp(-k * lam * res.t) * e[0]
    assert lam > 0
    assert np.all(e <= bound + 1e-9)
    assert np.all(np.diff(e) <= 1e-12)
    assert e[-1] < 1e-2 * e[0]


def test_first_order_closed_loop_is_hurwitz_only_with_pinning():
    A = gt.path_graph(4)
    assert np.linalg.eigvals(closed_loop_matrix_first_order(A, [1, 0, 0, 0], 1.0)).real.max() < 0
    assert np.linalg.eigvals(closed_loop_matrix_first_order(A, [0, 0, 0, 0], 1.0)).real.max() > -1e-12


def test_moving_leader_tracked_with_feedforward():
    sim, _, _ = _leader_follower(k=2.0, leader=StraightLine((0.0, 0.0), (0.5, 0.2)), t_final=40.0)
    res = sim.run()
    assert res.tracking_error[-1] < 1e-3


def test_leaderless_consensus_preserves_centroid_and_forms_shape():
    A = gt.cycle_graph(4)
    formation = FormationReference([(0.0, Circle(1.0))])
    ctrl = FirstOrderFormationControl(adjacency=A, formation=formation, gain=1.0)
    res = Simulation(SingleIntegrator(), ctrl, X0, t_final=15.0).run()
    # r is centred, so the centroid of xi = p - r is the centroid of p; consensus keeps it fixed
    np.testing.assert_allclose(res.points[-1].mean(axis=0), X0.mean(axis=0), atol=1e-9)
    assert res.formation_error[-1] < 1e-4
    final = res.points[-1] - res.points[-1].mean(axis=0)
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), 1.0, atol=1e-4)


def test_second_order_law_hurwitz_and_converges():
    A = gt.cycle_graph(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    M = closed_loop_matrix_second_order(A, b, kp=1.0, kv=2.0)
    assert np.linalg.eigvals(M).real.max() < 0
    ctrl = SecondOrderFormationControl(
        adjacency=A, formation=FormationReference([(0.0, Line(1.0))]),
        leader=StraightLine((0.0, 0.0), (0.3, 0.0)), pinning=b, kp=1.0, kv=2.0,
    )
    x0 = np.hstack([X0, np.zeros((4, 2))])
    res = Simulation(DoubleIntegrator(), ctrl, x0, t_final=40.0).run()
    assert res.tracking_error[-1] < 1e-3
    np.testing.assert_allclose(res.states[-1, :, 2:], [[0.3, 0.0]] * 4, atol=1e-3)  # velocities match leader


def test_inactive_agents_receive_no_command():
    sim, _, _ = _leader_follower()
    active = np.array([True, True, False, True])
    u = sim.controller(0.0, X0, None, active)
    np.testing.assert_array_equal(u[2], 0.0)
