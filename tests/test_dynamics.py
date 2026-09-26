import numpy as np

from formation_sim.dynamics import DoubleIntegrator, SingleIntegrator, Unicycle, rk4_step


class _Decay:
    """x_dot = -x, used to check the integrator's order of accuracy."""

    def deriv(self, x, u):
        return -x


def test_rk4_is_fourth_order():
    x0 = np.array([[1.0]])
    errors = []
    for dt in (0.2, 0.1):
        x = x0.copy()
        for _ in range(int(round(1.0 / dt))):
            x = rk4_step(_Decay(), x, None, dt)
        errors.append(abs(x[0, 0] - np.exp(-1.0)))
    assert errors[1] < 1e-6
    assert 12 < errors[0] / errors[1] < 20  # halving dt shrinks the error by ~2^4


def test_integrators():
    x = np.array([[0.0, 0.0], [1.0, 2.0]])
    u = np.array([[1.0, -1.0], [0.5, 0.0]])
    np.testing.assert_allclose(rk4_step(SingleIntegrator(), x, u, 0.1), x + 0.1 * u)

    xd = np.array([[0.0, 0.0, 1.0, 0.0]])
    ud = np.array([[0.0, 2.0]])
    out = rk4_step(DoubleIntegrator(), xd, ud, 0.5)
    # p = p0 + v0 t + a t^2 / 2, v = v0 + a t  (exact for constant input)
    np.testing.assert_allclose(out, [[0.5, 0.25, 1.0, 1.0]])


def test_unicycle_hand_point_is_single_integrator():
    """The near-identity transform makes the hand point's velocity equal the command."""
    rng = np.random.default_rng(0)
    dyn = Unicycle(hand_offset=0.3)
    x = np.column_stack([rng.normal(size=(5, 2)), rng.uniform(-np.pi, np.pi, 5)])
    u_point = rng.normal(size=(5, 2))
    vw = dyn.map_input(x, u_point)
    h = 1e-7
    numeric = (dyn.point(x + h * dyn.deriv(x, vw)) - dyn.point(x)) / h
    np.testing.assert_allclose(numeric, u_point, atol=1e-5)
    # the body is exactly hand_offset behind the controlled point
    np.testing.assert_allclose(np.linalg.norm(dyn.point(x) - dyn.body_position(x), axis=1), 0.3)
