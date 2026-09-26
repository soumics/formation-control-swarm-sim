import numpy as np
import pytest

from formation_sim.formations import (
    Circle,
    FormationReference,
    Line,
    StraightLine,
    Wedge,
    assign_slots,
)
from formation_sim.metrics import (
    first_crossing,
    formation_error,
    min_pairwise_distance,
    settling_time,
)

ALL = np.ones(5, dtype=bool)


@pytest.mark.parametrize("shape", [Line(1.0), Wedge(1.0), Circle(2.0)])
def test_shapes_are_centred(shape):
    np.testing.assert_allclose(shape(5).mean(axis=0), 0.0, atol=1e-12)


def test_circle_and_line_geometry():
    np.testing.assert_allclose(np.linalg.norm(Circle(2.0)(6), axis=1), 2.0)
    d = np.diff(Line(1.5)(4), axis=0)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.5)


def test_reference_blends_smoothly_and_derivative_matches():
    ref = FormationReference([(0.0, Line(1.0)), (10.0, Circle(2.0))], transition_time=4.0)
    np.testing.assert_allclose(ref.offsets(9.99, ALL), ref.offsets(0.0, ALL))
    np.testing.assert_allclose(ref.offsets(14.0, ALL), ref.offsets(30.0, ALL), atol=1e-12)
    # C1 blend: derivative continuous at the switch, finite-difference derivative consistent
    assert np.abs(ref.offsets_dot(10.0, ALL)).max() < 1e-3
    t, h = 12.0, 1e-3
    fd = (ref.offsets(t + h, ALL) - ref.offsets(t - h, ALL)) / (2 * h)
    np.testing.assert_allclose(ref.offsets_dot(t, ALL), fd, atol=1e-5)


def test_reference_rotates_with_leader_heading():
    ref = FormationReference([(0.0, Line(1.0))], align_with=StraightLine((0, 0), (0.0, 1.0)))
    r = ref.offsets(0.0, np.ones(3, dtype=bool))
    # a line abreast of a leader heading +y lies along the x axis
    np.testing.assert_allclose(r[:, 1], 0.0, atol=1e-12)


def test_dropout_respaces_survivors():
    ref = FormationReference([(0.0, Circle(1.0))])
    active = np.array([True, True, False, True])
    r = ref.offsets(0.0, active)
    np.testing.assert_array_equal(r[2], 0.0)
    pts = r[active]
    d = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
    np.testing.assert_allclose(d, d[0])  # three survivors evenly spaced (triangle)


def test_slot_assignment_is_optimal_permutation():
    slots = Line(1.0)(4)
    perm = np.array([2, 0, 3, 1])
    current = slots[perm] + np.array([10.0, -3.0])  # shifted copy, shuffled
    np.testing.assert_allclose(assign_slots(current, slots), slots[perm])


def test_formation_error_translation_invariant_and_zero_at_target():
    r = Circle(1.0)(5)
    assert formation_error(r + np.array([7.0, -2.0]), r, ALL) == pytest.approx(0.0, abs=1e-12)
    perturbed = r.copy()
    perturbed[0] += [0.5, 0.0]
    assert formation_error(perturbed, r, ALL) > 0.1


def test_settling_and_crossing_times():
    t = np.linspace(0, 10, 101)
    e = np.exp(-t)
    e[50] = 1.0  # a late spike
    assert first_crossing(t, e, 0.1) == pytest.approx(2.4)
    assert settling_time(t, e, 0.1) == pytest.approx(5.1)
    assert settling_time(t, np.ones_like(t), 0.1) is None


def test_min_pairwise_distance():
    p = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    assert min_pairwise_distance(p) == pytest.approx(1.0)
    assert min_pairwise_distance(p, np.array([True, True, False])) == pytest.approx(5.0)
