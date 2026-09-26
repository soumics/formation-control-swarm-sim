import numpy as np
import pytest

from formation_sim import graph_topology as gt


def test_path_laplacian_matches_hand_computation():
    L = gt.laplacian(gt.path_graph(3))
    expected = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=float)
    np.testing.assert_array_equal(L, expected)


@pytest.mark.parametrize(
    "A",
    [gt.complete_graph(5), gt.cycle_graph(6), gt.star_graph(4), gt.circulant_graph(8, 2), gt.path_graph(7)],
)
def test_laplacian_properties(A):
    L = gt.laplacian(A)
    np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)  # rows sum to zero
    np.testing.assert_allclose(L, L.T)  # undirected -> symmetric
    eig = np.linalg.eigvalsh(L)
    assert eig.min() > -1e-10  # positive semidefinite
    assert abs(eig.min()) < 1e-10  # 1 is in the null space


def test_known_spectra():
    # Complete graph K_n: eigenvalues {0, n, ..., n}; cycle C_n: 2 - 2 cos(2 pi k / n)
    np.testing.assert_allclose(np.linalg.eigvalsh(gt.laplacian(gt.complete_graph(5))), [0, 5, 5, 5, 5], atol=1e-12)
    n = 6
    expected = np.sort(2 - 2 * np.cos(2 * np.pi * np.arange(n) / n))
    np.testing.assert_allclose(np.linalg.eigvalsh(gt.laplacian(gt.cycle_graph(n))), expected, atol=1e-12)


def test_connectivity_and_fiedler_value():
    assert gt.is_connected(gt.path_graph(5))
    assert gt.algebraic_connectivity(gt.path_graph(5)) > 0
    split = gt.from_edges(4, [(0, 1), (2, 3)])
    assert not gt.is_connected(split)
    assert gt.algebraic_connectivity(split) == pytest.approx(0.0, abs=1e-12)
    assert gt.connected_components(split) == [[0, 1], [2, 3]]


def test_agent_removal_path_vs_circulant():
    active = np.array([True, True, False, True, True, True])
    survivors = np.flatnonzero(active)
    path = gt.path_graph(6)
    masked = gt.mask_agents(path, active)
    assert np.all(masked[2] == 0) and np.all(masked[:, 2] == 0)
    assert gt.connected_components(path, survivors) == [[0, 1], [3, 4, 5]]
    assert gt.is_connected(gt.circulant_graph(6, 2), survivors)


def test_pinned_laplacian_positive_definite_iff_pinned():
    A = gt.cycle_graph(5)
    assert np.linalg.eigvalsh(gt.pinned_laplacian(A, np.zeros(5))).min() == pytest.approx(0, abs=1e-12)
    b = np.zeros(5)
    b[0] = 1.0
    assert np.linalg.eigvalsh(gt.pinned_laplacian(A, b)).min() > 0


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        gt.from_edges(3, [(1, 1)])
    with pytest.raises(ValueError):
        gt.circulant_graph(6, 4)
    with pytest.raises(ValueError):
        gt.laplacian(np.eye(3))
