"""End-to-end: every shipped scenario runs, converges and stays collision-free."""

import numpy as np
import pytest

from formation_sim import scenarios
from formation_sim.metrics import summarize
from formation_sim.visualize import animate, plot_errors, plot_trajectories


@pytest.fixture(scope="module")
def results():
    return {name: builder().sim.run() for name, builder in scenarios.SCENARIOS.items()}


@pytest.mark.parametrize("name", list(scenarios.SCENARIOS))
def test_scenario_converges_without_collision(results, name):
    res = results[name]
    s = summarize(res, 0.05, scenarios.AGENT_RADIUS)
    assert s["formation_error_final_m"] < 0.05
    if res.tracking_error is not None:
        assert s["tracking_error_final_m"] < 0.05
    assert s["min_inter_agent_distance_m"] >= 2 * scenarios.AGENT_RADIUS - 1e-3
    if "min_obstacle_clearance_m" in s:
        assert s["min_obstacle_clearance_m"] >= scenarios.AGENT_RADIUS - 1e-3
    assert s.get("cbf_infeasible", 0) == 0


def test_dropout_path_splits_but_circulant_recovers():
    path = scenarios.dropout("path").sim.run()
    circ = scenarios.dropout("circulant").sim.run()
    assert circ.tracking_error[-1] < 0.01
    assert path.tracking_error[-1] > 0.1  # unpinned sub-group {3, 4, 5} cannot correct its offset
    assert not path.active[-1, 2] and np.all(path.active[0])


def test_plots_and_gif_are_written(results, tmp_path):
    res = results["circle"]
    plot_trajectories(res, "t").savefig(tmp_path / "traj.png")
    plot_errors(res, "e", safety_distance=0.4).savefig(tmp_path / "err.png")
    gif = animate(res, tmp_path / "a.gif", frame_dt=3.0)
    for f in ("traj.png", "err.png", "a.gif"):
        assert (tmp_path / f).stat().st_size > 1000
    assert gif.exists()
