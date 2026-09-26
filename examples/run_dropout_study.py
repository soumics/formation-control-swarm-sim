"""Robustness to agent loss: identical runs on a path graph vs. a circulant graph.

Agent 2 fails at t = 40 s. On the path graph it is a cut vertex, so the
survivors split into {0, 1} (still pinned to the leader) and {3, 4, 5}
(no longer pinned). On the circulant graph the survivors stay connected.

Writes ``dropout_comparison.png``, per-topology trajectory/error plots, a GIF
of the path-graph case, and a ``dropout`` entry in ``results.json``.

Usage::

    python examples/run_dropout_study.py [--outdir media] [--no-gif]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from formation_sim import graph_topology as gt  # noqa: E402
from formation_sim import scenarios  # noqa: E402
from formation_sim.metrics import first_crossing, settling_time, summarize  # noqa: E402
from formation_sim.visualize import animate, plot_comparison, plot_errors, plot_trajectories  # noqa: E402

TOPOLOGIES = ("path", "circulant")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("media"))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    runs, study = {}, {}
    for topo in TOPOLOGIES:
        sc = scenarios.dropout(topo)
        res = sc.sim.run()
        runs[topo] = (sc, res)
        (agent, t_fail), = sc.sim.dropouts.items()
        A, b = sc.sim.controller.adjacency, sc.sim.controller.b
        survivors = [i for i in range(len(A)) if i != agent]
        comps = gt.connected_components(A, survivors)
        thr = sc.settle_threshold
        te = res.tracking_error
        pre = res.t < t_fail
        entry = summarize(res, thr, scenarios.AGENT_RADIUS)
        entry.update(
            failed_agent=agent,
            failure_time_s=t_fail,
            lambda_min_L_plus_B_before=float(np.linalg.eigvalsh(gt.pinned_laplacian(A, b)).min()),
            algebraic_connectivity_before=gt.algebraic_connectivity(A),
            components_after_failure=comps,
            unpinned_components_after_failure=[c for c in comps if not np.any(b[c] > 0)],
            tracking_first_below_threshold_before_failure_s=first_crossing(res.t[pre], te[pre], thr),
            tracking_error_just_after_failure_m=float(te[np.searchsorted(res.t, t_fail)]),
            tracking_recovered_after_failure_s=(
                None if (s := settling_time(res.t, te, thr, after=t_fail)) is None else round(s - t_fail, 2)
            ),
        )
        study[topo] = entry

        print(f"\n=== {sc.title}")
        for k, v in entry.items():
            print(f"  {k:48s} {v:.4g}" if isinstance(v, float) else f"  {k:48s} {v}")

        fig = plot_trajectories(res, sc.title)
        fig.savefig(args.outdir / f"dropout_{topo}_trajectories.png", dpi=130, bbox_inches="tight")
        fig = plot_errors(res, sc.title, safety_distance=2 * scenarios.AGENT_RADIUS,
                          markers=[(t_fail, f" agent {agent} fails")])
        fig.savefig(args.outdir / f"dropout_{topo}_errors.png", dpi=130, bbox_inches="tight")
        if topo == "path" and not args.no_gif:
            animate(res, args.outdir / "dropout_path.gif", sc.title)

    t_fail = next(iter(runs["path"][0].sim.dropouts.values()))
    fig = plot_comparison(
        {f"{topo} graph": res for topo, (_, res) in runs.items()},
        title=f"Tracking error with agent 2 failing at t = {t_fail:g} s (same gains, same pinned agent)",
        markers=[(t_fail, " agent 2 fails")],
    )
    fig.savefig(args.outdir / "dropout_comparison.png", dpi=130, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close("all")

    results_path = args.outdir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["dropout"] = study
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nFigures and metrics written to {args.outdir}/")


if __name__ == "__main__":
    main()
