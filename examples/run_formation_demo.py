"""Run one or all formation-control scenarios end-to-end.

For each scenario this simulates the closed loop, prints headline metrics,
and writes to ``--outdir`` (default ``media/``):

* ``<name>_trajectories.png`` -- agent paths, obstacles, final graph
* ``<name>_errors.png``       -- formation/tracking error and safety distances vs. time
* ``<name>.gif``              -- animation (skip with ``--no-gif``)
* ``results.json``            -- all metrics (merged with any existing file)

Usage::

    python examples/run_formation_demo.py                  # all scenarios
    python examples/run_formation_demo.py --scenario wedge
    python examples/run_formation_demo.py --scenario circle --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from formation_sim import scenarios  # noqa: E402
from formation_sim.metrics import summarize  # noqa: E402
from formation_sim.visualize import animate, plot_errors, plot_trajectories  # noqa: E402


def run(name: str, outdir: Path, gif: bool, show: bool) -> dict:
    sc = scenarios.SCENARIOS[name]()
    t0 = time.perf_counter()
    res = sc.sim.run()
    wall = time.perf_counter() - t0
    summary = summarize(res, sc.settle_threshold, scenarios.AGENT_RADIUS)
    summary["wall_time_s"] = round(wall, 2)
    summary["title"] = sc.title

    print(f"\n=== {sc.title}")
    for k, v in summary.items():
        if k != "title":
            print(f"  {k:36s} {v:.4g}" if isinstance(v, float) else f"  {k:36s} {v}")

    outdir.mkdir(parents=True, exist_ok=True)
    fig = plot_trajectories(res, sc.title)
    fig.savefig(outdir / f"{name}_trajectories.png", dpi=130, bbox_inches="tight")
    fig = plot_errors(res, sc.title, safety_distance=2 * scenarios.AGENT_RADIUS,
                      agent_radius=scenarios.AGENT_RADIUS)
    fig.savefig(outdir / f"{name}_errors.png", dpi=130, bbox_inches="tight")
    if gif:
        animate(res, outdir / f"{name}.gif", sc.title, headings=sc.sim.dynamics.state_dim == 3)
    if show:
        plt.show()
    plt.close("all")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=[*scenarios.SCENARIOS, "all"], default="all")
    ap.add_argument("--outdir", type=Path, default=Path("media"))
    ap.add_argument("--no-gif", action="store_true", help="skip the (slower) GIF rendering")
    ap.add_argument("--show", action="store_true", help="also open interactive plot windows")
    args = ap.parse_args()

    names = list(scenarios.SCENARIOS) if args.scenario == "all" else [args.scenario]
    results_path = args.outdir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name in names:
        results[name] = run(name, args.outdir, gif=not args.no_gif, show=args.show)
    args.outdir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nFigures and metrics written to {args.outdir}/")


if __name__ == "__main__":
    main()
