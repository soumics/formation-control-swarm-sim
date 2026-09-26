#!/usr/bin/env bash
# Run the test suite and regenerate every figure, GIF and results.json in media/.
# Usage (from the repo root, with the environment activated): bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg
python -m pytest -q
python examples/run_formation_demo.py --outdir media
python examples/run_dropout_study.py --outdir media
