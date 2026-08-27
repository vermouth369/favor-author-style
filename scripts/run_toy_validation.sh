#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${ROOT}/benchmark/favor_bench_v02_1/scripts/validate_paper_alignment.py" \
  "${ROOT}/benchmark/favor_bench_v02_1"
python3 "${ROOT}/benchmark/favor_bench_v02_1/scripts/validate_artifacts.py" \
  --run_dir "${ROOT}/examples/toy_run"
