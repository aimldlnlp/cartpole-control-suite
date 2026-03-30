#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts_rerun_ekf_balanced}"
ESTIMATOR="${ESTIMATOR:-ekf}"
MC_SAMPLES="${MC_SAMPLES:-300}"
SUITE_CONTROLLERS="${SUITE_CONTROLLERS:-lqr,pfl,smc,ilqr,mpc}"
MC_CONTROLLERS="${MC_CONTROLLERS:-lqr,pfl,smc}"
RENDER_FORMATS="${RENDER_FORMATS:-gif,mp4}"
SKIP_TESTS="${SKIP_TESTS:-0}"

if command -v cartpole-bench >/dev/null 2>&1; then
  BENCH_CMD=(cartpole-bench)
else
  BENCH_CMD=(python -m cartpole_bench.cli)
fi

echo "Balanced cart-pole rerun"
echo "  output: ${OUTPUT_DIR}"
echo "  estimator: ${ESTIMATOR}"
echo "  suite controllers: ${SUITE_CONTROLLERS}"
echo "  Monte Carlo controllers: ${MC_CONTROLLERS}"
echo "  Monte Carlo samples: ${MC_SAMPLES}"
echo "  render formats: ${RENDER_FORMATS}"

if [[ "${SKIP_TESTS}" != "1" ]]; then
  pytest
fi

"${BENCH_CMD[@]}" run-suite \
  --suite nominal \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --output "${OUTPUT_DIR}"

"${BENCH_CMD[@]}" run-suite \
  --suite stress \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --output "${OUTPUT_DIR}"

"${BENCH_CMD[@]}" monte-carlo \
  --controllers "${MC_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --samples "${MC_SAMPLES}" \
  --output "${OUTPUT_DIR}"

"${BENCH_CMD[@]}" render \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --formats "${RENDER_FORMATS}" \
  --output "${OUTPUT_DIR}"
