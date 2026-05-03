#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_PATH="${LOG_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts_full_dense_cmu}"
ESTIMATOR="${ESTIMATOR:-ekf}"
MC_SAMPLES="${MC_SAMPLES:-1000}"
SUITE_CONTROLLERS="${SUITE_CONTROLLERS:-lqr,pfl,smc,ilqr,mpc}"
MC_CONTROLLERS="${MC_CONTROLLERS:-lqr,pfl,smc,ilqr,mpc}"
RENDER_FORMATS="${RENDER_FORMATS:-gif,mp4}"
SKIP_TESTS="${SKIP_TESTS:-0}"
THEME="${THEME:-paper_dense_cmu}"

if [[ -n "${LOG_PATH}" ]]; then
  mkdir -p "$(dirname "${LOG_PATH}")"
  exec > >(tee -a "${LOG_PATH}") 2>&1
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

phase_banner() {
  local phase_index="$1"
  local total_phases="$2"
  local label="$3"
  local percent=$(( phase_index * 100 / total_phases ))
  printf '\n[%s] ===== (%s/%s %s%%) %s =====\n' "$(timestamp)" "${phase_index}" "${total_phases}" "${percent}" "${label}"
}

if command -v cartpole-bench >/dev/null 2>&1; then
  BENCH_CMD=(cartpole-bench)
else
  export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
  BENCH_CMD=(python -m cartpole_bench.cli)
fi

echo "Dense full cart-pole pipeline"
echo "  output: ${OUTPUT_DIR}"
echo "  estimator: ${ESTIMATOR}"
echo "  suite controllers: ${SUITE_CONTROLLERS}"
echo "  Monte Carlo controllers: ${MC_CONTROLLERS}"
echo "  Monte Carlo samples: ${MC_SAMPLES}"
echo "  render formats: ${RENDER_FORMATS}"
echo "  render theme: ${THEME}"
if [[ -n "${LOG_PATH}" ]]; then
  echo "  log path: ${LOG_PATH}"
fi

TOTAL_PHASES=5

phase_banner 1 "${TOTAL_PHASES}" "pytest"
if [[ "${SKIP_TESTS}" != "1" ]]; then
  pytest
else
  echo "[$(timestamp)] Skipping pytest because SKIP_TESTS=1"
fi

phase_banner 2 "${TOTAL_PHASES}" "run nominal suite"
"${BENCH_CMD[@]}" run-suite \
  --suite nominal \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --output "${OUTPUT_DIR}"

phase_banner 3 "${TOTAL_PHASES}" "run stress suite"
"${BENCH_CMD[@]}" run-suite \
  --suite stress \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --output "${OUTPUT_DIR}"

phase_banner 4 "${TOTAL_PHASES}" "run Monte Carlo"
"${BENCH_CMD[@]}" monte-carlo \
  --controllers "${MC_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --samples "${MC_SAMPLES}" \
  --output "${OUTPUT_DIR}"

phase_banner 5 "${TOTAL_PHASES}" "render and sync README media"
"${BENCH_CMD[@]}" render \
  --controllers "${SUITE_CONTROLLERS}" \
  --estimator "${ESTIMATOR}" \
  --formats "${RENDER_FORMATS}" \
  --theme "${THEME}" \
  --output "${OUTPUT_DIR}"
python scripts/sync_readme_media.py --artifact-root "${OUTPUT_DIR}"

printf '\n[%s] ===== COMPLETE =====\n' "$(timestamp)"
echo "Artifacts: ${OUTPUT_DIR}"
