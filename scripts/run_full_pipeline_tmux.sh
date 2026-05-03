#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SESSION_NAME="${SESSION_NAME:-cartpole-full}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts_full_dense_cmu}"
ESTIMATOR="${ESTIMATOR:-ekf}"
MC_SAMPLES="${MC_SAMPLES:-1000}"
SUITE_CONTROLLERS="${SUITE_CONTROLLERS:-lqr,pfl,smc,ilqr,mpc}"
MC_CONTROLLERS="${MC_CONTROLLERS:-lqr,pfl,smc,ilqr,mpc}"
RENDER_FORMATS="${RENDER_FORMATS:-gif,mp4}"
SKIP_TESTS="${SKIP_TESTS:-0}"
THEME="${THEME:-paper_dense_cmu}"
LOG_DIR="${LOG_DIR:-logs}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="${LOG_DIR}/full_pipeline_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Choose a different SESSION_NAME or close it first."
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT_DIR}' && LOG_PATH='${LOG_PATH}' OUTPUT_DIR='${OUTPUT_DIR}' ESTIMATOR='${ESTIMATOR}' MC_SAMPLES='${MC_SAMPLES}' SUITE_CONTROLLERS='${SUITE_CONTROLLERS}' MC_CONTROLLERS='${MC_CONTROLLERS}' RENDER_FORMATS='${RENDER_FORMATS}' SKIP_TESTS='${SKIP_TESTS}' THEME='${THEME}' bash scripts/run_full_pipeline.sh"

echo "Started detached full pipeline session"
echo "  session: ${SESSION_NAME}"
echo "  log: ${ROOT_DIR}/${LOG_PATH}"
echo "  attach: tmux attach -t ${SESSION_NAME}"
echo "  tail: tail -f ${ROOT_DIR}/${LOG_PATH}"
