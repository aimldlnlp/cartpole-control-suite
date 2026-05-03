# Nonlinear Cart-Pole Benchmark

Five-controller nonlinear cart-pole benchmark with a shared swing-up pipeline, reproducible suites, dense paper-style figures, and synchronized comparison animations.

![Nominal full-task comparison](docs/media/side_by_side_nominal_comparison.gif)

Nominal full-task comparison across `LQR`, `PFL`, `SMC`, `iLQR`, and `MPC`.

## Snapshot

- Plant: one nonlinear cart-pole model with force and track limits
- Stack: `energy_pump -> capture_assist -> balance`
- Controllers: `LQR`, `PFL`, `SMC`, `iLQR`, `MPC`
- Estimators: `none`, `ekf`
- Outputs: CSV/JSON runs, summary tables, dense figures, GIF/MP4 animations

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Full Run In `tmux`

Run the exhaustive benchmark in a detached session:

```bash
bash scripts/run_full_pipeline_tmux.sh
```

This launches:

1. `pytest`
2. `nominal` suite for all five controllers
3. `stress` suite for all five controllers
4. Monte Carlo for all five controllers
5. render + README media sync

Progress is written to a timestamped log under `logs/`.

Useful commands:

```bash
tmux attach -t cartpole-full
tail -f logs/full_pipeline_YYYYMMDD_HHMMSS.log
```

Env overrides:

```bash
SESSION_NAME=cartpole-full OUTPUT_DIR=artifacts_full_dense_cmu MC_SAMPLES=1000 RENDER_FORMATS=gif,mp4 bash scripts/run_full_pipeline_tmux.sh
```

If you want the same pipeline without `tmux`:

```bash
bash scripts/run_full_pipeline.sh
```

## Core Commands

```bash
cartpole-bench run-suite --suite nominal --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --output artifacts
cartpole-bench run-suite --suite stress --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --output artifacts
cartpole-bench monte-carlo --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --samples 1000 --output artifacts
cartpole-bench render --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --formats gif,mp4 --theme paper_dense_cmu --output artifacts
```

## Visuals

<p align="center">
  <img src="docs/media/nominal_local_response.png" width="48%" />
  <img src="docs/media/full_task_handoff.png" width="48%" />
</p>
<p align="center">
  <img src="docs/media/stress_comparison.png" width="48%" />
  <img src="docs/media/metric_summary.png" width="48%" />
</p>
<p align="center">
  <img src="docs/media/handoff_focus.png" width="70%" />
</p>

## Layout

```text
configs/                Controller, estimator, experiment, and render configs
docs/media/             README-facing tracked media
scripts/                Full-run, sync, and helper scripts
src/cartpole_bench/     Installable package and CLI
tests/                  Unit, regression, and smoke tests
```

Generated `artifacts*/` directories are local outputs and are intentionally ignored by git.

## Notes

- This is an engineering benchmark, not a formal research benchmark.
- `iLQR` and `MPC` usually look strongest in the saved suites, but they cost much more runtime.
- Full-task control effort includes swing-up and capture, not just near-upright stabilization.

## License

[MIT](LICENSE)
