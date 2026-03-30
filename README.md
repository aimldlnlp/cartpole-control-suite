# Nonlinear Cart-Pole Benchmark

Portfolio-grade nonlinear cart-pole benchmark comparing five upright stabilizers under a shared swing-up pipeline, with optional EKF state estimation, reproducible suites, curated figures, and synchronized animations.

## Hero Visual

![Nominal full-task comparison](docs/media/side_by_side_nominal_comparison.gif)

Synchronized nominal full-task comparison across `LQR`, `PFL`, `SMC`, `iLQR`, and `MPC`.

## What This Repo Is

This repository packages a nonlinear cart-pole benchmark for control engineering and portfolio presentation. It is built around a single plant model, a shared swing-up and capture stack, and multiple near-upright stabilizers so that controller differences are evaluated under the same task setup.

The repo includes:

- full nonlinear cart-pole dynamics with track and force limits
- repeated-seed `nominal` and `stress` suites
- Monte Carlo robustness evaluation
- export of per-run CSV and JSON artifacts
- publication-style figure rendering
- animation rendering for single-controller and full-comparison views

It is an engineering benchmark, not a formal research benchmark.

## Control Stack

All controllers share the same hybrid pre-balance flow:

- `energy_pump`
- `capture_assist`
- `balance`

The balance-stage stabilizers currently implemented are:

- `LQR`
- `Feedback Linearization (PFL)`
- `Sliding Mode Control (SMC)`
- `Iterative LQR (iLQR)`
- `Model Predictive Control (MPC)`

The benchmark supports two estimator modes:

- `none`
- `ekf`

Each saved run records explicit diagnostics, including:

- success / failure status
- first balance entry time
- balance fraction
- track violation state
- failure reason
- controller debug payloads where applicable

## Controllers and Estimator Support

The project now reflects the full five-controller benchmark state:

- `LQR`, `PFL`, and `SMC` provide the fast classical baselines
- `iLQR` and `MPC` provide optimization-based stabilizers under the same hybrid switching logic
- `EKF` support is available as a first-class estimator path in both suites and rendering

The recommended final showcase path is:

- run `nominal + stress` for all five controllers
- run Monte Carlo for `LQR/PFL/SMC`
- render from the same artifact root

That split keeps the final artifact representative while avoiding unnecessary optimizer-heavy Monte Carlo runtime.

## Benchmark Suite

### Nominal

- `local_small_angle`
- `full_task_hanging`

### Stress

- `measurement_noise`
- `impulse_disturbance`
- `friction_and_damping`
- `large_angle_recovery`
- `parameter_mismatch`

### Monte Carlo

The Monte Carlo benchmark samples randomized conditions over:

- initial angle and angular rate
- disturbance settings
- friction and damping
- parameter mismatch
- estimator/controller combinations selected at the CLI

## Current Benchmark Snapshot

The current public-facing snapshot is based on the final merged `ekf` benchmark artifact used to curate the visuals in `docs/media/`.

Current saved results:

- Under `ekf`, all five controllers succeeded on the saved repeated-seed `nominal` and `stress` suites.
- Monte Carlo in the final showcase artifact is intentionally limited to the fast classical controllers:
  - `LQR = 0.81`
  - `PFL = 0.77`
  - `SMC = 0.71`
  - `samples = 300`
- `iLQR` and `MPC` are functionally healthy in `nominal + stress`, but Monte Carlo for them is intentionally omitted from the final showcase because of runtime cost.

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .[dev]
```

Run the test suite:

```bash
pytest
```

## Reproduction Commands

Run the nominal suite for all five controllers:

```bash
cartpole-bench run-suite --suite nominal --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --output artifacts
```

Run the stress suite for all five controllers:

```bash
cartpole-bench run-suite --suite stress --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --output artifacts
```

Run the time-balanced Monte Carlo pass:

```bash
cartpole-bench monte-carlo --controllers lqr,pfl,smc --estimator ekf --samples 300 --output artifacts
```

Render figures and animations from previously saved artifacts:

```bash
cartpole-bench render --controllers lqr,pfl,smc,ilqr,mpc --estimator ekf --formats gif,mp4 --output artifacts
```

Run the helper pipeline that matches the recommended final showcase flow:

```bash
bash scripts/run_balanced_pipeline.sh
```

`cartpole-bench all` still exists as a simulation-only convenience command, but it is not the recommended final showcase path because it applies one controller list across both suites and Monte Carlo.

## Visual Highlights

![Nominal local response](docs/media/nominal_local_response.png)

Near-upright recovery view showing angle, cart motion, force, and phase behavior across the five stabilizers.

![Full-task handoff](docs/media/full_task_handoff.png)

Shared swing-up and capture pipeline with aligned balance entry behavior across all five methods.

![Stress comparison](docs/media/stress_comparison.png)

Repeated-seed stress summary across measurement noise, disturbance, damping, large-angle recovery, and parameter mismatch.

![Metric summary](docs/media/metric_summary.png)

Compact publication-style summary of settling time, overshoot, steady-state error, and control effort.

![Handoff focus](docs/media/handoff_focus.png)

Focused view around the balance-entry window, useful for comparing post-capture behavior.

## Repository Layout

```text
configs/                JSON configs for controllers, estimators, experiments, and rendering
docs/media/             Curated README showcase assets
scripts/                Helper scripts for reruns and artifact merging
src/cartpole_bench/     Installable package and CLI
tests/                  Unit, regression, and smoke tests
```

Generated `artifacts*/` directories are local outputs and are intentionally not part of the tracked GitHub-facing repository contents.

## Limitations and Honesty Notes

- The benchmark is engineering-focused and optimized for reproducible comparison, not for formal scientific claims.
- Total control effort in the full task still includes the shared swing-up and capture phases, so it is not a pure stabilizer-only metric.
- `iLQR` and `MPC` are currently strongest in the saved suite benchmarks, but they remain much more expensive computationally than the classical baselines.
- The final public visuals intentionally suppress partial comparison figures. If a comparison output does not cover the selected controller set, it is skipped and removed.

## License

This project is released under the [MIT License](LICENSE).
