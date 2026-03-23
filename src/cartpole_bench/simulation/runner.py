from __future__ import annotations

from pathlib import Path

import numpy as np

from cartpole_bench.config import load_suite, load_system_params
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step
from cartpole_bench.metrics.core import compute_run_metrics
from cartpole_bench.metrics.summary import write_metric_summaries
from cartpole_bench.simulation.recorder import save_run_artifacts, write_manifest
from cartpole_bench.simulation.scenario import CONTROLLER_KEYS, build_hybrid_controller, resolve_plant_params
from cartpole_bench.types import ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.math import wrap_angle
from cartpole_bench.utils.paths import artifact_roots
from cartpole_bench.utils.seed import make_rng


def simulate_trajectory(
    scenario: ScenarioConfig,
    controller_key: str,
    nominal_params=None,
    controller_override=None,
    swingup_override=None,
    switch_override=None,
) -> tuple[TrajectoryResult, dict, object]:
    nominal_params = load_system_params() if nominal_params is None else nominal_params
    plant_params = resolve_plant_params(nominal_params, scenario)
    dynamics = CartPoleDynamics(plant_params)
    controller, controller_payload = build_hybrid_controller(
        controller_key,
        nominal_params,
        controller_override=controller_override,
        swingup_override=swingup_override,
        switch_override=switch_override,
    )
    controller.reset()

    sim_cfg = scenario.simulation_config(plant_params)
    rng = make_rng(sim_cfg.seed)
    state = np.asarray(scenario.initial_state, dtype=float)
    state[2] = wrap_angle(state[2])

    time = [0.0]
    states = [state.copy()]
    observations = [state.copy()]
    controls = [0.0]
    disturbances = [0.0]
    modes = ["initial"]

    invalid = False
    track_violation = False

    for step in range(sim_cfg.steps):
        t = step * sim_cfg.dt
        observed = state + rng.normal(0.0, scenario.noise.to_array())
        observed[2] = wrap_angle(observed[2])
        disturbance = dynamics.disturbance_force(scenario.disturbance, t)
        try:
            control, mode = controller.compute_control(t, observed, sim_cfg.dt)
            next_state = rk4_step(dynamics, state, control, sim_cfg.dt, disturbance)
        except FloatingPointError:
            invalid = True
            break

        time.append(t + sim_cfg.dt)
        states.append(next_state.copy())
        observations.append(observed.copy())
        controls.append(control)
        disturbances.append(disturbance)
        modes.append(mode)

        if not np.all(np.isfinite(next_state)) or not np.isfinite(control):
            invalid = True
            break
        if abs(next_state[0]) > plant_params.track_limit:
            track_violation = True
            state = next_state
            break
        state = next_state

    metrics, diagnosis = compute_run_metrics(
        np.asarray(time, dtype=float),
        np.asarray(states, dtype=float),
        np.asarray(controls, dtype=float),
        modes,
        plant_params.track_limit,
        controller.switch_time,
        invalid,
        track_violation,
    )

    result = TrajectoryResult(
        controller_name=controller.label,
        scenario_name=scenario.name,
        suite_name=scenario.suite_name,
        seed=scenario.seed,
        time=np.asarray(time, dtype=float),
        states=np.asarray(states, dtype=float),
        observations=np.asarray(observations, dtype=float),
        controls=np.asarray(controls, dtype=float),
        disturbances=np.asarray(disturbances, dtype=float),
        modes=modes,
        metrics=metrics,
        diagnosis=diagnosis,
        invalid=invalid,
        track_violation=track_violation,
    )
    return result, controller_payload, plant_params


def run_trajectory(
    scenario: ScenarioConfig,
    controller_key: str,
    output_dir: Path,
    cwd: Path,
) -> tuple[TrajectoryResult, dict[str, str]]:
    nominal_params = load_system_params()
    result, controller_payload, plant_params = simulate_trajectory(scenario, controller_key, nominal_params)
    roots = artifact_roots(output_dir)
    saved_paths = save_run_artifacts(result, scenario, plant_params, controller_payload, roots, cwd)
    return result, saved_paths


def run_suite(suite_name: str, output_dir: Path, cwd: Path | None = None) -> list[TrajectoryResult]:
    cwd = Path.cwd() if cwd is None else cwd
    roots = artifact_roots(output_dir)
    scenarios = load_suite(suite_name)
    results: list[TrajectoryResult] = []
    manifest_runs = []

    for scenario in scenarios:
        for controller_key in CONTROLLER_KEYS:
            result, saved_paths = run_trajectory(scenario, controller_key, output_dir, cwd)
            results.append(result)
            manifest_runs.append(
                {
                    "suite": suite_name,
                    "scenario": scenario.name,
                    "controller": result.controller_name,
                    "seed": result.seed,
                    **saved_paths,
                }
            )

    write_manifest(
        roots,
        suite_name,
        {
            "suite": suite_name,
            "runs": manifest_runs,
        },
    )
    write_metric_summaries(roots["tables"], results)
    return results
