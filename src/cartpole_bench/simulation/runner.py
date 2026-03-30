from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from cartpole_bench.config import load_estimator_config, load_suite, load_system_params
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step
from cartpole_bench.estimators import ExtendedKalmanFilter
from cartpole_bench.metrics.core import compute_run_metrics
from cartpole_bench.metrics.summary import write_metric_summaries
from cartpole_bench.simulation.recorder import save_run_artifacts, write_manifest
from cartpole_bench.simulation.scenario import CONTROLLER_KEYS, build_hybrid_controller, controller_label, resolve_plant_params
from cartpole_bench.types import ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.math import wrap_angle
from cartpole_bench.utils.paths import artifact_roots
from cartpole_bench.utils.progress import NullProgressReporter, PhaseTimer, ProgressEvent, ProgressReporter
from cartpole_bench.utils.seed import make_rng


def simulate_trajectory(
    scenario: ScenarioConfig,
    controller_key: str,
    nominal_params=None,
    controller_override=None,
    swingup_override=None,
    switch_override=None,
    estimator_name: str = "none",
    progress: ProgressReporter | None = None,
    run_index: int | None = None,
    run_total: int | None = None,
) -> tuple[TrajectoryResult, dict, object]:
    progress = NullProgressReporter() if progress is None else progress
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

    sim_cfg = scenario.simulation_config(plant_params, estimator_name=estimator_name)
    rng = make_rng(sim_cfg.seed)
    state = np.asarray(scenario.initial_state, dtype=float)
    state[2] = wrap_angle(state[2])
    observation = state.copy()
    estimate = observation.copy()
    estimator = None
    last_control = 0.0

    if estimator_name != "none":
        estimator_config = load_estimator_config(estimator_name)
        estimator = ExtendedKalmanFilter(estimator_config, plant_params, measurement_std=scenario.noise.to_array())
        estimator.reset(observation)
        estimate = estimator.step(observation, 0.0, 0.0)
        controller_payload["estimator"] = estimator_config.to_dict()
    else:
        controller_payload["estimator"] = {"name": "none", "gains": {}}

    time = [0.0]
    states = [state.copy()]
    observations = [observation.copy()]
    estimates = [estimate.copy()]
    controls = [0.0]
    disturbances = [0.0]
    modes = ["initial"]

    invalid = False
    track_violation = False
    timer = PhaseTimer()
    heartbeat_interval = max(25, sim_cfg.steps // 10)
    if sim_cfg.steps < 100:
        heartbeat_interval = max(10, sim_cfg.steps // 5)

    progress.emit(
        ProgressEvent(
            domain="trajectory",
            stage="start",
            current=run_index,
            total=run_total,
            elapsed_s=0.0,
            eta_s=None,
            context={
                "suite": scenario.suite_name,
                "scenario": scenario.name,
                "controller": controller.label,
                "estimator": estimator_name,
                "seed": scenario.seed,
                "note": f"steps={sim_cfg.steps}",
            },
        )
    )

    for step in range(sim_cfg.steps):
        t = step * sim_cfg.dt
        observed = state + rng.normal(0.0, scenario.noise.to_array())
        observed[2] = wrap_angle(observed[2])
        if estimator is not None:
            estimate = estimator.step(observed, last_control, sim_cfg.dt)
        else:
            estimate = observed.copy()
        disturbance = dynamics.disturbance_force(scenario.disturbance, t)
        try:
            control, mode = controller.compute_control(t, estimate, sim_cfg.dt)
            next_state = rk4_step(dynamics, state, control, sim_cfg.dt, disturbance)
        except FloatingPointError:
            invalid = True
            break

        time.append(t + sim_cfg.dt)
        states.append(next_state.copy())
        observations.append(observed.copy())
        estimates.append(estimate.copy())
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
        last_control = float(control)
        if (step + 1) % heartbeat_interval == 0 or step + 1 == sim_cfg.steps:
            progress.emit(
                ProgressEvent(
                    domain="trajectory",
                    stage="heartbeat",
                    current=run_index,
                    total=run_total,
                    elapsed_s=timer.elapsed(),
                    eta_s=(timer.elapsed() * (sim_cfg.steps - (step + 1)) / max(1, step + 1)),
                    context={
                        "suite": scenario.suite_name,
                        "scenario": scenario.name,
                        "controller": controller.label,
                        "estimator": estimator_name,
                        "seed": scenario.seed,
                        "note": f"step={step + 1}/{sim_cfg.steps} t={t + sim_cfg.dt:.2f}s",
                    },
                )
            )

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
        estimator_name=estimator_name,
        scenario_name=scenario.name,
        suite_name=scenario.suite_name,
        seed=scenario.seed,
        time=np.asarray(time, dtype=float),
        states=np.asarray(states, dtype=float),
        observations=np.asarray(observations, dtype=float),
        estimates=np.asarray(estimates, dtype=float),
        controls=np.asarray(controls, dtype=float),
        disturbances=np.asarray(disturbances, dtype=float),
        modes=modes,
        metrics=metrics,
        diagnosis=diagnosis,
        invalid=invalid,
        track_violation=track_violation,
    )
    controller_payload["controller_debug"] = controller.stabilizer.debug_summary()
    progress.emit(
        ProgressEvent(
            domain="trajectory",
            stage="done",
            current=run_index,
            total=run_total,
            elapsed_s=timer.elapsed(),
            eta_s=0.0,
            context={
                "suite": scenario.suite_name,
                "scenario": scenario.name,
                "controller": controller.label,
                "estimator": estimator_name,
                "seed": scenario.seed,
                "note": f"success={metrics.success} invalid={invalid} track_violation={track_violation}",
            },
        )
    )
    return result, controller_payload, plant_params


def run_trajectory(
    scenario: ScenarioConfig,
    controller_key: str,
    output_dir: Path,
    cwd: Path,
    estimator_name: str = "none",
    progress: ProgressReporter | None = None,
    run_index: int | None = None,
    run_total: int | None = None,
) -> tuple[TrajectoryResult, dict[str, str]]:
    nominal_params = load_system_params()
    result, controller_payload, plant_params = simulate_trajectory(
        scenario,
        controller_key,
        nominal_params,
        estimator_name=estimator_name,
        progress=progress,
        run_index=run_index,
        run_total=run_total,
    )
    roots = artifact_roots(output_dir)
    saved_paths = save_run_artifacts(result, scenario, plant_params, controller_payload, roots, cwd)
    return result, saved_paths


def run_suite(
    suite_name: str,
    output_dir: Path,
    cwd: Path | None = None,
    controllers: tuple[str, ...] = CONTROLLER_KEYS,
    estimator_name: str = "none",
    progress: ProgressReporter | None = None,
) -> list[TrajectoryResult]:
    progress = NullProgressReporter() if progress is None else progress
    cwd = Path.cwd() if cwd is None else cwd
    roots = artifact_roots(output_dir)
    scenarios = load_suite(suite_name)
    results: list[TrajectoryResult] = []
    manifest_runs = []
    total_runs = len(scenarios) * len(controllers)
    suite_timer = PhaseTimer()
    progress.emit(
        ProgressEvent(
            domain="suite",
            stage="start",
            current=0,
            total=total_runs,
            elapsed_s=0.0,
            eta_s=None,
            context={"suite": suite_name, "estimator": estimator_name, "note": f"controllers={','.join(controllers)}"},
        )
    )

    run_counter = 0
    for scenario in scenarios:
        for controller_key in controllers:
            run_counter += 1
            progress.emit(
                ProgressEvent(
                    domain="suite",
                    stage="run_start",
                    current=run_counter,
                    total=total_runs,
                    elapsed_s=suite_timer.elapsed(),
                    eta_s=suite_timer.eta(run_counter - 1, total_runs),
                    context={
                        "suite": suite_name,
                        "scenario": scenario.name,
                        "controller": controller_label(controller_key),
                        "estimator": estimator_name,
                        "seed": scenario.seed,
                    },
                )
            )
            run_started = perf_counter()
            result, saved_paths = run_trajectory(
                scenario,
                controller_key,
                output_dir,
                cwd,
                estimator_name=estimator_name,
                progress=progress,
                run_index=run_counter,
                run_total=total_runs,
            )
            results.append(result)
            manifest_runs.append(
                {
                    "suite": suite_name,
                    "scenario": scenario.name,
                    "controller": result.controller_name,
                    "estimator": result.estimator_name,
                    "seed": result.seed,
                    **saved_paths,
                }
            )
            progress.emit(
                ProgressEvent(
                    domain="suite",
                    stage="run_end",
                    current=run_counter,
                    total=total_runs,
                    elapsed_s=suite_timer.elapsed(),
                    eta_s=suite_timer.eta(run_counter, total_runs),
                    context={
                        "suite": suite_name,
                        "scenario": scenario.name,
                        "controller": result.controller_name,
                        "estimator": estimator_name,
                        "seed": result.seed,
                        "note": f"success={result.metrics.success} wall={perf_counter() - run_started:.1f}s",
                    },
                )
            )

    write_manifest(
        roots,
        f"{suite_name}_{estimator_name}",
        {
            "suite": suite_name,
            "estimator_name": estimator_name,
            "runs": manifest_runs,
        },
    )
    write_metric_summaries(roots["tables"], results)
    progress.emit(
        ProgressEvent(
            domain="suite",
            stage="done",
            current=total_runs,
            total=total_runs,
            elapsed_s=suite_timer.elapsed(),
            eta_s=0.0,
            context={"suite": suite_name, "estimator": estimator_name, "note": f"runs={len(results)}"},
        )
    )
    return results
