from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cartpole_bench import ARTIFACT_VERSION
from cartpole_bench.config import load_controller_config, load_suite, load_system_params
from cartpole_bench.simulation.recorder import write_manifest
from cartpole_bench.simulation.runner import simulate_trajectory
from cartpole_bench.types import ControllerConfig, ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.io import save_json
from cartpole_bench.utils.paths import artifact_roots
from cartpole_bench.utils.progress import NullProgressReporter, PhaseTimer, ProgressEvent, ProgressReporter
from cartpole_bench.utils.seed import make_rng


TUNING_PANEL = (
    "local_small_angle",
    "full_task_hanging",
    "measurement_noise",
    "impulse_disturbance",
    "parameter_mismatch",
)


def _unique_search_panel() -> list[ScenarioConfig]:
    scenarios = load_suite("nominal") + load_suite("stress")
    selected = []
    seen = set()
    for scenario_name in TUNING_PANEL:
        for scenario in scenarios:
            if scenario.name == scenario_name and scenario_name not in seen:
                selected.append(scenario)
                seen.add(scenario_name)
                break
    return selected


def _validation_panel() -> list[ScenarioConfig]:
    scenarios = load_suite("nominal") + load_suite("stress")
    return [scenario for scenario in scenarios if scenario.name in TUNING_PANEL]


def _numeric_spec(gains: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for key, value in gains.items():
        path = prefix + (key,)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            specs.append({"path": path, "kind": "scalar", "is_int": isinstance(value, int), "base": float(value)})
        elif isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
            for index, item in enumerate(value):
                specs.append(
                    {
                        "path": path + (str(index),),
                        "kind": "list",
                        "is_int": isinstance(item, int),
                        "base": float(item),
                    }
                )
    return specs


def _get_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = payload
    for part in path[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    leaf = path[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    else:
        current[leaf] = value


def _perturb_value(base: float, rng: np.random.Generator, iteration_fraction: float) -> float:
    if abs(base) < 1e-9:
        return float(rng.normal(0.0, 0.1 * max(0.25, 1.0 - 0.7 * iteration_fraction)))
    log_scale = 0.35 * max(0.25, 1.0 - 0.7 * iteration_fraction)
    return float(base * np.exp(rng.normal(0.0, log_scale)))


def _candidate_config(
    base: ControllerConfig,
    rng: np.random.Generator,
    iteration_fraction: float,
    anchor: ControllerConfig,
) -> ControllerConfig:
    specs = _numeric_spec(anchor.gains)
    candidate_payload = deepcopy(anchor.gains)
    selected = max(1, int(round(0.35 * len(specs))))
    indices = rng.choice(len(specs), size=selected, replace=False)

    for raw_index in np.atleast_1d(indices):
        spec = specs[int(raw_index)]
        base_value = float(_get_nested(anchor.gains, spec["path"]))
        candidate = _perturb_value(base_value, rng, iteration_fraction)
        if spec["is_int"]:
            candidate = max(1, int(round(candidate)))
        _set_nested(candidate_payload, spec["path"], candidate)

    # Keep dt absent unless the base config already defines it.
    if "dt" not in base.gains and "dt" in candidate_payload:
        candidate_payload.pop("dt", None)
    return replace(base, gains=candidate_payload)


def _evaluate_result_set(results: list[TrajectoryResult]) -> dict[str, float]:
    success_rate = float(np.mean([float(result.metrics.success) for result in results]))
    track_violation_rate = float(np.mean([float(result.track_violation) for result in results]))
    invalid_rate = float(np.mean([float(result.invalid) for result in results]))
    settling_values = np.asarray(
        [result.metrics.settling_time if result.metrics.settling_time is not None else np.nan for result in results],
        dtype=float,
    )
    overshoot_values = np.asarray([result.metrics.overshoot_deg for result in results], dtype=float)
    sse_values = np.asarray([result.metrics.steady_state_error_deg for result in results], dtype=float)
    effort_values = np.asarray([result.metrics.control_effort for result in results], dtype=float)
    metrics = {
        "success_rate": success_rate,
        "track_violation_rate": track_violation_rate,
        "invalid_rate": invalid_rate,
        "median_settling_time": float(np.nanmedian(settling_values)) if np.any(np.isfinite(settling_values)) else float("nan"),
        "median_overshoot_deg": float(np.median(overshoot_values)),
        "median_steady_state_error_deg": float(np.median(sse_values)),
        "median_control_effort": float(np.median(effort_values)),
    }
    metrics["objective"] = float(
        1000.0 * (1.0 - metrics["success_rate"])
        + 100.0 * metrics["track_violation_rate"]
        + 50.0 * metrics["invalid_rate"]
        + 25.0 * (metrics["median_settling_time"] if np.isfinite(metrics["median_settling_time"]) else 10.0)
        + 2.0 * metrics["median_overshoot_deg"]
        + 1.0 * metrics["median_steady_state_error_deg"]
        + 0.02 * metrics["median_control_effort"]
    )
    return metrics


def _run_panel(
    controller_key: str,
    controller_config: ControllerConfig,
    scenarios: list[ScenarioConfig],
    estimator_name: str,
) -> list[TrajectoryResult]:
    params = load_system_params()
    results = []
    for scenario in scenarios:
        result, _, _ = simulate_trajectory(
            scenario,
            controller_key,
            params,
            controller_override=controller_config,
            estimator_name=estimator_name,
        )
        results.append(result)
    return results


def tune_controller(
    controller_key: str,
    output_dir: Path,
    estimator_name: str = "none",
    budget: int = 40,
    seed: int = 0,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    progress = progress or NullProgressReporter()
    roots = artifact_roots(output_dir)
    tuning_dir = roots["base"] / "tuning" / controller_key / estimator_name
    tuning_dir.mkdir(parents=True, exist_ok=True)

    rng = make_rng(seed)
    base = load_controller_config(controller_key)
    search_panel = _unique_search_panel()
    validation_panel = _validation_panel()

    history_rows: list[dict[str, Any]] = []
    best = base
    best_stats = _evaluate_result_set(_run_panel(controller_key, base, search_panel, estimator_name))
    tuning_timer = PhaseTimer()
    total_iterations = max(1, int(budget))
    progress.emit(
        ProgressEvent(
            domain="tune",
            stage="start",
            current=0,
            total=total_iterations,
            context={
                "controller": controller_key,
                "estimator": estimator_name,
                "note": f"budget={int(budget)}",
            },
        )
    )

    for iteration in range(total_iterations):
        fraction = iteration / max(1, total_iterations - 1) if total_iterations > 1 else 0.0
        anchor = best if iteration >= max(1, total_iterations // 2) else base
        candidate = _candidate_config(base, rng, fraction, anchor)
        progress.emit(
            ProgressEvent(
                domain="tune",
                stage="iteration_start",
                current=iteration + 1,
                total=total_iterations,
                context={
                    "controller": controller_key,
                    "estimator": estimator_name,
                    "iteration": iteration,
                },
                elapsed_s=tuning_timer.elapsed(),
                eta_s=tuning_timer.eta(iteration, total_iterations),
            )
        )
        search_stats = _evaluate_result_set(_run_panel(controller_key, candidate, search_panel, estimator_name))
        improved = float(search_stats["objective"]) < float(best_stats["objective"])
        if improved:
            best = candidate
            best_stats = search_stats
        history_rows.append(
            {
                "iteration": iteration,
                "controller": controller_key,
                "estimator": estimator_name,
                "objective": float(search_stats["objective"]),
                "success_rate": float(search_stats["success_rate"]),
                "track_violation_rate": float(search_stats["track_violation_rate"]),
                "invalid_rate": float(search_stats["invalid_rate"]),
                "median_settling_time": float(search_stats["median_settling_time"]),
                "median_overshoot_deg": float(search_stats["median_overshoot_deg"]),
                "median_steady_state_error_deg": float(search_stats["median_steady_state_error_deg"]),
                "median_control_effort": float(search_stats["median_control_effort"]),
                "improved": improved,
                "candidate_gains": candidate.gains,
            }
        )
        progress.emit(
            ProgressEvent(
                domain="tune",
                stage="iteration_end",
                current=iteration + 1,
                total=total_iterations,
                context={
                    "controller": controller_key,
                    "estimator": estimator_name,
                    "iteration": iteration,
                    "note": f"obj={search_stats['objective']:.4f}, best={best_stats['objective']:.4f}",
                },
                elapsed_s=tuning_timer.elapsed(),
                eta_s=tuning_timer.eta(iteration + 1, total_iterations),
            )
        )

    progress.emit(
        ProgressEvent(
            domain="tune",
            stage="validation_start",
            current=total_iterations,
            total=total_iterations,
            context={"controller": controller_key, "estimator": estimator_name},
            elapsed_s=tuning_timer.elapsed(),
        )
    )
    validation_stats = _evaluate_result_set(_run_panel(controller_key, best, validation_panel, estimator_name))
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(tuning_dir / "tuning_history.csv", index=False)
    save_json(
        tuning_dir / "tuning_history.json",
        {"artifact_version": ARTIFACT_VERSION, "rows": history_frame.to_dict(orient="records")},
    )
    best_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "controller": controller_key,
        "estimator": estimator_name,
        "seed": seed,
        "budget": int(budget),
        "search_panel": [scenario.name for scenario in search_panel],
        "validation_panel": [scenario.name for scenario in validation_panel],
        "config": best.to_dict(),
        "validation_metrics": validation_stats,
    }
    save_json(tuning_dir / "best_config.json", best_payload)
    write_manifest(
        {"base": tuning_dir, "csv": tuning_dir, "json": tuning_dir, "figures": tuning_dir, "animations": tuning_dir, "tables": tuning_dir},
        "tuning",
        {
            "controller": controller_key,
            "estimator_name": estimator_name,
            "budget": int(budget),
            "seed": int(seed),
            "history_rows": len(history_rows),
            "best_objective": float(validation_stats["objective"]),
            "best_config": "best_config.json",
        },
    )
    progress.emit(
        ProgressEvent(
            domain="tune",
            stage="done",
            current=total_iterations,
            total=total_iterations,
            context={
                "controller": controller_key,
                "estimator": estimator_name,
                "note": f"best={validation_stats['objective']:.4f}",
            },
            elapsed_s=tuning_timer.elapsed(),
            eta_s=0.0,
        )
    )
    return {
        "best_config": best,
        "search_metrics": best_stats,
        "validation_metrics": validation_stats,
        "output_dir": tuning_dir,
    }
