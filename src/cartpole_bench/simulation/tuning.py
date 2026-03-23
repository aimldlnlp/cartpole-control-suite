from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from cartpole_bench.config import load_controller_config, load_suite, load_system_params
from cartpole_bench.simulation.runner import simulate_trajectory
from cartpole_bench.types import ControllerConfig, ScenarioConfig, TrajectoryResult


def _local_nominal_scenarios() -> list[ScenarioConfig]:
    return [scenario for scenario in load_suite("nominal") if scenario.name == "local_small_angle"]


def _score_result(result: TrajectoryResult, scenario: ScenarioConfig) -> float:
    settling = result.metrics.settling_time if result.metrics.settling_time is not None else scenario.horizon
    return (
        (0.0 if result.metrics.success else 1000.0)
        + (220.0 if result.track_violation else 0.0)
        + (120.0 if result.invalid else 0.0)
        + 50.0 * result.diagnosis.max_abs_x
        + 8.0 * result.metrics.final_abs_theta_deg
        + 5.0 * settling
        + 0.12 * result.metrics.control_effort
    )


def evaluate_controller_config(controller_key: str, config: ControllerConfig) -> dict[str, float]:
    params = load_system_params()
    scenarios = _local_nominal_scenarios()
    scores = []
    success = []
    settling = []
    for scenario in scenarios:
        result, _, _ = simulate_trajectory(scenario, controller_key, params, controller_override=config)
        scores.append(_score_result(result, scenario))
        success.append(float(result.metrics.success))
        settling.append(result.metrics.settling_time if result.metrics.settling_time is not None else scenario.horizon)
    return {
        "score": float(np.mean(scores)),
        "success_rate": float(np.mean(success)),
        "median_settling_time": float(np.median(settling)),
    }


def tune_pfl() -> tuple[ControllerConfig, dict[str, float]]:
    base = load_controller_config("pfl")
    best = base
    best_stats = evaluate_controller_config("pfl", base)
    for k_theta, k_theta_dot, k_x, k_x_dot, accel_limit, epsilon in product(
        [14.0, 18.0, 22.0],
        [6.0, 8.5, 11.0],
        [1.0, 1.6, 2.2],
        [2.0, 2.8, 3.6],
        [4.0, 6.0, 8.0],
        [0.05, 0.08, 0.12],
    ):
        gains = dict(base.gains)
        gains["k_theta"] = k_theta
        gains["k_theta_dot"] = k_theta_dot
        gains["k_x"] = k_x
        gains["k_x_dot"] = k_x_dot
        gains["desired_accel_limit"] = accel_limit
        gains["effective_mass_epsilon"] = epsilon
        candidate = replace(base, gains=gains)
        stats = evaluate_controller_config("pfl", candidate)
        if float(stats["score"]) < float(best_stats["score"]):
            best = candidate
            best_stats = stats
    return best, best_stats


def tune_smc() -> tuple[ControllerConfig, dict[str, float]]:
    base = load_controller_config("smc")
    best = base
    best_stats = evaluate_controller_config("smc", base)
    for surface_x, surface_v, reach, boundary, blend, center_x, center_v, settle_angle, trim_limit in product(
        [3.8, 4.6, 5.2],
        [2.8, 3.1, 3.6],
        [1.8, 2.2, 2.6],
        [0.20, 0.25, 0.35],
        [0.9, 1.0, 1.1],
        [2.2, 2.8, 3.4],
        [3.0, 3.6, 4.2],
        [5.0, 6.0, 7.0],
        [1.0, 1.5, 2.0],
    ):
        gains = dict(base.gains)
        gains["sliding_surface"] = [surface_x, surface_v, 4.6, 1.5]
        gains["boundary_layer"] = boundary
        gains["lqr_blend_weight"] = blend
        gains["reach_gain"] = reach
        gains["cart_center_gain"] = center_x
        gains["cart_velocity_gain"] = center_v
        gains["settle_angle_deg"] = settle_angle
        gains["settle_rate"] = 0.75
        gains["settle_cart_gain"] = center_x * 0.8
        gains["settle_cart_velocity_gain"] = center_v * 0.95
        gains["settle_trim_limit"] = trim_limit
        gains["robust_decay_position_scale"] = 0.18
        gains["robust_decay_velocity_scale"] = 0.3
        gains["robust_decay_angle_deg"] = 5.0
        gains["robust_decay_rate_scale"] = 0.4
        candidate = replace(base, gains=gains)
        stats = evaluate_controller_config("smc", candidate)
        if float(stats["score"]) < float(best_stats["score"]):
            best = candidate
            best_stats = stats
    return best, best_stats
