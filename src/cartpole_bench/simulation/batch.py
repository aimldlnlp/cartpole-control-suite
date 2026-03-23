from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cartpole_bench import ARTIFACT_VERSION
from cartpole_bench.config import load_monte_carlo_config, load_system_params
from cartpole_bench.metrics.summary import write_monte_carlo_summary
from cartpole_bench.simulation.recorder import write_manifest
from cartpole_bench.simulation.runner import simulate_trajectory
from cartpole_bench.simulation.scenario import CONTROLLER_KEYS
from cartpole_bench.types import BatchSummary, DisturbanceConfig, NoiseConfig, ScenarioConfig
from cartpole_bench.utils.io import save_json
from cartpole_bench.utils.paths import artifact_roots
from cartpole_bench.utils.seed import make_rng

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def resolve_execution_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "auto":
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unsupported device selection: {requested}")


def _sample_scenarios(samples: int) -> list[tuple[ScenarioConfig, dict[str, float]]]:
    cfg = load_monte_carlo_config()
    rng = make_rng(int(cfg["seed"]))
    base_params = load_system_params()
    scenarios: list[tuple[ScenarioConfig, dict[str, float]]] = []

    for index in range(samples):
        angle_deg = float(rng.uniform(*cfg["initial_angle_range_deg"]))
        angle_deg *= -1.0 if rng.random() < 0.5 else 1.0
        theta_std_deg = float(rng.uniform(*cfg["noise_angle_std_deg"]))
        disturbance = float(rng.uniform(*cfg["disturbance_magnitude"]))
        disturbance *= -1.0 if rng.random() < 0.5 else 1.0
        cart_friction = float(rng.uniform(*cfg["cart_friction"]))
        pivot_damping = float(rng.uniform(*cfg["pivot_damping"]))
        mass_scale = float(rng.uniform(*cfg["parameter_scale"]))
        pendulum_scale = float(rng.uniform(*cfg["parameter_scale"]))
        length_scale = float(rng.uniform(*cfg["parameter_scale"]))

        scenario = ScenarioConfig(
            name=f"sample_{index:05d}",
            suite_name="monte_carlo",
            horizon=float(cfg["horizon"]),
            dt=float(cfg["dt"]),
            initial_state=(
                0.0,
                0.0,
                np.deg2rad(angle_deg),
                float(rng.uniform(*cfg["initial_rate_range"])),
            ),
            seed=int(cfg["seed"]) + index,
            noise=NoiseConfig(
                state_std=(
                    float(cfg["noise_state_std"][0]),
                    float(cfg["noise_state_std"][1]),
                    np.deg2rad(theta_std_deg),
                    float(cfg["noise_state_std"][3]),
                )
            ),
            disturbance=DisturbanceConfig(
                kind="pulse" if abs(disturbance) > 1e-9 else "none",
                magnitude=disturbance,
                start_time=2.5,
                duration=0.2,
            ),
            plant_overrides={
                "M": base_params.M * mass_scale,
                "m": base_params.m * pendulum_scale,
                "l": base_params.l * length_scale,
                "cart_friction": cart_friction,
                "pivot_damping": pivot_damping,
            },
        )
        scenarios.append(
            (
                scenario,
                {
                    "initial_angle_deg": angle_deg,
                    "initial_rate": scenario.initial_state[3],
                    "noise_theta_std_deg": theta_std_deg,
                    "disturbance_force": disturbance,
                    "cart_friction": cart_friction,
                    "pivot_damping": pivot_damping,
                    "M": base_params.M * mass_scale,
                    "m": base_params.m * pendulum_scale,
                    "l": base_params.l * length_scale,
                },
            )
        )
    return scenarios


def run_monte_carlo(output_dir: Path, requested_device: str = "cpu", samples: int | None = None) -> list[BatchSummary]:
    roots = artifact_roots(output_dir)
    cfg = load_monte_carlo_config()
    sample_count = int(cfg["samples"]) if samples is None else int(samples)
    execution_device = resolve_execution_device(requested_device)
    scenarios = _sample_scenarios(sample_count)
    nominal_params = load_system_params()

    sample_rows: list[dict[str, float | str | bool | None]] = []
    summaries: list[BatchSummary] = []

    for controller_key in CONTROLLER_KEYS:
        controller_rows = []
        for index, (scenario, sample_meta) in enumerate(scenarios):
            result, _, _ = simulate_trajectory(scenario, controller_key, nominal_params)
            row = {
                "sample_id": index,
                "controller": result.controller_name,
                "device_requested": requested_device,
                "device_used": execution_device,
                "success": result.metrics.success,
                "settling_time": result.metrics.settling_time,
                "steady_state_error_deg": result.metrics.steady_state_error_deg,
                "control_effort": result.metrics.control_effort,
                "invalid": result.invalid,
                "track_violation": result.track_violation,
                "failure_reason": result.diagnosis.failure_reason,
                "first_balance_time": result.diagnosis.first_balance_time,
                **sample_meta,
            }
            sample_rows.append(row)
            controller_rows.append(row)

        success_values = np.asarray([float(row["success"]) for row in controller_rows], dtype=float)
        invalid_values = np.asarray([float(row["invalid"]) for row in controller_rows], dtype=float)
        settling_values = np.asarray(
            [
                float(row["settling_time"]) if row["settling_time"] is not None else np.nan
                for row in controller_rows
            ],
            dtype=float,
        )
        effort_values = np.asarray([float(row["control_effort"]) for row in controller_rows], dtype=float)
        sse_values = np.asarray([float(row["steady_state_error_deg"]) for row in controller_rows], dtype=float)
        summaries.append(
            BatchSummary(
                controller_name=str(controller_rows[0]["controller"]),
                samples=sample_count,
                success_rate=float(np.mean(success_values)),
                success_count=int(np.sum(success_values)),
                median_settling_time=(
                    float(np.nanmedian(settling_values)) if np.any(np.isfinite(settling_values)) else None
                ),
                median_control_effort=float(np.median(effort_values)),
                median_steady_state_error_deg=float(np.median(sse_values)),
                invalid_rate=float(np.mean(invalid_values)),
            )
        )

    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(roots["tables"] / "monte_carlo_samples.csv", index=False)
    save_json(
        roots["tables"] / "monte_carlo_samples.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "rows": sample_frame.to_dict(orient="records"),
        },
    )
    write_monte_carlo_summary(roots["tables"], summaries)
    write_manifest(
        roots,
        "monte_carlo",
        {
            "suite": "monte_carlo",
            "device_requested": requested_device,
            "device_used": execution_device,
            "samples": sample_count,
            "summary_rows": [summary.to_dict() for summary in summaries],
            "sample_table": str((roots["tables"] / "monte_carlo_samples.csv").relative_to(roots["base"])),
        },
    )
    return summaries
