from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cartpole_bench.animations.render import (
    _best_complete_seed_for_scenario,
    _cleanup_stale_animation_outputs,
    _eligible_animation_keys,
)
from cartpole_bench.plots.figures import _cleanup_stale_figure_outputs, _eligible_figure_keys


ALL_CONTROLLER_KEYS = ("lqr", "pfl", "smc", "ilqr", "mpc")
ALL_CONTROLLER_LABELS = [
    "LQR",
    "Feedback Linearization (PFL)",
    "Sliding Mode Control (SMC)",
    "Iterative LQR (iLQR)",
    "Model Predictive Control (MPC)",
]


def _make_run(
    controller_name: str,
    scenario_name: str,
    suite_name: str,
    seed: int,
    *,
    estimator_name: str = "ekf",
    disturbance_active: bool = False,
    first_balance_time: float = 0.5,
    json_path: str | None = None,
) -> dict:
    time = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    disturbance = np.array([0.0, 0.0, 0.6, 0.0], dtype=float) if disturbance_active else np.zeros_like(time)
    frame = pd.DataFrame(
        {
            "t": time,
            "x": np.linspace(0.0, 0.1, len(time)),
            "x_dot": np.zeros(len(time)),
            "theta": np.deg2rad(np.array([10.0, 5.0, 1.0, 0.2])),
            "theta_dot": np.array([0.6, 0.3, 0.1, 0.0]),
            "theta_deg": np.array([10.0, 5.0, 1.0, 0.2]),
            "u": np.array([1.0, 0.5, 0.2, 0.0]),
            "disturbance": disturbance,
            "energy_gap": np.array([1.0, 0.5, 0.1, 0.0]),
            "track_margin": np.array([0.9, 0.88, 0.87, 0.86]),
            "force_saturated": np.zeros(len(time)),
            "mode": ["capture_assist", "balance", "balance", "balance"],
        }
    )
    return {
        "metadata": {
            "controller_name": controller_name,
            "scenario_name": scenario_name,
            "suite_name": suite_name,
            "seed": seed,
            "estimator_name": estimator_name,
            "metrics": {
                "success": True,
                "settling_time": 1.0,
                "final_abs_theta_deg": 0.2,
            },
            "diagnosis": {
                "first_balance_time": first_balance_time,
                "max_abs_x": 0.1,
            },
            "plant_params": {
                "track_limit": 1.0,
            },
        },
        "summary": {
            "json_path": json_path or f"json/{controller_name.lower().replace(' ', '_')}_{scenario_name}_{seed}.json",
        },
        "frame": frame,
    }


def test_eligible_figure_keys_drop_partial_monte_carlo_outputs() -> None:
    runs = []
    for controller in ALL_CONTROLLER_LABELS:
        runs.append(_make_run(controller, "local_small_angle", "nominal", 7))
        runs.append(_make_run(controller, "full_task_hanging", "nominal", 11))
        runs.append(_make_run(controller, "impulse_disturbance", "stress", 22, disturbance_active=True))

    summary = pd.DataFrame(
        [
            {"suite": "nominal", "scenario": "local_small_angle", "controller": controller, "estimator": "ekf"}
            for controller in ALL_CONTROLLER_LABELS
        ]
        + [
            {"suite": "nominal", "scenario": "full_task_hanging", "controller": controller, "estimator": "ekf"}
            for controller in ALL_CONTROLLER_LABELS
        ]
        + [
            {"suite": "stress", "scenario": "impulse_disturbance", "controller": controller, "estimator": "ekf"}
            for controller in ALL_CONTROLLER_LABELS
        ]
    )
    monte_summary = pd.DataFrame(
        {
            "controller_name": ALL_CONTROLLER_LABELS[:3],
            "estimator_name": ["ekf", "ekf", "ekf"],
            "success_rate": [0.8, 0.7, 0.6],
        }
    )
    monte_samples = pd.DataFrame(
        {
            "controller": ALL_CONTROLLER_LABELS[:3],
            "estimator": ["ekf", "ekf", "ekf"],
            "success": [1, 1, 0],
            "settling_time": [1.0, 1.2, 1.5],
            "steady_state_error_deg": [0.1, 0.2, 0.3],
            "noise_theta_std_deg": [0.05, 0.05, 0.05],
            "disturbance_force": [0.2, 0.2, 0.2],
        }
    )

    eligible = _eligible_figure_keys(
        runs,
        summary,
        monte_summary,
        monte_samples,
        ALL_CONTROLLER_KEYS,
        "ekf",
        include_supplements=True,
    )

    assert "nominal_local_response" in eligible
    assert "full_task_handoff" in eligible
    assert "stress_comparison" in eligible
    assert "metric_summary" in eligible
    assert "cartpole_schematic" in eligible
    assert "handoff_focus" in eligible
    assert "constraint_usage" in eligible
    assert "energy_phase" in eligible
    assert "monte_carlo_overview" not in eligible
    assert "robustness_map" not in eligible


def test_cleanup_stale_figure_outputs_removes_non_compliant_paths(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    supplemental_dir = figures_dir / "supplemental"
    supplemental_dir.mkdir(parents=True)
    metric_summary = figures_dir / "metric_summary.png"
    monte_overview = figures_dir / "monte_carlo_overview.png"
    robustness_map = supplemental_dir / "robustness_map.png"
    for path in (metric_summary, monte_overview, robustness_map):
        path.write_text("placeholder")

    stale = _cleanup_stale_figure_outputs(tmp_path, ["metric_summary"])

    assert metric_summary.exists()
    assert not monte_overview.exists()
    assert not robustness_map.exists()
    assert monte_overview in stale
    assert robustness_map in stale


def test_best_complete_seed_for_scenario_prefers_complete_seed() -> None:
    runs = []
    for controller in ALL_CONTROLLER_LABELS[:-1]:
        runs.append(_make_run(controller, "full_task_hanging", "nominal", 1))
    for controller in ALL_CONTROLLER_LABELS:
        runs.append(_make_run(controller, "full_task_hanging", "nominal", 2))

    selected_seed = _best_complete_seed_for_scenario(runs, "full_task_hanging", ALL_CONTROLLER_LABELS)

    assert selected_seed == 2


def test_eligible_animation_keys_require_complete_coverage_for_comparisons() -> None:
    runs = []
    for controller in ALL_CONTROLLER_LABELS:
        runs.append(_make_run(controller, "full_task_hanging", "nominal", 2))
    for controller in ALL_CONTROLLER_LABELS[:-1]:
        runs.append(_make_run(controller, "impulse_disturbance", "stress", 3, disturbance_active=True))

    eligible = _eligible_animation_keys(runs, ALL_CONTROLLER_KEYS, include_supplements=True)

    for controller in ALL_CONTROLLER_LABELS:
        assert controller in eligible
    assert "nominal_comparison" in eligible
    assert "handoff_focus" in eligible
    assert "stress_comparison" not in eligible
    assert "disturbance_focus" not in eligible


def test_cleanup_stale_animation_outputs_prunes_metadata(tmp_path: Path) -> None:
    animations_dir = tmp_path / "animations"
    json_dir = tmp_path / "json"
    animations_dir.mkdir()
    json_dir.mkdir()

    stale_path = animations_dir / "side_by_side_nominal_comparison.gif"
    keep_path = animations_dir / "lqr_full_task_nominal.gif"
    stale_path.write_text("old")
    keep_path.write_text("keep")

    metadata_path = json_dir / "run.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_version": "v3",
                "render_paths": {
                    "comparison_gif": "animations/side_by_side_nominal_comparison.gif",
                    "single_gif": "animations/lqr_full_task_nominal.gif",
                },
            }
        )
    )
    runs = [
        _make_run(
            "LQR",
            "full_task_hanging",
            "nominal",
            7,
            json_path="json/run.json",
        )
    ]

    stale = _cleanup_stale_animation_outputs(tmp_path, runs, {stale_path})

    updated = json.loads(metadata_path.read_text())
    assert stale_path in stale
    assert not stale_path.exists()
    assert keep_path.exists()
    assert "comparison_gif" not in updated["render_paths"]
    assert updated["render_paths"]["single_gif"] == "animations/lqr_full_task_nominal.gif"
