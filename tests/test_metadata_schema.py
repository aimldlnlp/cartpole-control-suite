from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_system_params
from cartpole_bench.metrics.summary import write_metric_summaries
from cartpole_bench.simulation.recorder import save_run_artifacts
from cartpole_bench.types import RunDiagnosis, RunMetrics, TrajectoryResult
from cartpole_bench.types import DisturbanceConfig, NoiseConfig, ScenarioConfig
from cartpole_bench.utils.io import load_json
from cartpole_bench.utils.paths import artifact_roots


def test_trajectory_result_serializes_diagnosis() -> None:
    result = TrajectoryResult(
        controller_name="LQR",
        scenario_name="local_small_angle",
        suite_name="nominal",
        seed=1,
        time=np.array([0.0, 0.1]),
        states=np.zeros((2, 4)),
        observations=np.zeros((2, 4)),
        controls=np.zeros(2),
        disturbances=np.zeros(2),
        modes=["balance", "balance"],
        metrics=RunMetrics(
            settling_time=1.2,
            overshoot_deg=2.0,
            steady_state_error_deg=0.1,
            control_effort=1.0,
            success=True,
            max_abs_force=2.0,
            final_abs_theta_deg=0.1,
            switch_time=0.0,
        ),
        diagnosis=RunDiagnosis(
            failure_reason=None,
            first_balance_time=0.0,
            balance_fraction=1.0,
            min_abs_theta_deg=0.1,
            time_of_min_abs_theta=0.1,
            max_abs_x=0.05,
            max_abs_theta_dot=0.2,
        ),
    )
    payload = result.to_dict()
    assert "diagnosis" in payload
    assert payload["diagnosis"]["balance_fraction"] == 1.0


def test_saved_run_metadata_includes_artifact_version(tmp_path) -> None:
    result = TrajectoryResult(
        controller_name="LQR",
        scenario_name="local_small_angle",
        suite_name="nominal",
        seed=7,
        time=np.array([0.0, 0.1]),
        states=np.zeros((2, 4)),
        observations=np.zeros((2, 4)),
        controls=np.zeros(2),
        disturbances=np.zeros(2),
        modes=["balance", "balance"],
        metrics=RunMetrics(
            settling_time=0.1,
            overshoot_deg=1.0,
            steady_state_error_deg=0.05,
            control_effort=0.2,
            success=True,
            max_abs_force=0.0,
            final_abs_theta_deg=0.0,
            switch_time=0.0,
        ),
        diagnosis=RunDiagnosis(
            failure_reason=None,
            first_balance_time=0.0,
            balance_fraction=1.0,
            min_abs_theta_deg=0.0,
            time_of_min_abs_theta=0.1,
            max_abs_x=0.0,
            max_abs_theta_dot=0.0,
        ),
    )
    scenario = ScenarioConfig(
        name="local_small_angle",
        suite_name="nominal",
        horizon=1.0,
        dt=0.1,
        initial_state=(0.0, 0.0, 0.0, 0.0),
        seed=7,
        noise=NoiseConfig(),
        disturbance=DisturbanceConfig(),
    )
    roots = artifact_roots(tmp_path / "artifacts_v2")
    paths = save_run_artifacts(
        result,
        scenario,
        load_system_params(),
        {
            "controller": {"name": "lqr"},
            "swingup": {"name": "swingup"},
            "switch": {"enter_angle_deg": 12.0},
        },
        roots,
        tmp_path,
    )
    payload = load_json(roots["base"] / paths["json_path"])
    assert payload["artifact_version"] == "v2"


def test_metric_summary_json_includes_artifact_version_and_required_columns(tmp_path) -> None:
    result = TrajectoryResult(
        controller_name="LQR",
        scenario_name="local_small_angle",
        suite_name="nominal",
        seed=1,
        time=np.array([0.0, 0.1]),
        states=np.zeros((2, 4)),
        observations=np.zeros((2, 4)),
        controls=np.zeros(2),
        disturbances=np.zeros(2),
        modes=["balance", "balance"],
        metrics=RunMetrics(
            settling_time=0.1,
            overshoot_deg=1.0,
            steady_state_error_deg=0.05,
            control_effort=0.2,
            success=True,
            max_abs_force=0.0,
            final_abs_theta_deg=0.0,
            switch_time=0.0,
        ),
        diagnosis=RunDiagnosis(
            failure_reason=None,
            first_balance_time=0.0,
            balance_fraction=1.0,
            min_abs_theta_deg=0.0,
            time_of_min_abs_theta=0.1,
            max_abs_x=0.0,
            max_abs_theta_dot=0.0,
        ),
    )
    frame = write_metric_summaries(tmp_path, [result])
    payload = load_json(tmp_path / "metric_summary.json")

    assert payload["artifact_version"] == "v2"
    assert {
        "runs",
        "success_count",
        "success_rate",
        "handoff_rate",
        "settling_time_median",
        "overshoot_deg_median",
        "steady_state_error_deg_median",
        "control_effort_median",
        "first_balance_time_median",
        "min_abs_theta_deg_median",
        "balance_fraction_median",
        "invalid_rate",
        "track_violation_rate",
    }.issubset(set(frame.columns))
