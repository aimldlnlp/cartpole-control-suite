from __future__ import annotations

import warnings

import numpy as np

from cartpole_bench.metrics.summary import write_metric_summaries
from cartpole_bench.types import RunDiagnosis, RunMetrics, TrajectoryResult
from cartpole_bench.utils.io import load_json


def test_metric_summaries_handle_all_failure_runs_without_nan_warning(tmp_path) -> None:
    result = TrajectoryResult(
        controller_name="Iterative LQR (iLQR)",
        estimator_name="none",
        scenario_name="local_small_angle",
        suite_name="nominal",
        seed=7,
        time=np.array([0.0, 0.1]),
        states=np.zeros((2, 4)),
        observations=np.zeros((2, 4)),
        estimates=np.zeros((2, 4)),
        controls=np.zeros(2),
        disturbances=np.zeros(2),
        modes=["balance", "balance"],
        metrics=RunMetrics(
            settling_time=None,
            overshoot_deg=4.0,
            steady_state_error_deg=3.5,
            control_effort=1.2,
            success=False,
            max_abs_force=2.0,
            final_abs_theta_deg=3.5,
            switch_time=None,
        ),
        diagnosis=RunDiagnosis(
            failure_reason="did_not_meet_settling_window",
            first_balance_time=None,
            balance_fraction=0.0,
            min_abs_theta_deg=3.5,
            time_of_min_abs_theta=None,
            max_abs_x=0.2,
            max_abs_theta_dot=0.6,
        ),
        invalid=False,
        track_violation=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        frame = write_metric_summaries(tmp_path, [result])

    assert not caught
    assert frame.loc[0, "settling_time_median"] is None
    payload = load_json(tmp_path / "metric_summary.json")
    assert payload["rows"][0]["settling_time_median"] is None
    markdown = (tmp_path / "metric_summary.md").read_text(encoding="utf-8")
    assert "NA" in markdown
