from __future__ import annotations

import numpy as np

from cartpole_bench.metrics.core import compute_run_metrics


def test_metrics_return_expected_shapes() -> None:
    time = np.linspace(0.0, 2.0, 201)
    states = np.zeros((201, 4), dtype=float)
    states[:, 2] = np.linspace(0.1, 0.0, 201)
    controls = np.zeros(201, dtype=float)
    modes = ["balance"] * len(time)
    metrics, diagnosis = compute_run_metrics(time, states, controls, modes, 2.4, 0.5, False, False)
    assert metrics.overshoot_deg > 0.0
    assert metrics.steady_state_error_deg >= 0.0
    assert metrics.control_effort == 0.0
    assert diagnosis.first_balance_time == 0.0


def test_settling_time_is_measured_from_handoff_when_available() -> None:
    time = np.linspace(0.0, 3.0, 301)
    states = np.zeros((301, 4), dtype=float)
    states[:120, 2] = np.deg2rad(18.0)
    controls = np.zeros(301, dtype=float)
    modes = ["energy_pump"] * 100 + ["capture_assist"] * 20 + ["balance"] * 181

    metrics, diagnosis = compute_run_metrics(time, states, controls, modes, 2.4, 1.2, False, False)

    assert diagnosis.first_balance_time is not None
    assert metrics.settling_time is not None
    assert metrics.settling_time < 0.1
