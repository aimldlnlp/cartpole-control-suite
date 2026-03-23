from __future__ import annotations

from cartpole_bench.config import load_suite


def test_nominal_suite_expands_repeated_seeds() -> None:
    scenarios = load_suite("nominal")
    local = [scenario.seed for scenario in scenarios if scenario.name == "local_small_angle"]
    full = [scenario.seed for scenario in scenarios if scenario.name == "full_task_hanging"]

    assert local == [7, 8, 9, 10, 11]
    assert full == [11, 12, 13, 14, 15]


def test_stress_suite_expands_all_repeated_scenarios() -> None:
    scenarios = load_suite("stress")
    counts = {}
    for scenario in scenarios:
        counts[scenario.name] = counts.get(scenario.name, 0) + 1

    assert counts == {
        "measurement_noise": 5,
        "impulse_disturbance": 5,
        "friction_and_damping": 5,
        "large_angle_recovery": 5,
        "parameter_mismatch": 5,
    }
