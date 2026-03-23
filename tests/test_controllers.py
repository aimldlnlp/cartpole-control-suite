from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_controller_config, load_suite, load_system_params
from cartpole_bench.controllers.smc import SlidingModeController
from cartpole_bench.simulation.runner import simulate_trajectory


def test_all_controllers_remain_finite_on_nominal_local_case() -> None:
    scenario = load_suite("nominal")[0]
    params = load_system_params()
    initial_angle = abs(float(scenario.initial_state[2]))
    for controller_key in ("lqr", "pfl", "smc"):
        result, _, _ = simulate_trajectory(scenario, controller_key, params)
        assert np.all(np.isfinite(result.states))
        assert np.all(np.isfinite(result.controls))
        assert abs(float(result.states[-1, 2])) < initial_angle + 0.2


def test_smc_applies_recentering_control_near_upright() -> None:
    params = load_system_params()
    smc_config = load_controller_config("smc")
    lqr_config = load_controller_config("lqr")
    controller = SlidingModeController(smc_config, params, lqr_config)

    positive_state = np.array([0.18, 0.14, np.deg2rad(1.5), 0.05], dtype=float)
    negative_state = np.array([-0.18, -0.14, np.deg2rad(-1.5), -0.05], dtype=float)

    assert controller.compute_control(0.0, positive_state) < 0.0
    assert controller.compute_control(0.0, negative_state) > 0.0
