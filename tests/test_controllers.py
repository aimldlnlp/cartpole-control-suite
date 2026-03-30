from __future__ import annotations

import numpy as np
import pytest

from cartpole_bench.config import load_controller_config, load_suite, load_system_params
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step
from cartpole_bench.controllers.ilqr import IterativeLQRController
from cartpole_bench.controllers.mpc import ModelPredictiveController
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


def test_ilqr_and_mpc_produce_finite_bounded_controls_near_upright() -> None:
    params = load_system_params()
    state = np.array([0.08, -0.05, np.deg2rad(3.0), 0.12], dtype=float)
    ilqr = IterativeLQRController(load_controller_config("ilqr"), params)
    mpc = ModelPredictiveController(load_controller_config("mpc"), params)

    for controller in (ilqr, mpc):
        control = controller.compute_control(0.0, state, dt=0.01)
        assert np.isfinite(control)
        assert abs(control) <= controller.config.force_limit + 1e-9
        debug = controller.debug_summary()
        assert debug["solve_calls"] == 1
        assert debug["used_runtime_dt"] == 0.01
        assert debug["median_solve_ms"] is not None
        assert debug["median_iterations"] is not None
        assert debug["median_objective"] is not None


def test_ilqr_and_mpc_stay_inside_track_on_short_near_upright_rollout() -> None:
    params = load_system_params()
    dynamics = CartPoleDynamics(params)
    initial_state = np.array([0.05, 0.0, np.deg2rad(8.0), 0.0], dtype=float)
    dt = 0.02

    for controller_cls, key in (
        (IterativeLQRController, "ilqr"),
        (ModelPredictiveController, "mpc"),
    ):
        controller = controller_cls(load_controller_config(key), params)
        state = initial_state.copy()
        max_abs_x = abs(float(state[0]))
        for step in range(25):
            control = controller.compute_control(step * dt, state, dt=dt)
            assert np.isfinite(control)
            state = rk4_step(dynamics, state, control, dt)
            max_abs_x = max(max_abs_x, abs(float(state[0])))
        assert np.all(np.isfinite(state))
        assert max_abs_x < params.track_limit
        assert abs(float(state[2])) < abs(float(initial_state[2])) + 0.1


@pytest.mark.slow
def test_ilqr_and_mpc_succeed_on_most_nominal_local_small_angle_ekf_runs() -> None:
    params = load_system_params()
    local_scenarios = [
        scenario
        for scenario in load_suite("nominal")
        if scenario.name == "local_small_angle"
    ]

    for controller_key in ("ilqr", "mpc"):
        results = [
            simulate_trajectory(
                scenario,
                controller_key,
                params,
                estimator_name="ekf",
            )[0]
            for scenario in local_scenarios
        ]
        success_count = sum(int(result.metrics.success) for result in results)
        track_violations = sum(int(result.track_violation) for result in results)
        assert success_count >= 4
        assert track_violations == 0
