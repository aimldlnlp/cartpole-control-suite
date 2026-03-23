from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_controller_config, load_switch_config, load_system_params
from cartpole_bench.controllers.swingup import EnergySwingUpController
from cartpole_bench.dynamics.cartpole import CartPoleDynamics


def _build_swingup() -> EnergySwingUpController:
    params = load_system_params()
    return EnergySwingUpController(
        load_controller_config("swingup"),
        CartPoleDynamics(params),
        load_controller_config("lqr"),
    )


def test_capture_assist_is_requested_in_large_angle_recovery_zone() -> None:
    swingup = _build_swingup()
    state = np.array([0.0, 0.0, np.deg2rad(25.0), 0.0], dtype=float)
    assert swingup.wants_capture_assist(state, load_switch_config())


def test_swingup_rail_guard_pushes_cart_back_toward_center() -> None:
    swingup = _build_swingup()
    state = np.array([1.9, 0.4, np.pi, 0.0], dtype=float)
    control = swingup.compute_control(0.0, state)
    assert control < 0.0
