from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_controller_config, load_switch_config, load_system_params
from cartpole_bench.controllers.hybrid import HybridController
from cartpole_bench.controllers.lqr import LQRController
from cartpole_bench.controllers.swingup import EnergySwingUpController
from cartpole_bench.dynamics.cartpole import CartPoleDynamics


def test_hybrid_controller_enters_and_exits_balance_mode() -> None:
    params = load_system_params()
    lqr_config = load_controller_config("lqr")
    swingup = EnergySwingUpController(load_controller_config("swingup"), CartPoleDynamics(params), lqr_config)
    stabilizer = LQRController(lqr_config, params)
    hybrid = HybridController(swingup, stabilizer, load_switch_config())

    balance_state = np.array([0.0, 0.0, np.deg2rad(4.0), 0.0], dtype=float)
    mode = "energy_pump"
    for index in range(20):
        _, mode = hybrid.compute_control(index * 0.01, balance_state, 0.01)
    assert mode == "balance"

    far_state = np.array([0.0, 0.0, np.deg2rad(45.0), 0.0], dtype=float)
    for index in range(30):
        _, mode = hybrid.compute_control(index * 0.01, far_state, 0.01)
    assert mode in {"energy_pump", "capture_assist"}
