from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_system_params
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step


def test_upright_equilibrium_is_nearly_stationary() -> None:
    dynamics = CartPoleDynamics(load_system_params())
    state = np.zeros(4, dtype=float)
    deriv = dynamics.derivatives(state, 0.0, 0.0)
    assert np.allclose(deriv, 0.0, atol=1e-10)


def test_rk4_step_stays_finite() -> None:
    dynamics = CartPoleDynamics(load_system_params())
    state = np.array([0.05, 0.0, 0.15, 0.0], dtype=float)
    next_state = rk4_step(dynamics, state, 0.0, 0.01, 0.0)
    assert np.all(np.isfinite(next_state))
