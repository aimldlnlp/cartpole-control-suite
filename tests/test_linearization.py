from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_system_params
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.linearize import finite_difference_state_space, upright_state_space


def test_analytic_linearization_matches_finite_difference() -> None:
    params = load_system_params()
    dynamics = CartPoleDynamics(params)
    analytic_a, analytic_b = upright_state_space(params)
    fd_a, fd_b = finite_difference_state_space(dynamics)
    assert np.allclose(analytic_a, fd_a, atol=1e-4)
    assert np.allclose(analytic_b, fd_b, atol=1e-4)
