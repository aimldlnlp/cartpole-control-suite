from __future__ import annotations

import numpy as np

from cartpole_bench.dynamics.cartpole import CartPoleDynamics


def rk4_step(
    dynamics: CartPoleDynamics,
    state: np.ndarray,
    control: float,
    dt: float,
    disturbance_force: float = 0.0,
) -> np.ndarray:
    k1 = dynamics.derivatives(state, control, disturbance_force)
    k2 = dynamics.derivatives(state + 0.5 * dt * k1, control, disturbance_force)
    k3 = dynamics.derivatives(state + 0.5 * dt * k2, control, disturbance_force)
    k4 = dynamics.derivatives(state + dt * k3, control, disturbance_force)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return dynamics.post_step(next_state)
