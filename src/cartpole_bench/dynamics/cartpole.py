from __future__ import annotations

from dataclasses import replace

import numpy as np

from cartpole_bench.types import CartPoleParams, DisturbanceConfig
from cartpole_bench.utils.math import wrap_angle


class CartPoleDynamics:
    """Full nonlinear underactuated cart-pole dynamics."""

    def __init__(self, params: CartPoleParams) -> None:
        self.params = params

    def with_overrides(self, overrides: dict[str, float]) -> "CartPoleDynamics":
        return CartPoleDynamics(replace(self.params, **overrides))

    def accelerations(
        self,
        state: np.ndarray,
        control: float,
        disturbance_force: float = 0.0,
    ) -> np.ndarray:
        x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
        p = self.params

        a11 = p.M + p.m
        a12 = -p.m * p.l * np.cos(theta)
        a21 = -p.m * p.l * np.cos(theta)
        a22 = p.pendulum_mass_matrix

        rhs1 = (
            control
            + disturbance_force
            - p.cart_friction * x_dot
            - p.m * p.l * theta_dot * theta_dot * np.sin(theta)
        )
        rhs2 = p.m * p.g * p.l * np.sin(theta) - p.pivot_damping * theta_dot

        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-9:
            raise FloatingPointError("Cart-pole mass matrix became singular.")

        inv = np.array([[a22, -a12], [-a21, a11]], dtype=float) / det
        accel = inv @ np.array([rhs1, rhs2], dtype=float)
        return accel

    def derivatives(
        self,
        state: np.ndarray,
        control: float,
        disturbance_force: float = 0.0,
    ) -> np.ndarray:
        x_ddot, theta_ddot = self.accelerations(state, control, disturbance_force)
        return np.array([state[1], x_ddot, state[3], theta_ddot], dtype=float)

    def post_step(self, state: np.ndarray) -> np.ndarray:
        updated = np.asarray(state, dtype=float).copy()
        updated[2] = wrap_angle(updated[2])
        return updated

    def pendulum_energy_from_downward(self, state: np.ndarray) -> float:
        theta = float(state[2])
        theta_dot = float(state[3])
        kinetic = 0.5 * self.params.pendulum_mass_matrix * theta_dot * theta_dot
        potential = self.params.m * self.params.g * self.params.l * (1.0 + np.cos(theta))
        return kinetic + potential

    def desired_upright_energy(self) -> float:
        return 2.0 * self.params.m * self.params.g * self.params.l

    def disturbance_force(self, cfg: DisturbanceConfig, t: float) -> float:
        if cfg.kind == "none":
            return 0.0
        if cfg.kind == "pulse":
            end_time = cfg.start_time + cfg.duration
            return cfg.magnitude if cfg.start_time <= t <= end_time else 0.0
        raise ValueError(f"Unsupported disturbance kind: {cfg.kind}")

    def upright_equilibrium(self) -> np.ndarray:
        return np.zeros(4, dtype=float)
