from __future__ import annotations

import numpy as np

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.controllers.lqr import LQRController
from cartpole_bench.types import CartPoleParams, ControllerConfig
from cartpole_bench.utils.math import clamp, wrap_angle


class PartialFeedbackLinearizationController(BaseController):
    def __init__(self, config: ControllerConfig, params: CartPoleParams, lqr_config: ControllerConfig) -> None:
        super().__init__("Feedback Linearization (PFL)", config)
        self.params = params
        self.lqr = LQRController(lqr_config, params)
        self.lqr_assist_weight = float(config.gains.get("lqr_assist_weight", 0.55))

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
        p = self.params
        g = self.config.gains
        theta_err = float(wrap_angle(theta))
        angle_term = 0.65 * theta_err + 0.35 * np.sin(theta_err)

        desired_cart_accel = (
            -g["k_x"] * x
            - g["k_x_dot"] * x_dot
            - g["k_theta"] * angle_term
            - g["k_theta_dot"] * theta_dot
        )
        desired_cart_accel = clamp(
            float(desired_cart_accel),
            -g["desired_accel_limit"],
            g["desired_accel_limit"],
        )

        inertia = p.pendulum_mass_matrix
        coupling = p.m * p.l * np.cos(theta)
        effective_mass = (p.M + p.m) - (coupling * coupling) / max(inertia, 1e-6)
        effective_mass = max(effective_mass, g["effective_mass_epsilon"])
        rhs_theta = p.m * p.g * p.l * np.sin(theta) - p.pivot_damping * theta_dot
        control = (
            effective_mass * desired_cart_accel
            - (coupling / max(inertia, 1e-6)) * rhs_theta
            + p.cart_friction * x_dot
            + p.m * p.l * theta_dot * theta_dot * np.sin(theta)
        )
        nominal = self.lqr.compute_control(t, state, dt)
        blended = (1.0 - self.lqr_assist_weight) * float(control) + self.lqr_assist_weight * nominal
        return self.saturate(float(blended))
