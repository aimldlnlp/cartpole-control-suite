from __future__ import annotations

import numpy as np

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.controllers.lqr import LQRController
from cartpole_bench.types import CartPoleParams, ControllerConfig
from cartpole_bench.utils.math import smooth_sat, wrap_angle


class SlidingModeController(BaseController):
    def __init__(self, config: ControllerConfig, params: CartPoleParams, lqr_config: ControllerConfig) -> None:
        super().__init__("Sliding Mode Control (SMC)", config)
        self.surface = np.asarray(config.gains["sliding_surface"], dtype=float)
        self.reach_gain = float(config.gains["reach_gain"])
        self.boundary_layer = float(config.gains["boundary_layer"])
        self.lqr_blend_weight = float(config.gains["lqr_blend_weight"])
        self.cart_center_gain = float(config.gains.get("cart_center_gain", 0.0))
        self.cart_velocity_gain = float(config.gains.get("cart_velocity_gain", 0.0))
        self.settle_angle = np.deg2rad(float(config.gains.get("settle_angle_deg", 6.0)))
        self.settle_rate = float(config.gains.get("settle_rate", 0.75))
        self.settle_cart_gain = float(config.gains.get("settle_cart_gain", 2.2))
        self.settle_cart_velocity_gain = float(config.gains.get("settle_cart_velocity_gain", 3.4))
        self.settle_trim_limit = float(config.gains.get("settle_trim_limit", 1.5))
        self.robust_decay_position_scale = float(config.gains.get("robust_decay_position_scale", 0.18))
        self.robust_decay_velocity_scale = float(config.gains.get("robust_decay_velocity_scale", 0.25))
        self.robust_decay_angle = np.deg2rad(float(config.gains.get("robust_decay_angle_deg", 5.0)))
        self.robust_decay_rate_scale = float(config.gains.get("robust_decay_rate_scale", 0.35))
        self.lqr = LQRController(lqr_config, params)

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        x = np.asarray(state, dtype=float)
        theta_err = float(wrap_angle(x[2]))
        sliding_state = np.array([x[0], x[1], theta_err, x[3]], dtype=float)
        sigma = float(self.surface @ sliding_state)
        nominal = self.lqr.compute_control(t, x, dt)
        state_envelope = (
            abs(float(x[0])) / max(self.robust_decay_position_scale, 1e-6)
            + abs(float(x[1])) / max(self.robust_decay_velocity_scale, 1e-6)
            + abs(theta_err) / max(self.robust_decay_angle, 1e-6)
            + abs(float(x[3])) / max(self.robust_decay_rate_scale, 1e-6)
        )
        robust_weight = float(np.tanh(state_envelope))
        robust = self.reach_gain * robust_weight * float(smooth_sat(sigma, self.boundary_layer))
        centering = self.cart_center_gain * float(x[0]) + self.cart_velocity_gain * float(x[1])
        far_control = self.lqr_blend_weight * nominal - robust - centering
        theta_gate = np.clip(1.0 - abs(theta_err) / max(self.settle_angle, 1e-6), 0.0, 1.0)
        rate_gate = np.clip(1.0 - abs(float(x[3])) / max(self.settle_rate, 1e-6), 0.0, 1.0)
        settle_weight = float(np.sqrt(theta_gate * rate_gate))
        settle_trim = self.settle_cart_gain * float(x[0]) + self.settle_cart_velocity_gain * float(x[1])
        settle_trim = float(np.clip(settle_trim, -self.settle_trim_limit, self.settle_trim_limit))
        near_control = nominal - settle_trim
        control = (1.0 - settle_weight) * far_control + settle_weight * near_control
        return self.saturate(control)
