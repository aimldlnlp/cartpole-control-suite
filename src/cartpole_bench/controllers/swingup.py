from __future__ import annotations

import numpy as np

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.controllers.lqr import LQRController
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.types import ControllerConfig
from cartpole_bench.utils.math import clamp


class EnergySwingUpController(BaseController):
    def __init__(
        self,
        config: ControllerConfig,
        dynamics: CartPoleDynamics,
        capture_lqr_config: ControllerConfig,
    ) -> None:
        super().__init__("Swing-Up", config)
        self.dynamics = dynamics
        self.capture_lqr = LQRController(capture_lqr_config, dynamics.params)

    def energy_gap(self, state: np.ndarray) -> float:
        return self.dynamics.desired_upright_energy() - self.dynamics.pendulum_energy_from_downward(state)

    def energy_gap_ratio(self, state: np.ndarray) -> float:
        desired = max(self.dynamics.desired_upright_energy(), 1e-6)
        return self.energy_gap(state) / desired

    def wants_capture_assist(self, state: np.ndarray, switch_config) -> bool:
        theta = float(state[2])
        theta_dot = float(state[3])
        angle_deg = abs(np.rad2deg(theta))
        rate = abs(theta_dot)
        near_top = switch_config.capture_min_angle_deg < angle_deg <= switch_config.capture_entry_angle_deg
        energy_ready = abs(self.energy_gap_ratio(state)) < switch_config.capture_energy_tolerance_ratio
        approaching_upright = theta * theta_dot < 0.0 and angle_deg <= switch_config.capture_release_angle_deg
        return (
            angle_deg <= switch_config.capture_entry_angle_deg
            and rate <= self.config.gains["capture_rate_limit"]
            and (near_top or energy_ready or approaching_upright)
        )

    def _rail_guard(self, x: float, x_dot: float, gain_scale: float = 1.0) -> float:
        gains = self.config.gains
        track_limit = max(self.dynamics.params.track_limit, 1e-6)
        position_term = -gain_scale * gains.get("rail_guard_gain", 0.0) * np.tanh(x / (0.35 * track_limit))
        velocity_term = -gain_scale * gains.get("rail_guard_velocity_gain", 0.0) * x_dot
        return float(position_term + velocity_term)

    def _recenter_control(self, x: float, x_dot: float) -> float:
        gains = self.config.gains
        control = -gains.get("recenter_position_gain", 6.0) * x - gains.get("recenter_velocity_gain", 4.5) * x_dot
        return self.saturate(float(control))

    def _needs_recenter(self, x: float) -> bool:
        track_limit = max(self.dynamics.params.track_limit, 1e-6)
        threshold = self.config.gains.get("recenter_position_threshold_ratio", 0.68) * track_limit
        return abs(x) >= threshold

    def compute_capture_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        gains = self.config.gains
        x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
        if self._needs_recenter(x):
            return self._recenter_control(x, x_dot)
        cart_target = clamp(
            gains.get("capture_target_theta_gain", 0.6) * theta
            + gains.get("capture_target_theta_dot_gain", 0.18) * theta_dot,
            -gains.get("capture_target_limit", 0.45),
            gains.get("capture_target_limit", 0.45),
        )
        capture_pd = (
            -gains["capture_theta_gain"] * np.sin(theta)
            - gains["capture_theta_dot_gain"] * theta_dot
            - gains["capture_x_gain"] * (x - cart_target)
            - gains["capture_x_dot_gain"] * x_dot
        )
        lqr_assist = self.capture_lqr.compute_control(t, np.asarray([x, x_dot, theta, theta_dot], dtype=float), dt)
        lqr_full_angle = np.deg2rad(gains.get("capture_lqr_full_angle_deg", 20.0))
        blend_angle = np.deg2rad(gains.get("capture_blend_angle_deg", 35.0))
        if abs(theta) <= lqr_full_angle:
            blend = 1.0
        else:
            upright_fraction = clamp(1.0 - abs(theta) / max(blend_angle, 1e-6), 0.0, 1.0)
            blend = gains.get("capture_balance_blend", 0.65) * upright_fraction
        rail_term = self._rail_guard(x, x_dot, gain_scale=gains.get("capture_rail_guard_scale", 1.35))
        control = (1.0 - blend) * capture_pd + blend * lqr_assist + rail_term
        return self.saturate(float(control))

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        gains = self.config.gains
        x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
        if self._needs_recenter(x):
            return self._recenter_control(x, x_dot)
        current_energy = self.dynamics.pendulum_energy_from_downward(state)
        desired_energy = self.dynamics.desired_upright_energy()
        energy_gap = desired_energy - current_energy
        pump_phase = theta_dot * np.cos(theta)
        phase_drive = np.tanh(gains["energy_velocity_gain"] * pump_phase)
        angle_drive = np.sin(theta)
        track_soft_limit = max(gains.get("energy_soft_limit_ratio", 0.85) * self.dynamics.params.track_limit, 1e-6)
        edge_ratio = min(abs(x) / track_soft_limit, 1.5)
        edge_scale = max(0.25, 1.0 - 0.65 * edge_ratio * edge_ratio)
        swing_drive = 0.55 * angle_drive + 0.45 * phase_drive
        swing_term = gains["energy_gain"] * energy_gap * swing_drive * edge_scale
        center_term = -gains["cart_position_gain"] * x - gains["cart_velocity_gain"] * x_dot
        rail_term = self._rail_guard(x, x_dot, gain_scale=gains.get("energy_rail_guard_scale", 0.55))
        return self.saturate(float(swing_term + center_term + rail_term))
