from __future__ import annotations

import math

import numpy as np

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.types import SwitchConfig


class HybridController:
    def __init__(
        self,
        swingup_controller: BaseController,
        stabilizer: BaseController,
        switch_config: SwitchConfig,
    ) -> None:
        self.swingup_controller = swingup_controller
        self.stabilizer = stabilizer
        self.switch_config = switch_config
        self.stage = "energy_pump"
        self.time_inside_balance_gate = 0.0
        self.time_outside_balance = 0.0
        self.switch_time: float | None = None

    @property
    def label(self) -> str:
        return self.stabilizer.label

    def reset(self) -> None:
        self.swingup_controller.reset()
        self.stabilizer.reset()
        self.stage = "energy_pump"
        self.time_inside_balance_gate = 0.0
        self.time_outside_balance = 0.0
        self.switch_time = None

    def _in_balance_gate(self, state: np.ndarray) -> bool:
        angle = abs(float(state[2]))
        rate = abs(float(state[3]))
        return angle < math.radians(self.switch_config.enter_angle_deg) and rate < self.switch_config.enter_rate

    def _outside_balance_gate(self, state: np.ndarray) -> bool:
        angle = abs(float(state[2]))
        return angle > math.radians(self.switch_config.exit_angle_deg)

    def _release_capture(self, state: np.ndarray) -> bool:
        angle = abs(float(state[2]))
        return angle > math.radians(self.switch_config.capture_release_angle_deg)

    def compute_control(self, t: float, state: np.ndarray, dt: float) -> tuple[float, str]:
        if self.stage == "balance":
            if self._outside_balance_gate(state):
                self.time_outside_balance += dt
            else:
                self.time_outside_balance = 0.0
            if self.time_outside_balance >= self.switch_config.exit_hold_time:
                self.stage = "capture_assist"
                self.time_inside_balance_gate = 0.0
                self.time_outside_balance = 0.0
        else:
            in_balance_gate = self._in_balance_gate(state)
            if in_balance_gate:
                self.time_inside_balance_gate += dt
            else:
                self.time_inside_balance_gate = 0.0

            if self.time_inside_balance_gate >= self.switch_config.enter_hold_time:
                self.stage = "balance"
                if self.switch_time is None:
                    self.switch_time = max(0.0, t + dt - self.switch_config.enter_hold_time)
            else:
                if self.stage == "capture_assist" and self._release_capture(state):
                    self.stage = "energy_pump"
                # If the state is already inside the balance gate, do not detour back into
                # capture assist while waiting out the entry hold time.
                if (
                    not in_balance_gate
                    and self.stage != "capture_assist"
                    and self.swingup_controller.wants_capture_assist(state, self.switch_config)
                ):
                    self.stage = "capture_assist"

        if self.stage == "balance":
            return self.stabilizer.compute_control(t, state), "balance"
        if self.stage == "capture_assist":
            return self.swingup_controller.compute_capture_control(t, state), "capture_assist"
        return self.swingup_controller.compute_control(t, state), "energy_pump"
