from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from cartpole_bench.types import ControllerConfig
from cartpole_bench.utils.math import clamp


class BaseController(ABC):
    def __init__(self, label: str, config: ControllerConfig) -> None:
        self.label = label
        self.config = config

    def reset(self) -> None:
        return None

    def debug_summary(self) -> dict[str, float | int | bool | None]:
        return {}

    def switch_overrides(self) -> dict[str, float]:
        return {}

    def saturate(self, control: float) -> float:
        return clamp(control, -self.config.force_limit, self.config.force_limit)

    def wants_capture_assist(self, state: np.ndarray, switch_config) -> bool:
        return False

    def compute_capture_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        return self.compute_control(t, state, dt)

    @abstractmethod
    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        raise NotImplementedError
