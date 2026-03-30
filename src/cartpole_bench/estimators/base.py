from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    def reset(self, initial_state: np.ndarray | None = None) -> None:
        return None

    @abstractmethod
    def step(self, observation: np.ndarray, control: float, dt: float) -> np.ndarray:
        raise NotImplementedError
