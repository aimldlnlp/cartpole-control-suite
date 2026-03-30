from __future__ import annotations

import numpy as np
from scipy import linalg

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.dynamics.linearize import upright_state_space
from cartpole_bench.types import CartPoleParams, ControllerConfig


class LQRController(BaseController):
    def __init__(self, config: ControllerConfig, params: CartPoleParams) -> None:
        super().__init__("LQR", config)
        A, B = upright_state_space(params)
        q_vec = np.asarray(config.gains["Q"], dtype=float)
        Q = np.diag(q_vec)
        R = np.array([[float(config.gains["R"])]], dtype=float)
        P = linalg.solve_continuous_are(A, B, Q, R)
        self.K = np.asarray(np.linalg.solve(R, B.T @ P), dtype=float).reshape(-1)

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        x = np.asarray(state, dtype=float).reshape(-1)
        control = -float(self.K @ x)
        return self.saturate(control)
