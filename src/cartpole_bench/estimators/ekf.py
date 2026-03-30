from __future__ import annotations

import numpy as np

from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step
from cartpole_bench.estimators.base import BaseEstimator
from cartpole_bench.types import CartPoleParams, EstimatorConfig
from cartpole_bench.utils.math import wrap_angle


class ExtendedKalmanFilter(BaseEstimator):
    def __init__(
        self,
        config: EstimatorConfig,
        params: CartPoleParams,
        measurement_std: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.dynamics = CartPoleDynamics(params)
        gains = config.gains
        process_std = np.asarray(gains["process_std"], dtype=float)
        initial_cov_diag = np.asarray(gains["initial_cov_diag"], dtype=float)
        self.measurement_floor = float(gains["measurement_floor"])
        self.Q = np.diag(np.square(process_std))
        measurement_std = np.zeros(4, dtype=float) if measurement_std is None else np.asarray(measurement_std, dtype=float)
        self.R = np.diag(np.maximum(np.square(measurement_std), self.measurement_floor))
        self.P0 = np.diag(initial_cov_diag)
        self.H = np.eye(4, dtype=float)
        self.x = np.zeros(4, dtype=float)
        self.P = self.P0.copy()

    def reset(self, initial_state: np.ndarray | None = None) -> None:
        self.x = np.zeros(4, dtype=float) if initial_state is None else np.asarray(initial_state, dtype=float).copy()
        self.x[2] = wrap_angle(self.x[2])
        self.P = self.P0.copy()

    def _predict_state(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        return rk4_step(self.dynamics, state, float(control), dt, 0.0)

    def _jacobian(self, state: np.ndarray, control: float, dt: float, eps: float = 1e-5) -> np.ndarray:
        A = np.zeros((4, 4), dtype=float)
        for index in range(4):
            delta = np.zeros(4, dtype=float)
            delta[index] = eps
            forward = self._predict_state(state + delta, control, dt)
            backward = self._predict_state(state - delta, control, dt)
            A[:, index] = (forward - backward) / (2.0 * eps)
        return A

    def step(self, observation: np.ndarray, control: float, dt: float) -> np.ndarray:
        observation = np.asarray(observation, dtype=float).copy()
        observation[2] = wrap_angle(observation[2])

        if dt <= 0.0:
            A = np.eye(4, dtype=float)
            x_pred = self.x.copy()
            P_pred = self.P.copy()
        else:
            A = self._jacobian(self.x, control, dt)
            x_pred = self._predict_state(self.x, control, dt)
            P_pred = A @ self.P @ A.T + self.Q

        innovation = observation - x_pred
        innovation[2] = wrap_angle(innovation[2])
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.pinv(S)

        self.x = x_pred + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(4, dtype=float) - K @ self.H) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()
