from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import signal
from scipy.optimize import minimize

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.dynamics.linearize import discrete_lqr_terminal_cost, upright_state_space
from cartpole_bench.types import CartPoleParams, ControllerConfig


class ModelPredictiveController(BaseController):
    def __init__(self, config: ControllerConfig, params: CartPoleParams) -> None:
        super().__init__("Model Predictive Control (MPC)", config)
        self.params = params
        gains = config.gains
        self.Q = np.diag(np.asarray(gains["Q"], dtype=float))
        self.Qf = np.diag(np.asarray(gains["Qf"], dtype=float))
        self.R = float(gains["R"])
        self.du_weight = float(gains["du_weight"])
        self.horizon_steps = int(gains["horizon_steps"])
        self.solver_maxiter = int(gains["solver_maxiter"])
        self.solver_ftol = float(gains["solver_ftol"])
        self.default_dt = float(gains.get("dt", 0.01))
        self.soft_track_weight = float(gains.get("soft_track_weight", 0.0))
        self.soft_track_limit_ratio = float(gains.get("soft_track_limit_ratio", 1.0))
        self.soft_angle_weight = float(gains.get("soft_angle_weight", 0.0))
        self.soft_angle_limit = np.deg2rad(float(gains.get("soft_angle_limit_deg", 180.0)))
        self._A_c, self._B_c = upright_state_space(params)
        self._current_dt: float | None = None
        self.A_d = np.eye(4, dtype=float)
        self.B_d = np.zeros((4, 1), dtype=float)
        self._terminal_cost_matrix = self.Qf.copy()
        self._terminal_feedback = np.zeros((1, 4), dtype=float)
        self._u_guess = np.zeros(self.horizon_steps, dtype=float)
        self._last_control = 0.0
        self._used_runtime_dt = self.default_dt
        self.reset()

    def reset(self) -> None:
        self._u_guess = np.zeros(self.horizon_steps, dtype=float)
        self._last_control = 0.0
        self._used_runtime_dt = self.default_dt
        self._current_dt = None
        self.A_d = np.eye(4, dtype=float)
        self.B_d = np.zeros((4, 1), dtype=float)
        self._terminal_cost_matrix = self.Qf.copy()
        self._terminal_feedback = np.zeros((1, 4), dtype=float)
        self._solve_calls = 0
        self._solver_failure_count = 0
        self._solve_times_ms: list[float] = []
        self._iteration_counts: list[int] = []
        self._objectives: list[float] = []
        self._max_abs_command = 0.0

    def debug_summary(self) -> dict[str, float | int | bool | None]:
        return {
            "solver_name": "slsqp_mpc",
            "solve_calls": self._solve_calls,
            "solver_failure_count": self._solver_failure_count,
            "median_solve_ms": float(np.median(self._solve_times_ms)) if self._solve_times_ms else None,
            "max_solve_ms": float(np.max(self._solve_times_ms)) if self._solve_times_ms else None,
            "median_iterations": float(np.median(self._iteration_counts)) if self._iteration_counts else None,
            "max_iterations": int(np.max(self._iteration_counts)) if self._iteration_counts else None,
            "median_objective": float(np.median(self._objectives)) if self._objectives else None,
            "max_abs_command": float(self._max_abs_command),
            "used_runtime_dt": float(self._used_runtime_dt),
        }

    def switch_overrides(self) -> dict[str, float]:
        return {
            "enter_angle_deg": 6.0,
            "enter_rate": 0.35,
            "enter_hold_time": 0.20,
            "exit_angle_deg": 10.0,
            "exit_hold_time": 0.05,
        }

    def _track_limit_soft(self) -> float:
        return max(1e-6, self.soft_track_limit_ratio * self.params.track_limit)

    def _ensure_discretized(self, dt: float) -> None:
        if self._current_dt is not None and abs(self._current_dt - dt) <= 1e-12:
            return
        try:
            A_d, B_d, P, K = discrete_lqr_terminal_cost(self.params, dt, self.Q, self.R)
        except np.linalg.LinAlgError:
            A_d, B_d, _, _, _ = signal.cont2discrete((self._A_c, self._B_c, np.eye(4), np.zeros((4, 1))), dt)
            self.A_d = np.asarray(A_d, dtype=float)
            self.B_d = np.asarray(B_d, dtype=float).reshape(4, 1)
            self._terminal_cost_matrix = self.Qf.copy()
            self._terminal_feedback = np.zeros((1, 4), dtype=float)
        else:
            self.A_d = A_d
            self.B_d = B_d
            self._terminal_cost_matrix = P
            self._terminal_feedback = K
        self._current_dt = dt

    def _lqr_seed_controls(self, x0: np.ndarray) -> np.ndarray:
        controls = np.zeros(self.horizon_steps, dtype=float)
        state = np.asarray(x0, dtype=float).reshape(4)
        for stage in range(self.horizon_steps):
            control = self.saturate(-float((self._terminal_feedback @ state).item()))
            controls[stage] = control
            state = self.A_d @ state + self.B_d[:, 0] * control
        return controls

    def _rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((len(controls) + 1, 4), dtype=float)
        states[0] = x0
        for index, control in enumerate(controls):
            states[index + 1] = self.A_d @ states[index] + self.B_d[:, 0] * float(control)
        return states

    def _soft_state_penalty(self, state: np.ndarray) -> float:
        penalty = 0.0
        x_excess = abs(float(state[0])) - self._track_limit_soft()
        if x_excess > 0.0:
            penalty += self.soft_track_weight * x_excess * x_excess
        theta_excess = abs(float(state[2])) - self.soft_angle_limit
        if theta_excess > 0.0:
            penalty += self.soft_angle_weight * theta_excess * theta_excess
        return penalty

    def _objective(self, controls: np.ndarray, x0: np.ndarray) -> float:
        states = self._rollout(x0, controls)
        total = 0.0
        previous = self._last_control
        for state, control in zip(states[:-1], controls, strict=True):
            du = float(control - previous)
            total += 0.5 * float(state.T @ self.Q @ state)
            total += 0.5 * self.R * float(control * control)
            total += 0.5 * self.du_weight * du * du
            total += self._soft_state_penalty(state)
            previous = float(control)
        total += 0.5 * float(states[-1].T @ self._terminal_cost_matrix @ states[-1]) + self._soft_state_penalty(states[-1])
        return total

    def _accepted_rollout(self, controls: np.ndarray, x0: np.ndarray) -> tuple[bool, float]:
        if not np.all(np.isfinite(controls)):
            return False, float("inf")
        states = self._rollout(x0, controls)
        if not np.all(np.isfinite(states)):
            return False, float("inf")
        if np.max(np.abs(states[:, 0])) > 1.05 * self.params.track_limit:
            return False, float("inf")
        return True, self._objective(controls, x0)

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        runtime_dt = self.default_dt if dt is None else float(dt)
        self._used_runtime_dt = runtime_dt
        self._ensure_discretized(runtime_dt)
        self._solve_calls += 1
        solve_started = perf_counter()

        x0 = np.asarray(state, dtype=float).reshape(4)
        bounds = [(-self.config.force_limit, self.config.force_limit)] * self.horizon_steps
        lqr_seed = self._lqr_seed_controls(x0)
        initial = 0.5 * self._u_guess.copy() + 0.5 * lqr_seed

        result = minimize(
            lambda u: self._objective(np.asarray(u, dtype=float), x0),
            initial,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.solver_maxiter, "ftol": self.solver_ftol, "disp": False},
        )

        solver_failed = False
        chosen_controls: np.ndarray
        chosen_objective: float

        if result.success and np.all(np.isfinite(result.x)):
            candidate_controls = np.asarray(result.x, dtype=float)
            accepted, objective = self._accepted_rollout(candidate_controls, x0)
            if accepted:
                chosen_controls = candidate_controls
                chosen_objective = objective
            else:
                solver_failed = True
                chosen_controls = initial
                accepted, objective = self._accepted_rollout(chosen_controls, x0)
                if not accepted:
                    chosen_controls = np.zeros(self.horizon_steps, dtype=float)
                    _, objective = self._accepted_rollout(chosen_controls, x0)
                chosen_objective = objective
        else:
            solver_failed = True
            chosen_controls = initial
            accepted, objective = self._accepted_rollout(chosen_controls, x0)
            if not accepted:
                chosen_controls = np.zeros(self.horizon_steps, dtype=float)
                _, objective = self._accepted_rollout(chosen_controls, x0)
            chosen_objective = objective

        if solver_failed:
            self._solver_failure_count += 1

        issued_control = self.saturate(float(chosen_controls[0] if len(chosen_controls) else 0.0))
        self._u_guess[:-1] = chosen_controls[1:]
        self._u_guess[-1] = chosen_controls[-1] if len(chosen_controls) else 0.0
        self._last_control = issued_control
        self._max_abs_command = max(self._max_abs_command, abs(issued_control))
        self._solve_times_ms.append((perf_counter() - solve_started) * 1000.0)
        self._iteration_counts.append(int(getattr(result, "nit", 0)))
        self._objectives.append(float(chosen_objective))
        return issued_control
