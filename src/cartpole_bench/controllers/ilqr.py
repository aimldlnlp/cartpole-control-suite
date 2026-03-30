from __future__ import annotations

from time import perf_counter

import numpy as np

from cartpole_bench.controllers.base import BaseController
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.dynamics.integrators import rk4_step
from cartpole_bench.dynamics.linearize import discrete_lqr_terminal_cost
from cartpole_bench.types import CartPoleParams, ControllerConfig


class IterativeLQRController(BaseController):
    def __init__(self, config: ControllerConfig, params: CartPoleParams) -> None:
        super().__init__("Iterative LQR (iLQR)", config)
        self.params = params
        self.dynamics = CartPoleDynamics(params)
        gains = config.gains
        self.Q = np.diag(np.asarray(gains["Q"], dtype=float))
        self.Qf = np.diag(np.asarray(gains["Qf"], dtype=float))
        self.R = float(gains["R"])
        self.du_weight = float(gains.get("du_weight", 0.0))
        self.horizon_steps = int(gains["horizon_steps"])
        self.max_iterations = int(gains["max_iterations"])
        self.regularization = float(gains["regularization"])
        self.regularization_scale_up = float(gains["regularization_scale_up"])
        self.regularization_scale_down = float(gains["regularization_scale_down"])
        self.state_epsilon = float(gains["state_epsilon"])
        self.default_dt = float(gains.get("dt", 0.01))
        self.line_search_alphas = tuple(float(alpha) for alpha in gains.get("line_search_alphas", [1.0, 0.5, 0.25]))
        self.soft_track_weight = float(gains.get("soft_track_weight", 0.0))
        self.soft_track_limit_ratio = float(gains.get("soft_track_limit_ratio", 1.0))
        self.soft_angle_weight = float(gains.get("soft_angle_weight", 0.0))
        self.soft_angle_limit = np.deg2rad(float(gains.get("soft_angle_limit_deg", 180.0)))
        self._seed_A_d = np.eye(4, dtype=float)
        self._seed_B_d = np.zeros((4, 1), dtype=float)
        self._terminal_dt: float | None = None
        self._terminal_cost_matrix = self.Qf.copy()
        self._terminal_feedback = np.zeros((1, 4), dtype=float)
        self._u_nominal = np.zeros(self.horizon_steps, dtype=float)
        self._last_control = 0.0
        self._used_runtime_dt = self.default_dt
        self.reset()

    def reset(self) -> None:
        self._u_nominal = np.zeros(self.horizon_steps, dtype=float)
        self._last_control = 0.0
        self._used_runtime_dt = self.default_dt
        self._terminal_dt = None
        self._terminal_cost_matrix = self.Qf.copy()
        self._terminal_feedback = np.zeros((1, 4), dtype=float)
        self._seed_A_d = np.eye(4, dtype=float)
        self._seed_B_d = np.zeros((4, 1), dtype=float)
        self._solve_calls = 0
        self._solver_failure_count = 0
        self._solve_times_ms: list[float] = []
        self._iteration_counts: list[int] = []
        self._objectives: list[float] = []
        self._max_abs_command = 0.0

    def debug_summary(self) -> dict[str, float | int | bool | None]:
        return {
            "solver_name": "ilqr",
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

    def _ensure_terminal_controller(self, dt: float) -> None:
        if self._terminal_dt is not None and abs(self._terminal_dt - dt) <= 1e-12:
            return
        try:
            A_d, B_d, P, K = discrete_lqr_terminal_cost(self.params, dt, self.Q, self.R)
        except np.linalg.LinAlgError:
            self._seed_A_d = np.eye(4, dtype=float)
            self._seed_B_d = np.zeros((4, 1), dtype=float)
            self._terminal_cost_matrix = self.Qf.copy()
            self._terminal_feedback = np.zeros((1, 4), dtype=float)
        else:
            self._seed_A_d = A_d
            self._seed_B_d = B_d
            self._terminal_cost_matrix = P
            self._terminal_feedback = K
        self._terminal_dt = dt

    def _lqr_seed_controls(self, x0: np.ndarray, dt: float) -> np.ndarray:
        self._ensure_terminal_controller(dt)
        controls = np.zeros(self.horizon_steps, dtype=float)
        state = np.asarray(x0, dtype=float).reshape(4)
        for stage in range(self.horizon_steps):
            control = self.saturate(-float((self._terminal_feedback @ state).item()))
            controls[stage] = control
            state = self._seed_A_d @ state + self._seed_B_d[:, 0] * control
        return controls

    def _dynamics_step(self, state: np.ndarray, control: float, dt: float) -> np.ndarray:
        return rk4_step(self.dynamics, state, float(control), dt, 0.0)

    def _linearize(self, state: np.ndarray, control: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=float)
        eps = self.state_epsilon
        A = np.zeros((4, 4), dtype=float)
        for index in range(4):
            delta = np.zeros(4, dtype=float)
            delta[index] = eps
            forward = self._dynamics_step(state + delta, control, dt)
            backward = self._dynamics_step(state - delta, control, dt)
            A[:, index] = (forward - backward) / (2.0 * eps)
        control_eps = max(eps, 1e-6)
        B = (
            self._dynamics_step(state, control + control_eps, dt) - self._dynamics_step(state, control - control_eps, dt)
        )[:, None] / (2.0 * control_eps)
        return A, B

    def _soft_state_terms(self, state: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        l_x = np.zeros(4, dtype=float)
        l_xx = np.zeros((4, 4), dtype=float)
        penalty = 0.0

        track_limit = self._track_limit_soft()
        x_value = float(state[0])
        x_excess = abs(x_value) - track_limit
        if x_excess > 0.0:
            penalty += self.soft_track_weight * x_excess * x_excess
            l_x[0] += 2.0 * self.soft_track_weight * x_excess * np.sign(x_value)
            l_xx[0, 0] += 2.0 * self.soft_track_weight

        theta_value = float(state[2])
        theta_excess = abs(theta_value) - self.soft_angle_limit
        if theta_excess > 0.0:
            penalty += self.soft_angle_weight * theta_excess * theta_excess
            l_x[2] += 2.0 * self.soft_angle_weight * theta_excess * np.sign(theta_value)
            l_xx[2, 2] += 2.0 * self.soft_angle_weight

        return penalty, l_x, l_xx

    def _stage_terms(self, state: np.ndarray, control: float, previous_control: float) -> tuple[float, np.ndarray, np.ndarray, float, float]:
        state = np.asarray(state, dtype=float)
        du = float(control - previous_control)
        stage_cost = 0.5 * float(state.T @ self.Q @ state) + 0.5 * self.R * float(control * control)
        stage_cost += 0.5 * self.du_weight * du * du
        soft_penalty, soft_lx, soft_lxx = self._soft_state_terms(state)
        stage_cost += soft_penalty
        l_x = self.Q @ state + soft_lx
        l_xx = self.Q + soft_lxx
        l_u = self.R * float(control) + self.du_weight * du
        l_uu = self.R + self.du_weight
        return stage_cost, l_x, l_xx, l_u, l_uu

    def _terminal_terms(self, state: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        soft_penalty, soft_lx, soft_lxx = self._soft_state_terms(state)
        terminal_cost = 0.5 * float(state.T @ self._terminal_cost_matrix @ state) + soft_penalty
        return (
            terminal_cost,
            self._terminal_cost_matrix @ state + soft_lx,
            self._terminal_cost_matrix + soft_lxx,
        )

    def _cost(self, states: np.ndarray, controls: np.ndarray, initial_control: float) -> float:
        total = 0.0
        previous_control = float(initial_control)
        for state, control in zip(states[:-1], controls, strict=True):
            stage_cost, _, _, _, _ = self._stage_terms(state, float(control), previous_control)
            total += stage_cost
            previous_control = float(control)
        terminal_cost, _, _ = self._terminal_terms(states[-1])
        total += terminal_cost
        return total

    def _rollout(self, x0: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
        states = np.zeros((len(controls) + 1, 4), dtype=float)
        states[0] = x0
        for index, control in enumerate(controls):
            states[index + 1] = self._dynamics_step(states[index], float(control), dt)
        return states

    def _states_within_guard(self, states: np.ndarray) -> bool:
        if not np.all(np.isfinite(states)):
            return False
        return bool(np.max(np.abs(states[:, 0])) <= 1.05 * self.params.track_limit)

    def compute_control(self, t: float, state: np.ndarray, dt: float | None = None) -> float:
        runtime_dt = self.default_dt if dt is None else float(dt)
        self._used_runtime_dt = runtime_dt
        self._ensure_terminal_controller(runtime_dt)
        self._solve_calls += 1
        solve_started = perf_counter()

        x0 = np.asarray(state, dtype=float).reshape(4)
        lqr_seed = self._lqr_seed_controls(x0, runtime_dt)
        controls = 0.5 * self._u_nominal.copy() + 0.5 * lqr_seed
        states = self._rollout(x0, controls, runtime_dt)
        best_cost = self._cost(states, controls, self._last_control)
        regularization = self.regularization
        iterations_run = 0

        for _ in range(self.max_iterations):
            iterations_run += 1
            A_seq = []
            B_seq = []
            for stage in range(self.horizon_steps):
                A, B = self._linearize(states[stage], controls[stage], runtime_dt)
                A_seq.append(A)
                B_seq.append(B)

            k_seq = np.zeros(self.horizon_steps, dtype=float)
            K_seq = np.zeros((self.horizon_steps, 1, 4), dtype=float)
            _, V_x, V_xx = self._terminal_terms(states[-1])
            backward_ok = True

            for stage in range(self.horizon_steps - 1, -1, -1):
                x_stage = states[stage]
                u_stage = float(controls[stage])
                previous_control = self._last_control if stage == 0 else float(controls[stage - 1])
                _, l_x, l_xx, l_u, l_uu = self._stage_terms(x_stage, u_stage, previous_control)
                A = A_seq[stage]
                B = B_seq[stage]

                Q_x = l_x + A.T @ V_x
                Q_u = np.array([l_u], dtype=float) + B.T @ V_x
                Q_xx = l_xx + A.T @ V_xx @ A
                Q_ux = B.T @ V_xx @ A
                Q_uu = np.array([[l_uu]], dtype=float) + B.T @ V_xx @ B + regularization * np.eye(1)

                try:
                    inv_Q_uu = np.linalg.inv(Q_uu)
                except np.linalg.LinAlgError:
                    backward_ok = False
                    break

                k = -inv_Q_uu @ Q_u
                K = -inv_Q_uu @ Q_ux
                k_seq[stage] = float(k.item())
                K_seq[stage] = K

                V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                V_xx = 0.5 * (V_xx + V_xx.T)

            if not backward_ok:
                regularization *= self.regularization_scale_up
                continue

            improved = False
            for alpha in self.line_search_alphas:
                candidate_controls = np.zeros_like(controls)
                candidate_states = np.zeros_like(states)
                candidate_states[0] = x0
                for stage in range(self.horizon_steps):
                    delta_x = candidate_states[stage] - states[stage]
                    feedback = float((K_seq[stage] @ delta_x).item())
                    candidate_u = controls[stage] + alpha * k_seq[stage] + feedback
                    candidate_controls[stage] = self.saturate(float(candidate_u))
                    candidate_states[stage + 1] = self._dynamics_step(candidate_states[stage], candidate_controls[stage], runtime_dt)

                if not self._states_within_guard(candidate_states) or not np.all(np.isfinite(candidate_controls)):
                    continue

                candidate_cost = self._cost(candidate_states, candidate_controls, self._last_control)
                if np.isfinite(candidate_cost) and candidate_cost < best_cost:
                    states = candidate_states
                    controls = candidate_controls
                    best_cost = candidate_cost
                    regularization = max(1e-8, regularization * self.regularization_scale_down)
                    improved = True
                    break

            if not improved:
                regularization *= self.regularization_scale_up
                if regularization > 1e6:
                    break

        solve_failed = not np.isfinite(best_cost) or not self._states_within_guard(states) or not np.all(np.isfinite(controls))
        if solve_failed:
            self._solver_failure_count += 1
            controls = np.zeros_like(self._u_nominal)
            states = self._rollout(x0, controls, runtime_dt)
            best_cost = self._cost(states, controls, self._last_control)

        issued_control = self.saturate(float(controls[0] if len(controls) else 0.0))
        self._u_nominal[:-1] = controls[1:]
        self._u_nominal[-1] = controls[-1] if len(controls) else 0.0
        self._last_control = issued_control
        self._max_abs_command = max(self._max_abs_command, abs(issued_control))
        self._solve_times_ms.append((perf_counter() - solve_started) * 1000.0)
        self._iteration_counts.append(iterations_run)
        self._objectives.append(float(best_cost))
        return issued_control
