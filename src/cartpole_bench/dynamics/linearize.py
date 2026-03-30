from __future__ import annotations

import numpy as np
from scipy import linalg, signal

from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.types import CartPoleParams
from cartpole_bench.utils.math import finite_difference_jacobian


def upright_state_space(params: CartPoleParams) -> tuple[np.ndarray, np.ndarray]:
    a = params.M + params.m
    coupling = params.m * params.l
    inertia = params.pendulum_mass_matrix
    det = a * inertia - coupling * coupling
    if abs(det) < 1e-9:
        raise FloatingPointError("Linearization is singular for the current parameters.")

    A = np.zeros((4, 4), dtype=float)
    B = np.zeros((4, 1), dtype=float)

    A[0, 1] = 1.0
    A[2, 3] = 1.0
    A[1, 1] = -(inertia * params.cart_friction) / det
    A[1, 2] = (coupling * params.m * params.g * params.l) / det
    A[1, 3] = -(coupling * params.pivot_damping) / det
    A[3, 1] = -(coupling * params.cart_friction) / det
    A[3, 2] = (a * params.m * params.g * params.l) / det
    A[3, 3] = -(a * params.pivot_damping) / det

    B[1, 0] = inertia / det
    B[3, 0] = coupling / det
    return A, B


def discrete_lqr_terminal_cost(
    params: CartPoleParams,
    dt: float,
    Q: np.ndarray,
    R: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A_c, B_c = upright_state_space(params)
    A_d, B_d, _, _, _ = signal.cont2discrete((A_c, B_c, np.eye(4), np.zeros((4, 1))), dt)
    A_d = np.asarray(A_d, dtype=float)
    B_d = np.asarray(B_d, dtype=float).reshape(4, 1)
    R_matrix = np.asarray([[float(R)]], dtype=float) if np.isscalar(R) else np.asarray(R, dtype=float)
    P = np.asarray(linalg.solve_discrete_are(A_d, B_d, np.asarray(Q, dtype=float), R_matrix), dtype=float)
    K = np.asarray(np.linalg.solve(B_d.T @ P @ B_d + R_matrix, B_d.T @ P @ A_d), dtype=float)
    return A_d, B_d, P, K


def finite_difference_state_space(
    dynamics: CartPoleDynamics,
    equilibrium: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    equilibrium = np.zeros(4, dtype=float) if equilibrium is None else np.asarray(equilibrium, dtype=float)

    def state_func(x: np.ndarray) -> np.ndarray:
        return dynamics.derivatives(x, 0.0, 0.0)

    def input_func(u_array: np.ndarray) -> np.ndarray:
        return dynamics.derivatives(equilibrium, float(u_array[0]), 0.0)

    A = finite_difference_jacobian(state_func, equilibrium)
    B = finite_difference_jacobian(input_func, np.array([0.0], dtype=float))
    return A, B
