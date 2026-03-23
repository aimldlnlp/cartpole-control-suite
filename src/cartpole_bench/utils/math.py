from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def smooth_sat(value: float | np.ndarray, boundary_layer: float) -> float | np.ndarray:
    if boundary_layer <= 0.0:
        return np.sign(value)
    return np.tanh(np.asarray(value) / boundary_layer)


def deg2rad(value: float) -> float:
    return math.radians(value)


def rad2deg(value: float | np.ndarray) -> float | np.ndarray:
    return np.rad2deg(value)


def finite_difference_jacobian(
    func,
    x: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    base = np.asarray(func(x), dtype=float)
    jac = np.zeros((base.size, x.size), dtype=float)
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = epsilon
        upper = np.asarray(func(x + delta), dtype=float)
        lower = np.asarray(func(x - delta), dtype=float)
        jac[:, index] = (upper - lower) / (2.0 * epsilon)
    return jac
