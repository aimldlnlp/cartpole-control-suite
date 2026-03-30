from __future__ import annotations

import numpy as np

from cartpole_bench.config import load_estimator_config, load_system_params
from cartpole_bench.estimators.ekf import ExtendedKalmanFilter


def test_ekf_estimates_remain_finite_and_reduce_angle_error_on_synthetic_noisy_sequence() -> None:
    params = load_system_params()
    config = load_estimator_config("ekf")
    measurement_std = np.array([0.01, 0.015, np.deg2rad(2.5), 0.04], dtype=float)
    estimator = ExtendedKalmanFilter(config, params, measurement_std=measurement_std)
    true_state = np.array([0.02, 0.0, np.deg2rad(1.0), 0.0], dtype=float)
    estimator.reset(true_state)

    rng = np.random.default_rng(5)
    observations = np.asarray([true_state + rng.normal(0.0, measurement_std) for _ in range(60)], dtype=float)
    estimates = np.asarray([estimator.step(obs, 0.0, 0.01) for obs in observations], dtype=float)

    assert np.all(np.isfinite(estimates))
    obs_error = np.mean(np.abs(observations[:, 2] - true_state[2]))
    est_error = np.mean(np.abs(estimates[:, 2] - true_state[2]))
    assert est_error <= obs_error + 1e-9
