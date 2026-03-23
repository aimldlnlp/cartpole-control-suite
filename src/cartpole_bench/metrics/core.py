from __future__ import annotations

import numpy as np

from cartpole_bench.types import RunDiagnosis, RunMetrics
from cartpole_bench.utils.math import rad2deg, wrap_angle


STATE_THRESHOLDS = {
    "x": 0.05,
    "x_dot": 0.1,
    "theta": np.deg2rad(2.0),
    "theta_dot": 0.2,
}


def stable_window_mask(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    return (
        (np.abs(states[:, 0]) < STATE_THRESHOLDS["x"])
        & (np.abs(states[:, 1]) < STATE_THRESHOLDS["x_dot"])
        & (np.abs(wrap_angle(states[:, 2])) < STATE_THRESHOLDS["theta"])
        & (np.abs(states[:, 3]) < STATE_THRESHOLDS["theta_dot"])
    )


def compute_settling_time(time: np.ndarray, states: np.ndarray, window_seconds: float = 1.0) -> float | None:
    if len(time) < 2:
        return None
    dt = float(time[1] - time[0])
    window_steps = max(1, int(round(window_seconds / dt)))
    mask = stable_window_mask(states).astype(int)
    if mask.size < window_steps:
        return None
    rolling = np.convolve(mask, np.ones(window_steps, dtype=int), mode="valid")
    indices = np.flatnonzero(rolling == window_steps)
    if indices.size == 0:
        return None
    return float(time[int(indices[0])])


def _evaluation_start_index(time: np.ndarray, switch_time: float | None, first_balance_time: float | None) -> int:
    if len(time) == 0:
        return 0
    start_time = first_balance_time if first_balance_time is not None else switch_time
    if start_time is None:
        return 0
    return int(np.clip(np.searchsorted(time, start_time, side="left"), 0, len(time) - 1))


def compute_run_metrics(
    time: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    modes: list[str],
    track_limit: float,
    switch_time: float | None,
    invalid: bool,
    track_violation: bool,
) -> tuple[RunMetrics, RunDiagnosis]:
    time = np.asarray(time, dtype=float)
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    dt = float(time[1] - time[0]) if len(time) > 1 else 0.0
    abs_theta_deg = np.abs(rad2deg(wrap_angle(states[:, 2])))
    max_abs_x = float(np.max(np.abs(states[:, 0]))) if len(states) else float("nan")
    max_abs_theta_dot = float(np.max(np.abs(states[:, 3]))) if len(states) else float("nan")
    min_idx = int(np.argmin(abs_theta_deg)) if len(abs_theta_deg) else 0
    balance_mask = np.asarray([mode == "balance" for mode in modes], dtype=bool)
    capture_mask = np.asarray([mode == "capture_assist" for mode in modes], dtype=bool)
    first_balance_time = float(time[int(np.flatnonzero(balance_mask)[0])]) if np.any(balance_mask) else None
    balance_fraction = float(np.mean(balance_mask)) if len(balance_mask) else 0.0
    eval_start = _evaluation_start_index(time, switch_time, first_balance_time)
    eval_time = time[eval_start:]
    eval_states = states[eval_start:]

    settling_time = compute_settling_time(eval_time, eval_states)
    if settling_time is not None and len(eval_time):
        settling_time -= float(eval_time[0])
    overshoot_deg = (
        float(np.max(np.abs(rad2deg(wrap_angle(eval_states[:, 2]))))) if len(eval_states) else float("nan")
    )

    final_steps = max(1, int(round(1.0 / dt))) if dt > 0.0 else 1
    final_slice = states[-final_steps:]
    steady_state_error_deg = float(np.mean(np.abs(rad2deg(wrap_angle(final_slice[:, 2])))))
    control_effort = float(np.sum(np.square(controls)) * dt)
    max_abs_force = float(np.max(np.abs(controls))) if len(controls) else 0.0
    final_abs_theta_deg = float(abs(rad2deg(wrap_angle(states[-1, 2])))) if len(states) else float("nan")
    final_stable = bool(np.all(stable_window_mask(final_slice)))

    success = bool(
        not invalid
        and not track_violation
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(controls))
        and settling_time is not None
        and final_stable
    )

    if success:
        failure_reason = None
    elif invalid:
        failure_reason = "invalid_state_or_control"
    elif track_violation:
        failure_reason = "track_limit_exceeded"
    elif first_balance_time is None and np.any(capture_mask):
        failure_reason = "capture_assist_no_handoff"
    elif first_balance_time is None:
        failure_reason = "no_balance_handoff"
    elif settling_time is None:
        failure_reason = "did_not_meet_settling_window"
    elif not final_stable:
        failure_reason = "left_stable_region"
    else:
        failure_reason = "unknown_failure"

    metrics = RunMetrics(
        settling_time=settling_time,
        overshoot_deg=overshoot_deg,
        steady_state_error_deg=steady_state_error_deg,
        control_effort=control_effort,
        success=success,
        max_abs_force=max_abs_force,
        final_abs_theta_deg=final_abs_theta_deg,
        switch_time=switch_time,
    )
    diagnosis = RunDiagnosis(
        failure_reason=failure_reason,
        first_balance_time=first_balance_time,
        balance_fraction=balance_fraction,
        min_abs_theta_deg=float(abs_theta_deg[min_idx]) if len(abs_theta_deg) else float("nan"),
        time_of_min_abs_theta=float(time[min_idx]) if len(time) else None,
        max_abs_x=max_abs_x,
        max_abs_theta_dot=max_abs_theta_dot,
    )
    return metrics, diagnosis
