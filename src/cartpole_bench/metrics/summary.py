from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cartpole_bench import ARTIFACT_VERSION
from cartpole_bench.types import BatchSummary, TrajectoryResult
from cartpole_bench.utils.io import save_json


def results_to_frame(results: Iterable[TrajectoryResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "suite": result.suite_name,
                "scenario": result.scenario_name,
                "controller": result.controller_name,
                "estimator": result.estimator_name,
                "settling_time": result.metrics.settling_time,
                "overshoot_deg": result.metrics.overshoot_deg,
                "steady_state_error_deg": result.metrics.steady_state_error_deg,
                "control_effort": result.metrics.control_effort,
                "success": result.metrics.success,
                "success_rate": float(result.metrics.success),
                "max_abs_force": result.metrics.max_abs_force,
                "final_abs_theta_deg": result.metrics.final_abs_theta_deg,
                "switch_time": result.metrics.switch_time,
                "failure_reason": result.diagnosis.failure_reason,
                "first_balance_time": result.diagnosis.first_balance_time,
                "balance_fraction": result.diagnosis.balance_fraction,
                "min_abs_theta_deg": result.diagnosis.min_abs_theta_deg,
                "time_of_min_abs_theta": result.diagnosis.time_of_min_abs_theta,
                "max_abs_x": result.diagnosis.max_abs_x,
                "max_abs_theta_dot": result.diagnosis.max_abs_theta_dot,
                "handoff": float(result.diagnosis.first_balance_time is not None),
                "invalid": float(result.invalid),
                "track_violation": float(result.track_violation),
                "seed": result.seed,
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_to_frame(summaries: Iterable[BatchSummary]) -> pd.DataFrame:
    return pd.DataFrame([summary.to_dict() for summary in summaries])


def _safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return None
    return float(finite.mean())


def _safe_median(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return None
    return float(finite.median())


def aggregate_metric_table(results: Iterable[TrajectoryResult]) -> pd.DataFrame:
    frame = results_to_frame(results)
    if frame.empty:
        return frame
    rows = []
    grouped = frame.groupby(["suite", "scenario", "controller", "estimator"], dropna=False, sort=True)
    for (suite, scenario, controller, estimator), group in grouped:
        rows.append(
            {
                "suite": suite,
                "scenario": scenario,
                "controller": controller,
                "estimator": estimator,
                "runs": int(group["seed"].count()),
                "success_count": int(pd.to_numeric(group["success"], errors="coerce").fillna(0.0).sum()),
                "success_rate": _safe_mean(group["success_rate"]),
                "handoff_rate": _safe_mean(group["handoff"]),
                "settling_time_median": _safe_median(group["settling_time"]),
                "overshoot_deg_median": _safe_median(group["overshoot_deg"]),
                "steady_state_error_deg_median": _safe_median(group["steady_state_error_deg"]),
                "control_effort_median": _safe_median(group["control_effort"]),
                "first_balance_time_median": _safe_median(group["first_balance_time"]),
                "min_abs_theta_deg_median": _safe_median(group["min_abs_theta_deg"]),
                "balance_fraction_median": _safe_median(group["balance_fraction"]),
                "invalid_rate": _safe_mean(group["invalid"]),
                "track_violation_rate": _safe_mean(group["track_violation"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["suite", "scenario", "controller", "estimator"], ignore_index=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "| empty |\n| --- |\n| no data |"
    headers = list(frame.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
                continue
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator, *body])


def write_metric_summaries(output_dir: Path, results: Iterable[TrajectoryResult]) -> pd.DataFrame:
    frame = aggregate_metric_table(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metric_summary.csv"
    json_path = output_dir / "metric_summary.json"
    md_path = output_dir / "metric_summary.md"
    frame.to_csv(csv_path, index=False)
    save_json(
        json_path,
        {
            "artifact_version": ARTIFACT_VERSION,
            "rows": frame.to_dict(orient="records"),
        },
    )
    md_path.write_text(_markdown_table(frame), encoding="utf-8")
    return frame


def write_monte_carlo_summary(output_dir: Path, summaries: Iterable[BatchSummary]) -> pd.DataFrame:
    frame = monte_carlo_to_frame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "monte_carlo_summary.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if {"controller_name", "estimator_name"}.issubset(existing.columns) and not frame.empty:
            pairs = set(zip(frame["controller_name"], frame["estimator_name"], strict=True))
            existing = existing[
                ~existing.apply(lambda row: (row["controller_name"], row["estimator_name"]) in pairs, axis=1)
            ]
            frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(csv_path, index=False)
    save_json(
        output_dir / "monte_carlo_summary.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "rows": frame.to_dict(orient="records"),
        },
    )
    return frame
