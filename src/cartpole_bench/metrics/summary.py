from __future__ import annotations

from pathlib import Path
from typing import Iterable

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


def aggregate_metric_table(results: Iterable[TrajectoryResult]) -> pd.DataFrame:
    frame = results_to_frame(results)
    if frame.empty:
        return frame
    return (
        frame.groupby(["suite", "scenario", "controller"], as_index=False)
        .agg(
            runs=("seed", "count"),
            success_count=("success", "sum"),
            success_rate=("success_rate", "mean"),
            handoff_rate=("handoff", "mean"),
            settling_time_median=("settling_time", "median"),
            overshoot_deg_median=("overshoot_deg", "median"),
            steady_state_error_deg_median=("steady_state_error_deg", "median"),
            control_effort_median=("control_effort", "median"),
            first_balance_time_median=("first_balance_time", "median"),
            min_abs_theta_deg_median=("min_abs_theta_deg", "median"),
            balance_fraction_median=("balance_fraction", "median"),
            invalid_rate=("invalid", "mean"),
            track_violation_rate=("track_violation", "mean"),
        )
        .sort_values(["suite", "scenario", "controller"])
    )


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
    frame.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    save_json(
        output_dir / "monte_carlo_summary.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "rows": frame.to_dict(orient="records"),
        },
    )
    return frame
