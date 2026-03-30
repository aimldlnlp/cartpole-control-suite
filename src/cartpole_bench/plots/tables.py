from __future__ import annotations

from pathlib import Path

from cartpole_bench.metrics.summary import write_metric_summaries
from cartpole_bench.simulation.recorder import load_saved_runs
from cartpole_bench.types import RunDiagnosis, RunMetrics, TrajectoryResult


def refresh_metric_tables(base_dir: Path) -> None:
    loaded = load_saved_runs(base_dir, suites={"nominal", "stress"})
    results = []
    for item in loaded:
        metadata = item["metadata"]
        frame = item["frame"]
        results.append(
            TrajectoryResult(
                controller_name=metadata["controller_name"],
                estimator_name=metadata.get("estimator_name", "none"),
                scenario_name=metadata["scenario_name"],
                suite_name=metadata["suite_name"],
                seed=int(metadata["seed"]),
                time=frame["t"].to_numpy(),
                states=frame[["x", "x_dot", "theta", "theta_dot"]].to_numpy(),
                observations=frame[["x_obs", "x_dot_obs", "theta_obs", "theta_dot_obs"]].to_numpy()
                if {"x_obs", "x_dot_obs", "theta_obs", "theta_dot_obs"}.issubset(frame.columns)
                else frame[["x", "x_dot", "theta", "theta_dot"]].to_numpy(),
                estimates=frame[["x_est", "x_dot_est", "theta_est", "theta_dot_est"]].to_numpy()
                if {"x_est", "x_dot_est", "theta_est", "theta_dot_est"}.issubset(frame.columns)
                else frame[["x", "x_dot", "theta", "theta_dot"]].to_numpy(),
                controls=frame["u"].to_numpy(),
                disturbances=frame["disturbance"].to_numpy() if "disturbance" in frame else frame["u"].to_numpy() * 0.0,
                modes=frame["mode"].astype(str).tolist(),
                metrics=RunMetrics(**metadata["metrics"]),
                diagnosis=RunDiagnosis(**metadata["diagnosis"]),
                invalid=bool(metadata.get("invalid", False)),
                track_violation=bool(metadata.get("track_violation", False)),
            )
        )
    write_metric_summaries(base_dir / "tables", results)
