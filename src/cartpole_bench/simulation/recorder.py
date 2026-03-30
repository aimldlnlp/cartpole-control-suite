from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cartpole_bench import ARTIFACT_VERSION
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.types import CartPoleParams, ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.io import load_json, save_json, try_git_commit
from cartpole_bench.utils.math import wrap_angle


def result_stem(result: TrajectoryResult) -> str:
    scenario = result.scenario_name.replace(" ", "_")
    controller = (
        result.controller_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    estimator = result.estimator_name.lower().replace(" ", "_")
    return f"{result.suite_name}__{scenario}__{controller}__{estimator}__seed{result.seed}"


def trajectory_frame(
    result: TrajectoryResult,
    plant_params: CartPoleParams,
    controller_payload: dict[str, Any],
) -> pd.DataFrame:
    dynamics = CartPoleDynamics(plant_params)
    switch_cfg = controller_payload["switch"]
    swing_cfg = controller_payload.get("swingup", {})
    theta = result.states[:, 2]
    theta_est = result.estimates[:, 2]
    enter_angle_deg = float(switch_cfg.get("enter_angle_deg", 12.0))
    enter_rate = float(switch_cfg.get("enter_rate", 1.0))
    capture_min_angle_deg = float(switch_cfg.get("capture_min_angle_deg", 15.0))
    capture_release_angle_deg = float(switch_cfg.get("capture_release_angle_deg", 65.0))
    capture_rate_limit = float(swing_cfg.get("gains", {}).get("capture_rate_limit", swing_cfg.get("capture_rate_limit", 2.8)))
    force_limit = float(controller_payload["controller"].get("force_limit", plant_params.force_limit))
    state_theta_deg = np.rad2deg(np.arctan2(np.sin(theta), np.cos(theta)))
    estimate_theta_deg = np.rad2deg(np.arctan2(np.sin(theta_est), np.cos(theta_est)))
    energy_gap = np.asarray(
        [dynamics.desired_upright_energy() - dynamics.pendulum_energy_from_downward(state) for state in result.states],
        dtype=float,
    )
    track_margin = np.asarray(plant_params.track_limit - np.abs(result.states[:, 0]), dtype=float)
    force_saturated = np.asarray(np.abs(np.abs(result.controls) - force_limit) <= 1e-6, dtype=float)
    in_balance_gate = np.asarray(
        [
            (
                abs(float(state[2])) < np.deg2rad(enter_angle_deg)
                and abs(float(state[3])) < enter_rate
            )
            for state in result.states
        ],
        dtype=float,
    )
    in_capture_window = np.asarray(
        [
            (
                capture_min_angle_deg <= abs(np.rad2deg(float(state[2])))
                <= capture_release_angle_deg
                and abs(float(state[3])) <= capture_rate_limit
            )
            for state in result.states
        ],
        dtype=float,
    )
    frame = pd.DataFrame(
        {
            "t": result.time,
            "x": result.states[:, 0],
            "x_dot": result.states[:, 1],
            "theta": result.states[:, 2],
            "theta_dot": result.states[:, 3],
            "x_obs": result.observations[:, 0],
            "x_dot_obs": result.observations[:, 1],
            "theta_obs": result.observations[:, 2],
            "theta_dot_obs": result.observations[:, 3],
            "x_est": result.estimates[:, 0],
            "x_dot_est": result.estimates[:, 1],
            "theta_est": result.estimates[:, 2],
            "theta_dot_est": result.estimates[:, 3],
            "theta_deg": state_theta_deg,
            "theta_est_deg": estimate_theta_deg,
            "u": result.controls,
            "disturbance": result.disturbances,
            "energy_gap": energy_gap,
            "track_margin": track_margin,
            "force_saturated": force_saturated,
            "in_balance_gate": in_balance_gate,
            "in_capture_window": in_capture_window,
            "mode": result.modes,
            "controller": result.controller_name,
            "estimator": result.estimator_name,
            "scenario": result.scenario_name,
            "suite": result.suite_name,
            "seed": result.seed,
        }
    )
    return frame


def save_run_artifacts(
    result: TrajectoryResult,
    scenario: ScenarioConfig,
    plant_params: CartPoleParams,
    controller_payload: dict[str, Any],
    roots: dict[str, Path],
    cwd: Path,
) -> dict[str, str]:
    stem = result_stem(result)
    csv_path = roots["csv"] / f"{stem}.csv"
    json_path = roots["json"] / f"{stem}.json"

    trajectory_frame(result, plant_params, controller_payload).to_csv(csv_path, index=False)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "controller_name": result.controller_name,
        "estimator_name": result.estimator_name,
        "scenario_name": result.scenario_name,
        "suite_name": result.suite_name,
        "seed": result.seed,
        "plant_params": plant_params.to_dict(),
        "controller_config": controller_payload["controller"],
        "swingup_config": controller_payload["swingup"],
        "switch_config": controller_payload["switch"],
        "estimator_config": controller_payload.get("estimator", {"name": "none", "gains": {}}),
        "controller_debug": controller_payload.get("controller_debug", {}),
        "scenario_config": scenario.to_dict(),
        "metrics": result.metrics.to_dict(),
        "diagnosis": result.diagnosis.to_dict(),
        "invalid": result.invalid,
        "track_violation": result.track_violation,
        "csv_path": str(csv_path.relative_to(roots["base"])),
        "render_paths": {},
        "git_commit": try_git_commit(cwd),
    }
    save_json(json_path, metadata)
    return {
        "csv_path": str(csv_path.relative_to(roots["base"])),
        "json_path": str(json_path.relative_to(roots["base"])),
    }


def write_manifest(roots: dict[str, Path], name: str, payload: dict[str, Any]) -> Path:
    path = roots["json"] / f"manifest_{name}.json"
    save_json(path, {**payload, "artifact_version": ARTIFACT_VERSION})
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    return load_json(path)


def update_metadata_render_paths(base_dir: Path, relative_json_path: str, render_paths: dict[str, str]) -> None:
    json_path = base_dir / relative_json_path
    metadata = load_json(json_path)
    metadata.setdefault("artifact_version", ARTIFACT_VERSION)
    metadata.setdefault("render_paths", {}).update(render_paths)
    save_json(json_path, metadata)


def prune_metadata_render_paths(base_dir: Path, relative_json_path: str, stale_relative_paths: set[str]) -> None:
    if not stale_relative_paths:
        return
    json_path = base_dir / relative_json_path
    metadata = load_json(json_path)
    metadata.setdefault("artifact_version", ARTIFACT_VERSION)
    render_paths = metadata.setdefault("render_paths", {})
    stale_keys = [key for key, relative_path in render_paths.items() if relative_path in stale_relative_paths]
    for key in stale_keys:
        render_paths.pop(key, None)
    save_json(json_path, metadata)


def _normalized_diagnosis(metadata: dict[str, Any]) -> dict[str, Any]:
    diagnosis = dict(metadata.get("diagnosis", {}))
    metrics = metadata.get("metrics", {})
    diagnosis.setdefault("failure_reason", None)
    diagnosis.setdefault("first_balance_time", metrics.get("switch_time"))
    diagnosis.setdefault("balance_fraction", 0.0)
    diagnosis.setdefault("min_abs_theta_deg", metrics.get("final_abs_theta_deg", float("nan")))
    diagnosis.setdefault("time_of_min_abs_theta", None)
    diagnosis.setdefault("max_abs_x", float("nan"))
    diagnosis.setdefault("max_abs_theta_dot", float("nan"))
    return diagnosis


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    normalized.setdefault("artifact_version", "legacy")
    normalized["diagnosis"] = _normalized_diagnosis(metadata)
    normalized.setdefault("invalid", False)
    normalized.setdefault("track_violation", False)
    normalized.setdefault("render_paths", {})
    normalized.setdefault("estimator_name", "none")
    normalized.setdefault("estimator_config", {"name": "none", "gains": {}})
    normalized.setdefault("controller_debug", {})
    return normalized


def load_saved_runs(base_dir: Path, suites: set[str] | None = None) -> list[dict[str, Any]]:
    json_dir = base_dir / "json"
    manifests = sorted(json_dir.glob("manifest_*.json"))
    loaded: list[dict[str, Any]] = []
    for manifest_path in manifests:
        manifest = load_json(manifest_path)
        manifest.setdefault("artifact_version", "legacy")
        suite = manifest.get("suite")
        if suites is not None and suite not in suites:
            continue
        for run in manifest.get("runs", []):
            csv_path = base_dir / run["csv_path"]
            json_path = base_dir / run["json_path"]
            metadata = _normalize_metadata(load_json(json_path))
            frame = pd.read_csv(csv_path)
            legacy_pairs = {
                "x_obs": "x",
                "x_dot_obs": "x_dot",
                "theta_obs": "theta",
                "theta_dot_obs": "theta_dot",
                "x_est": "x",
                "x_dot_est": "x_dot",
                "theta_est": "theta",
                "theta_dot_est": "theta_dot",
                "theta_deg": None,
                "theta_est_deg": None,
                "energy_gap": None,
                "track_margin": None,
                "force_saturated": None,
                "in_balance_gate": None,
                "in_capture_window": None,
                "estimator": None,
            }
            for column, source in legacy_pairs.items():
                if column in frame:
                    continue
                if source is None:
                    if column == "theta_deg":
                        frame[column] = np.rad2deg(np.arctan2(np.sin(frame["theta"]), np.cos(frame["theta"])))
                    elif column == "theta_est_deg":
                        frame[column] = np.rad2deg(np.arctan2(np.sin(frame["theta_est"]), np.cos(frame["theta_est"])))
                    elif column == "track_margin":
                        track_limit = float(metadata.get("plant_params", {}).get("track_limit", 0.0))
                        frame[column] = track_limit - frame["x"].abs()
                    elif column == "force_saturated":
                        force_limit = float(metadata.get("plant_params", {}).get("force_limit", 0.0))
                        frame[column] = (frame["u"].abs() >= force_limit - 1e-6).astype(float)
                    elif column == "estimator":
                        frame[column] = metadata.get("estimator_name", "none")
                    else:
                        frame[column] = 0.0
                else:
                    frame[column] = frame[source]
            loaded.append(
                {
                    "manifest": manifest,
                    "summary": run,
                    "metadata": metadata,
                    "frame": frame,
                }
            )
    return loaded
