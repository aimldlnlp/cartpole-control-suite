from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cartpole_bench import ARTIFACT_VERSION
from cartpole_bench.types import CartPoleParams, ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.io import load_json, save_json, try_git_commit


def result_stem(result: TrajectoryResult) -> str:
    scenario = result.scenario_name.replace(" ", "_")
    controller = (
        result.controller_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    return f"{result.suite_name}__{scenario}__{controller}__seed{result.seed}"


def trajectory_frame(result: TrajectoryResult) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "t": result.time,
            "x": result.states[:, 0],
            "x_dot": result.states[:, 1],
            "theta": result.states[:, 2],
            "theta_dot": result.states[:, 3],
            "u": result.controls,
            "disturbance": result.disturbances,
            "mode": result.modes,
            "controller": result.controller_name,
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

    trajectory_frame(result).to_csv(csv_path, index=False)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "controller_name": result.controller_name,
        "scenario_name": result.scenario_name,
        "suite_name": result.suite_name,
        "seed": result.seed,
        "plant_params": plant_params.to_dict(),
        "controller_config": controller_payload["controller"],
        "swingup_config": controller_payload["swingup"],
        "switch_config": controller_payload["switch"],
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
            loaded.append(
                {
                    "manifest": manifest,
                    "summary": run,
                    "metadata": metadata,
                    "frame": pd.read_csv(csv_path),
                }
            )
    return loaded
