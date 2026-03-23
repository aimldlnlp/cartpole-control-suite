from __future__ import annotations

from pathlib import Path
from typing import Any

from cartpole_bench.types import (
    CartPoleParams,
    ControllerConfig,
    DisturbanceConfig,
    NoiseConfig,
    RenderThemeConfig,
    ScenarioConfig,
    SwitchConfig,
    VideoRenderConfig,
)
from cartpole_bench.utils.io import load_json


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_system_params(config_root: Path = CONFIG_ROOT) -> CartPoleParams:
    payload = load_json(config_root / "system" / "default.json")
    return CartPoleParams(**payload)


def load_controller_config(name: str, config_root: Path = CONFIG_ROOT) -> ControllerConfig:
    payload = load_json(config_root / "controllers" / f"{name}.json")
    gains = {key: value for key, value in payload.items() if key != "force_limit"}
    return ControllerConfig(name=name, gains=gains, force_limit=payload["force_limit"])


def load_switch_config() -> SwitchConfig:
    return SwitchConfig()


def build_scenarios(payload: dict[str, Any], suite_name: str) -> list[ScenarioConfig]:
    noise_payload = payload.get("noise", {})
    disturbance_payload = payload.get("disturbance", {})
    if "seeds" in payload:
        seeds = [int(seed) for seed in payload["seeds"]]
    elif "repetitions" in payload:
        base_seed = int(payload.get("seed", 0))
        repetitions = int(payload["repetitions"])
        seeds = [base_seed + offset for offset in range(repetitions)]
    else:
        seeds = [int(payload["seed"])]

    scenarios = []
    for seed in seeds:
        scenarios.append(
            ScenarioConfig(
                name=payload["name"],
                suite_name=suite_name,
                horizon=payload["horizon"],
                dt=payload["dt"],
                initial_state=tuple(payload["initial_state"]),
                seed=seed,
                noise=NoiseConfig(state_std=tuple(noise_payload.get("state_std", [0.0] * 4))),
                disturbance=DisturbanceConfig(**disturbance_payload),
                plant_overrides=payload.get("plant_overrides", {}),
            )
        )
    return scenarios


def load_suite(name: str, config_root: Path = CONFIG_ROOT) -> list[ScenarioConfig]:
    payload = load_json(config_root / "experiments" / f"{name}.json")
    suite_name = payload["suite_name"]
    scenarios: list[ScenarioConfig] = []
    for item in payload["scenarios"]:
        scenarios.extend(build_scenarios(item, suite_name))
    return scenarios


def load_monte_carlo_config(config_root: Path = CONFIG_ROOT) -> dict[str, Any]:
    return load_json(config_root / "experiments" / "monte_carlo.json")


def load_theme_config(theme: str = "paper_white", config_root: Path = CONFIG_ROOT) -> RenderThemeConfig:
    payload = load_json(config_root / "rendering" / "theme.json")
    theme_payload = payload["themes"][theme]
    return RenderThemeConfig(name=theme, **theme_payload)


def load_video_config(config_root: Path = CONFIG_ROOT) -> VideoRenderConfig:
    payload = load_json(config_root / "rendering" / "video.json")
    return VideoRenderConfig(
        fps_mp4=int(payload["fps_mp4"]),
        fps_gif=int(payload["fps_gif"]),
        canvas_size_single=tuple(payload["canvas_size_single"]),
        canvas_size_comparison=tuple(payload["canvas_size_comparison"]),
        duration_profiles=payload["duration_profiles"],
    )
