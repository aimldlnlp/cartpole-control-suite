from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CartPoleParams:
    M: float
    m: float
    l: float
    g: float
    cart_friction: float
    pivot_damping: float
    force_limit: float
    track_limit: float
    pendulum_inertia: float = 0.0

    @property
    def pendulum_mass_matrix(self) -> float:
        return self.pendulum_inertia + self.m * self.l * self.l

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NoiseConfig:
    state_std: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.state_std, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"state_std": list(self.state_std)}


@dataclass(slots=True)
class DisturbanceConfig:
    kind: str = "none"
    magnitude: float = 0.0
    start_time: float = 0.0
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SwitchConfig:
    capture_min_angle_deg: float = 15.0
    capture_entry_angle_deg: float = 55.0
    capture_release_angle_deg: float = 65.0
    capture_energy_tolerance_ratio: float = 0.35
    enter_angle_deg: float = 12.0
    enter_rate: float = 1.0
    enter_hold_time: float = 0.15
    exit_angle_deg: float = 20.0
    exit_hold_time: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ControllerConfig:
    name: str
    gains: dict[str, Any]
    force_limit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EstimatorConfig:
    name: str
    gains: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationConfig:
    horizon: float
    dt: float
    seed: int
    force_limit: float
    track_limit: float
    estimator_name: str = "none"

    @property
    def steps(self) -> int:
        return int(round(self.horizon / self.dt))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    suite_name: str
    horizon: float
    dt: float
    initial_state: tuple[float, float, float, float]
    seed: int
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    plant_overrides: dict[str, Any] = field(default_factory=dict)

    def simulation_config(self, params: CartPoleParams, estimator_name: str = "none") -> SimulationConfig:
        return SimulationConfig(
            horizon=self.horizon,
            dt=self.dt,
            seed=self.seed,
            force_limit=params.force_limit,
            track_limit=params.track_limit,
            estimator_name=estimator_name,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["noise"] = self.noise.to_dict()
        payload["disturbance"] = self.disturbance.to_dict()
        return payload


@dataclass(slots=True)
class RunDiagnosis:
    failure_reason: str | None
    first_balance_time: float | None
    balance_fraction: float
    min_abs_theta_deg: float
    time_of_min_abs_theta: float | None
    max_abs_x: float
    max_abs_theta_dot: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StepResult:
    t: float
    state: np.ndarray
    control: float
    mode: str
    disturbance: float
    observed_state: np.ndarray


@dataclass(slots=True)
class RunMetrics:
    settling_time: float | None
    overshoot_deg: float
    steady_state_error_deg: float
    control_effort: float
    success: bool
    max_abs_force: float
    final_abs_theta_deg: float
    switch_time: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryResult:
    controller_name: str
    estimator_name: str
    scenario_name: str
    suite_name: str
    seed: int
    time: np.ndarray
    states: np.ndarray
    observations: np.ndarray
    estimates: np.ndarray
    controls: np.ndarray
    disturbances: np.ndarray
    modes: list[str]
    metrics: RunMetrics
    diagnosis: RunDiagnosis
    invalid: bool = False
    track_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_name": self.controller_name,
            "estimator_name": self.estimator_name,
            "scenario_name": self.scenario_name,
            "suite_name": self.suite_name,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "diagnosis": self.diagnosis.to_dict(),
            "invalid": self.invalid,
            "track_violation": self.track_violation,
        }


@dataclass(slots=True)
class BatchSummary:
    controller_name: str
    estimator_name: str
    samples: int
    success_rate: float
    success_count: int
    median_settling_time: float | None
    median_control_effort: float
    median_steady_state_error_deg: float
    invalid_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RenderThemeConfig:
    name: str
    font_family: str
    dpi: int
    line_width: float
    axes_line_width: float
    background_color: str
    panel_color: str
    text_color: str
    grid_color: str
    spine_color: str
    accent_color: str
    muted_color: str
    controller_colors: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoRenderConfig:
    fps_mp4: int
    fps_gif: int
    canvas_size_single: tuple[float, float]
    canvas_size_comparison: tuple[float, float]
    duration_profiles: dict[str, dict[str, float]]

    def profile(self, name: str) -> dict[str, float]:
        if name not in self.duration_profiles:
            available = ", ".join(sorted(self.duration_profiles))
            raise KeyError(f"Unknown duration profile '{name}'. Available: {available}")
        return self.duration_profiles[name]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
