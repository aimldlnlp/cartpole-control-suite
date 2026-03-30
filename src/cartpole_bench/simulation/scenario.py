from __future__ import annotations

from dataclasses import replace

from cartpole_bench.config import load_controller_config, load_switch_config
from cartpole_bench.controllers.base import BaseController
from cartpole_bench.controllers.hybrid import HybridController
from cartpole_bench.controllers.ilqr import IterativeLQRController
from cartpole_bench.controllers.lqr import LQRController
from cartpole_bench.controllers.mpc import ModelPredictiveController
from cartpole_bench.controllers.pfl import PartialFeedbackLinearizationController
from cartpole_bench.controllers.smc import SlidingModeController
from cartpole_bench.controllers.swingup import EnergySwingUpController
from cartpole_bench.dynamics.cartpole import CartPoleDynamics
from cartpole_bench.types import CartPoleParams, ControllerConfig, ScenarioConfig


CONTROLLER_KEYS = ("lqr", "pfl", "smc", "ilqr", "mpc")


def controller_label(key: str) -> str:
    return {
        "lqr": "LQR",
        "pfl": "Feedback Linearization (PFL)",
        "smc": "Sliding Mode Control (SMC)",
        "ilqr": "Iterative LQR (iLQR)",
        "mpc": "Model Predictive Control (MPC)",
    }[key]


def resolve_plant_params(base_params: CartPoleParams, scenario: ScenarioConfig) -> CartPoleParams:
    if not scenario.plant_overrides:
        return base_params
    return replace(base_params, **scenario.plant_overrides)


def build_stabilizer(
    key: str,
    model_params: CartPoleParams,
    controller_override: ControllerConfig | None = None,
) -> tuple[BaseController, dict]:
    if key == "lqr":
        config = controller_override or load_controller_config("lqr")
        return LQRController(config, model_params), config.to_dict()
    if key == "pfl":
        config = controller_override or load_controller_config("pfl")
        lqr_config = load_controller_config("lqr")
        return PartialFeedbackLinearizationController(config, model_params, lqr_config), config.to_dict()
    if key == "smc":
        smc_config = controller_override or load_controller_config("smc")
        lqr_config = load_controller_config("lqr")
        return SlidingModeController(smc_config, model_params, lqr_config), smc_config.to_dict()
    if key == "ilqr":
        config = controller_override or load_controller_config("ilqr")
        return IterativeLQRController(config, model_params), config.to_dict()
    if key == "mpc":
        config = controller_override or load_controller_config("mpc")
        return ModelPredictiveController(config, model_params), config.to_dict()
    raise ValueError(f"Unknown controller key: {key}")


def build_hybrid_controller(
    key: str,
    model_params: CartPoleParams,
    controller_override: ControllerConfig | None = None,
    swingup_override: ControllerConfig | None = None,
    switch_override=None,
) -> tuple[HybridController, dict]:
    swing_config = swingup_override or load_controller_config("swingup")
    lqr_config = load_controller_config("lqr")
    swingup = EnergySwingUpController(swing_config, CartPoleDynamics(model_params), lqr_config)
    stabilizer, controller_cfg = build_stabilizer(key, model_params, controller_override=controller_override)
    switch_config = switch_override or load_switch_config()
    hybrid = HybridController(swingup, stabilizer, switch_config)
    return hybrid, {
        "controller": controller_cfg,
        "swingup": swing_config.to_dict(),
        "switch": hybrid.switch_config.to_dict(),
    }
