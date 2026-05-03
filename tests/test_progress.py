from __future__ import annotations

from pathlib import Path

import numpy as np

import cartpole_bench.animations.render as render_module
import cartpole_bench.plots.figures as figures_module
import cartpole_bench.simulation.batch as batch_module
import cartpole_bench.simulation.runner as runner_module
from cartpole_bench.config import load_suite
from cartpole_bench.plots.figures import generate_figures
from cartpole_bench.simulation.batch import run_monte_carlo
from cartpole_bench.simulation.runner import run_suite, simulate_trajectory
from cartpole_bench.simulation.tuning import tune_controller
from cartpole_bench.types import RunDiagnosis, RunMetrics, ScenarioConfig, TrajectoryResult
from cartpole_bench.utils.progress import LineProgressReporter, ProgressEvent, format_eta, format_percent


class CollectingReporter:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _fake_result(controller_name: str = "LQR", estimator_name: str = "none", seed: int = 0) -> TrajectoryResult:
    return TrajectoryResult(
        controller_name=controller_name,
        estimator_name=estimator_name,
        scenario_name="local_small_angle",
        suite_name="nominal",
        seed=seed,
        time=np.asarray([0.0, 0.1]),
        states=np.zeros((2, 4), dtype=float),
        observations=np.zeros((2, 4), dtype=float),
        estimates=np.zeros((2, 4), dtype=float),
        controls=np.zeros(2, dtype=float),
        disturbances=np.zeros(2, dtype=float),
        modes=["initial", "balance"],
        metrics=RunMetrics(
            settling_time=0.2,
            overshoot_deg=0.1,
            steady_state_error_deg=0.1,
            control_effort=0.2,
            success=True,
            max_abs_force=0.3,
            final_abs_theta_deg=0.1,
            switch_time=0.1,
        ),
        diagnosis=RunDiagnosis(
            failure_reason=None,
            first_balance_time=0.1,
            balance_fraction=1.0,
            min_abs_theta_deg=0.1,
            time_of_min_abs_theta=0.1,
            max_abs_x=0.02,
            max_abs_theta_dot=0.05,
        ),
        invalid=False,
        track_violation=False,
    )


def test_format_eta_and_line_progress_reporter(capsys) -> None:
    assert format_eta(0.2) == "<1s"
    assert format_eta(12.4) == "12.4s"
    assert format_eta(192.0) == "3m12s"
    assert format_eta(479.6) == "8m00s"
    assert format_eta(3840.0) == "1h04m"
    assert format_percent(1, 3) == "33%"
    assert format_percent(3, 3) == "100%"

    reporter = LineProgressReporter()
    reporter.emit(
        ProgressEvent(
            domain="suite",
            stage="start",
            current=1,
            total=3,
            elapsed_s=12.4,
            eta_s=48.0,
            context={"suite": "nominal", "controller": "lqr", "seed": 3},
        )
    )
    captured = capsys.readouterr()
    assert "[suite 33% 1/3] start" in captured.err
    assert "suite=nominal" in captured.err
    assert "controller=lqr" in captured.err
    assert "elapsed=12.4s" in captured.err
    assert "eta=48.0s" in captured.err


def test_simulate_trajectory_emits_heartbeat_events() -> None:
    scenario = load_suite("nominal")[0]
    reporter = CollectingReporter()
    simulate_trajectory(scenario, "lqr", estimator_name="none", progress=reporter, run_index=1, run_total=1)
    stages = [event.stage for event in reporter.events if event.domain == "trajectory"]
    assert stages[0] == "start"
    assert "heartbeat" in stages
    assert stages[-1] == "done"


def test_run_suite_emits_suite_progress(tmp_path: Path, monkeypatch) -> None:
    reporter = CollectingReporter()
    scenario = ScenarioConfig(
        name="local_small_angle",
        suite_name="nominal",
        horizon=0.1,
        dt=0.05,
        initial_state=(0.0, 0.0, 0.0, 0.0),
        seed=7,
    )

    def fake_run_trajectory(*args, **kwargs):
        return _fake_result(seed=scenario.seed), {"csv_path": "csv/file.csv", "json_path": "json/file.json"}

    monkeypatch.setattr(runner_module, "load_suite", lambda suite_name: [scenario])
    monkeypatch.setattr(runner_module, "run_trajectory", fake_run_trajectory)

    results = run_suite("nominal", tmp_path / "artifacts", controllers=("lqr",), progress=reporter)
    assert len(results) == 1
    suite_events = [event for event in reporter.events if event.domain == "suite"]
    assert suite_events[0].stage == "start"
    assert any(event.stage == "run_start" for event in suite_events)
    assert any(event.stage == "run_end" for event in suite_events)
    assert suite_events[-1].stage == "done"


def test_run_monte_carlo_emits_sample_progress(tmp_path: Path, monkeypatch) -> None:
    reporter = CollectingReporter()
    scenarios = [
        (
            ScenarioConfig(
                name=f"sample_{index:05d}",
                suite_name="monte_carlo",
                horizon=0.1,
                dt=0.05,
                initial_state=(0.0, 0.0, 0.0, 0.0),
                seed=index,
            ),
            {
                "initial_angle_deg": 0.0,
                "initial_rate": 0.0,
                "noise_theta_std_deg": 0.0,
                "disturbance_force": 0.0,
                "cart_friction": 0.0,
                "pivot_damping": 0.0,
                "M": 1.0,
                "m": 0.1,
                "l": 0.5,
            },
        )
        for index in range(3)
    ]

    monkeypatch.setattr(batch_module, "_sample_scenarios", lambda samples: scenarios[:samples])
    monkeypatch.setattr(batch_module, "simulate_trajectory", lambda *args, **kwargs: (_fake_result(), {}, None))

    summaries = run_monte_carlo(
        tmp_path / "artifacts",
        samples=3,
        controllers=("lqr",),
        estimator_name="none",
        progress=reporter,
    )
    assert len(summaries) == 1
    stages = [event.stage for event in reporter.events if event.domain == "monte_carlo"]
    assert stages[0] == "start"
    assert "controller_start" in stages
    assert stages.count("sample") == 3
    assert "controller_end" in stages
    assert stages[-1] == "done"


def test_tune_controller_emits_iteration_progress(tmp_path: Path) -> None:
    reporter = CollectingReporter()
    tune_controller("lqr", tmp_path / "artifacts", budget=2, seed=1, progress=reporter)
    stages = [event.stage for event in reporter.events if event.domain == "tune"]
    assert stages[0] == "start"
    assert stages.count("iteration_start") == 2
    assert stages.count("iteration_end") == 2
    assert "validation_start" in stages
    assert stages[-1] == "done"


def test_generate_figures_emits_item_progress(tmp_path: Path, monkeypatch) -> None:
    reporter = CollectingReporter()
    run = {"metadata": {"controller_name": "LQR", "estimator_name": "none"}}

    monkeypatch.setattr(figures_module, "refresh_metric_tables", lambda base_dir: None)
    monkeypatch.setattr(figures_module, "load_saved_runs", lambda base_dir, suites: [run])
    monkeypatch.setattr(
        figures_module,
        "_eligible_figure_keys",
        lambda *args, **kwargs: [
            "nominal_local_response",
            "full_task_handoff",
            "stress_comparison",
            "metric_summary",
            "monte_carlo_overview",
        ],
    )
    monkeypatch.setattr(figures_module, "_cleanup_stale_figure_outputs", lambda *args, **kwargs: set())
    monkeypatch.setattr(figures_module, "_figure_nominal_local_response", lambda *args: tmp_path / "fig1.png")
    monkeypatch.setattr(figures_module, "_figure_full_task_handoff", lambda *args: tmp_path / "fig2.png")
    monkeypatch.setattr(figures_module, "_figure_stress_comparison", lambda *args: tmp_path / "fig3.png")
    monkeypatch.setattr(figures_module, "_figure_metric_summary", lambda *args: tmp_path / "fig4.png")
    monkeypatch.setattr(figures_module, "_figure_monte_carlo_overview", lambda *args: tmp_path / "fig5.png")

    created = generate_figures(
        tmp_path / "artifacts",
        include_supplements=False,
        controllers=("lqr",),
        progress=reporter,
    )
    assert len(created) == 5
    stages = [event.stage for event in reporter.events if event.domain == "render_figures"]
    assert stages[0] == "start"
    assert stages.count("item_end") == 5
    assert stages[-1] == "done"


def test_render_animations_emits_item_progress(tmp_path: Path, monkeypatch) -> None:
    reporter = CollectingReporter()
    run = {"metadata": {"controller_name": "LQR", "estimator_name": "none", "scenario_name": "full_task_hanging", "seed": 0}}

    monkeypatch.setattr(render_module, "load_saved_runs", lambda base_dir, suites: [run])
    monkeypatch.setattr(render_module, "_eligible_animation_keys", lambda *args, **kwargs: ["LQR", "nominal_comparison", "stress_comparison"])
    monkeypatch.setattr(render_module, "_cleanup_stale_animation_outputs", lambda *args, **kwargs: set())
    monkeypatch.setattr(render_module, "_representative_run", lambda *args: run)
    monkeypatch.setattr(render_module, "_best_seed_for_scenario", lambda *args: 0)
    monkeypatch.setattr(render_module, "_runs_for_seed", lambda *args: [run])
    monkeypatch.setattr(render_module, "_render_single", lambda *args: tmp_path / "single.gif")
    monkeypatch.setattr(render_module, "_render_comparison", lambda *args, **kwargs: tmp_path / "comparison.gif")

    def fake_focus(*args, **kwargs):
        return tmp_path / "focus.gif"

    monkeypatch.setattr(render_module, "_render_focus_comparison", fake_focus)

    created = render_module.render_animations(
        tmp_path / "artifacts",
        include_supplements=False,
        controllers=("lqr",),
        progress=reporter,
    )
    assert len(created) == 3
    stages = [event.stage for event in reporter.events if event.domain == "render_animations"]
    assert stages[0] == "start"
    assert stages.count("item_end") == 3
    assert stages[-1] == "done"
