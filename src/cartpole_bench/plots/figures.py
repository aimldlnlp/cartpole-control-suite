from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

from cartpole_bench.plots.style import (
    MODE_COLORS,
    add_event_band,
    add_panel_title,
    apply_theme,
    controller_badge,
    controller_color,
    make_density_cmap,
    plot_percentile_band,
    save_figure,
    soften,
    style_axis,
)
from cartpole_bench.plots.tables import refresh_metric_tables
from cartpole_bench.simulation.recorder import load_saved_runs
from cartpole_bench.utils.progress import NullProgressReporter, PhaseTimer, ProgressEvent, ProgressReporter


CONTROLLER_LABELS = {
    "lqr": "LQR",
    "pfl": "Feedback Linearization (PFL)",
    "smc": "Sliding Mode Control (SMC)",
    "ilqr": "Iterative LQR (iLQR)",
    "mpc": "Model Predictive Control (MPC)",
}
CONTROLLER_ORDER = [CONTROLLER_LABELS[key] for key in ("lqr", "pfl", "smc", "ilqr", "mpc")]
FIGURE_FILENAMES = {
    "nominal_local_response": "nominal_local_response.png",
    "full_task_handoff": "full_task_handoff.png",
    "stress_comparison": "stress_comparison.png",
    "metric_summary": "metric_summary.png",
    "monte_carlo_overview": "monte_carlo_overview.png",
}
SUPPLEMENTAL_FIGURE_FILENAMES = {
    "handoff_focus": "handoff_focus.png",
    "constraint_usage": "constraint_usage.png",
    "robustness_map": "robustness_map.png",
    "energy_phase": "energy_phase.png",
}
FAILURE_ORDER = [
    "success",
    "did_not_meet_settling_window",
    "left_stable_region",
    "track_limit_exceeded",
    "no_balance_handoff",
    "capture_assist_no_handoff",
    "invalid_state_or_control",
    "unknown_failure",
]
FAILURE_SHORT_LABELS = {
    "success": "ok",
    "did_not_meet_settling_window": "settle",
    "left_stable_region": "left",
    "track_limit_exceeded": "track",
    "no_balance_handoff": "handoff",
    "capture_assist_no_handoff": "capture",
    "invalid_state_or_control": "invalid",
    "unknown_failure": "unknown",
}
SCENARIO_SHORT_LABELS = {
    "local_small_angle": "local",
    "full_task_hanging": "full task",
    "measurement_noise": "noise",
    "impulse_disturbance": "impulse",
    "friction_and_damping": "friction",
    "large_angle_recovery": "large-angle",
    "parameter_mismatch": "mismatch",
}


def _short_label(controller: str) -> str:
    return {
        "LQR": "LQR",
        "Feedback Linearization (PFL)": "PFL",
        "Sliding Mode Control (SMC)": "SMC",
        "Iterative LQR (iLQR)": "iLQR",
        "Model Predictive Control (MPC)": "MPC",
    }.get(controller, controller)


def _selected_labels(controllers: tuple[str, ...]) -> list[str]:
    return [CONTROLLER_LABELS[key] for key in controllers if key in CONTROLLER_LABELS]


def _scenario_label(name: str) -> str:
    return SCENARIO_SHORT_LABELS.get(name, name.replace("_", " "))


def _required_labels(controllers: tuple[str, ...]) -> list[str]:
    return _selected_labels(controllers)


def _filter_runs(runs: list[dict], controllers: tuple[str, ...], estimator_name: str) -> list[dict]:
    allowed = set(_selected_labels(controllers))
    return [
        run
        for run in runs
        if run["metadata"]["controller_name"] in allowed and run["metadata"].get("estimator_name", "none") == estimator_name
    ]


def _require_render_runs(base_dir: Path, runs: list[dict]) -> None:
    if runs:
        return
    raise RuntimeError(
        f"No saved run artifacts were found in '{base_dir}' for the requested controller/estimator selection."
    )


def _stack_on_time_grid(runs: list[dict], value_key: str, *, time_key: str = "t") -> tuple[np.ndarray, np.ndarray]:
    grid = runs[0]["frame"][time_key].to_numpy(dtype=float)
    stacked = []
    for run in runs:
        frame = run["frame"]
        x = frame[time_key].to_numpy(dtype=float)
        y = frame[value_key].to_numpy(dtype=float)
        stacked.append(np.interp(grid, x, y))
    return grid, np.asarray(stacked, dtype=float)


def _aligned_samples(
    runs: list[dict],
    value_key: str,
    center_key: str,
    before: float,
    after: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(runs[0]["frame"]["t"].to_numpy(dtype=float))))
    grid = np.arange(-before, after + 0.5 * dt, dt)
    samples = []
    for run in runs:
        center = run["metadata"]["diagnosis"].get(center_key)
        if center is None:
            continue
        frame = run["frame"]
        rel_t = frame["t"].to_numpy(dtype=float) - float(center)
        values = frame[value_key].to_numpy(dtype=float)
        mask = (rel_t >= -before) & (rel_t <= after)
        if np.count_nonzero(mask) < 2:
            continue
        samples.append(np.interp(grid, rel_t[mask], values[mask]))
    return grid, np.asarray(samples, dtype=float) if samples else np.empty((0, len(grid)), dtype=float)


def _mode_alignment(runs: list[dict], before: float, after: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dt = float(np.median(np.diff(runs[0]["frame"]["t"].to_numpy(dtype=float))))
    grid = np.arange(-before, after + 0.5 * dt, dt)
    mode_names = ("energy_pump", "capture_assist", "balance")
    counts = {mode: [] for mode in mode_names}
    for run in runs:
        center = run["metadata"]["diagnosis"].get("first_balance_time")
        if center is None:
            continue
        frame = run["frame"]
        rel_t = frame["t"].to_numpy(dtype=float) - float(center)
        mode_series = frame["mode"].astype(str).tolist()
        mode_grid = {mode: np.zeros(len(grid), dtype=float) for mode in mode_names}
        for index, target_t in enumerate(grid):
            src_index = int(np.argmin(np.abs(rel_t - target_t)))
            mode = mode_series[src_index]
            if mode in mode_grid:
                mode_grid[mode][index] = 1.0
        for mode in mode_names:
            counts[mode].append(mode_grid[mode])
    return grid, {mode: np.mean(rows, axis=0) if rows else np.zeros(len(grid), dtype=float) for mode, rows in counts.items()}


def _shared_legend(
    fig,
    theme_cfg,
    controllers: list[str],
    *,
    loc: str = "upper right",
    x: float = 0.985,
    y: float = 0.985,
    ncol: int | None = None,
    fontsize: float = 7.8,
    handlelength: float = 1.7,
    columnspacing: float = 0.8,
    handletextpad: float = 0.5,
) -> None:
    handles = [Line2D([0], [0], color=controller_color(theme_cfg, controller), linewidth=1.9) for controller in controllers]
    fig.legend(
        handles,
        [_short_label(controller) for controller in controllers],
        loc=loc,
        bbox_to_anchor=(x, y),
        fontsize=fontsize,
        ncol=ncol or max(1, len(handles)),
        handlelength=handlelength,
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        frameon=False,
    )


def _representative_run(runs: list[dict], scenario: str, controller: str) -> dict:
    candidates = [
        run
        for run in runs
        if run["metadata"]["scenario_name"] == scenario and run["metadata"]["controller_name"] == controller
    ]
    successful = [run for run in candidates if run["metadata"]["metrics"]["success"]]
    pool = successful or candidates
    if not pool:
        raise KeyError(f"Missing run for {scenario=} {controller=}")
    settling = [
        run["metadata"]["metrics"]["settling_time"] if run["metadata"]["metrics"]["settling_time"] is not None else 999.0
        for run in pool
    ]
    target = float(np.median(settling))
    return min(
        pool,
        key=lambda run: abs(
            (float(run["metadata"]["metrics"]["settling_time"]) if run["metadata"]["metrics"]["settling_time"] is not None else 999.0)
            - target
        ),
    )


def _summary(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "tables" / "metric_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normalize_metric_values(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    normalized = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return normalized
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if abs(high - low) < 1e-9:
        normalized[finite] = 0.72
        return normalized
    scaled = (values - low) / (high - low)
    normalized[finite] = scaled[finite] if higher_is_better else 1.0 - scaled[finite]
    return normalized


def _draw_metric_heatmap(
    ax,
    normalized: pd.DataFrame,
    display: pd.DataFrame,
    theme_cfg,
    *,
    title: str | None,
    subtitle: str | None = None,
    x_rotation: float = 0.0,
    value_fontsize: float = 7.1,
    title_pad: float = 4.5,
    subtitle_y: float = 1.005,
) -> None:
    style_axis(ax, theme_cfg)
    add_panel_title(ax, title, subtitle=subtitle, theme_cfg=theme_cfg, title_pad=title_pad, subtitle_y=subtitle_y)
    if normalized.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8.0, color=theme_cfg.muted_color)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    cmap = make_density_cmap(theme_cfg)
    matrix = np.nan_to_num(normalized.to_numpy(dtype=float), nan=0.0)
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(normalized.columns)))
    ax.set_xticklabels(list(normalized.columns), rotation=x_rotation, ha="right" if x_rotation else "center", fontsize=7.7)
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels(list(normalized.index), fontsize=7.9)
    ax.set_xticks(np.arange(-0.5, len(normalized.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(normalized.index), 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=soften(theme_cfg.spine_color, 0.45), linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, row_label in enumerate(normalized.index):
        for col_index, col_label in enumerate(normalized.columns):
            value = float(matrix[row_index, col_index])
            text_color = "#FFFFFF" if value >= 0.58 else theme_cfg.text_color
            ax.text(
                col_index,
                row_index,
                str(display.loc[row_label, col_label]),
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color=text_color,
            )


def _build_profile_tables(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, bool, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_rows: dict[str, list[float]] = {}
    display_rows: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        row_key = str(row["label"])
        numeric_values: list[float] = []
        display_values: list[str] = []
        for column, _, _, fmt in metrics:
            value = row.get(column)
            numeric_values.append(float(value) if value is not None and pd.notna(value) else np.nan)
            if value is None or pd.isna(value):
                display_values.append("NA")
            elif fmt == "pct":
                display_values.append(f"{100.0 * float(value):.0f}%")
            elif fmt == "float2":
                display_values.append(f"{float(value):.2f}")
            elif fmt == "float1":
                display_values.append(f"{float(value):.1f}")
            else:
                display_values.append(str(value))
        value_rows[row_key] = numeric_values
        display_rows[row_key] = display_values

    columns = [label for _, label, _, _ in metrics]
    value_frame = pd.DataFrame.from_dict(value_rows, orient="index", columns=columns)
    normalized = pd.DataFrame(index=value_frame.index, columns=value_frame.columns, dtype=float)
    for column, label, higher_is_better, _ in metrics:
        normalized[label] = _normalize_metric_values(value_frame[label].to_numpy(dtype=float), higher_is_better=higher_is_better)
    display = pd.DataFrame.from_dict(display_rows, orient="index", columns=columns)
    return normalized, display


def _monte_summary(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = base_dir / "tables" / "monte_carlo_summary.csv"
    sample_path = base_dir / "tables" / "monte_carlo_samples.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    samples = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    return summary, samples


def _scenario_has_all_labels(runs: list[dict], scenario_name: str, required_labels: list[str]) -> bool:
    labels = {
        run["metadata"].get("controller_name")
        for run in runs
        if run.get("metadata", {}).get("scenario_name") == scenario_name and run.get("metadata", {}).get("controller_name")
    }
    return set(required_labels).issubset(labels)


def _summary_has_all_labels(summary: pd.DataFrame, required_labels: list[str], estimator_name: str) -> bool:
    if summary.empty or "controller" not in summary:
        return False
    filtered = summary.copy()
    if "estimator" in filtered:
        filtered = filtered[filtered["estimator"] == estimator_name]
    labels = set(filtered["controller"].astype(str))
    return set(required_labels).issubset(labels)


def _stress_summary_has_full_grid(summary: pd.DataFrame, required_labels: list[str], estimator_name: str) -> bool:
    if summary.empty or "suite" not in summary or "controller" not in summary or "scenario" not in summary:
        return False
    filtered = summary[summary["suite"] == "stress"].copy()
    if "estimator" in filtered:
        filtered = filtered[filtered["estimator"] == estimator_name]
    if filtered.empty:
        return False
    scenarios = sorted(set(filtered["scenario"].astype(str)))
    if not scenarios:
        return False
    for controller in required_labels:
        controller_rows = filtered[filtered["controller"] == controller]
        if controller_rows.empty:
            return False
        controller_scenarios = set(controller_rows["scenario"].astype(str))
        if set(scenarios) - controller_scenarios:
            return False
    return True


def _monte_summary_has_all_labels(
    summary: pd.DataFrame,
    samples: pd.DataFrame,
    required_labels: list[str],
    estimator_name: str,
) -> bool:
    if summary.empty or samples.empty:
        return False
    if "controller_name" not in summary or "controller" not in samples:
        return False
    filtered_summary = summary.copy()
    if "estimator_name" in filtered_summary:
        filtered_summary = filtered_summary[filtered_summary["estimator_name"] == estimator_name]
    filtered_samples = samples.copy()
    if "estimator" in filtered_samples:
        filtered_samples = filtered_samples[filtered_samples["estimator"] == estimator_name]
    summary_labels = set(filtered_summary["controller_name"].astype(str))
    sample_labels = set(filtered_samples["controller"].astype(str))
    required = set(required_labels)
    return required.issubset(summary_labels) and required.issubset(sample_labels)


def _aligned_samples_exist_for_all_labels(
    runs: list[dict],
    scenario_name: str,
    required_labels: list[str],
    *,
    value_key: str,
    center_key: str,
    before: float,
    after: float,
) -> bool:
    for controller in required_labels:
        controller_runs = [
            run
            for run in runs
            if run.get("metadata", {}).get("scenario_name") == scenario_name
            and run.get("metadata", {}).get("controller_name") == controller
        ]
        if not controller_runs:
            return False
        _, samples = _aligned_samples(controller_runs, value_key, center_key, before, after)
        if samples.size == 0:
            return False
    return True


def _runs_have_all_labels(runs: list[dict], required_labels: list[str]) -> bool:
    labels = {run.get("metadata", {}).get("controller_name") for run in runs if run.get("metadata", {}).get("controller_name")}
    return set(required_labels).issubset(labels)


def _figure_output_paths(base_dir: Path) -> dict[str, Path]:
    figures_dir = base_dir / "figures"
    supplemental_dir = figures_dir / "supplemental"
    return {
        "nominal_local_response": figures_dir / FIGURE_FILENAMES["nominal_local_response"],
        "full_task_handoff": figures_dir / FIGURE_FILENAMES["full_task_handoff"],
        "stress_comparison": figures_dir / FIGURE_FILENAMES["stress_comparison"],
        "metric_summary": figures_dir / FIGURE_FILENAMES["metric_summary"],
        "monte_carlo_overview": figures_dir / FIGURE_FILENAMES["monte_carlo_overview"],
        "handoff_focus": supplemental_dir / SUPPLEMENTAL_FIGURE_FILENAMES["handoff_focus"],
        "constraint_usage": supplemental_dir / SUPPLEMENTAL_FIGURE_FILENAMES["constraint_usage"],
        "robustness_map": supplemental_dir / SUPPLEMENTAL_FIGURE_FILENAMES["robustness_map"],
        "energy_phase": supplemental_dir / SUPPLEMENTAL_FIGURE_FILENAMES["energy_phase"],
    }


def _eligible_figure_keys(
    runs: list[dict],
    summary: pd.DataFrame,
    monte_summary: pd.DataFrame,
    monte_samples: pd.DataFrame,
    controllers: tuple[str, ...],
    estimator_name: str,
    *,
    include_supplements: bool,
) -> list[str]:
    required_labels = _required_labels(controllers)
    eligible: list[str] = []

    if _scenario_has_all_labels(runs, "local_small_angle", required_labels):
        eligible.append("nominal_local_response")
    if _scenario_has_all_labels(runs, "full_task_hanging", required_labels):
        eligible.append("full_task_handoff")
    if _stress_summary_has_full_grid(summary, required_labels, estimator_name):
        eligible.append("stress_comparison")
    if _summary_has_all_labels(summary, required_labels, estimator_name):
        eligible.append("metric_summary")
    if _monte_summary_has_all_labels(monte_summary, monte_samples, required_labels, estimator_name):
        eligible.append("monte_carlo_overview")

    if include_supplements:
        if _aligned_samples_exist_for_all_labels(
            runs,
            "full_task_hanging",
            required_labels,
            value_key="theta_deg",
            center_key="first_balance_time",
            before=1.0,
            after=1.2,
        ):
            eligible.append("handoff_focus")
        if _runs_have_all_labels(runs, required_labels):
            eligible.append("constraint_usage")
        if _monte_summary_has_all_labels(monte_summary, monte_samples, required_labels, estimator_name):
            eligible.append("robustness_map")
        if _scenario_has_all_labels(runs, "full_task_hanging", required_labels):
            eligible.append("energy_phase")

    return eligible


def _cleanup_stale_figure_outputs(base_dir: Path, eligible_keys: list[str]) -> set[Path]:
    managed_paths = _figure_output_paths(base_dir)
    stale_paths = {path for key, path in managed_paths.items() if key not in set(eligible_keys) and path.exists()}
    for path in stale_paths:
        path.unlink()
    return stale_paths


def generate_figures(
    base_dir: Path,
    theme: str = "paper_dense_cmu",
    include_supplements: bool = True,
    controllers: tuple[str, ...] = tuple(CONTROLLER_LABELS),
    estimator_name: str = "none",
    progress: ProgressReporter | None = None,
) -> list[Path]:
    progress = progress or NullProgressReporter()
    theme_cfg = apply_theme(theme)
    refresh_metric_tables(base_dir)
    runs = load_saved_runs(base_dir, suites={"nominal", "stress"})
    runs = _filter_runs(runs, controllers, estimator_name)
    _require_render_runs(base_dir, runs)
    summary = _summary(base_dir)
    monte_summary, monte_samples = _monte_summary(base_dir)
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    eligible_keys = _eligible_figure_keys(
        runs,
        summary,
        monte_summary,
        monte_samples,
        controllers,
        estimator_name,
        include_supplements=include_supplements,
    )
    _cleanup_stale_figure_outputs(base_dir, eligible_keys)
    total_items = len(eligible_keys)
    timer = PhaseTimer()
    progress.emit(
        ProgressEvent(
            domain="render_figures",
            stage="start",
            current=0,
            total=total_items,
            context={"estimator": estimator_name, "note": f"theme={theme}"},
        )
    )

    created: list[Path] = []
    supplemental_dir = figures_dir / "supplemental"
    if any(key in SUPPLEMENTAL_FIGURE_FILENAMES for key in eligible_keys):
        supplemental_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "nominal_local_response": lambda: _figure_nominal_local_response(runs, figures_dir, theme_cfg),
        "full_task_handoff": lambda: _figure_full_task_handoff(runs, figures_dir, theme_cfg),
        "stress_comparison": lambda: _figure_stress_comparison(runs, figures_dir, theme_cfg, base_dir),
        "metric_summary": lambda: _figure_metric_summary(runs, figures_dir, theme_cfg, base_dir),
        "monte_carlo_overview": lambda: _figure_monte_carlo_overview(figures_dir, theme_cfg, base_dir, controllers, estimator_name),
        "handoff_focus": lambda: _figure_handoff_focus(runs, supplemental_dir, theme_cfg),
        "constraint_usage": lambda: _figure_constraint_usage(runs, supplemental_dir, theme_cfg),
        "robustness_map": lambda: _figure_robustness_map(supplemental_dir, theme_cfg, base_dir, controllers, estimator_name),
        "energy_phase": lambda: _figure_energy_phase(runs, supplemental_dir, theme_cfg),
    }
    item_names = {**FIGURE_FILENAMES, **SUPPLEMENTAL_FIGURE_FILENAMES}

    def _capture(stage_index: int, item_name: str, build) -> None:
        created_path = build()
        created.append(created_path)
        progress.emit(
            ProgressEvent(
                domain="render_figures",
                stage="item_end",
                current=stage_index,
                total=total_items,
                context={"item_name": item_name, "estimator": estimator_name},
                elapsed_s=timer.elapsed(),
                eta_s=timer.eta(stage_index, total_items),
            )
        )

    for item_index, key in enumerate(eligible_keys, start=1):
        _capture(item_index, item_names[key], builders[key])
    progress.emit(
        ProgressEvent(
            domain="render_figures",
            stage="done",
            current=len(created),
            total=total_items,
            context={"estimator": estimator_name},
            elapsed_s=timer.elapsed(),
            eta_s=0.0,
        )
    )
    return created


def _figure_nominal_local_response(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    scenario_runs = [run for run in runs if run["metadata"]["scenario_name"] == "local_small_angle"]
    controllers = [controller for controller in CONTROLLER_ORDER if any(run["metadata"]["controller_name"] == controller for run in scenario_runs)]

    fig = plt.figure(figsize=(12.4, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.92], height_ratios=[1.0, 1.0])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.11, wspace=0.18, hspace=0.24)
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_cart = fig.add_subplot(gs[0, 1], sharex=ax_angle)
    ax_force = fig.add_subplot(gs[1, 0], sharex=ax_angle)
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_profile = fig.add_subplot(gs[:, 2])
    for ax in (ax_angle, ax_cart, ax_force, ax_phase, ax_profile):
        style_axis(ax, theme_cfg)

    profile_rows = []
    for controller in controllers:
        controller_runs = [run for run in scenario_runs if run["metadata"]["controller_name"] == controller]
        color = controller_color(theme_cfg, controller)
        time, angle_samples = _stack_on_time_grid(controller_runs, "theta_deg")
        _, cart_samples = _stack_on_time_grid(controller_runs, "x")
        _, force_samples = _stack_on_time_grid(controller_runs, "u")
        _, theta_samples = _stack_on_time_grid(controller_runs, "theta_deg")
        _, rate_samples = _stack_on_time_grid(controller_runs, "theta_dot")

        for run in controller_runs:
            frame = run["frame"]
            ax_angle.plot(frame["t"], frame["theta_deg"], color=color, linewidth=0.8, alpha=0.14)
            ax_cart.plot(frame["t"], frame["x"], color=color, linewidth=0.8, alpha=0.14)
            ax_force.plot(frame["t"], frame["u"], color=color, linewidth=0.8, alpha=0.14)
        plot_percentile_band(ax_angle, time, angle_samples, color, label=_short_label(controller))
        plot_percentile_band(ax_cart, time, cart_samples, color)
        plot_percentile_band(ax_force, time, force_samples, color)
        ax_phase.plot(np.nanmedian(theta_samples, axis=0), np.nanmedian(rate_samples, axis=0), color=color, linewidth=1.9)
        ax_phase.scatter(
            np.nanmedian(theta_samples[:, -1]),
            np.nanmedian(rate_samples[:, -1]),
            color=color,
            s=16,
            zorder=4,
        )
        settling_values = pd.to_numeric(
            pd.Series([run["metadata"]["metrics"]["settling_time"] for run in controller_runs]),
            errors="coerce",
        )
        profile_rows.append(
            {
                "label": _short_label(controller),
                "settling_time": float(np.nanmedian(settling_values)),
                "max_abs_x": float(np.nanmedian([run["metadata"]["diagnosis"]["max_abs_x"] for run in controller_runs])),
                "control_effort": float(np.nanmedian([run["metadata"]["metrics"]["control_effort"] for run in controller_runs])),
                "balance_fraction": float(np.nanmedian([run["metadata"]["diagnosis"]["balance_fraction"] for run in controller_runs])),
            }
        )

    add_panel_title(ax_angle, "Angle", theme_cfg=theme_cfg, title_pad=7.0)
    add_panel_title(ax_cart, "Cart", theme_cfg=theme_cfg, title_pad=7.0)
    add_panel_title(ax_force, "Force", theme_cfg=theme_cfg, title_pad=7.0)
    add_panel_title(ax_phase, "Phase", theme_cfg=theme_cfg, title_pad=7.0)
    ax_angle.set_ylabel("Angle [deg]")
    ax_cart.set_ylabel("Position [m]")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_xlabel("Time [s]")
    ax_phase.set_xlabel("Angle [deg]")
    ax_phase.set_ylabel("Angular rate [rad/s]")
    ax_angle.tick_params(labelbottom=False)
    ax_cart.tick_params(labelbottom=False)
    profile_frame = pd.DataFrame(profile_rows)
    normalized, display = _build_profile_tables(
        profile_frame,
        [
            ("settling_time", "Settle", False, "float1"),
            ("max_abs_x", "|x|", False, "float2"),
            ("control_effort", "Effort", False, "float1"),
            ("balance_fraction", "Bal", True, "pct"),
        ],
    )
    _draw_metric_heatmap(
        ax_profile,
        normalized,
        display,
        theme_cfg,
        title="Profile",
        value_fontsize=7.0,
        title_pad=8.0,
    )
    _shared_legend(fig, theme_cfg, controllers, x=0.985, y=0.984, fontsize=7.4, handlelength=1.6, columnspacing=0.72)
    fig.suptitle("Nominal Local Response", x=0.07, y=0.982, ha="left", fontsize=11.5, color=theme_cfg.text_color)
    fig.text(0.07, 0.945, "Seed-cloud medians", ha="left", va="top", fontsize=7.2, color=theme_cfg.muted_color)
    path = figures_dir / FIGURE_FILENAMES["nominal_local_response"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_full_task_handoff(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    scenario_runs = [run for run in runs if run["metadata"]["scenario_name"] == "full_task_hanging"]
    controllers = [controller for controller in CONTROLLER_ORDER if any(run["metadata"]["controller_name"] == controller for run in scenario_runs)]

    fig = plt.figure(figsize=(12.6, 7.1))
    gs = fig.add_gridspec(4, 2, width_ratios=[4.2, 1.45], height_ratios=[2.0, 1.25, 1.25, 0.75])
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_cart = fig.add_subplot(gs[1, 0], sharex=ax_angle)
    ax_force = fig.add_subplot(gs[2, 0], sharex=ax_angle)
    ax_mode = fig.add_subplot(gs[3, 0], sharex=ax_angle)
    ax_profile = fig.add_subplot(gs[:, 1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.12, hspace=0.24, wspace=0.18)
    for ax in (ax_angle, ax_cart, ax_force):
        style_axis(ax, theme_cfg)
        add_event_band(ax, -0.15, 0.15, theme_cfg, alpha=0.07)
        ax.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9, alpha=0.75)
    style_axis(ax_profile, theme_cfg)

    mode_rows = []
    aligned_controllers = []
    grid = None
    profile_rows = []
    for controller in controllers:
        controller_runs = [run for run in scenario_runs if run["metadata"]["controller_name"] == controller]
        color = controller_color(theme_cfg, controller)
        grid, angle_samples = _aligned_samples(controller_runs, "theta_deg", "first_balance_time", 1.0, 1.2)
        _, cart_samples = _aligned_samples(controller_runs, "x", "first_balance_time", 1.0, 1.2)
        _, force_samples = _aligned_samples(controller_runs, "u", "first_balance_time", 1.0, 1.2)
        if angle_samples.size == 0:
            continue
        plot_percentile_band(ax_angle, grid, angle_samples, color, label=_short_label(controller))
        plot_percentile_band(ax_cart, grid, cart_samples, color)
        plot_percentile_band(ax_force, grid, force_samples, color)

        _, mode_density = _mode_alignment(controller_runs, 1.0, 1.2)
        dominant = np.argmax(
            np.vstack([mode_density["energy_pump"], mode_density["capture_assist"], mode_density["balance"]]),
            axis=0,
        )
        mode_rows.append(dominant)
        aligned_controllers.append(controller)
        entry_values = pd.to_numeric(
            pd.Series([run["metadata"]["diagnosis"]["first_balance_time"] for run in controller_runs]),
            errors="coerce",
        )
        settling_values = pd.to_numeric(
            pd.Series([run["metadata"]["metrics"]["settling_time"] for run in controller_runs]),
            errors="coerce",
        )
        profile_rows.append(
            {
                "label": _short_label(controller),
                "first_balance_time": float(np.nanmedian(entry_values)),
                "settling_time": float(np.nanmedian(settling_values)),
                "balance_fraction": float(np.nanmedian([run["metadata"]["diagnosis"]["balance_fraction"] for run in controller_runs])),
                "max_abs_force": float(np.nanmedian([run["metadata"]["metrics"]["max_abs_force"] for run in controller_runs])),
            }
        )

    add_panel_title(ax_angle, "Angle", theme_cfg=theme_cfg, title_pad=7.0)
    add_panel_title(ax_cart, "Cart", theme_cfg=theme_cfg, title_pad=7.0)
    add_panel_title(ax_force, "Force", theme_cfg=theme_cfg, title_pad=7.0)
    ax_angle.set_ylabel("Angle [deg]")
    ax_cart.set_ylabel("Cart [m]")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_xlabel("Time from balance entry [s]")
    ax_angle.tick_params(labelbottom=False)
    ax_cart.tick_params(labelbottom=False)

    if grid is not None and mode_rows:
        mode_matrix = np.asarray(mode_rows, dtype=float)
        cmap = ListedColormap([MODE_COLORS["energy_pump"], MODE_COLORS["capture_assist"], MODE_COLORS["balance"]])
        ax_mode.imshow(
            mode_matrix,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
            extent=[grid[0], grid[-1], -0.5, len(mode_rows) - 0.5],
            vmin=0,
            vmax=2,
        )
    ax_mode.set_facecolor(theme_cfg.panel_color)
    ax_mode.set_yticks(np.arange(len(aligned_controllers)))
    ax_mode.set_yticklabels([_short_label(controller) for controller in aligned_controllers], fontsize=8.5)
    ax_mode.set_xlabel("Time from balance entry [s]")
    ax_mode.tick_params(axis="x", labelsize=8.5)
    for spine in ax_mode.spines.values():
        spine.set_visible(False)
    profile_frame = pd.DataFrame(profile_rows)
    normalized, display = _build_profile_tables(
        profile_frame,
        [
            ("first_balance_time", "Entry", False, "float1"),
            ("settling_time", "Settle", False, "float1"),
            ("balance_fraction", "Bal", True, "pct"),
            ("max_abs_force", "|u|", False, "float1"),
        ],
    )
    _draw_metric_heatmap(
        ax_profile,
        normalized,
        display,
        theme_cfg,
        title="Profile",
        value_fontsize=6.9,
        title_pad=8.0,
    )
    _shared_legend(fig, theme_cfg, controllers, x=0.985, y=0.984, fontsize=7.4, handlelength=1.6, columnspacing=0.72)
    fig.suptitle("Full-Task Handoff", x=0.075, y=0.982, ha="left", fontsize=11.5, color=theme_cfg.text_color)
    fig.text(0.075, 0.945, "Aligned to balance entry", ha="left", va="top", fontsize=7.2, color=theme_cfg.muted_color)
    path = figures_dir / FIGURE_FILENAMES["full_task_handoff"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_stress_comparison(runs: list[dict], figures_dir: Path, theme_cfg, base_dir: Path) -> Path:
    summary = _summary(base_dir)
    stress = summary[summary["suite"] == "stress"].copy()
    if stress.empty:
        stress = pd.DataFrame(columns=["controller", "scenario", "success_rate", "estimator"])
    estimator = runs[0]["metadata"].get("estimator_name", "none")
    stress = stress[stress.get("estimator", "none") == estimator] if "estimator" in stress else stress
    controllers = [controller for controller in CONTROLLER_ORDER if controller in set(stress.get("controller", []))]
    scenarios = list(dict.fromkeys(stress.get("scenario", pd.Series(dtype=str)).tolist()))

    success = np.zeros((len(controllers), len(scenarios)), dtype=float)
    settling = np.full((len(controllers), len(scenarios)), np.nan, dtype=float)
    failure_index = np.zeros((len(controllers), len(scenarios)), dtype=float)
    for row_index, controller in enumerate(controllers):
        for col_index, scenario in enumerate(scenarios):
            subset = stress[(stress["controller"] == controller) & (stress["scenario"] == scenario)]
            if not subset.empty:
                success[row_index, col_index] = float(subset["success_rate"].iloc[0])
                settling[row_index, col_index] = float(subset["settling_time_median"].iloc[0]) if "settling_time_median" in subset else np.nan
            scenario_runs = [
                run
                for run in runs
                if run["metadata"]["controller_name"] == controller and run["metadata"]["scenario_name"] == scenario
            ]
            reasons = [
                run["metadata"]["diagnosis"]["failure_reason"] if not run["metadata"]["metrics"]["success"] else "success"
                for run in scenario_runs
            ]
            if reasons:
                dominant = max(set(reasons), key=reasons.count)
                failure_index[row_index, col_index] = FAILURE_ORDER.index(dominant) if dominant in FAILURE_ORDER else FAILURE_ORDER.index("unknown_failure")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.23, wspace=0.14)
    if controllers and scenarios:
        row_labels = [_short_label(controller) for controller in controllers]
        col_labels = [_scenario_label(scenario) for scenario in scenarios]
        success_norm = pd.DataFrame(success, index=row_labels, columns=col_labels)
        success_display = success_norm.apply(lambda column: column.map(lambda value: f"{value:.2f}"))
        _draw_metric_heatmap(
            axes[0],
            success_norm,
            success_display,
            theme_cfg,
            title="Success",
            x_rotation=12,
            value_fontsize=7.0,
            title_pad=7.0,
        )

        settling_frame = pd.DataFrame(settling, index=row_labels, columns=col_labels)
        settling_norm = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
        for scenario_label in col_labels:
            settling_norm[scenario_label] = _normalize_metric_values(
                settling_frame[scenario_label].to_numpy(dtype=float),
                higher_is_better=False,
            )
        settling_display = settling_frame.apply(
            lambda column: column.map(lambda value: "NA" if pd.isna(value) else f"{float(value):.1f}")
        )
        _draw_metric_heatmap(
            axes[1],
            settling_norm,
            settling_display,
            theme_cfg,
            title="Settle",
            x_rotation=12,
            value_fontsize=6.9,
            title_pad=7.0,
        )

        failure_frame = pd.DataFrame(
            [
                [FAILURE_SHORT_LABELS[FAILURE_ORDER[int(value)]] for value in row]
                for row in failure_index
            ],
            index=row_labels,
            columns=col_labels,
        )
        failure_norm = pd.DataFrame(
            np.where(np.asarray(failure_frame == "ok"), 1.0, 0.28),
            index=row_labels,
            columns=col_labels,
        )
        _draw_metric_heatmap(
            axes[2],
            failure_norm,
            failure_frame,
            theme_cfg,
            title="Failure",
            x_rotation=12,
            value_fontsize=6.8,
            title_pad=7.0,
        )
        for ax in axes:
            ax.tick_params(axis="x", labelsize=7.4)
    fig.suptitle("Stress Comparison", x=0.065, y=0.982, ha="left", fontsize=11.5, color=theme_cfg.text_color)
    fig.text(0.065, 0.942, "Success, settle median, and dominant failure", ha="left", va="top", fontsize=7.2, color=theme_cfg.muted_color)
    path = figures_dir / FIGURE_FILENAMES["stress_comparison"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_metric_summary(runs: list[dict], figures_dir: Path, theme_cfg, base_dir: Path) -> Path:
    summary = _summary(base_dir)
    estimator = runs[0]["metadata"].get("estimator_name", "none")
    summary = summary[summary.get("estimator", "none") == estimator] if "estimator" in summary else summary
    controller_order = [controller for controller in CONTROLLER_ORDER if controller in set(summary.get("controller", []))]
    profile_rows = []
    for controller in controller_order:
        subset = summary[summary["controller"] == controller]
        profile_rows.append(
            {
                "label": _short_label(controller),
                "success_rate": float(subset["success_rate"].mean()) if "success_rate" in subset else np.nan,
                "settling_time_median": float(subset["settling_time_median"].median()) if "settling_time_median" in subset else np.nan,
                "steady_state_error_deg_median": float(subset["steady_state_error_deg_median"].median()) if "steady_state_error_deg_median" in subset else np.nan,
                "control_effort_median": float(subset["control_effort_median"].median()) if "control_effort_median" in subset else np.nan,
                "handoff_rate": float(subset["handoff_rate"].mean()) if "handoff_rate" in subset else np.nan,
                "balance_fraction_median": float(subset["balance_fraction_median"].median()) if "balance_fraction_median" in subset else np.nan,
            }
        )

    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.77, bottom=0.18)
    normalized, display = _build_profile_tables(
        pd.DataFrame(profile_rows),
        [
            ("success_rate", "Succ", True, "pct"),
            ("settling_time_median", "Settle", False, "float1"),
            ("steady_state_error_deg_median", "SSE", False, "float2"),
            ("control_effort_median", "Effort", False, "float1"),
            ("handoff_rate", "Hand", True, "pct"),
            ("balance_fraction_median", "Bal", True, "pct"),
        ],
    )
    _draw_metric_heatmap(
        ax,
        normalized,
        display,
        theme_cfg,
        title=None,
        value_fontsize=7.0,
    )
    fig.suptitle("Metric Summary", x=0.09, y=0.982, ha="left", fontsize=11.5, color=theme_cfg.text_color)
    fig.text(0.09, 0.942, "Aggregated nominal + stress medians", ha="left", va="top", fontsize=7.2, color=theme_cfg.muted_color)
    path = figures_dir / FIGURE_FILENAMES["metric_summary"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_monte_carlo_overview(
    figures_dir: Path,
    theme_cfg,
    base_dir: Path,
    controllers: tuple[str, ...],
    estimator_name: str,
) -> Path:
    summary, samples = _monte_summary(base_dir)
    labels = _required_labels(controllers)
    if not _monte_summary_has_all_labels(summary, samples, labels, estimator_name):
        raise RuntimeError("Monte Carlo overview requires complete controller coverage for the selected render set.")
    if not summary.empty and "estimator_name" in summary:
        summary = summary[summary["estimator_name"] == estimator_name]
    if not samples.empty and "estimator" in samples:
        samples = samples[samples["estimator"] == estimator_name]
    summary = summary[summary["controller_name"].isin(labels)] if not summary.empty else summary
    samples = samples[samples["controller"].isin(labels)] if not samples.empty else samples

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.14, wspace=0.16)
    for ax in axes:
        style_axis(ax, theme_cfg)

    if not summary.empty:
        controller_order = [controller for controller in CONTROLLER_ORDER if controller in set(summary["controller_name"])]
        values = [float(summary.loc[summary["controller_name"] == controller, "success_rate"].iloc[0]) for controller in controller_order]
        x = np.arange(len(controller_order))
        axes[0].bar(
            x,
            values,
            color=[soften(controller_color(theme_cfg, controller), 0.35) for controller in controller_order],
            edgecolor=[controller_color(theme_cfg, controller) for controller in controller_order],
            linewidth=1.0,
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([_short_label(controller) for controller in controller_order], fontsize=8.8)
        axes[0].set_ylim(0.0, 1.0)
        add_panel_title(axes[0], "Success", theme_cfg=theme_cfg)

    if not samples.empty:
        add_panel_title(axes[1], "Settling ECDF", theme_cfg=theme_cfg)
        add_panel_title(axes[2], "SSE Distribution", theme_cfg=theme_cfg)
        for controller in [controller for controller in CONTROLLER_ORDER if controller in set(samples["controller"])]:
            color = controller_color(theme_cfg, controller)
            subset = samples[(samples["controller"] == controller) & samples["settling_time"].notna()]
            if not subset.empty:
                values = np.sort(subset["settling_time"].to_numpy(dtype=float))
                ecdf = np.arange(1, len(values) + 1) / len(values)
                axes[1].plot(values, ecdf, color=color, linewidth=1.7, label=_short_label(controller))
            sse_subset = samples[samples["controller"] == controller]["steady_state_error_deg"].dropna().to_numpy(dtype=float)
            if sse_subset.size:
                axes[2].hist(
                    sse_subset,
                    bins=18,
                    alpha=0.18,
                    color=color,
                    edgecolor=color,
                    linewidth=0.8,
                    density=True,
                )
        handles, labels_for_legend = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels_for_legend, loc="lower right", fontsize=7.5, frameon=False)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("CDF")
        axes[2].set_xlabel("SSE [deg]")
        axes[2].set_ylabel("density")

    fig.suptitle("Monte Carlo Overview", x=0.07, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["monte_carlo_overview"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_handoff_focus(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    scenario_runs = [run for run in runs if run["metadata"]["scenario_name"] == "full_task_hanging"]
    controllers = [controller for controller in CONTROLLER_ORDER if any(run["metadata"]["controller_name"] == controller for run in scenario_runs)]
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 5.8), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12, hspace=0.20)
    for ax, title in zip(axes, ("Angle", "Cart", "Force"), strict=True):
        style_axis(ax, theme_cfg)
        add_panel_title(ax, title, theme_cfg=theme_cfg)
        add_event_band(ax, -0.1, 0.1, theme_cfg, alpha=0.07)
        ax.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9)

    for controller in controllers:
        controller_runs = [run for run in scenario_runs if run["metadata"]["controller_name"] == controller]
        color = controller_color(theme_cfg, controller)
        grid, angle_samples = _aligned_samples(controller_runs, "theta_deg", "first_balance_time", 1.0, 1.2)
        _, cart_samples = _aligned_samples(controller_runs, "x", "first_balance_time", 1.0, 1.2)
        _, force_samples = _aligned_samples(controller_runs, "u", "first_balance_time", 1.0, 1.2)
        if angle_samples.size == 0:
            continue
        plot_percentile_band(axes[0], grid, angle_samples, color, label=_short_label(controller))
        plot_percentile_band(axes[1], grid, cart_samples, color)
        plot_percentile_band(axes[2], grid, force_samples, color)

    axes[0].set_ylabel("Angle [deg]")
    axes[1].set_ylabel("Cart [m]")
    axes[2].set_ylabel("Force [N]")
    axes[2].set_xlabel("Time from balance entry [s]")
    _shared_legend(fig, theme_cfg, controllers)
    fig.suptitle("Handoff Focus", x=0.09, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["handoff_focus"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_constraint_usage(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    rows = []
    for run in runs:
        frame = run["frame"]
        rows.append(
            {
                "controller": run["metadata"]["controller_name"],
                "force_saturation_rate": float(frame["force_saturated"].mean()),
                "peak_track_usage": float(1.0 - np.min(frame["track_margin"]) / max(run["metadata"]["plant_params"]["track_limit"], 1e-6)),
                "max_abs_theta_dot": float(np.max(np.abs(frame["theta_dot"]))),
                "balance_fraction": float(run["metadata"]["diagnosis"]["balance_fraction"]),
            }
        )
    frame = pd.DataFrame(rows)
    controller_order = [controller for controller in CONTROLLER_ORDER if controller in set(frame["controller"])]
    metrics = [
        ("force_saturation_rate", "Sat. rate"),
        ("peak_track_usage", "Track usage"),
        ("max_abs_theta_dot", "Max |theta_dot|"),
        ("balance_fraction", "Balance frac"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.9))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.11, wspace=0.18, hspace=0.22)
    for ax, (column, title) in zip(axes.ravel(), metrics, strict=True):
        style_axis(ax, theme_cfg)
        add_panel_title(ax, title, theme_cfg=theme_cfg)
        for index, controller in enumerate(controller_order):
            values = frame.loc[frame["controller"] == controller, column].to_numpy(dtype=float)
            color = controller_color(theme_cfg, controller)
            ax.scatter(np.full_like(values, index, dtype=float), values, color=color, s=20, alpha=0.55)
            ax.scatter(index, float(np.median(values)), color=color, s=48, edgecolors=theme_cfg.background_color, linewidths=0.8)
        ax.set_xticks(np.arange(len(controller_order)))
        ax.set_xticklabels([_short_label(controller) for controller in controller_order], fontsize=8.8)
    fig.suptitle("Constraint Usage", x=0.08, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["constraint_usage"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_robustness_map(
    figures_dir: Path,
    theme_cfg,
    base_dir: Path,
    controllers: tuple[str, ...],
    estimator_name: str,
) -> Path:
    summary, samples = _monte_summary(base_dir)
    labels = _required_labels(controllers)
    if not _monte_summary_has_all_labels(summary, samples, labels, estimator_name):
        raise RuntimeError("Robustness map requires complete Monte Carlo coverage for the selected render set.")
    if "estimator" in samples:
        samples = samples[samples["estimator"] == estimator_name]
    samples = samples[samples["controller"].isin(labels)]

    selected = [controller for controller in CONTROLLER_ORDER if controller in set(samples["controller"])]
    fig, axes = plt.subplots(1, len(selected), figsize=(3.2 * len(selected), 3.3))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.18, wspace=0.18)
    noise_bins = pd.cut(samples["noise_theta_std_deg"], bins=4, labels=False, include_lowest=True)
    disturbance_bins = pd.cut(samples["disturbance_force"].abs(), bins=4, labels=False, include_lowest=True)
    samples = samples.assign(noise_bin=noise_bins, disturbance_bin=disturbance_bins)
    cmap = LinearSegmentedColormap.from_list("robust", ["#F6F6F6", soften(theme_cfg.accent_color, 0.55), "#6C976A"])
    for ax, controller in zip(axes, selected, strict=True):
        style_axis(ax, theme_cfg)
        subset = samples[samples["controller"] == controller]
        pivot = subset.pivot_table(
            index="noise_bin",
            columns="disturbance_bin",
            values="success",
            aggfunc="mean",
        ).sort_index(ascending=True)
        mat = pivot.to_numpy(dtype=float) if not pivot.empty else np.zeros((4, 4), dtype=float)
        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_xlabel("|disturbance| bin")
        ax.set_ylabel("noise bin")
        controller_badge(ax, _short_label(controller), controller_color(theme_cfg, controller), theme_cfg)
    fig.suptitle("Robustness Map", x=0.06, y=0.96, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["robustness_map"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_energy_phase(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    scenario_runs = [run for run in runs if run["metadata"]["scenario_name"] == "full_task_hanging"]
    controllers = [controller for controller in CONTROLLER_ORDER if any(run["metadata"]["controller_name"] == controller for run in scenario_runs)]
    selected = [_representative_run(scenario_runs, "full_task_hanging", controller) for controller in controllers]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.16, wspace=0.16)
    for ax in axes:
        style_axis(ax, theme_cfg)
    add_panel_title(axes[0], "Energy Gap", theme_cfg=theme_cfg)
    add_panel_title(axes[1], "Phase", theme_cfg=theme_cfg)
    add_panel_title(axes[2], "Track Margin", theme_cfg=theme_cfg)

    for run in selected:
        controller = run["metadata"]["controller_name"]
        color = controller_color(theme_cfg, controller)
        frame = run["frame"]
        axes[0].plot(frame["t"], frame["energy_gap"], color=color, linewidth=1.7)
        axes[1].plot(frame["theta_deg"], frame["theta_dot"], color=color, linewidth=1.7)
        axes[2].plot(frame["t"], frame["track_margin"], color=color, linewidth=1.7)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("J")
    axes[1].set_xlabel("Angle [deg]")
    axes[1].set_ylabel("Angular rate [rad/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Margin [m]")
    _shared_legend(fig, theme_cfg, controllers, x=0.985, y=0.975)
    fig.suptitle("Energy and Phase", x=0.07, y=0.97, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["energy_phase"]
    save_figure(fig, path, theme_cfg)
    return path
