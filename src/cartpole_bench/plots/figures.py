from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

from cartpole_bench.plots.style import (
    MODE_COLORS,
    add_panel_title,
    apply_theme,
    controller_color,
    save_figure,
    soften,
    style_axis,
)
from cartpole_bench.plots.tables import refresh_metric_tables
from cartpole_bench.simulation.recorder import load_saved_runs


CONTROLLER_ORDER = ["LQR", "Feedback Linearization (PFL)", "Sliding Mode Control (SMC)"]
FIGURE_FILENAMES = {
    "nominal_local_response": "nominal_local_response.png",
    "full_task_handoff": "full_task_handoff.png",
    "stress_comparison": "stress_comparison.png",
    "metric_summary": "metric_summary.png",
    "monte_carlo_overview": "monte_carlo_overview.png",
}
SUPPLEMENTAL_FIGURE_FILENAMES = {
    "cartpole_schematic": "cartpole_schematic.png",
    "handoff_focus": "handoff_focus.png",
}


def _short_label(controller: str) -> str:
    return {
        "LQR": "LQR",
        "Feedback Linearization (PFL)": "PFL",
        "Sliding Mode Control (SMC)": "SMC",
    }.get(controller, controller)


def _ordered_controllers(runs: list[dict]) -> list[str]:
    present = {run["metadata"]["controller_name"] for run in runs}
    return [controller for controller in CONTROLLER_ORDER if controller in present]


def _legend_handles(theme_cfg, controllers: list[str]) -> tuple[list[Line2D], list[str]]:
    handles = [Line2D([0], [0], color=controller_color(theme_cfg, controller), linewidth=1.8) for controller in controllers]
    labels = [_short_label(controller) for controller in controllers]
    return handles, labels


def _add_shared_legend(
    fig,
    theme_cfg,
    controllers: list[str],
    *,
    x: float = 0.985,
    y: float = 0.988,
    loc: str = "upper right",
    ncol: int | None = None,
) -> None:
    handles, labels = _legend_handles(theme_cfg, controllers)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=(x, y),
        ncol=max(1, ncol if ncol is not None else len(labels)),
        fontsize=8.0,
        handlelength=1.9,
        columnspacing=1.0,
    )


def _run_score(run: dict) -> tuple[float, float, float]:
    metadata = run["metadata"]
    metrics = metadata["metrics"]
    diagnosis = metadata["diagnosis"]
    return (
        0.0 if metrics["success"] else 1.0,
        float(metrics["settling_time"]) if metrics["settling_time"] is not None else 999.0,
        float(metrics["final_abs_theta_deg"]) + 0.25 * float(diagnosis["max_abs_x"]),
    )


def _representative_run(runs: list[dict], scenario: str, controller: str) -> dict:
    candidates = [
        run
        for run in runs
        if run["metadata"]["scenario_name"] == scenario and run["metadata"]["controller_name"] == controller
    ]
    if not candidates:
        raise KeyError(f"Missing representative run for scenario={scenario}, controller={controller}")
    successful = [run for run in candidates if run["metadata"]["metrics"]["success"]]
    if successful:
        settling = np.asarray([run["metadata"]["metrics"]["settling_time"] for run in successful], dtype=float)
        median_settling = float(np.median(settling))
        return min(successful, key=lambda run: abs(float(run["metadata"]["metrics"]["settling_time"]) - median_settling))
    return min(candidates, key=_run_score)


def _representative_seed(runs: list[dict], scenario: str) -> int:
    seed_rows = {}
    for run in runs:
        if run["metadata"]["scenario_name"] != scenario:
            continue
        seed = int(run["metadata"]["seed"])
        seed_rows.setdefault(seed, {"success": 0, "handoff": 0, "score": 0.0})
        seed_rows[seed]["success"] += int(bool(run["metadata"]["metrics"]["success"]))
        seed_rows[seed]["handoff"] += int(run["metadata"]["diagnosis"]["first_balance_time"] is not None)
        seed_rows[seed]["score"] += _run_score(run)[2]
    return max(seed_rows, key=lambda seed: (seed_rows[seed]["success"], seed_rows[seed]["handoff"], -seed_rows[seed]["score"]))


def _short_scenario_label(scenario: str) -> str:
    return {
        "measurement_noise": "Noise",
        "impulse_disturbance": "Impulse",
        "friction_and_damping": "Friction",
        "large_angle_recovery": "Large Angle",
        "parameter_mismatch": "Mismatch",
    }.get(scenario, scenario.replace("_", " ").title())


def _window_around_time(frame: pd.DataFrame, center_time: float, before: float, after: float) -> pd.DataFrame:
    cropped = frame[(frame["t"] >= center_time - before) & (frame["t"] <= center_time + after)].copy()
    cropped["t_rel"] = cropped["t"] - center_time
    return cropped


def _require_render_runs(base_dir: Path, runs: list[dict]) -> None:
    if runs:
        return
    raise RuntimeError(
        f"No saved nominal/stress run artifacts were found in '{base_dir}'. "
        "The render command reads previously generated CSV/JSON run data from the same artifact root that it writes figures into. "
        "Use an artifact directory that already contains suite outputs, such as your existing benchmark root, or rerun the suites first."
    )


def generate_figures(base_dir: Path, theme: str = "paper_white", include_supplements: bool = True) -> list[Path]:
    theme_cfg = apply_theme(theme)
    refresh_metric_tables(base_dir)
    runs = load_saved_runs(base_dir, suites={"nominal", "stress"})
    _require_render_runs(base_dir, runs)
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    created = [
        _figure_nominal_local_response(runs, figures_dir, theme_cfg),
        _figure_full_task_handoff(runs, figures_dir, theme_cfg),
        _figure_stress_comparison(figures_dir, theme_cfg, base_dir),
        _figure_metric_summary(figures_dir, theme_cfg, base_dir),
        _figure_monte_carlo_overview(figures_dir, theme_cfg, base_dir),
    ]
    if include_supplements:
        supplemental_dir = figures_dir / "supplemental"
        supplemental_dir.mkdir(parents=True, exist_ok=True)
        created.extend(
            [
                _figure_cartpole_schematic(supplemental_dir, theme_cfg),
                _figure_handoff_focus(runs, supplemental_dir, theme_cfg),
            ]
        )
    return created


def _figure_nominal_local_response(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    controllers = _ordered_controllers(runs)
    selected = [_representative_run(runs, "local_small_angle", controller) for controller in controllers]

    fig = plt.figure(figsize=(11.7, 6.7))
    gs = fig.add_gridspec(2, 2)
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_cart = fig.add_subplot(gs[0, 1])
    ax_force = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.11, wspace=0.16, hspace=0.24)
    axes = [ax_angle, ax_cart, ax_force, ax_phase]
    for ax in axes:
        style_axis(ax, theme_cfg)

    phase_angles = []
    phase_rates = []
    for run in selected:
        controller = run["metadata"]["controller_name"]
        color = controller_color(theme_cfg, controller)
        frame = run["frame"]
        theta = frame["theta"].to_numpy(dtype=float)
        theta_deg = np.rad2deg(np.arctan2(np.sin(theta), np.cos(theta)))
        x = frame["t"].to_numpy(dtype=float)
        theta_rate = frame["theta_dot"].to_numpy(dtype=float)
        ax_angle.plot(x, theta_deg, color=color, linewidth=1.55)
        ax_cart.plot(x, frame["x"].to_numpy(dtype=float), color=color, linewidth=1.55)
        ax_force.plot(x, frame["u"].to_numpy(dtype=float), color=color, linewidth=1.55)
        ax_phase.plot(theta_deg, theta_rate, color=color, linewidth=1.55)
        phase_angles.extend(theta_deg.tolist())
        phase_rates.extend(theta_rate.tolist())

    add_panel_title(ax_angle, "Angle", theme_cfg=theme_cfg)
    add_panel_title(ax_cart, "Cart", theme_cfg=theme_cfg)
    add_panel_title(ax_force, "Force", theme_cfg=theme_cfg)
    add_panel_title(ax_phase, "Phase", theme_cfg=theme_cfg)

    ax_angle.set_ylabel("Angle [deg]")
    ax_cart.set_ylabel("Position [m]")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_xlabel("Time [s]")
    ax_phase.set_xlabel("Angle [deg]")
    ax_phase.set_ylabel("Angular rate [rad/s]")
    ax_angle.tick_params(labelbottom=False)
    ax_cart.tick_params(labelbottom=False)

    if phase_angles and phase_rates:
        angle_pad = 0.75
        rate_pad = 0.05
        ax_phase.set_xlim(min(phase_angles) - angle_pad, max(phase_angles) + angle_pad)
        ax_phase.set_ylim(min(phase_rates) - rate_pad, max(phase_rates) + rate_pad)

    _add_shared_legend(fig, theme_cfg, controllers)
    fig.suptitle("Nominal Local Response", x=0.075, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["nominal_local_response"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_full_task_handoff(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    controllers = _ordered_controllers(runs)
    selected = [_representative_run(runs, "full_task_hanging", controller) for controller in controllers]

    fig = plt.figure(figsize=(11.7, 6.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 1.25, 1.25, 0.55])
    ax_angle = fig.add_subplot(gs[0, 0])
    ax_cart = fig.add_subplot(gs[1, 0], sharex=ax_angle)
    ax_force = fig.add_subplot(gs[2, 0], sharex=ax_angle)
    ax_modes = fig.add_subplot(gs[3, 0], sharex=ax_angle)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.11, hspace=0.28)
    for ax in (ax_angle, ax_cart, ax_force):
        style_axis(ax, theme_cfg)
    ax_modes.set_facecolor(theme_cfg.panel_color)

    handoff_times = []
    mode_to_index = {"energy_pump": 1, "capture_assist": 2, "balance": 3, "initial": 0}
    t_max = min(8.0, max(float(run["frame"]["t"].max()) for run in selected))
    max_steps = max(int((run["frame"]["t"] <= t_max).sum()) for run in selected)
    mode_matrix = np.zeros((len(selected), max_steps))

    for row_index, run in enumerate(selected):
        controller = run["metadata"]["controller_name"]
        color = controller_color(theme_cfg, controller)
        frame = run["frame"]
        theta_deg = np.rad2deg(np.unwrap(frame["theta"].to_numpy(dtype=float)))
        theta_deg = theta_deg - theta_deg[-1]
        time = frame["t"].to_numpy(dtype=float)
        ax_angle.plot(time, theta_deg, color=color, linewidth=1.55)
        ax_cart.plot(time, frame["x"].to_numpy(dtype=float), color=color, linewidth=1.55)
        ax_force.plot(time, frame["u"].to_numpy(dtype=float), color=color, linewidth=1.55)

        handoff_time = run["metadata"]["diagnosis"]["first_balance_time"]
        if handoff_time is not None:
            handoff_time = float(handoff_time)
            handoff_times.append(handoff_time)
            ax_angle.axvline(handoff_time, color=color, linestyle="--", linewidth=0.75, alpha=0.6)

        cropped_modes = frame.loc[frame["t"] <= t_max, "mode"].astype(str).tolist()
        mode_values = [mode_to_index.get(mode, 0) for mode in cropped_modes]
        mode_matrix[row_index, : len(mode_values)] = mode_values

    if handoff_times:
        handoff_center = float(np.median(handoff_times))
        for ax in (ax_angle, ax_cart, ax_force):
            ax.axvspan(handoff_center - 0.15, handoff_center + 0.15, color=theme_cfg.accent_color, alpha=0.06, linewidth=0.0)

    ax_angle.set_ylabel("Angle [deg]")
    ax_cart.set_ylabel("Cart [m]")
    ax_force.set_ylabel("Force [N]")
    ax_modes.set_yticks(np.arange(len(selected)))
    ax_modes.set_yticklabels([_short_label(controller) for controller in controllers])
    ax_modes.set_xlabel("Time [s]")
    cmap = ListedColormap(
        [
            soften(MODE_COLORS["initial"], 0.2),
            MODE_COLORS["energy_pump"],
            MODE_COLORS["capture_assist"],
            MODE_COLORS["balance"],
        ]
    )
    extent = [0.0, t_max, -0.5, len(selected) - 0.5]
    ax_modes.imshow(mode_matrix, aspect="auto", cmap=cmap, interpolation="nearest", origin="lower", extent=extent, vmin=0, vmax=3)
    for row_boundary in np.arange(-0.5, len(selected), 1.0):
        ax_modes.axhline(row_boundary, color=theme_cfg.background_color, linewidth=1.1)
    ax_modes.spines[:].set_visible(False)
    ax_modes.tick_params(axis="both", length=0, labelsize=8.5)
    ax_angle.tick_params(labelbottom=False)
    ax_cart.tick_params(labelbottom=False)
    ax_angle.set_xlim(0.0, t_max)

    _add_shared_legend(fig, theme_cfg, controllers)
    fig.suptitle("Full-Task Handoff", x=0.08, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["full_task_handoff"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_stress_comparison(figures_dir: Path, theme_cfg, base_dir: Path) -> Path:
    summary = pd.read_csv(base_dir / "tables" / "metric_summary.csv")
    stress = summary[summary["suite"] == "stress"].copy()
    controllers = [controller for controller in CONTROLLER_ORDER if controller in set(stress["controller"])]
    scenarios = list(dict.fromkeys(stress["scenario"].tolist()))
    success = (
        stress.pivot(index="controller", columns="scenario", values="success_rate")
        .reindex(index=controllers, columns=scenarios)
        .fillna(0.0)
    )
    settling = (
        stress.pivot(index="controller", columns="scenario", values="settling_time_median")
        .reindex(index=controllers, columns=scenarios)
        .astype(float)
    )

    finite_settling = settling.to_numpy(dtype=float)
    finite_values = finite_settling[np.isfinite(finite_settling)]
    if finite_values.size:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        if high > low:
            normalized = 1.0 - (settling - low) / (high - low)
        else:
            normalized = pd.DataFrame(0.65, index=settling.index, columns=settling.columns)
    else:
        normalized = pd.DataFrame(0.5, index=settling.index, columns=settling.columns)
    normalized = normalized.fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.6, 3.9))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.84, bottom=0.20)
    cmap = LinearSegmentedColormap.from_list(
        "stress_settling",
        ["#F7F7F7", soften(theme_cfg.accent_color, 0.58), "#7FA67B"],
    )
    ax.imshow(normalized.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([_short_scenario_label(scenario) for scenario in scenarios], fontsize=9)
    ax.set_yticks(np.arange(len(controllers)))
    ax.set_yticklabels([_short_label(controller) for controller in controllers], fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(scenarios), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(controllers), 1.0), minor=True)
    ax.grid(which="minor", color=theme_cfg.background_color, linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index, controller in enumerate(controllers):
        for col_index, scenario in enumerate(scenarios):
            success_rate = success.loc[controller, scenario]
            ax.text(
                col_index,
                row_index,
                f"{success_rate:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=theme_cfg.text_color,
            )
    fig.text(
        0.10,
        0.06,
        "text = success rate, color = median settling time",
        fontsize=8,
        color=theme_cfg.muted_color,
        ha="left",
    )
    fig.suptitle("Stress Comparison", x=0.10, y=0.965, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["stress_comparison"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_metric_summary(figures_dir: Path, theme_cfg, base_dir: Path) -> Path:
    summary = pd.read_csv(base_dir / "tables" / "metric_summary.csv")
    nominal = summary[summary["suite"] == "nominal"].copy()
    controllers = [controller for controller in CONTROLLER_ORDER if controller in set(nominal["controller"])]
    metric_specs = [
        ("settling_time_median", "Settling"),
        ("overshoot_deg_median", "Overshoot"),
        ("steady_state_error_deg_median", "SSE"),
        ("control_effort_median", "Effort"),
    ]
    y_labels = {
        "settling_time_median": "s",
        "overshoot_deg_median": "deg",
        "steady_state_error_deg_median": "deg",
        "control_effort_median": "u^2 dt",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 6.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.11, wspace=0.14, hspace=0.18)
    for ax, (column, label) in zip(axes.ravel(), metric_specs, strict=True):
        style_axis(ax, theme_cfg)
        add_panel_title(ax, label, theme_cfg=theme_cfg)
        metric_data = []
        for controller in controllers:
            values = nominal[nominal["controller"] == controller][column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                metric_data.append((controller, np.nan, np.nan, np.nan))
            else:
                metric_data.append((controller, float(np.min(values)), float(np.median(values)), float(np.max(values))))
        finite_values = [value for row in metric_data for value in row[1:] if np.isfinite(value)]
        if finite_values:
            low = min(finite_values)
            high = max(finite_values)
            pad = 0.04 * (high - low if high > low else 1.0)
            ax.set_ylim(low - pad, high + pad)
        for index, (controller, low, mid, high) in enumerate(metric_data):
            color = controller_color(theme_cfg, controller)
            ax.vlines(index, low, high, color=color, linewidth=1.6, alpha=0.42)
            ax.scatter(index, mid, color=color, s=38, zorder=3)
        ax.set_xticks(np.arange(len(controllers)))
        ax.set_xticklabels([_short_label(controller) for controller in controllers], fontsize=9)
        ax.set_ylabel(y_labels[column], fontsize=9)

    fig.suptitle("Nominal Metrics", x=0.08, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["metric_summary"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_monte_carlo_overview(figures_dir: Path, theme_cfg, base_dir: Path) -> Path:
    summary_path = base_dir / "tables" / "monte_carlo_summary.csv"
    sample_path = base_dir / "tables" / "monte_carlo_samples.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    samples = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    controllers = [
        controller
        for controller in CONTROLLER_ORDER
        if controller in set(summary.get("controller_name", [])) or controller in set(samples.get("controller", []))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.14, wspace=0.16)
    for ax in axes:
        style_axis(ax, theme_cfg)

    if not summary.empty:
        x = np.arange(len(controllers))
        values = [float(summary.loc[summary["controller_name"] == controller, "success_rate"].iloc[0]) for controller in controllers]
        bars = axes[0].bar(
            x,
            values,
            width=0.58,
            color=[soften(controller_color(theme_cfg, controller), 0.3) for controller in controllers],
            edgecolor=[controller_color(theme_cfg, controller) for controller in controllers],
            linewidth=1.0,
        )
        for bar, value in zip(bars, values, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2.0, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=7.8, color=theme_cfg.text_color)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([_short_label(controller) for controller in controllers], fontsize=9)
        axes[0].set_ylim(0.0, 1.0)
        add_panel_title(axes[0], "Success", theme_cfg=theme_cfg)

    if not samples.empty:
        legend_handles = []
        legend_labels = []
        for controller in controllers:
            subset = samples[(samples["controller"] == controller) & samples["settling_time"].notna()].copy()
            if subset.empty:
                continue
            values = np.sort(subset["settling_time"].to_numpy(dtype=float))
            ecdf = np.arange(1, len(values) + 1) / len(values)
            color = controller_color(theme_cfg, controller)
            line, = axes[1].plot(values, ecdf, color=color, linewidth=1.5)
            legend_handles.append(line)
            legend_labels.append(_short_label(controller))
        add_panel_title(axes[1], "Settling", theme_cfg=theme_cfg)
        axes[1].set_xlabel("Time [s]", fontsize=9)
        axes[1].set_ylabel("CDF", fontsize=9)
        if legend_handles:
            axes[1].legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                fontsize=7.7,
                ncol=3,
                handlelength=1.8,
                columnspacing=0.9,
            )

        grouped = [
            samples.loc[samples["controller"] == controller, "steady_state_error_deg"].dropna().to_numpy(dtype=float)
            for controller in controllers
        ]
        if any(len(values) for values in grouped):
            bp = axes[2].boxplot(grouped, patch_artist=True, labels=[_short_label(controller) for controller in controllers], showfliers=False, widths=0.5)
            for patch, controller in zip(bp["boxes"], controllers, strict=True):
                patch.set(facecolor=soften(controller_color(theme_cfg, controller), 0.84), edgecolor=controller_color(theme_cfg, controller), linewidth=1.0)
            for median, controller in zip(bp["medians"], controllers, strict=True):
                median.set(color=controller_color(theme_cfg, controller), linewidth=1.4)
            for whisker, controller in zip(bp["whiskers"], np.repeat(controllers, 2), strict=True):
                whisker.set(color=controller_color(theme_cfg, controller), linewidth=0.9, alpha=0.8)
            for cap, controller in zip(bp["caps"], np.repeat(controllers, 2), strict=True):
                cap.set(color=controller_color(theme_cfg, controller), linewidth=0.9, alpha=0.8)
            add_panel_title(axes[2], "SSE", theme_cfg=theme_cfg)
            axes[2].set_ylabel("SSE [deg]", fontsize=9)

    fig.suptitle("Monte Carlo Overview", x=0.07, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / FIGURE_FILENAMES["monte_carlo_overview"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_cartpole_schematic(figures_dir: Path, theme_cfg) -> Path:
    fig, ax = plt.subplots(figsize=(8.1, 4.6))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.08)
    ax.set_facecolor(theme_cfg.panel_color)
    ax.set_xlim(-2.2, 2.4)
    ax.set_ylim(-0.55, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ground_y = 0.0
    ax.plot([-2.0, 2.0], [ground_y, ground_y], color=theme_cfg.text_color, linewidth=1.0, alpha=0.7)

    cart_x = 0.0
    cart_y = 0.08
    cart_w = 0.8
    cart_h = 0.28
    pend_len = 1.45
    theta = np.deg2rad(16.0)
    pivot = (cart_x, cart_y + cart_h)
    bob = (pivot[0] + pend_len * np.sin(theta), pivot[1] + pend_len * np.cos(theta))
    lqr_color = controller_color(theme_cfg, "LQR")
    smc_color = controller_color(theme_cfg, "Sliding Mode Control (SMC)")

    cart = patches.FancyBboxPatch(
        (cart_x - cart_w / 2, cart_y),
        cart_w,
        cart_h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.4,
        edgecolor=theme_cfg.text_color,
        facecolor=soften(lqr_color, 0.86),
    )
    ax.add_patch(cart)
    ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color=lqr_color, linewidth=2.0)
    ax.add_patch(patches.Circle(bob, 0.08, edgecolor=lqr_color, facecolor=theme_cfg.panel_color, linewidth=1.4))
    ax.add_patch(patches.Arc((pivot[0], pivot[1]), 0.9, 0.9, theta1=74, theta2=90, color=theme_cfg.accent_color, linewidth=1.0))

    ax.annotate("", xy=(1.45, cart_y + 0.15), xytext=(0.55, cart_y + 0.15), arrowprops={"arrowstyle": "->", "color": smc_color, "linewidth": 1.4})
    ax.text(1.5, cart_y + 0.18, r"$u$", fontsize=11, color=theme_cfg.text_color, va="bottom")

    ax.annotate("", xy=(0.0, -0.12), xytext=(0.7, -0.12), arrowprops={"arrowstyle": "<->", "color": theme_cfg.muted_color, "linewidth": 1.0})
    ax.text(0.35, -0.2, r"$x$", fontsize=11, color=theme_cfg.text_color, ha="center")
    ax.text(pivot[0] + 0.18, pivot[1] + 0.35, r"$\theta$", fontsize=11, color=theme_cfg.text_color)
    ax.text(cart_x, cart_y + 0.14, "cart", fontsize=9, color=theme_cfg.text_color, ha="center", va="center")
    ax.text(bob[0] + 0.18, bob[1] + 0.02, "pendulum", fontsize=9, color=theme_cfg.text_color, va="center")
    ax.plot([pivot[0], pivot[0]], [pivot[1], pivot[1] + 1.55], color=theme_cfg.muted_color, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.text(pivot[0] - 0.05, pivot[1] + 1.65, "upright", fontsize=8.5, color=theme_cfg.muted_color, ha="right")
    ax.text(pivot[0] + 0.33, pivot[1] - 0.55, "downward", fontsize=8.2, color=theme_cfg.muted_color, ha="left")
    ax.text(-1.95, 2.08, r"State: $[x,\ \dot{x},\ \theta,\ \dot{\theta}]$", fontsize=9, color=theme_cfg.text_color, ha="left")

    ax.text(-1.95, 1.85, "Modes", fontsize=8.5, color=theme_cfg.muted_color, ha="left")
    mode_x = -1.95
    for name, color in [("energy pump", MODE_COLORS["energy_pump"]), ("capture assist", MODE_COLORS["capture_assist"]), ("balance", MODE_COLORS["balance"])]:
        ax.add_patch(patches.Rectangle((mode_x, 1.58), 0.18, 0.08, facecolor=color, edgecolor="none"))
        ax.text(mode_x + 0.28, 1.61, name, fontsize=8.3, color=theme_cfg.text_color, va="center", ha="left")
        mode_x += 1.05

    fig.suptitle("Cart-Pole Schematic", x=0.06, y=0.97, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["cartpole_schematic"]
    save_figure(fig, path, theme_cfg)
    return path


def _figure_handoff_focus(runs: list[dict], figures_dir: Path, theme_cfg) -> Path:
    controllers = _ordered_controllers(runs)
    selected = [_representative_run(runs, "full_task_hanging", controller) for controller in controllers]
    windowed = []
    for run in selected:
        handoff_time = run["metadata"]["diagnosis"]["first_balance_time"]
        if handoff_time is None:
            continue
        windowed.append((run["metadata"]["controller_name"], _window_around_time(run["frame"], float(handoff_time), 1.0, 1.2)))

    fig, axes = plt.subplots(3, 1, figsize=(10.3, 5.6), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.12, hspace=0.22)
    y_labels = ["Angle [deg]", "Cart [m]", "Force [N]"]
    titles = ["Angle", "Cart", "Force"]

    for ax, title, ylabel in zip(axes, titles, y_labels, strict=True):
        style_axis(ax, theme_cfg)
        add_panel_title(ax, title, theme_cfg=theme_cfg)
        ax.axvline(0.0, color=theme_cfg.accent_color, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Time from balance entry [s]")

    for controller, frame in windowed:
        color = controller_color(theme_cfg, controller)
        theta_deg = np.rad2deg(np.unwrap(frame["theta"].to_numpy(dtype=float)))
        theta_deg = theta_deg - theta_deg[np.argmin(np.abs(frame["t_rel"].to_numpy(dtype=float)))]
        axes[0].plot(frame["t_rel"], theta_deg, color=color, linewidth=1.75)
        axes[1].plot(frame["t_rel"], frame["x"], color=color, linewidth=1.75)
        axes[2].plot(frame["t_rel"], frame["u"], color=color, linewidth=1.75)

    axes[0].set_xlim(-1.0, 1.2)
    _add_shared_legend(fig, theme_cfg, [controller for controller, _ in windowed])
    fig.suptitle("Handoff Focus", x=0.09, y=0.975, ha="left", fontsize=12.2, color=theme_cfg.text_color)
    path = figures_dir / SUPPLEMENTAL_FIGURE_FILENAMES["handoff_focus"]
    save_figure(fig, path, theme_cfg)
    return path
