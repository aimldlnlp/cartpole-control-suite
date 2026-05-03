from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cartpole_bench.animations.artist import CartPoleArtist
from cartpole_bench.config import load_video_config
from cartpole_bench.plots.style import add_panel_title, apply_theme, controller_badge, controller_color, soften, style_axis
from cartpole_bench.simulation.recorder import load_saved_runs, prune_metadata_render_paths, update_metadata_render_paths
from cartpole_bench.utils.progress import NullProgressReporter, PhaseTimer, ProgressEvent, ProgressReporter


CONTROLLER_LABELS = {
    "lqr": "LQR",
    "pfl": "Feedback Linearization (PFL)",
    "smc": "Sliding Mode Control (SMC)",
    "ilqr": "Iterative LQR (iLQR)",
    "mpc": "Model Predictive Control (MPC)",
}
CONTROLLER_ORDER = [CONTROLLER_LABELS[key] for key in ("lqr", "pfl", "smc", "ilqr", "mpc")]
ANIMATION_STEMS = {
    "LQR": "lqr_full_task_nominal",
    "Feedback Linearization (PFL)": "pfl_full_task_nominal",
    "Sliding Mode Control (SMC)": "smc_full_task_nominal",
    "Iterative LQR (iLQR)": "ilqr_full_task_nominal",
    "Model Predictive Control (MPC)": "mpc_full_task_nominal",
    "nominal_comparison": "side_by_side_nominal_comparison",
    "stress_comparison": "side_by_side_stress_comparison",
}
SUPPLEMENTAL_ANIMATION_STEMS = {
    "handoff_focus": "handoff_focus_comparison",
    "disturbance_focus": "disturbance_focus_comparison",
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
        f"No saved nominal/stress run artifacts were found in '{base_dir}' for the requested controller/estimator selection."
    )


def _run_score(run: dict) -> tuple[float, float, float]:
    metadata = run["metadata"]
    return (
        0.0 if metadata["metrics"]["success"] else 1.0,
        float(metadata["metrics"]["settling_time"]) if metadata["metrics"]["settling_time"] is not None else 999.0,
        float(metadata["metrics"]["final_abs_theta_deg"]) + 0.25 * float(metadata["diagnosis"]["max_abs_x"]),
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
        raise KeyError(f"Missing run for scenario={scenario}, controller={controller}")
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


def _scenario_has_controller(runs: list[dict], scenario: str, controller: str) -> bool:
    return any(
        run["metadata"]["scenario_name"] == scenario and run["metadata"]["controller_name"] == controller
        for run in runs
    )


def _seed_has_all_labels(runs: list[dict], scenario_name: str, seed: int, required_labels: list[str]) -> bool:
    labels = {
        run["metadata"]["controller_name"]
        for run in runs
        if run["metadata"]["scenario_name"] == scenario_name and int(run["metadata"]["seed"]) == seed
    }
    return set(required_labels).issubset(labels)


def _best_complete_seed_for_scenario(runs: list[dict], scenario: str, required_labels: list[str]) -> int | None:
    seed_rows: dict[int, dict[str, float]] = {}
    for run in runs:
        if run["metadata"]["scenario_name"] != scenario:
            continue
        seed = int(run["metadata"]["seed"])
        seed_rows.setdefault(seed, {"success": 0.0, "handoff": 0.0, "score": 0.0})
        metrics = run["metadata"].get("metrics", {})
        diagnosis = run["metadata"].get("diagnosis", {})
        seed_rows[seed]["success"] += float(metrics.get("success", False))
        seed_rows[seed]["handoff"] += float(diagnosis.get("first_balance_time") is not None)
        score = _run_score(run)[2] if "metrics" in run["metadata"] and "diagnosis" in run["metadata"] else 0.0
        seed_rows[seed]["score"] += score
    if not seed_rows:
        return None
    complete_seeds = [seed for seed in seed_rows if _seed_has_all_labels(runs, scenario, seed, required_labels)]
    if not complete_seeds:
        return None
    return max(complete_seeds, key=lambda seed: (seed_rows[seed]["success"], seed_rows[seed]["handoff"], -seed_rows[seed]["score"]))


def _best_seed_for_scenario(runs: list[dict], scenario: str, required_labels: list[str] | None = None) -> int | None:
    if required_labels is None:
        required_labels = sorted(
            {
                run["metadata"]["controller_name"]
                for run in runs
                if run["metadata"]["scenario_name"] == scenario
            }
        )
    return _best_complete_seed_for_scenario(runs, scenario, required_labels)


def _runs_for_seed(runs: list[dict], scenario: str, seed: int) -> list[dict]:
    selected = [run for run in runs if run["metadata"]["scenario_name"] == scenario and int(run["metadata"]["seed"]) == seed]
    return sorted(selected, key=lambda run: CONTROLLER_ORDER.index(run["metadata"]["controller_name"]))


def _focus_crop_has_all_labels(
    selected_runs: list[dict],
    center_key: str,
    before: float,
    after: float,
    required_labels: list[str],
) -> bool:
    labels = {run["metadata"]["controller_name"] for run in selected_runs}
    if set(required_labels) - labels:
        return False
    for run in selected_runs:
        frame = run["frame"]
        if center_key == "disturbance_time":
            disturbance = frame["disturbance"].to_numpy(dtype=float)
            active = np.flatnonzero(np.abs(disturbance) > 1e-9)
            center = float(frame["t"].iloc[active[0]]) if active.size else float(frame["t"].iloc[0])
        else:
            center_value = run["metadata"]["diagnosis"].get(center_key)
            if center_value is None:
                return False
            center = float(center_value)
        rel = frame["t"].to_numpy(dtype=float) - center
        if np.count_nonzero((rel >= -before) & (rel <= after)) == 0:
            return False
    return True


def _animation_output_paths(base_dir: Path, controllers: tuple[str, ...], formats: tuple[str, ...]) -> dict[str, set[Path]]:
    animations_dir = base_dir / "animations"
    supplemental_dir = animations_dir / "supplemental"
    required_labels = _required_labels(controllers)
    output_paths: dict[str, set[Path]] = {}
    for controller in required_labels:
        if controller in ANIMATION_STEMS:
            output_paths[controller] = {
                animations_dir / f"{ANIMATION_STEMS[controller]}.{fmt}"
                for fmt in formats
            }
    output_paths["nominal_comparison"] = {
        animations_dir / f"{ANIMATION_STEMS['nominal_comparison']}.{fmt}"
        for fmt in formats
    }
    output_paths["stress_comparison"] = {
        animations_dir / f"{ANIMATION_STEMS['stress_comparison']}.{fmt}"
        for fmt in formats
    }
    output_paths["handoff_focus"] = {
        supplemental_dir / f"{SUPPLEMENTAL_ANIMATION_STEMS['handoff_focus']}.{fmt}"
        for fmt in formats
    }
    output_paths["disturbance_focus"] = {
        supplemental_dir / f"{SUPPLEMENTAL_ANIMATION_STEMS['disturbance_focus']}.{fmt}"
        for fmt in formats
    }
    return output_paths


def _eligible_animation_keys(
    runs: list[dict],
    controllers: tuple[str, ...],
    *,
    include_supplements: bool,
) -> list[str]:
    required_labels = _required_labels(controllers)
    eligible: list[str] = []
    for controller in [label for label in CONTROLLER_ORDER if label in required_labels]:
        if _scenario_has_controller(runs, "full_task_hanging", controller):
            eligible.append(controller)

    nominal_seed = _best_seed_for_scenario(runs, "full_task_hanging", required_labels)
    stress_seed = _best_seed_for_scenario(runs, "impulse_disturbance", required_labels)
    nominal_runs = _runs_for_seed(runs, "full_task_hanging", nominal_seed) if nominal_seed is not None else []
    stress_runs = _runs_for_seed(runs, "impulse_disturbance", stress_seed) if stress_seed is not None else []

    if nominal_seed is not None:
        eligible.append("nominal_comparison")
    if stress_seed is not None:
        eligible.append("stress_comparison")
    if include_supplements and nominal_runs and _focus_crop_has_all_labels(nominal_runs, "first_balance_time", 1.0, 1.2, required_labels):
        eligible.append("handoff_focus")
    if include_supplements and stress_runs and _focus_crop_has_all_labels(stress_runs, "disturbance_time", 1.0, 1.4, required_labels):
        eligible.append("disturbance_focus")
    return eligible


def _cleanup_stale_animation_outputs(base_dir: Path, all_runs: list[dict], stale_paths: set[Path]) -> set[Path]:
    existing = {path for path in stale_paths if path.exists()}
    stale_relative_paths = {str(path.relative_to(base_dir)) for path in existing}
    for path in existing:
        path.unlink()
    if stale_relative_paths:
        seen_json: set[str] = set()
        for run in all_runs:
            relative_json_path = run["summary"]["json_path"]
            if relative_json_path in seen_json:
                continue
            prune_metadata_render_paths(base_dir, relative_json_path, stale_relative_paths)
            seen_json.add(relative_json_path)
    return existing


def _build_timeline(length: int, fps: int, profile: dict[str, float]) -> list[int]:
    playback_seconds = profile["playback_seconds"]
    play_frames = max(2, int(round(playback_seconds * fps)))
    intro_frames = max(1, int(round(profile["title_hold"] * fps)))
    outro_frames = max(1, int(round(profile["outro_hold"] * fps)))
    indices = np.linspace(0, length - 1, play_frames).astype(int).tolist()
    return [0] * intro_frames + indices + [length - 1] * outro_frames


def _save_animation_bundle(fig, update_frame, total_samples: int, stem: Path, formats: tuple[str, ...], video_cfg, duration_profile: str) -> list[Path]:
    created = []
    profile = video_cfg.profile(duration_profile)
    for fmt in formats:
        fps = video_cfg.fps_gif if fmt == "gif" else video_cfg.fps_mp4
        timeline = _build_timeline(total_samples, fps, profile)
        anim = animation.FuncAnimation(fig, lambda i: update_frame(timeline[i]), frames=len(timeline), interval=1000 / fps, blit=False)
        path = stem.with_suffix(f".{fmt}")
        if fmt == "gif":
            writer = animation.PillowWriter(fps=fps)
        elif fmt == "mp4":
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError("MP4 export requested, but ffmpeg is not available.")
            writer = animation.FFMpegWriter(fps=fps, bitrate=2800)
        else:
            raise ValueError(f"Unsupported animation format: {fmt}")
        anim.save(path, writer=writer)
        created.append(path)
    plt.close(fig)
    return created


def render_animations(
    base_dir: Path,
    formats: tuple[str, ...] = ("gif",),
    theme: str = "paper_dense_cmu",
    duration_profile: str = "extended_gif",
    include_supplements: bool = True,
    controllers: tuple[str, ...] = tuple(CONTROLLER_LABELS),
    estimator_name: str = "none",
    progress: ProgressReporter | None = None,
) -> list[Path]:
    progress = progress or NullProgressReporter()
    theme_cfg = apply_theme(theme)
    video_cfg = load_video_config()
    all_runs = load_saved_runs(base_dir, suites={"nominal", "stress"})
    runs = _filter_runs(all_runs, controllers, estimator_name)
    _require_render_runs(base_dir, runs)
    animations_dir = base_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)
    required_labels = _required_labels(controllers)
    eligible_keys = _eligible_animation_keys(runs, controllers, include_supplements=include_supplements)
    managed_paths = _animation_output_paths(base_dir, controllers, formats)
    stale_paths = set().union(*(paths for key, paths in managed_paths.items() if key not in set(eligible_keys)))
    _cleanup_stale_animation_outputs(base_dir, all_runs, stale_paths)

    created = []
    total_items = len(eligible_keys)
    timer = PhaseTimer()
    progress.emit(
        ProgressEvent(
            domain="render_animations",
            stage="start",
            current=0,
            total=total_items,
            context={"estimator": estimator_name, "note": f"formats={','.join(formats)}"},
        )
    )
    item_index = 0
    nominal_seed = _best_seed_for_scenario(runs, "full_task_hanging", required_labels)
    stress_seed = _best_seed_for_scenario(runs, "impulse_disturbance", required_labels)
    nominal_runs = _runs_for_seed(runs, "full_task_hanging", nominal_seed) if nominal_seed is not None else []
    stress_runs = _runs_for_seed(runs, "impulse_disturbance", stress_seed) if stress_seed is not None else []
    supplemental_dir = animations_dir / "supplemental"
    if any(key in {"handoff_focus", "disturbance_focus"} for key in eligible_keys):
        supplemental_dir.mkdir(parents=True, exist_ok=True)

    for key in eligible_keys:
        if key in ANIMATION_STEMS and key in required_labels:
            run = _representative_run(runs, "full_task_hanging", key)
            created.append(_render_single(run, animations_dir, base_dir, formats, theme_cfg, video_cfg, duration_profile))
            item_name = f"{ANIMATION_STEMS[key]}.{formats[0]}"
        elif key == "nominal_comparison":
            created.append(
                _render_comparison(
                    nominal_runs,
                    "Nominal",
                    ANIMATION_STEMS["nominal_comparison"],
                    animations_dir,
                    base_dir,
                    formats,
                    theme_cfg,
                    video_cfg,
                    duration_profile,
                    show_disturbance=False,
                )
            )
            item_name = f"{ANIMATION_STEMS['nominal_comparison']}.{formats[0]}"
        elif key == "stress_comparison":
            created.append(
                _render_comparison(
                    stress_runs,
                    "Stress · Impulse",
                    ANIMATION_STEMS["stress_comparison"],
                    animations_dir,
                    base_dir,
                    formats,
                    theme_cfg,
                    video_cfg,
                    duration_profile,
                    show_disturbance=True,
                )
            )
            item_name = f"{ANIMATION_STEMS['stress_comparison']}.{formats[0]}"
        elif key == "handoff_focus":
            created.append(
                _render_focus_comparison(
                    nominal_runs,
                    "Handoff Focus",
                    "first_balance_time",
                    1.0,
                    1.2,
                    supplemental_dir / SUPPLEMENTAL_ANIMATION_STEMS["handoff_focus"],
                    base_dir,
                    formats,
                    theme_cfg,
                    video_cfg,
                    duration_profile,
                )
            )
            item_name = f"{SUPPLEMENTAL_ANIMATION_STEMS['handoff_focus']}.{formats[0]}"
        elif key == "disturbance_focus":
            created.append(
                _render_focus_comparison(
                    stress_runs,
                    "Disturbance Focus",
                    "disturbance_time",
                    1.0,
                    1.4,
                    supplemental_dir / SUPPLEMENTAL_ANIMATION_STEMS["disturbance_focus"],
                    base_dir,
                    formats,
                    theme_cfg,
                    video_cfg,
                    duration_profile,
                )
            )
            item_name = f"{SUPPLEMENTAL_ANIMATION_STEMS['disturbance_focus']}.{formats[0]}"
        else:
            continue
        item_index += 1
        progress.emit(
            ProgressEvent(
                domain="render_animations",
                stage="item_end",
                current=item_index,
                total=total_items,
                context={
                    "item_name": item_name,
                    "estimator": estimator_name,
                    "note": f"formats={','.join(formats)}",
                },
                elapsed_s=timer.elapsed(),
                eta_s=timer.eta(item_index, total_items),
            )
        )
    progress.emit(
        ProgressEvent(
            domain="render_animations",
            stage="done",
            current=len(created),
            total=total_items,
            context={"estimator": estimator_name},
            elapsed_s=timer.elapsed(),
            eta_s=0.0,
        )
    )
    return created


def _render_single(run: dict, output_dir: Path, base_dir: Path, formats: tuple[str, ...], theme_cfg, video_cfg, duration_profile: str) -> Path:
    frame = run["frame"]
    metadata = run["metadata"]
    controller = metadata["controller_name"]
    color = controller_color(theme_cfg, controller)
    time = frame["t"].to_numpy(dtype=float)
    angle = frame["theta_deg"].to_numpy(dtype=float)
    force = frame["u"].to_numpy(dtype=float)
    energy = frame["energy_gap"].to_numpy(dtype=float)

    fig = plt.figure(figsize=video_cfg.canvas_size_single)
    gs = fig.add_gridspec(3, 5, width_ratios=[1.45, 1.45, 1.45, 1.05, 1.05], height_ratios=[1.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.08, wspace=0.10, hspace=0.18)
    ax_anim = fig.add_subplot(gs[:, :3])
    ax_angle = fig.add_subplot(gs[0, 3:])
    ax_force = fig.add_subplot(gs[1, 3:], sharex=ax_angle)
    ax_energy = fig.add_subplot(gs[2, 3:], sharex=ax_angle)
    for ax, title in zip((ax_angle, ax_force, ax_energy), ("Angle", "Force", "Energy gap"), strict=True):
        style_axis(ax, theme_cfg)
        add_panel_title(ax, title, theme_cfg=theme_cfg)

    artist = CartPoleArtist(
        ax=ax_anim,
        track_limit=metadata["plant_params"]["track_limit"],
        pendulum_length=0.82,
        title="",
        color=color,
        text_color=theme_cfg.text_color,
        panel_color=theme_cfg.panel_color,
        background_color=theme_cfg.background_color,
    )
    controller_badge(ax_anim, f"{_short_label(controller)} · {metadata.get('estimator_name', 'none')}", color, theme_cfg)

    traces = [(ax_angle, angle), (ax_force, force), (ax_energy, energy)]
    progress_lines = []
    points = []
    for ax, series in traces:
        ax.plot(time, series, color=soften(color, 0.82), linewidth=1.0)
        line, = ax.plot([], [], color=color, linewidth=1.7)
        point, = ax.plot([], [], "o", color=color, markersize=3.0)
        progress_lines.append(line)
        points.append(point)
    ax_energy.set_xlabel("Time [s]")
    ax_angle.tick_params(labelbottom=False)
    ax_force.tick_params(labelbottom=False)
    status_text = fig.text(0.04, 0.955, "", fontsize=7.3, color=theme_cfg.muted_color, ha="left", va="top")

    stem = output_dir / ANIMATION_STEMS[controller]

    def update_frame(sample_index: int) -> None:
        row = frame.iloc[sample_index]
        state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
        artist.update(state, float(row["t"]), str(row["mode"]))
        for line, point, (_, series) in zip(progress_lines, points, traces, strict=True):
            line.set_data(time[: sample_index + 1], series[: sample_index + 1])
            point.set_data([time[sample_index]], [series[sample_index]])
        status_text.set_text(f"{str(row['mode']).replace('_', ' ')} · sat={float(row['force_saturated']):.0f}")

    created_paths = _save_animation_bundle(fig, update_frame, len(frame), stem, formats, video_cfg, duration_profile)
    update_metadata_render_paths(
        base_dir,
        run["summary"]["json_path"],
        {f"{stem.name}_{path.suffix.lstrip('.')}": str(path.relative_to(base_dir)) for path in created_paths},
    )
    return created_paths[0]


def _render_comparison(
    selected: list[dict],
    title: str,
    stem_name: str,
    output_dir: Path,
    base_dir: Path,
    formats: tuple[str, ...],
    theme_cfg,
    video_cfg,
    duration_profile: str,
    *,
    show_disturbance: bool,
) -> Path:
    fig = plt.figure(figsize=video_cfg.canvas_size_comparison)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95])
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.08, wspace=0.05, hspace=0.18)

    top_axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(3)]
    ax_trace = fig.add_subplot(gs[2, :])
    style_axis(ax_trace, theme_cfg)
    add_panel_title(ax_trace, "Angle trace", theme_cfg=theme_cfg, title_pad=6.5)
    ax_trace.set_xlabel("Time [s]")
    ax_trace.set_ylabel("deg")

    artists = []
    trace_lines = []
    trace_points = []
    legend_handles = []
    legend_labels = []
    max_len = max(len(run["frame"]) for run in selected)

    for index, ax in enumerate(top_axes):
        if index < len(selected):
            run = selected[index]
            controller = run["metadata"]["controller_name"]
            color = controller_color(theme_cfg, controller)
            artist = CartPoleArtist(
                ax=ax,
                track_limit=run["metadata"]["plant_params"]["track_limit"],
                pendulum_length=0.78,
                title="",
                color=color,
                text_color=theme_cfg.text_color,
                panel_color=theme_cfg.panel_color,
                background_color=theme_cfg.background_color,
            )
            controller_badge(ax, _short_label(controller), color, theme_cfg, y=0.90, fontsize=7.2, pad=0.16)
            artists.append((artist, run))
            frame = run["frame"]
            time = frame["t"].to_numpy(dtype=float)
            angle = frame["theta_deg"].to_numpy(dtype=float)
            base_line, = ax_trace.plot(time, angle, color=soften(color, 0.84), linewidth=1.0)
            line, = ax_trace.plot([], [], color=color, linewidth=1.5)
            point, = ax_trace.plot([], [], "o", color=color, markersize=2.8)
            trace_lines.append((line, run))
            trace_points.append((point, run))
            legend_handles.append(Line2D([0], [0], color=color, linewidth=1.6))
            legend_labels.append(_short_label(controller))
        else:
            ax.set_facecolor(theme_cfg.panel_color)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.08, 0.80, "Shared trace below", transform=ax.transAxes, fontsize=8.1, color=theme_cfg.muted_color, ha="left")

    current_time_line = ax_trace.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9)
    if show_disturbance:
        disturbance_runs = [run for run in selected if np.any(np.abs(run["frame"]["disturbance"].to_numpy(dtype=float)) > 1e-9)]
        if disturbance_runs:
            disturbance_frame = disturbance_runs[0]["frame"]
            mask = np.abs(disturbance_frame["disturbance"].to_numpy(dtype=float)) > 1e-9
            if np.any(mask):
                times = disturbance_frame["t"].to_numpy(dtype=float)[mask]
                ax_trace.axvspan(float(times[0]), float(times[-1]), color=theme_cfg.accent_color, alpha=0.08, linewidth=0.0)
    fig.text(0.04, 0.982, title, fontsize=10.4, color=theme_cfg.text_color, ha="left", va="top")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.956),
        fontsize=7.2,
        ncol=min(5, max(1, len(legend_labels))),
        handlelength=1.6,
        columnspacing=0.75,
        handletextpad=0.45,
        frameon=False,
    )

    stem = output_dir / stem_name

    def update_frame(sample_index: int) -> None:
        current_time = 0.0
        for artist, run in artists:
            frame = run["frame"]
            idx = min(sample_index, len(frame) - 1)
            row = frame.iloc[idx]
            state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
            artist.update(state, float(row["t"]), str(row["mode"]))
            current_time = max(current_time, float(row["t"]))
        for line, run in trace_lines:
            frame = run["frame"]
            idx = min(sample_index, len(frame) - 1)
            line.set_data(frame["t"].to_numpy(dtype=float)[: idx + 1], frame["theta_deg"].to_numpy(dtype=float)[: idx + 1])
        for point, run in trace_points:
            frame = run["frame"]
            idx = min(sample_index, len(frame) - 1)
            point.set_data([float(frame["t"].iloc[idx])], [float(frame["theta_deg"].iloc[idx])])
        current_time_line.set_xdata([current_time, current_time])

    created_paths = _save_animation_bundle(fig, update_frame, max_len, stem, formats, video_cfg, duration_profile)
    for run in selected:
        update_metadata_render_paths(
            base_dir,
            run["summary"]["json_path"],
            {f"{stem.name}_{path.suffix.lstrip('.')}": str(path.relative_to(base_dir)) for path in created_paths},
        )
    return created_paths[0]


def _render_focus_comparison(
    selected: list[dict],
    title: str,
    center_key: str,
    before: float,
    after: float,
    stem: Path,
    base_dir: Path,
    formats: tuple[str, ...],
    theme_cfg,
    video_cfg,
    duration_profile: str,
) -> Path:
    crops = []
    for run in selected:
        frame = run["frame"].copy()
        if center_key == "disturbance_time":
            disturbance = frame["disturbance"].to_numpy(dtype=float)
            active = np.flatnonzero(np.abs(disturbance) > 1e-9)
            center = float(frame["t"].iloc[active[0]]) if active.size else float(frame["t"].iloc[0])
        else:
            center = run["metadata"]["diagnosis"].get(center_key)
            if center is None:
                continue
            center = float(center)
        rel = frame["t"].to_numpy(dtype=float) - center
        crop = frame[(rel >= -before) & (rel <= after)].copy()
        if crop.empty:
            continue
        crop["t_rel"] = crop["t"] - center
        crops.append((run, crop))

    if not crops:
        raise RuntimeError(f"No runs were eligible for focus animation '{title}'.")

    fig = plt.figure(figsize=video_cfg.canvas_size_comparison)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95])
    fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.08, wspace=0.05, hspace=0.18)
    top_axes = [fig.add_subplot(gs[row, col]) for row in range(2) for col in range(3)]
    ax_trace = fig.add_subplot(gs[2, :])
    style_axis(ax_trace, theme_cfg)
    add_panel_title(ax_trace, "Angle around event", theme_cfg=theme_cfg, title_pad=6.5)
    ax_trace.set_xlabel("Time from event [s]")
    ax_trace.set_ylabel("deg")

    artists = []
    trace_lines = []
    trace_points = []
    legend_handles = []
    legend_labels = []
    max_len = max(len(crop) for _, crop in crops)
    for index, ax in enumerate(top_axes):
        if index < len(crops):
            run, crop = crops[index]
            controller = run["metadata"]["controller_name"]
            color = controller_color(theme_cfg, controller)
            artist = CartPoleArtist(
                ax=ax,
                track_limit=run["metadata"]["plant_params"]["track_limit"],
                pendulum_length=0.78,
                title="",
                color=color,
                text_color=theme_cfg.text_color,
                panel_color=theme_cfg.panel_color,
                background_color=theme_cfg.background_color,
            )
            controller_badge(ax, _short_label(controller), color, theme_cfg, y=0.90, fontsize=7.2, pad=0.16)
            artists.append((artist, crop))
            ax_trace.plot(crop["t_rel"], crop["theta_deg"], color=soften(color, 0.83), linewidth=1.0)
            line, = ax_trace.plot([], [], color=color, linewidth=1.5)
            point, = ax_trace.plot([], [], "o", color=color, markersize=2.8)
            trace_lines.append((line, crop))
            trace_points.append((point, crop))
            legend_handles.append(Line2D([0], [0], color=color, linewidth=1.6))
            legend_labels.append(_short_label(controller))
        else:
            ax.set_axis_off()
    event_line = ax_trace.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9)
    fig.text(0.04, 0.982, title, fontsize=10.4, color=theme_cfg.text_color, ha="left", va="top")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.956),
        fontsize=7.2,
        ncol=min(5, max(1, len(legend_labels))),
        handlelength=1.6,
        columnspacing=0.75,
        handletextpad=0.45,
        frameon=False,
    )

    def update_frame(sample_index: int) -> None:
        for artist, crop in artists:
            idx = min(sample_index, len(crop) - 1)
            row = crop.iloc[idx]
            state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
            artist.update(state, float(row["t_rel"]), str(row["mode"]))
        for line, crop in trace_lines:
            idx = min(sample_index, len(crop) - 1)
            line.set_data(crop["t_rel"].to_numpy(dtype=float)[: idx + 1], crop["theta_deg"].to_numpy(dtype=float)[: idx + 1])
        for point, crop in trace_points:
            idx = min(sample_index, len(crop) - 1)
            point.set_data([float(crop["t_rel"].iloc[idx])], [float(crop["theta_deg"].iloc[idx])])
        event_line.set_xdata([0.0, 0.0])

    created_paths = _save_animation_bundle(fig, update_frame, max_len, stem, formats, video_cfg, duration_profile)
    for run, _crop in crops:
        update_metadata_render_paths(
            base_dir,
            run["summary"]["json_path"],
            {f"{stem.name}_{path.suffix.lstrip('.')}": str(path.relative_to(base_dir)) for path in created_paths},
        )
    return created_paths[0]
