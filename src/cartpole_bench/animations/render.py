from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from cartpole_bench.animations.artist import CartPoleArtist
from cartpole_bench.config import load_video_config
from cartpole_bench.plots.style import add_panel_title, apply_theme, controller_color, soften, style_axis
from cartpole_bench.simulation.recorder import load_saved_runs, update_metadata_render_paths


ANIMATION_STEMS = {
    "LQR": "lqr_full_task_nominal",
    "Feedback Linearization (PFL)": "pfl_full_task_nominal",
    "Sliding Mode Control (SMC)": "smc_full_task_nominal",
    "nominal_comparison": "side_by_side_nominal_comparison",
    "stress_comparison": "side_by_side_stress_comparison",
}
SUPPLEMENTAL_ANIMATION_STEMS = {
    "handoff_focus": "handoff_focus_comparison",
}
CONTROLLER_ORDER = ["LQR", "Feedback Linearization (PFL)", "Sliding Mode Control (SMC)"]


def _short_label(controller: str) -> str:
    return {
        "LQR": "LQR",
        "Feedback Linearization (PFL)": "PFL",
        "Sliding Mode Control (SMC)": "SMC",
    }.get(controller, controller)


def _mode_label(mode: str) -> str:
    return {
        "energy_pump": "swing-up",
        "capture_assist": "capture",
        "balance": "balance",
    }.get(mode, mode.replace("_", " "))


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
    if not candidates:
        raise KeyError(f"Missing run for scenario={scenario}, controller={controller}. Generate the required suite before rendering.")
    successful = [run for run in candidates if run["metadata"]["metrics"]["success"]]
    if successful:
        settling = np.asarray([run["metadata"]["metrics"]["settling_time"] for run in successful], dtype=float)
        median_settling = float(np.median(settling))
        return min(successful, key=lambda run: abs(float(run["metadata"]["metrics"]["settling_time"]) - median_settling))
    return min(candidates, key=_run_score)


def _best_seed_for_scenario(runs: list[dict], scenario: str) -> int:
    seed_rows = {}
    for run in runs:
        if run["metadata"]["scenario_name"] != scenario:
            continue
        seed = int(run["metadata"]["seed"])
        seed_rows.setdefault(seed, {"success": 0, "handoff": 0, "score": 0.0})
        seed_rows[seed]["success"] += int(bool(run["metadata"]["metrics"]["success"]))
        seed_rows[seed]["handoff"] += int(run["metadata"]["diagnosis"]["first_balance_time"] is not None)
        seed_rows[seed]["score"] += _run_score(run)[2]
    if not seed_rows:
        raise KeyError(f"Missing runs for scenario={scenario}. Generate the required suite before rendering.")
    return max(seed_rows, key=lambda seed: (seed_rows[seed]["success"], seed_rows[seed]["handoff"], -seed_rows[seed]["score"]))


def _runs_for_seed(runs: list[dict], scenario: str, seed: int) -> list[dict]:
    selected = [run for run in runs if run["metadata"]["scenario_name"] == scenario and int(run["metadata"]["seed"]) == seed]
    return sorted(selected, key=lambda run: CONTROLLER_ORDER.index(run["metadata"]["controller_name"]))


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
                raise RuntimeError(
                    "MP4 export requested, but ffmpeg is not available to Matplotlib. "
                    "Install ffmpeg or omit mp4 from --formats."
                )
            writer = animation.FFMpegWriter(fps=fps, bitrate=2800)
        else:
            raise ValueError(f"Unsupported animation format: {fmt}")
        anim.save(path, writer=writer)
        created.append(path)
    plt.close(fig)
    return created


def _windowed_frame(run: dict, center_time: float, before: float, after: float):
    frame = run["frame"]
    cropped = frame[(frame["t"] >= center_time - before) & (frame["t"] <= center_time + after)].copy()
    cropped["t_rel"] = cropped["t"] - center_time
    return cropped


def _require_render_runs(base_dir: Path, runs: list[dict]) -> None:
    if runs:
        return
    raise RuntimeError(
        f"No saved nominal/stress run artifacts were found in '{base_dir}'. "
        "The render command reads previously generated CSV/JSON run data from the same artifact root that it writes animations into. "
        "Use an artifact directory that already contains suite outputs, or rerun the suites first."
    )


def render_animations(
    base_dir: Path,
    formats: tuple[str, ...] = ("gif",),
    theme: str = "paper_white",
    duration_profile: str = "extended_gif",
    include_supplements: bool = True,
) -> list[Path]:
    theme_cfg = apply_theme(theme)
    video_cfg = load_video_config()
    runs = load_saved_runs(base_dir, suites={"nominal", "stress"})
    _require_render_runs(base_dir, runs)
    animations_dir = base_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for controller in CONTROLLER_ORDER:
        run = _representative_run(runs, "full_task_hanging", controller)
        created.append(_render_single(run, animations_dir, base_dir, formats, theme_cfg, video_cfg, duration_profile))

    nominal_seed = _best_seed_for_scenario(runs, "full_task_hanging")
    stress_seed = _best_seed_for_scenario(runs, "impulse_disturbance")
    created.append(
        _render_comparison(
            _runs_for_seed(runs, "full_task_hanging", nominal_seed),
            "Nominal",
            ANIMATION_STEMS["nominal_comparison"],
            animations_dir,
            base_dir,
            formats,
            theme_cfg,
            video_cfg,
            duration_profile,
        )
    )
    created.append(
        _render_comparison(
            _runs_for_seed(runs, "impulse_disturbance", stress_seed),
            "Stress · Impulse",
            ANIMATION_STEMS["stress_comparison"],
            animations_dir,
            base_dir,
            formats,
            theme_cfg,
            video_cfg,
            duration_profile,
        )
    )

    if include_supplements:
        supplemental_dir = animations_dir / "supplemental"
        supplemental_dir.mkdir(parents=True, exist_ok=True)
        created.append(
            _render_handoff_focus_comparison(
                _runs_for_seed(runs, "full_task_hanging", nominal_seed),
                supplemental_dir,
                base_dir,
                formats,
                theme_cfg,
                video_cfg,
                duration_profile,
            )
        )
    return created


def _render_single(run: dict, output_dir: Path, base_dir: Path, formats: tuple[str, ...], theme_cfg, video_cfg, duration_profile: str) -> Path:
    frame = run["frame"]
    metadata = run["metadata"]
    controller = metadata["controller_name"]
    color = controller_color(theme_cfg, controller)
    time = frame["t"].to_numpy(dtype=float)
    angle = np.rad2deg(frame["theta"].to_numpy(dtype=float))
    cart = frame["x"].to_numpy(dtype=float)

    fig = plt.figure(figsize=video_cfg.canvas_size_single)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 1.35, 1.35, 1.35, 0.92], height_ratios=[1.0, 1.0])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.08, wspace=0.08, hspace=0.18)
    ax_anim = fig.add_subplot(gs[:, :4])
    ax_angle = fig.add_subplot(gs[0, 4])
    ax_cart = fig.add_subplot(gs[1, 4], sharex=ax_angle)
    for ax in (ax_angle, ax_cart):
        style_axis(ax, theme_cfg)

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
    fig.text(0.04, 0.965, f"{_short_label(controller)} · Full Task", fontsize=10.4, color=theme_cfg.text_color, ha="left", va="top")
    status_text = fig.text(0.04, 0.935, "", fontsize=7.2, color=theme_cfg.muted_color, ha="left", va="top")

    trace_axes = [(ax_angle, angle, "Angle"), (ax_cart, cart, "Cart")]
    progress_lines = []
    points = []
    for ax, series, title in trace_axes:
        ax.plot(time, series, color=soften(color, 0.8), linewidth=1.05)
        line, = ax.plot([], [], color=color, linewidth=1.7)
        point, = ax.plot([], [], "o", color=color, markersize=3.2)
        add_panel_title(ax, title, theme_cfg=theme_cfg)
        progress_lines.append(line)
        points.append(point)
    ax_cart.set_xlabel("Time [s]", fontsize=8.2)
    ax_angle.tick_params(labelbottom=False)

    stem = output_dir / ANIMATION_STEMS[controller]

    def update_frame(sample_index: int):
        row = frame.iloc[sample_index]
        state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
        artist.update(state, float(row["t"]), str(row["mode"]))
        for line, point, (_, series, _title) in zip(progress_lines, points, trace_axes, strict=True):
            line.set_data(time[: sample_index + 1], series[: sample_index + 1])
            point.set_data([time[sample_index]], [series[sample_index]])
        status_text.set_text(_mode_label(str(row["mode"])))

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
) -> Path:
    fig = plt.figure(figsize=video_cfg.canvas_size_comparison)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 0.92])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.10, wspace=0.04, hspace=0.14)
    ax_trace = fig.add_subplot(gs[1, :])
    style_axis(ax_trace, theme_cfg)
    add_panel_title(ax_trace, "Angle", theme_cfg=theme_cfg)
    ax_trace.set_xlabel("Time [s]", fontsize=8.5)
    ax_trace.set_ylabel("deg", fontsize=8.5)

    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    artists = []
    trace_lines = []
    trace_points = []
    legend_handles = []
    legend_labels = []
    max_len = max(len(run["frame"]) for run in selected)

    for ax, run in zip(axes, selected, strict=True):
        controller = run["metadata"]["controller_name"]
        color = controller_color(theme_cfg, controller)
        artist = CartPoleArtist(
            ax=ax,
            track_limit=run["metadata"]["plant_params"]["track_limit"],
            pendulum_length=0.8,
            title=_short_label(controller),
            color=color,
            text_color=theme_cfg.text_color,
            panel_color=theme_cfg.panel_color,
            background_color=theme_cfg.background_color,
        )
        artists.append(artist)
        frame = run["frame"]
        time = frame["t"].to_numpy(dtype=float)
        angle = np.rad2deg(frame["theta"].to_numpy(dtype=float))
        base_line, = ax_trace.plot(time, angle, color=soften(color, 0.82), linewidth=1.0)
        line, = ax_trace.plot([], [], color=color, linewidth=1.5)
        point, = ax_trace.plot([], [], "o", color=color, markersize=3.0)
        trace_lines.append(line)
        trace_points.append(point)
        legend_handles.append(base_line)
        legend_labels.append(_short_label(controller))

    current_time_line = ax_trace.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9)
    ax_trace.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=7.6,
        ncol=3,
        handlelength=1.8,
        columnspacing=0.9,
    )
    fig.text(0.04, 0.965, title, fontsize=10.4, color=theme_cfg.text_color, ha="left", va="top")

    stem = output_dir / stem_name

    def update_frame(sample_index: int):
        current_time = 0.0
        for artist, run, line, point in zip(artists, selected, trace_lines, trace_points, strict=True):
            frame = run["frame"]
            idx = min(sample_index, len(frame) - 1)
            row = frame.iloc[idx]
            state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
            artist.update(state, float(row["t"]), str(row["mode"]))
            time = frame["t"].to_numpy(dtype=float)
            angle = np.rad2deg(frame["theta"].to_numpy(dtype=float))
            line.set_data(time[: idx + 1], angle[: idx + 1])
            point.set_data([time[idx]], [angle[idx]])
            current_time = max(current_time, float(time[idx]))
        current_time_line.set_xdata([current_time, current_time])

    created_paths = _save_animation_bundle(fig, update_frame, max_len, stem, formats, video_cfg, duration_profile)
    for run in selected:
        update_metadata_render_paths(
            base_dir,
            run["summary"]["json_path"],
            {f"{stem.name}_{path.suffix.lstrip('.')}": str(path.relative_to(base_dir)) for path in created_paths},
        )
    return created_paths[0]


def _render_handoff_focus_comparison(
    selected: list[dict],
    output_dir: Path,
    base_dir: Path,
    formats: tuple[str, ...],
    theme_cfg,
    video_cfg,
    duration_profile: str,
) -> Path:
    windowed = []
    for run in selected:
        handoff_time = run["metadata"]["diagnosis"]["first_balance_time"]
        if handoff_time is None:
            continue
        cropped = _windowed_frame(run, float(handoff_time), before=1.0, after=1.2)
        if not cropped.empty:
            windowed.append({"run": run, "frame": cropped})
    if not windowed:
        raise RuntimeError("Cannot render handoff focus GIF because no selected run contains a balance handoff.")

    fig = plt.figure(figsize=video_cfg.canvas_size_comparison)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 0.92])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.10, wspace=0.04, hspace=0.14)
    ax_trace = fig.add_subplot(gs[1, :])
    style_axis(ax_trace, theme_cfg)
    add_panel_title(ax_trace, "Angle", theme_cfg=theme_cfg)
    ax_trace.set_xlabel("Time from balance entry [s]", fontsize=8.5)
    ax_trace.set_ylabel("deg", fontsize=8.5)

    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    artists = []
    trace_lines = []
    trace_points = []
    legend_handles = []
    legend_labels = []
    max_len = max(len(item["frame"]) for item in windowed)

    for ax, item in zip(axes, windowed, strict=True):
        run = item["run"]
        frame = item["frame"]
        controller = run["metadata"]["controller_name"]
        color = controller_color(theme_cfg, controller)
        artist = CartPoleArtist(
            ax=ax,
            track_limit=run["metadata"]["plant_params"]["track_limit"],
            pendulum_length=0.8,
            title=_short_label(controller),
            color=color,
            text_color=theme_cfg.text_color,
            panel_color=theme_cfg.panel_color,
            background_color=theme_cfg.background_color,
        )
        artists.append(artist)
        time_rel = frame["t_rel"].to_numpy(dtype=float)
        angle = np.rad2deg(frame["theta"].to_numpy(dtype=float))
        base_line, = ax_trace.plot(time_rel, angle, color=soften(color, 0.82), linewidth=1.0)
        line, = ax_trace.plot([], [], color=color, linewidth=1.5)
        point, = ax_trace.plot([], [], "o", color=color, markersize=3.0)
        trace_lines.append(line)
        trace_points.append(point)
        legend_handles.append(base_line)
        legend_labels.append(_short_label(controller))

    current_time_line = ax_trace.axvline(0.0, color=theme_cfg.accent_color, linestyle="--", linewidth=0.9)
    ax_trace.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        fontsize=7.6,
        ncol=3,
        handlelength=1.8,
        columnspacing=0.9,
    )
    ax_trace.set_xlim(-1.0, 1.2)
    fig.text(0.04, 0.965, "Handoff Focus", fontsize=10.4, color=theme_cfg.text_color, ha="left", va="top")

    stem = output_dir / SUPPLEMENTAL_ANIMATION_STEMS["handoff_focus"]

    def update_frame(sample_index: int):
        current_time = 0.0
        for artist, item, line, point in zip(artists, windowed, trace_lines, trace_points, strict=True):
            frame = item["frame"]
            idx = min(sample_index, len(frame) - 1)
            row = frame.iloc[idx]
            state = row[["x", "x_dot", "theta", "theta_dot"]].to_numpy(dtype=float)
            artist.update(state, float(row["t_rel"]), str(row["mode"]))
            time_rel = frame["t_rel"].to_numpy(dtype=float)
            angle = np.rad2deg(frame["theta"].to_numpy(dtype=float))
            line.set_data(time_rel[: idx + 1], angle[: idx + 1])
            point.set_data([time_rel[idx]], [angle[idx]])
            current_time = max(current_time, float(time_rel[idx]))
        current_time_line.set_xdata([current_time, current_time])

    created_paths = _save_animation_bundle(fig, update_frame, max_len, stem, formats, video_cfg, duration_profile)
    for item in windowed:
        update_metadata_render_paths(
            base_dir,
            item["run"]["summary"]["json_path"],
            {f"{stem.name}_{path.suffix.lstrip('.')}": str(path.relative_to(base_dir)) for path in created_paths},
        )
    return created_paths[0]
