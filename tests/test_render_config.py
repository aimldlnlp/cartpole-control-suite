from __future__ import annotations

from cartpole_bench.cli import build_parser
from cartpole_bench.config import load_theme_config, load_video_config
from cartpole_bench.animations.render import ANIMATION_STEMS, SUPPLEMENTAL_ANIMATION_STEMS
from cartpole_bench.plots.figures import FIGURE_FILENAMES, SUPPLEMENTAL_FIGURE_FILENAMES


def test_render_configs_load_expected_defaults() -> None:
    theme = load_theme_config()
    video = load_video_config()
    assert theme.name == "paper_dense_cmu"
    assert theme.font_family == "CMU Serif"
    assert theme.background_color == "#FBFAF6"
    assert "LQR" in theme.controller_colors
    assert video.profile("extended")["playback_seconds"] > 10.0
    assert video.profile("extended_gif")["playback_seconds"] == 12.8


def test_render_output_names_do_not_use_numeric_fig_anim_prefixes() -> None:
    assert all(not name.startswith("fig0") for name in FIGURE_FILENAMES.values())
    assert all(not stem.startswith("anim0") for stem in ANIMATION_STEMS.values())
    assert all(not name.startswith("fig0") for name in SUPPLEMENTAL_FIGURE_FILENAMES.values())
    assert all(not stem.startswith("anim0") for stem in SUPPLEMENTAL_ANIMATION_STEMS.values())


def test_cli_defaults_to_artifacts_v2_output_root() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-suite", "--suite", "nominal"])
    assert args.output == "artifacts_v3"


def test_cli_render_defaults_use_gif_first_muted_theme() -> None:
    parser = build_parser()
    args = parser.parse_args(["render"])
    assert args.formats == "gif"
    assert args.theme == "paper_dense_cmu"
    assert args.duration_profile == "extended_gif"
    assert args.estimator == "none"
    assert "ilqr" in args.controllers
    assert args.no_supplements is False
    assert args.quiet is False


def test_cli_render_split_commands_exist() -> None:
    parser = build_parser()
    fig_args = parser.parse_args(["render-figures", "--quiet"])
    anim_args = parser.parse_args(["render-animations", "--quiet"])
    assert fig_args.command == "render-figures"
    assert anim_args.command == "render-animations"
    assert fig_args.quiet is True
    assert anim_args.quiet is True


def test_render_output_names_include_new_controller_stems() -> None:
    assert ANIMATION_STEMS["Iterative LQR (iLQR)"] == "ilqr_full_task_nominal"
    assert ANIMATION_STEMS["Model Predictive Control (MPC)"] == "mpc_full_task_nominal"


def test_cli_can_disable_supplemental_outputs() -> None:
    parser = build_parser()
    args = parser.parse_args(["render", "--no-supplements"])
    assert args.no_supplements is True


def test_cli_all_is_simulation_only() -> None:
    parser = build_parser()
    args = parser.parse_args(["all", "--quiet"])
    assert args.command == "all"
    assert args.quiet is True
    assert not hasattr(args, "formats")
    assert not hasattr(args, "theme")
    assert not hasattr(args, "duration_profile")
