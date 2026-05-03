from __future__ import annotations

import argparse
from pathlib import Path

from cartpole_bench.animations.render import render_animations
from cartpole_bench.config import parse_controller_keys
from cartpole_bench.plots.figures import generate_figures
from cartpole_bench.plots.tables import refresh_metric_tables
from cartpole_bench.simulation.batch import run_monte_carlo
from cartpole_bench.simulation.runner import run_suite
from cartpole_bench.simulation.scenario import CONTROLLER_KEYS
from cartpole_bench.simulation.tuning import tune_controller
from cartpole_bench.utils.progress import LineProgressReporter, NullProgressReporter, ProgressEvent, ProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio-grade nonlinear cart-pole benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_output = "artifacts_v3"

    run_suite_parser = subparsers.add_parser("run-suite", help="Run a named experiment suite")
    run_suite_parser.add_argument("--suite", choices=("nominal", "stress"), required=True)
    run_suite_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    run_suite_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    run_suite_parser.add_argument("--output", default=default_output, help="Artifact directory")
    run_suite_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    monte_carlo_parser = subparsers.add_parser("monte-carlo", help="Run the Monte Carlo benchmark")
    monte_carlo_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    monte_carlo_parser.add_argument("--samples", type=int, default=1000)
    monte_carlo_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    monte_carlo_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    monte_carlo_parser.add_argument("--output", default=default_output, help="Artifact directory")
    monte_carlo_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    render_parser = subparsers.add_parser("render", help="Render figures and animations from saved artifacts")
    render_parser.add_argument("--formats", default="gif", help="Comma-separated animation formats")
    render_parser.add_argument("--theme", default="paper_dense_cmu", help="Rendering theme name")
    render_parser.add_argument("--duration-profile", default="extended_gif", help="Animation duration profile")
    render_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    render_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    render_parser.add_argument("--no-supplements", action="store_true", help="Skip supplemental figures and animations")
    render_parser.add_argument("--output", default=default_output, help="Artifact directory")
    render_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    render_figures_parser = subparsers.add_parser("render-figures", help="Render figures from saved artifacts")
    render_figures_parser.add_argument("--theme", default="paper_dense_cmu", help="Rendering theme name")
    render_figures_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    render_figures_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    render_figures_parser.add_argument("--no-supplements", action="store_true", help="Skip supplemental figures")
    render_figures_parser.add_argument("--output", default=default_output, help="Artifact directory")
    render_figures_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    render_animations_parser = subparsers.add_parser("render-animations", help="Render animations from saved artifacts")
    render_animations_parser.add_argument("--formats", default="gif", help="Comma-separated animation formats")
    render_animations_parser.add_argument("--theme", default="paper_dense_cmu", help="Rendering theme name")
    render_animations_parser.add_argument("--duration-profile", default="extended_gif", help="Animation duration profile")
    render_animations_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    render_animations_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    render_animations_parser.add_argument("--no-supplements", action="store_true", help="Skip supplemental animations")
    render_animations_parser.add_argument("--output", default=default_output, help="Artifact directory")
    render_animations_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    all_parser = subparsers.add_parser("all", help="Run all suites and Monte Carlo without rendering")
    all_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    all_parser.add_argument("--samples", type=int, default=1000)
    all_parser.add_argument("--controllers", default=",".join(CONTROLLER_KEYS), help="Comma-separated controller keys")
    all_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    all_parser.add_argument("--output", default=default_output, help="Artifact directory")
    all_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    tune_parser = subparsers.add_parser("tune", help="Tune a controller on the fixed benchmark panel")
    tune_parser.add_argument("--controller", choices=CONTROLLER_KEYS, required=True)
    tune_parser.add_argument("--estimator", choices=("none", "ekf"), default="none")
    tune_parser.add_argument("--budget", type=int, default=40)
    tune_parser.add_argument("--seed", type=int, default=0)
    tune_parser.add_argument("--output", default=default_output, help="Artifact directory")
    tune_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def _resolve_output(path_str: str) -> Path:
    return Path(path_str).resolve()


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(fmt.strip() for fmt in raw.split(",") if fmt.strip())
    if not formats:
        raise ValueError("At least one animation format must be specified.")
    return formats


def _make_reporter(quiet: bool) -> ProgressReporter:
    if quiet:
        return NullProgressReporter()
    return LineProgressReporter()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = _resolve_output(args.output)
    controllers = parse_controller_keys(getattr(args, "controllers", None), CONTROLLER_KEYS)
    progress = _make_reporter(getattr(args, "quiet", False))

    if args.command == "run-suite":
        results = run_suite(
            args.suite,
            output_dir,
            Path.cwd(),
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        refresh_metric_tables(output_dir)
        print(f"Completed {args.suite} suite with {len(results)} runs.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "monte-carlo":
        summaries = run_monte_carlo(
            output_dir,
            requested_device=args.device,
            samples=args.samples,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        print(f"Completed Monte Carlo benchmark for {len(summaries)} controllers.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "render-figures":
        include_supplements = not args.no_supplements
        figure_paths = generate_figures(
            output_dir,
            theme=args.theme,
            include_supplements=include_supplements,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        print(f"Rendered {len(figure_paths)} figures.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "render-animations":
        formats = _parse_formats(args.formats)
        include_supplements = not args.no_supplements
        animation_paths = render_animations(
            output_dir,
            formats=formats,
            theme=args.theme,
            duration_profile=args.duration_profile,
            include_supplements=include_supplements,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        print(f"Rendered {len(animation_paths)} animation storyboards.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "render":
        formats = _parse_formats(args.formats)
        include_supplements = not args.no_supplements
        progress.emit(ProgressEvent(domain="render", stage="figures"))
        figure_paths = generate_figures(
            output_dir,
            theme=args.theme,
            include_supplements=include_supplements,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        progress.emit(ProgressEvent(domain="render", stage="animations"))
        animation_paths = render_animations(
            output_dir,
            formats=formats,
            theme=args.theme,
            duration_profile=args.duration_profile,
            include_supplements=include_supplements,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        progress.emit(
            ProgressEvent(
                domain="render",
                stage="done",
                context={"note": f"figures={len(figure_paths)}, animations={len(animation_paths)}"},
            )
        )
        print(f"Rendered {len(figure_paths)} figures and {len(animation_paths)} animation storyboards.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "all":
        run_suite(
            "nominal",
            output_dir,
            Path.cwd(),
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        run_suite(
            "stress",
            output_dir,
            Path.cwd(),
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        refresh_metric_tables(output_dir)
        run_monte_carlo(
            output_dir,
            requested_device=args.device,
            samples=args.samples,
            controllers=controllers,
            estimator_name=args.estimator,
            progress=progress,
        )
        print("Completed cart-pole simulations.")
        print("Next step: run `cartpole-bench render`, `render-figures`, or `render-animations`.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "tune":
        summary = tune_controller(
            controller_key=args.controller,
            output_dir=output_dir,
            estimator_name=args.estimator,
            budget=args.budget,
            seed=args.seed,
            progress=progress,
        )
        print(f"Tuned {args.controller} with estimator={args.estimator}.")
        print(f"Best objective: {summary['validation_metrics']['objective']:.4f}")
        print(f"Artifacts: {summary['output_dir']}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
