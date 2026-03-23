from __future__ import annotations

import argparse
from pathlib import Path

from cartpole_bench.animations.render import render_animations
from cartpole_bench.plots.figures import generate_figures
from cartpole_bench.plots.tables import refresh_metric_tables
from cartpole_bench.simulation.batch import run_monte_carlo
from cartpole_bench.simulation.runner import run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portfolio-grade nonlinear cart-pole benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_output = "artifacts_v2"

    run_suite_parser = subparsers.add_parser("run-suite", help="Run a named experiment suite")
    run_suite_parser.add_argument("--suite", choices=("nominal", "stress"), required=True)
    run_suite_parser.add_argument("--output", default=default_output, help="Artifact directory")

    monte_carlo_parser = subparsers.add_parser("monte-carlo", help="Run the Monte Carlo benchmark")
    monte_carlo_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    monte_carlo_parser.add_argument("--samples", type=int, default=1000)
    monte_carlo_parser.add_argument("--output", default=default_output, help="Artifact directory")

    render_parser = subparsers.add_parser("render", help="Render figures and animations from saved artifacts")
    render_parser.add_argument("--formats", default="gif", help="Comma-separated animation formats")
    render_parser.add_argument("--theme", default="paper_white", help="Rendering theme name")
    render_parser.add_argument("--duration-profile", default="extended_gif", help="Animation duration profile")
    render_parser.add_argument("--no-supplements", action="store_true", help="Skip supplemental figures and animations")
    render_parser.add_argument("--output", default=default_output, help="Artifact directory")

    all_parser = subparsers.add_parser("all", help="Run all suites, Monte Carlo, and rendering")
    all_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    all_parser.add_argument("--samples", type=int, default=1000)
    all_parser.add_argument("--formats", default="gif", help="Comma-separated animation formats")
    all_parser.add_argument("--theme", default="paper_white", help="Rendering theme name")
    all_parser.add_argument("--duration-profile", default="extended_gif", help="Animation duration profile")
    all_parser.add_argument("--no-supplements", action="store_true", help="Skip supplemental figures and animations")
    all_parser.add_argument("--output", default=default_output, help="Artifact directory")
    return parser


def _resolve_output(path_str: str) -> Path:
    return Path(path_str).resolve()


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(fmt.strip() for fmt in raw.split(",") if fmt.strip())
    if not formats:
        raise ValueError("At least one animation format must be specified.")
    return formats


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = _resolve_output(args.output)

    if args.command == "run-suite":
        results = run_suite(args.suite, output_dir, Path.cwd())
        refresh_metric_tables(output_dir)
        print(f"Completed {args.suite} suite with {len(results)} runs.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "monte-carlo":
        summaries = run_monte_carlo(output_dir, requested_device=args.device, samples=args.samples)
        print(f"Completed Monte Carlo benchmark for {len(summaries)} controllers.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "render":
        formats = _parse_formats(args.formats)
        include_supplements = not args.no_supplements
        figure_paths = generate_figures(output_dir, theme=args.theme, include_supplements=include_supplements)
        animation_paths = render_animations(
            output_dir,
            formats=formats,
            theme=args.theme,
            duration_profile=args.duration_profile,
            include_supplements=include_supplements,
        )
        print(f"Rendered {len(figure_paths)} figures and {len(animation_paths)} animation storyboards.")
        print(f"Artifacts: {output_dir}")
        return 0

    if args.command == "all":
        run_suite("nominal", output_dir, Path.cwd())
        run_suite("stress", output_dir, Path.cwd())
        refresh_metric_tables(output_dir)
        run_monte_carlo(output_dir, requested_device=args.device, samples=args.samples)
        formats = _parse_formats(args.formats)
        include_supplements = not args.no_supplements
        figure_paths = generate_figures(output_dir, theme=args.theme, include_supplements=include_supplements)
        animation_paths = render_animations(
            output_dir,
            formats=formats,
            theme=args.theme,
            duration_profile=args.duration_profile,
            include_supplements=include_supplements,
        )
        print("Completed the full cart-pole benchmark.")
        print(f"Figures: {len(figure_paths)} | Animations: {len(animation_paths)}")
        print(f"Artifacts: {output_dir}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
