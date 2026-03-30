#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cartpole_bench.plots.tables import refresh_metric_tables


CLASSIC_CONTROLLERS = {
    "LQR",
    "Feedback Linearization (PFL)",
    "Sliding Mode Control (SMC)",
}
OPTIMIZER_CONTROLLERS = {
    "Iterative LQR (iLQR)",
    "Model Predictive Control (MPC)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge classic and optimizer cart-pole artifacts into one output root.")
    parser.add_argument("--classic-root", default="artifacts_rerun_ekf_balanced", help="Artifact root with classic-controller runs.")
    parser.add_argument("--optimizer-root", default="artifacts_optimizer_eval", help="Artifact root with optimizer runs.")
    parser.add_argument("--output-root", default="artifacts_final_ekf_merged", help="Merged artifact root to create/update.")
    parser.add_argument("--estimator", default="ekf", help="Estimator name to merge.")
    parser.add_argument(
        "--include-monte-carlo",
        action="store_true",
        help="Also copy Monte Carlo manifests/tables from the classic root when present.",
    )
    return parser.parse_args()


def load_manifest(root: Path, name: str) -> dict | None:
    path = root / "json" / f"manifest_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_key(run: dict) -> tuple[str, str, str, str, int]:
    return (
        str(run["suite"]),
        str(run["scenario"]),
        str(run["controller"]),
        str(run["estimator"]),
        int(run["seed"]),
    )


def ensure_dirs(root: Path) -> None:
    for name in ("csv", "json", "tables", "figures", "animations"):
        (root / name).mkdir(parents=True, exist_ok=True)


def copy_relpath(source_root: Path, output_root: Path, relative_path: str) -> None:
    src = source_root / relative_path
    dst = output_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def selected_runs(manifest: dict | None, allowed_controllers: set[str], estimator_name: str) -> list[dict]:
    if manifest is None:
        return []
    return [
        run
        for run in manifest.get("runs", [])
        if run["controller"] in allowed_controllers and run["estimator"] == estimator_name
    ]


def merge_suite_manifest(
    *,
    classic_root: Path,
    optimizer_root: Path,
    output_root: Path,
    estimator_name: str,
    suite_name: str,
) -> int:
    manifest_name = f"{suite_name}_{estimator_name}"
    classic_manifest = load_manifest(classic_root, manifest_name)
    optimizer_manifest = load_manifest(optimizer_root, manifest_name)

    classic_runs = selected_runs(classic_manifest, CLASSIC_CONTROLLERS, estimator_name)
    optimizer_runs = selected_runs(optimizer_manifest, OPTIMIZER_CONTROLLERS, estimator_name)

    merged: dict[tuple[str, str, str, str, int], dict] = {}

    for run in classic_runs:
        copy_relpath(classic_root, output_root, run["csv_path"])
        copy_relpath(classic_root, output_root, run["json_path"])
        merged[run_key(run)] = run

    for run in optimizer_runs:
        copy_relpath(optimizer_root, output_root, run["csv_path"])
        copy_relpath(optimizer_root, output_root, run["json_path"])
        merged[run_key(run)] = run

    template = optimizer_manifest or classic_manifest
    if template is None:
        return 0

    final_runs = [
        merged[key]
        for key in sorted(
            merged.keys(),
            key=lambda key: (key[1], key[2], key[4]),
        )
    ]

    payload = {
        "artifact_version": template.get("artifact_version", "v3"),
        "suite": template["suite"],
        "estimator_name": estimator_name,
        "runs": final_runs,
    }
    manifest_path = output_root / "json" / f"manifest_{manifest_name}.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return len(final_runs)


def copy_monte_carlo(classic_root: Path, output_root: Path, estimator_name: str) -> list[str]:
    copied: list[str] = []
    manifest_name = f"manifest_monte_carlo_{estimator_name}.json"
    manifest_path = classic_root / "json" / manifest_name
    if manifest_path.exists():
        shutil.copy2(manifest_path, output_root / "json" / manifest_name)
        copied.append(str((output_root / "json" / manifest_name).relative_to(output_root)))

    for name in (
        "monte_carlo_summary.csv",
        "monte_carlo_summary.json",
        "monte_carlo_samples.csv",
    ):
        source = classic_root / "tables" / name
        if source.exists():
            destination = output_root / "tables" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(output_root)))
    return copied


def main() -> int:
    args = parse_args()
    classic_root = Path(args.classic_root).resolve()
    optimizer_root = Path(args.optimizer_root).resolve()
    output_root = Path(args.output_root).resolve()
    estimator_name = str(args.estimator)

    ensure_dirs(output_root)

    nominal_count = merge_suite_manifest(
        classic_root=classic_root,
        optimizer_root=optimizer_root,
        output_root=output_root,
        estimator_name=estimator_name,
        suite_name="nominal",
    )
    stress_count = merge_suite_manifest(
        classic_root=classic_root,
        optimizer_root=optimizer_root,
        output_root=output_root,
        estimator_name=estimator_name,
        suite_name="stress",
    )

    copied_mc = copy_monte_carlo(classic_root, output_root, estimator_name) if args.include_monte_carlo else []

    refresh_metric_tables(output_root)

    print("Merged final artifact root")
    print(f"  classic_root: {classic_root}")
    print(f"  optimizer_root: {optimizer_root}")
    print(f"  output_root: {output_root}")
    print(f"  estimator: {estimator_name}")
    print(f"  nominal_runs: {nominal_count}")
    print(f"  stress_runs: {stress_count}")
    if copied_mc:
        print(f"  copied_monte_carlo_files: {len(copied_mc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
