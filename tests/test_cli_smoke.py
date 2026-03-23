from __future__ import annotations

from pathlib import Path

from cartpole_bench.cli import main


def test_run_suite_cli_smoke(tmp_path: Path) -> None:
    exit_code = main(["run-suite", "--suite", "nominal", "--output", str(tmp_path / "artifacts")])
    assert exit_code == 0
    assert (tmp_path / "artifacts" / "tables" / "metric_summary.csv").exists()
