from __future__ import annotations

from pathlib import Path

import cartpole_bench.cli as cli_module
from cartpole_bench.cli import main


def test_run_suite_cli_smoke(tmp_path: Path) -> None:
    exit_code = main(
        ["run-suite", "--suite", "nominal", "--controllers", "lqr", "--output", str(tmp_path / "artifacts"), "--quiet"]
    )
    assert exit_code == 0
    assert (tmp_path / "artifacts" / "tables" / "metric_summary.csv").exists()


def test_all_cli_does_not_render_outputs(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    artifact_dir = tmp_path / "artifacts"

    def fake_run_suite(*args, **kwargs):
        calls.append(f"suite:{args[0]}")
        return []

    def fake_refresh_metric_tables(output_dir):
        calls.append("tables")
        (Path(output_dir) / "tables").mkdir(parents=True, exist_ok=True)

    def fake_run_monte_carlo(*args, **kwargs):
        calls.append("monte-carlo")
        return []

    def fail_render(*args, **kwargs):
        raise AssertionError("render should not be called by `all`")

    monkeypatch.setattr(cli_module, "run_suite", fake_run_suite)
    monkeypatch.setattr(cli_module, "refresh_metric_tables", fake_refresh_metric_tables)
    monkeypatch.setattr(cli_module, "run_monte_carlo", fake_run_monte_carlo)
    monkeypatch.setattr(cli_module, "generate_figures", fail_render)
    monkeypatch.setattr(cli_module, "render_animations", fail_render)

    exit_code = main(["all", "--controllers", "lqr", "--samples", "1", "--output", str(artifact_dir), "--quiet"])
    assert exit_code == 0
    assert calls == ["suite:nominal", "suite:stress", "tables", "monte-carlo"]
    assert not (artifact_dir / "figures").exists()
    assert not (artifact_dir / "animations").exists()
