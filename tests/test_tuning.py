from __future__ import annotations

from pathlib import Path

from cartpole_bench.simulation.tuning import tune_controller


def test_tune_controller_writes_expected_artifacts(tmp_path: Path) -> None:
    summary = tune_controller("lqr", tmp_path / "artifacts", budget=2, seed=3)
    output_dir = Path(summary["output_dir"])
    assert (output_dir / "tuning_history.csv").exists()
    assert (output_dir / "tuning_history.json").exists()
    assert (output_dir / "best_config.json").exists()
    assert (output_dir / "manifest_tuning.json").exists()
