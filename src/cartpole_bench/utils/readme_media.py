from __future__ import annotations

import shutil
from pathlib import Path


README_MEDIA_MAP = {
    "animations/side_by_side_nominal_comparison.gif": "side_by_side_nominal_comparison.gif",
    "figures/nominal_local_response.png": "nominal_local_response.png",
    "figures/full_task_handoff.png": "full_task_handoff.png",
    "figures/stress_comparison.png": "stress_comparison.png",
    "figures/metric_summary.png": "metric_summary.png",
    "figures/supplemental/handoff_focus.png": "handoff_focus.png",
}


def sync_readme_media(artifact_root: Path, docs_media_dir: Path) -> list[Path]:
    artifact_root = artifact_root.resolve()
    docs_media_dir.mkdir(parents=True, exist_ok=True)
    missing = [relative for relative in README_MEDIA_MAP if not (artifact_root / relative).exists()]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Missing README media in '{artifact_root}': {missing_str}")

    copied: list[Path] = []
    for relative, filename in README_MEDIA_MAP.items():
        source = artifact_root / relative
        destination = docs_media_dir / filename
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied
