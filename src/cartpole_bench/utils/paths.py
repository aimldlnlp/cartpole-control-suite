from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_roots(base: Path) -> dict[str, Path]:
    return {
        "base": ensure_directory(base),
        "csv": ensure_directory(base / "csv"),
        "json": ensure_directory(base / "json"),
        "figures": ensure_directory(base / "figures"),
        "animations": ensure_directory(base / "animations"),
        "tables": ensure_directory(base / "tables"),
    }
