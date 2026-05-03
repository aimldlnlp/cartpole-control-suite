from __future__ import annotations

from pathlib import Path

import pytest

from cartpole_bench.utils.readme_media import README_MEDIA_MAP, sync_readme_media


def test_sync_readme_media_copies_expected_files(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    docs_media = tmp_path / "docs" / "media"
    for relative in README_MEDIA_MAP:
        path = artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    copied = sync_readme_media(artifact_root, docs_media)

    assert len(copied) == len(README_MEDIA_MAP)
    for relative, filename in README_MEDIA_MAP.items():
        assert (docs_media / filename).read_text(encoding="utf-8") == relative


def test_sync_readme_media_raises_for_missing_inputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    docs_media = tmp_path / "docs" / "media"
    artifact_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        sync_readme_media(artifact_root, docs_media)
