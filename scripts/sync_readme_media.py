#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cartpole_bench.utils.readme_media import sync_readme_media


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy curated render outputs into docs/media for the README.")
    parser.add_argument("--artifact-root", default="artifacts_full_dense_cmu", help="Artifact root to copy from.")
    parser.add_argument("--docs-media", default=str(REPO_ROOT / "docs" / "media"), help="Destination docs/media directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    docs_media_dir = Path(args.docs_media).resolve()
    copied = sync_readme_media(artifact_root, docs_media_dir)
    print("Synced README media")
    print(f"  artifact_root: {artifact_root}")
    print(f"  docs_media: {docs_media_dir}")
    print(f"  files: {len(copied)}")
    for path in copied:
        print(f"  - {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
