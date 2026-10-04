"""Provenance records written next to every pipeline artifact."""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from streamrank import __version__


def git_sha(repo_dir: Path | None = None) -> str:
    """Return the current commit SHA, with a `-dirty` suffix for uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH is intended
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def combined_hash(paths: list[Path]) -> str:
    """Hash several files into one digest that changes when any file or name changes."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def write_manifest(
    out_dir: Path,
    *,
    stage: str,
    config: Mapping[str, Any],
    data_hash: str,
    seed: int | None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write `manifest.json` describing how the artifacts in `out_dir` were produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "package_version": __version__,
        "config": dict(config),
        "data_hash": data_hash,
        "seed": seed,
        **(extra or {}),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return path


def read_manifest(out_dir: Path) -> dict[str, Any]:
    """Read the manifest in `out_dir`."""
    data: dict[str, Any] = json.loads((out_dir / "manifest.json").read_text())
    return data
