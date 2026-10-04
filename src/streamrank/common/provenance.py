"""Provenance records written next to every pipeline artifact."""

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from streamrank import __version__

MANIFEST: Final = "manifest.json"
_PACKAGE_DIR: Final = Path(__file__).resolve().parent.parent
_CORE_KEYS: Final = frozenset(
    {"stage", "created_at", "git_sha", "package_version", "config", "data_hash", "seed", "outputs"}
)


class ProvenanceError(RuntimeError):
    """Raised when an artifact does not match its manifest."""


def git_sha(repo_dir: Path | None = None) -> str:
    """Return the commit SHA of the repo holding this package, `-dirty` if it has changes.

    The `GIT_SHA` environment variable overrides the lookup (for containers without `.git`).
    """
    if repo_dir is None and (env_sha := os.environ.get("GIT_SHA")):
        return env_sha
    cwd = repo_dir or _PACKAGE_DIR
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH is intended
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=cwd,
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


def combined_hash(paths: Sequence[Path]) -> str:
    """Hash several files into one digest that changes when any file or name changes."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def begin_stage(out_dir: Path) -> None:
    """Prepare `out_dir` for a new run: remove the old manifest so a crash leaves none."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).unlink(missing_ok=True)


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write Parquet through a temporary file so readers never see a partial file."""
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        df.write_parquet(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(path)


def write_manifest(
    out_dir: Path,
    *,
    stage: str,
    config: Mapping[str, Any],
    data_hash: str,
    seed: int | None,
    outputs: Sequence[Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the manifest describing how the artifacts in `out_dir` were produced.

    `outputs` are hashed so later stages can confirm they read exactly these files.
    """
    extra = dict(extra or {})
    if clash := _CORE_KEYS & extra.keys():
        raise ValueError(f"extra keys collide with core manifest fields: {sorted(clash)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "package_version": __version__,
        "config": dict(config),
        "data_hash": data_hash,
        "seed": seed,
        "outputs": {p.name: file_sha256(p) for p in outputs},
        **extra,
    }
    path = out_dir / MANIFEST
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def read_manifest(out_dir: Path) -> dict[str, Any]:
    """Read the manifest in `out_dir`."""
    data: dict[str, Any] = json.loads((out_dir / MANIFEST).read_text())
    return data


def verify_outputs(out_dir: Path) -> dict[str, Any]:
    """Return the manifest after checking every recorded output still matches its hash."""
    path = out_dir / MANIFEST
    if not path.exists():
        raise ProvenanceError(f"{out_dir} has no manifest; the stage did not finish")
    manifest = read_manifest(out_dir)
    for name, expected in manifest.get("outputs", {}).items():
        file = out_dir / name
        if not file.exists() or file_sha256(file) != expected:
            raise ProvenanceError(f"{file} is missing or differs from its manifest")
    return manifest
