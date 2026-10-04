import subprocess
from pathlib import Path

import polars as pl
import pytest

from streamrank.common import provenance
from streamrank.common.provenance import (
    ProvenanceError,
    begin_stage,
    combined_hash,
    file_sha256,
    git_sha,
    read_manifest,
    verify_outputs,
    write_manifest,
    write_parquet_atomic,
)


def test_combined_hash_changes_with_content_and_order_free(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("one")
    b.write_text("two")
    h = combined_hash([a, b])
    assert h == combined_hash([b, a])
    b.write_text("three")
    assert combined_hash([a, b]) != h
    assert len(file_sha256(a)) == 64


def test_manifest_roundtrip(tmp_path: Path) -> None:
    write_manifest(tmp_path, stage="x", config={"k": 1}, data_hash="h", seed=3, extra={"n": 2})
    m = read_manifest(tmp_path)
    assert m["stage"] == "x"
    assert m["config"] == {"k": 1}
    assert m["seed"] == 3
    assert m["n"] == 2
    assert m["git_sha"]


def test_git_sha_in_repo_and_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.chdir(tmp_path)  # the package's own repo is used, not the working directory
    assert git_sha() != "unknown"
    assert git_sha(tmp_path) == "unknown"


def test_git_sha_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_SHA", "abc123")
    assert git_sha() == "abc123"


def test_extra_cannot_override_core_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collide"):
        write_manifest(tmp_path, stage="x", config={}, data_hash="h", seed=1, extra={"seed": 2})


def test_outputs_are_verified(tmp_path: Path) -> None:
    begin_stage(tmp_path)
    with pytest.raises(ProvenanceError, match="no manifest"):
        verify_outputs(tmp_path)
    path = tmp_path / "t.parquet"
    write_parquet_atomic(pl.DataFrame({"a": [1, 2]}), path)
    write_manifest(tmp_path, stage="x", config={}, data_hash="h", seed=1, outputs=[path])
    assert verify_outputs(tmp_path)["outputs"]["t.parquet"]
    write_parquet_atomic(pl.DataFrame({"a": [3]}), path)
    with pytest.raises(ProvenanceError, match="differs"):
        verify_outputs(tmp_path)
    begin_stage(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_atomic_write_cleans_up_on_failure(tmp_path: Path) -> None:
    bad = pl.DataFrame({"a": [object()]}, schema={"a": pl.Object})
    with pytest.raises(Exception):  # noqa: B017 - any write failure
        write_parquet_atomic(bad, tmp_path / "x.parquet")
    assert not list(tmp_path.iterdir())


def test_git_sha_marks_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    assert git_sha(tmp_path) == "unknown"  # no commits yet
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-q", "-m", "init")
    assert not git_sha(tmp_path).endswith("-dirty")
    (tmp_path / "f").write_text("2")
    assert git_sha(tmp_path).endswith("-dirty")


def test_git_sha_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: object, **__: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert git_sha() == "unknown"
