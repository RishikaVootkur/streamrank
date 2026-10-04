import hashlib
import zipfile
from pathlib import Path

import pytest

from streamrank.data import download as dl
from streamrank.data.columns import RAW_COLUMNS


def _make_archive(path: Path) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, cols in RAW_COLUMNS.items():
            zf.writestr(f"{dl.ARCHIVE_DIR}/{name}", ",".join(cols) + "\n")
        zf.writestr(f"{dl.ARCHIVE_DIR}/README.txt", "readme")
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def test_md5sum_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"streamrank" * 1000)
    assert (
        dl.md5sum(f, chunk_size=7) == hashlib.md5(f.read_bytes(), usedforsecurity=False).hexdigest()
    )


def test_parse_md5_file() -> None:
    line = "D472BE332D4DAA821EDC399621853B57  ml-32m.zip\n"
    assert dl.parse_md5_file(line) == dl.EXPECTED_MD5
    with pytest.raises(ValueError, match="not an MD5"):
        dl.parse_md5_file("<html>not found</html>")


def test_verify_rejects_wrong_hash(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"data")
    with pytest.raises(dl.ChecksumError):
        dl.verify(f, "0" * 32)


def test_ensure_dataset_downloads_verifies_and_extracts(tmp_path: Path) -> None:
    source = tmp_path / "remote.zip"
    md5 = _make_archive(source)
    data_dir = tmp_path / "data"
    raw = dl.ensure_dataset(data_dir, expected_md5=md5, url=source.as_uri())
    assert sorted(p.name for p in raw.iterdir()) == sorted(RAW_COLUMNS)
    assert (raw / "ratings.csv").read_text().startswith("userId,movieId")
    assert not list(data_dir.glob("*.part"))

    # A second run reuses the verified archive without fetching again.
    source.unlink()
    assert dl.ensure_dataset(data_dir, expected_md5=md5, url=source.as_uri()) == raw


def test_ensure_dataset_replaces_corrupt_archive(tmp_path: Path) -> None:
    source = tmp_path / "remote.zip"
    md5 = _make_archive(source)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ml-32m.zip").write_bytes(b"truncated")
    dl.ensure_dataset(data_dir, expected_md5=md5, url=source.as_uri())
    assert dl.md5sum(data_dir / "ml-32m.zip") == md5


def test_ensure_dataset_fails_on_bad_download(tmp_path: Path) -> None:
    source = tmp_path / "remote.zip"
    _make_archive(source)
    with pytest.raises(dl.ChecksumError):
        dl.ensure_dataset(tmp_path / "data", expected_md5="0" * 32, url=source.as_uri())


def test_fetch_published_md5_reads_file_url(tmp_path: Path) -> None:
    f = tmp_path / "x.md5"
    f.write_text(f"{dl.EXPECTED_MD5}  ml-32m.zip\n")
    assert dl.fetch_published_md5(f.as_uri()) == dl.EXPECTED_MD5
