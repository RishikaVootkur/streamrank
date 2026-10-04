"""Download MovieLens 32M from GroupLens and verify it against the published MD5."""

import argparse
import hashlib
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.data.columns import RAW_COLUMNS

DATASET_URL: Final = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
CHECKSUM_URL: Final = DATASET_URL + ".md5"
EXPECTED_MD5: Final = "d472be332d4daa821edc399621853b57"
ARCHIVE_DIR: Final = "ml-32m"
_MD5_HEX_LEN: Final = 32

log = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its expected checksum."""


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_md5_file(text: str) -> str:
    """Extract the hash from an `md5sum`-style line such as `<hash>  <file>`."""
    token = text.strip().split()[0].lower()
    if len(token) != _MD5_HEX_LEN or any(c not in "0123456789abcdef" for c in token):
        raise ValueError(f"not an MD5 checksum line: {text!r}")
    return token


def fetch_published_md5(url: str = CHECKSUM_URL, timeout: float = 30.0) -> str:
    """Download and parse the checksum file published next to the archive."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return parse_md5_file(resp.read().decode())


def verify(path: Path, expected: str) -> None:
    """Raise `ChecksumError` unless `path` has MD5 `expected`."""
    actual = md5sum(path)
    if actual != expected:
        raise ChecksumError(f"{path} has MD5 {actual}, expected {expected}")


def download(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Stream `url` to `dest` through a temporary file so partial downloads never remain."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def extract(archive: Path, raw_dir: Path) -> Path:
    """Extract the CSV files from the archive into `raw_dir` and return that folder."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in RAW_COLUMNS:
            member = f"{ARCHIVE_DIR}/{name}"
            target = raw_dir / name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
    return raw_dir


def ensure_dataset(
    data_dir: Path, expected_md5: str = EXPECTED_MD5, url: str = DATASET_URL
) -> Path:
    """Download (if needed), verify, and extract MovieLens 32M. Returns the raw CSV folder.

    Re-running is cheap: a verified archive is not downloaded again, and CSVs are only
    extracted when missing.
    """
    archive = data_dir / "ml-32m.zip"
    raw_dir = data_dir / "raw"
    if archive.exists():
        try:
            verify(archive, expected_md5)
            log.info("archive already present and verified", extra={"path": str(archive)})
        except ChecksumError:
            log.warning("archive checksum mismatch, downloading again")
            archive.unlink()
    if not archive.exists():
        log.info("downloading", extra={"url": url})
        download(url, archive)
        verify(archive, expected_md5)
        log.info("checksum verified", extra={"md5": expected_md5})
    if not all((raw_dir / name).exists() for name in RAW_COLUMNS):
        extract(archive, raw_dir)
        log.info("extracted", extra={"raw_dir": str(raw_dir)})
    return raw_dir


def main(argv: list[str] | None = None) -> None:
    """Download and verify MovieLens 32M into the data directory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    published = fetch_published_md5()
    if published != EXPECTED_MD5:
        raise ChecksumError(f"published MD5 {published} differs from pinned {EXPECTED_MD5}")
    ensure_dataset(args.data_dir or settings.data_dir)


if __name__ == "__main__":
    main()
