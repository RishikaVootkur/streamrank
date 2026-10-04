from pathlib import Path

import pytest

from streamrank.common.config import Settings, get_settings


def test_defaults_without_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("REDIS_PORT", "SEED", "DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.redis_port == 6379
    assert settings.seed == 42
    assert settings.data_dir == Path("data")


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_PORT", "6400")
    monkeypatch.setenv("DATA_DIR", "/tmp/sr")
    settings = Settings(_env_file=None)
    assert settings.redis_port == 6400
    assert settings.data_dir == Path("/tmp/sr")


def test_env_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_PASSWORD=secret\nPOSTGRES_PORT=5555\n")
    settings = Settings(_env_file=env_file)
    assert settings.postgres_dsn.endswith("@localhost:5555/mlflow")
    assert "secret" not in repr(settings)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
