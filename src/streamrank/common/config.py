"""Project settings loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    seed: int = 42
    log_level: str = "INFO"

    mlflow_tracking_uri: str = "http://localhost:5001"

    redis_host: str = "localhost"
    redis_port: int = 6379

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "mlflow"
    postgres_password: SecretStr = SecretStr("")
    postgres_db: str = "mlflow"

    @property
    def postgres_dsn(self) -> str:
        """Connection string for the MLflow Postgres backend."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
