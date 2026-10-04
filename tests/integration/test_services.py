import psycopg
import pytest
import redis

from streamrank.common.config import get_settings

pytestmark = pytest.mark.integration


def test_redis_is_reachable() -> None:
    settings = get_settings()
    client = redis.Redis(host=settings.redis_host, port=settings.redis_port)
    assert client.ping()


def test_postgres_is_reachable() -> None:
    with psycopg.connect(get_settings().postgres_dsn, connect_timeout=5) as conn:
        row = conn.execute("SELECT 1").fetchone()
    assert row == (1,)


def test_mlflow_logs_and_reads_runs() -> None:
    import mlflow  # noqa: PLC0415 - heavy import only needed here
    from mlflow.tracking import MlflowClient  # noqa: PLC0415

    uri = get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("integration-check")
    with mlflow.start_run() as run:
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 0.5)
        mlflow.log_text("hello", "note.txt")
    fetched = MlflowClient(uri).get_run(run.info.run_id)
    assert fetched.data.params["p"] == "1"
    assert fetched.data.metrics["m"] == 0.5
    assert mlflow.artifacts.load_text(f"runs:/{run.info.run_id}/note.txt") == "hello"
