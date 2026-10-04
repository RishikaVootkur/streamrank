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
