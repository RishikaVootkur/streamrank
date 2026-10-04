import json
import logging

import pytest

from streamrank.common.logging import configure_logging


def test_logs_are_json_with_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("debug")
    logging.getLogger("streamrank.test").info("hello", extra={"user_id": 5})
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["message"] == "hello"
    assert record["level"] == "INFO"
    assert record["logger"] == "streamrank.test"
    assert record["user_id"] == 5


def test_exceptions_are_included(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("streamrank.test").exception("failed")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert "ValueError: boom" in record["exc_info"]
