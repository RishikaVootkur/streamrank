import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from streamrank.demo.view import ServiceError, history_rows, max_len, parse_user_id, recommend
from streamrank.serving.state import UserState


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/recommendations/1?"):
            body = json.dumps({"source": "personalized", "items": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(503)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def api() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_recommend_returns_body(api: str) -> None:
    assert recommend(api, 1, 10)["source"] == "personalized"


def test_recommend_wraps_http_errors(api: str) -> None:
    with pytest.raises(ServiceError, match="503"):
        recommend(api, 2, 10)


def test_recommend_wraps_connection_errors() -> None:
    with pytest.raises(ServiceError, match="unavailable"):
        recommend("http://127.0.0.1:9", 1, 10, timeout=1)


def test_history_rows_newest_first() -> None:
    state = UserState(
        tokens=np.array([1, 2, 3]),
        positive=np.array([1, 0, 1]),
        gaps=np.zeros(3, dtype=np.int64),
        last_ts=100,
        seen=np.array([0, 1, 2]),
    )
    titles = np.array(["A", "B", "C"])
    assert history_rows(state, titles, n=2) == [("C", "liked"), ("B", "rated")]


@pytest.mark.parametrize(
    ("text", "expected"), [(" 42 ", 42), ("x1", None), ("²", None), ("", None)]
)
def test_parse_user_id(text: str, expected: int | None) -> None:
    assert parse_user_id(text) == expected


def test_max_len_reads_serving_meta(tmp_path: Path) -> None:
    assert max_len(tmp_path) == 200
    (tmp_path / "serving.json").write_text(json.dumps({"max_len": 50}))
    assert max_len(tmp_path) == 50
