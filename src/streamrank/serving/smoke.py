"""End-to-end smoke test against the running API (`make smoke`)."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any

OK = HTTPStatus.OK


def _open(url: str, timeout: float) -> Any:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"only http(s) URLs are allowed: {url}")
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - scheme checked above


def get(url: str, timeout: float = 10.0) -> tuple[int, Any]:
    try:
        with _open(url, timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, None


def wait_ready(base: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if get(f"{base}/readyz", timeout=2)[0] == OK:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    raise SystemExit(f"API at {base} not ready after {seconds:.0f} s")


def check_response(body: Any, k: int, source: str) -> list[str]:
    """Problems with one recommendation response (empty when it is valid)."""
    problems = []
    if body is None:
        return ["no body"]
    items = body.get("items", [])
    if len(items) != k:
        problems.append(f"expected {k} items, got {len(items)}")
    if len({i["item_id"] for i in items}) != len(items):
        problems.append("duplicate items")
    if body.get("source") != source:
        problems.append(f"expected source {source}, got {body.get('source')}")
    if "total" not in body.get("timings_ms", {}):
        problems.append("missing timings")
    return problems


def main(argv: list[str] | None = None) -> None:
    """Check readiness, a known user, an unknown user, and the metrics endpoint."""
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--users", type=Path, default=Path("artifacts/serving/loadtest_users.json"))
    p.add_argument("--wait-seconds", type=float, default=180)
    args = p.parse_args(argv)
    wait_ready(args.base_url, args.wait_seconds)
    known = int(json.loads(args.users.read_text())[0])
    failures: dict[str, list[str]] = {}
    for label, user, source in (("known", known, "personalized"), ("unknown", 10**9, "popular")):
        status, body = get(f"{args.base_url}/recommendations/{user}?k=10")
        problems = [f"status {status}"] if status != OK else check_response(body, 10, source)
        if problems:
            failures[label] = problems
        else:
            first = body["items"][0]
            print(
                f"{label} user {user}: {first['title']!r} first, "
                f"{body['timings_ms']['total']:.1f} ms"
            )
    status, _ = get(f"{args.base_url}/livez")
    if status != OK:
        failures["livez"] = [f"status {status}"]
    with _open(f"{args.base_url}/metrics/", 5) as resp:
        if b"recommend_latency_seconds" not in resp.read():
            failures["metrics"] = ["latency histogram missing"]
    if failures:
        print(json.dumps(failures, indent=2))
        sys.exit(1)
    print("smoke test passed")


if __name__ == "__main__":
    main()
