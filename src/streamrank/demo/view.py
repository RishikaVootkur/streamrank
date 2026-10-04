"""Pure helpers behind the demo page: API calls, history rows, and input parsing."""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from streamrank.serving.state import UserState


class ServiceError(RuntimeError):
    """The API could not be reached or answered with an error."""


def recommend(api: str, user_id: int, k: int, timeout: float = 5.0) -> dict[str, Any]:
    """GET /recommendations/{user_id}; raises ServiceError on network or HTTP errors."""
    url = f"{api}/recommendations/{user_id}?k={k}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            body: dict[str, Any] = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise ServiceError(f"API answered {exc.code} for user {user_id}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ServiceError(f"API unavailable at {api}") from exc
    return body


def history_rows(state: UserState, titles: np.ndarray, n: int = 15) -> list[tuple[str, str]]:
    """The user's last `n` events, newest first, as (title, "liked" or "rated")."""
    tokens, positive = state.tokens[-n:][::-1], state.positive[-n:][::-1]
    return [
        (str(titles[t - 1]), "liked" if p else "rated")
        for t, p in zip(tokens, positive, strict=True)
    ]


def parse_user_id(text: str) -> int | None:
    """A user ID typed by hand, or None when the text is not a plain non-negative integer."""
    text = text.strip()
    return int(text) if text.isascii() and text.isdigit() else None


def max_len(serving_dir: Path, default: int = 200) -> int:
    """Sequence length the serving state was built with (from serving.json)."""
    try:
        return int(json.loads((serving_dir / "serving.json").read_text())["max_len"])
    except (OSError, KeyError, ValueError):
        return default
