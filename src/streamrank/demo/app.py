"""StreamRank demo: pick a user, see their recent history and recommendations, simulate a click.

Run with `streamlit run src/streamrank/demo/app.py` (or `make up`, which starts it at
port 8501). A simulated click appends the event to the user's serving state in Redis, the
same update the streaming job makes, so the next recommendation reflects it.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import redis
import streamlit as st

from streamrank.demo.view import ServiceError, history_rows, max_len, parse_user_id, recommend
from streamrank.serving.state import append_event, read

API = os.environ.get("API_URL", "http://localhost:8000")
SERVING_DIR = Path(os.environ.get("SERVING_DIR", "artifacts/serving"))


@st.cache_resource
def catalog() -> tuple[np.ndarray, dict[int, int], int]:
    items = np.load(SERVING_DIR / "items.npz", allow_pickle=True)
    ids = items["item_ids"]
    return items["titles"], {int(v): i for i, v in enumerate(ids)}, max_len(SERVING_DIR)


@st.cache_resource
def redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def main() -> None:
    st.set_page_config(page_title="StreamRank", layout="wide")
    st.title("StreamRank")
    st.caption(
        "Two-stage movie recommendations: two-tower retrieval, LambdaMART ranking, "
        "live session state."
    )
    titles, row_of, seq_len = catalog()
    users = json.loads((SERVING_DIR / "loadtest_users.json").read_text())
    user_id = int(st.sidebar.selectbox("Validation user", users[:200], index=0))
    custom = st.sidebar.text_input("Or any user ID (unknown IDs get popular movies)")
    if custom.strip():
        typed = parse_user_id(custom)
        if typed is None:
            st.sidebar.warning("User IDs are whole numbers.")
        else:
            user_id = typed
    k = st.sidebar.slider("Recommendations", 5, 20, 10)

    try:
        state = read(redis_client(), user_id)
    except redis.RedisError as exc:
        st.error(f"Redis unavailable: {exc}")
        st.stop()
    left, right = st.columns(2)
    with left:
        st.subheader("Recent history")
        if state.known:
            recent = history_rows(state, titles)
            st.table({"Movie": [r[0] for r in recent], "": [r[1] for r in recent]})
            st.caption(f"{state.seen.size:,} movies rated in total")
        else:
            st.info("No history: this user gets the popularity fallback.")
    with right:
        st.subheader("Recommended now")
        t0 = time.perf_counter()
        try:
            body = recommend(API, user_id, k)
        except ServiceError as exc:
            st.error(str(exc))
            st.stop()
        elapsed = (time.perf_counter() - t0) * 1000
        st.caption(
            f"source: {body['source']}, {elapsed:.0f} ms round trip, "
            f"{body['timings_ms']['total']:.1f} ms in the service"
        )
        for i, item in enumerate(body["items"], start=1):
            cols = st.columns([6, 1])
            cols[0].write(f"{i}. {item['title']}")
            if cols[1].button("Like", key=f"like-{item['item_id']}"):
                token = row_of[int(item["item_id"])] + 1
                last = state.last_ts or int(time.time())
                added = append_event(
                    redis_client(),
                    user_id,
                    token=token,
                    positive=True,
                    ts=last + 60,
                    max_len=seq_len,
                )
                if added:
                    st.rerun()
                st.toast("Already in this user's history.")
        with st.expander("Stage timings (ms)"):
            st.json(body["timings_ms"])


if __name__ == "__main__":
    main()
