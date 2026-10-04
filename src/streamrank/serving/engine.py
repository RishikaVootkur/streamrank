"""The recommendation pipeline for one request, independent of the web framework.

1. Read the user's recent events and seen set from Redis.
2. Unknown user: return the recent-popularity fallback.
3. Encode the left-padded sequence with the ONNX user tower (same padding as training).
4. Retrieve candidates from FAISS, fetching past the user's seen items, keep the top K.
5. Compute ranker features (same formulas as training) and score with the ONNX ranker.
"""

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import redis

from streamrank.ranking.online import ItemTable, request_features
from streamrank.retrieval.index import load as load_index
from streamrank.serving.state import UserState, read

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
UserFeatureFn = Callable[[int], Mapping[str, float | None]]
CANDIDATES = 200
# Fetch cap keeps tail latency bounded: users with more than MAX_FETCH - CANDIDATES seen
# items may get fewer than CANDIDATES unseen candidates.
MAX_FETCH = 4096


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@dataclass
class Artifacts:
    """Everything loaded from the serving artifact directory."""

    user_tower: ort.InferenceSession
    ranker: ort.InferenceSession
    index: Any
    item_ids: IntArray
    titles: npt.NDArray[np.object_]
    year: npt.NDArray[np.float64]
    genres: FloatArray
    has_movie: npt.NDArray[np.bool_]
    profile_users: IntArray
    profile_year: npt.NDArray[np.float64]
    profile_genres: FloatArray
    fallback: IntArray
    max_len: int
    cutoff_ts: int

    @classmethod
    def load(cls, root: Path) -> "Artifacts":
        meta = json.loads((root / "serving.json").read_text())
        items = np.load(root / "items.npz", allow_pickle=True)
        prof = np.load(root / "profiles.npz")
        index = load_index(root / "items.faiss").index  # restores search-time parameters
        return cls(
            user_tower=_session(root / "user_tower.onnx"),
            ranker=_session(root / "ranker.onnx"),
            index=index,
            item_ids=items["item_ids"],
            titles=items["titles"],
            year=items["year"],
            genres=items["genres"],
            has_movie=items["has_movie"],
            profile_users=prof["user_ids"],
            profile_year=prof["mean_year"],
            profile_genres=prof["genres"],
            fallback=np.load(root / "fallback.npy"),
            max_len=int(meta["max_len"]),
            cutoff_ts=int(meta["cutoff_ts"]),
        )


@dataclass
class Recommendation:
    """A ranked list plus how it was produced."""

    item_ids: list[int]
    titles: list[str]
    scores: list[float]
    source: str  # "personalized" or "popular"
    timings_ms: dict[str, float] = field(default_factory=dict)


def left_pad(state: UserState, max_len: int) -> dict[str, IntArray]:
    """The last `max_len` events, left-padded with zeros (the training query layout)."""
    k = min(state.tokens.size, max_len)
    out = {name: np.zeros((1, max_len), dtype=np.int64) for name in ("tokens", "positive", "gaps")}
    if k:
        out["tokens"][0, max_len - k :] = state.tokens[-k:]
        out["positive"][0, max_len - k :] = state.positive[-k:]
        out["gaps"][0, max_len - k :] = state.gaps[-k:]
    return out


def retrieve(
    index: Any, user: FloatArray, seen: IntArray, n_items: int, k: int = CANDIDATES
) -> tuple[IntArray, FloatArray]:
    """Top-k unseen catalog rows and scores, fetching k + |seen| (capped) from the index."""
    fetch = int(min(n_items, k + seen.size, max(MAX_FETCH, k)))
    scores, rows = index.search(np.ascontiguousarray(user, dtype=np.float32), fetch)
    rows, scores = rows[0], scores[0]
    keep = (rows >= 0) & ~np.isin(rows, seen, assume_unique=False)
    return rows[keep][:k].astype(np.int64), scores[keep][:k].astype(np.float32)


class Engine:
    """Runs the request pipeline; all CPU work is synchronous (call it from a thread)."""

    def __init__(
        self,
        art: Artifacts,
        client: redis.Redis,
        user_features: UserFeatureFn,
        items: ItemTable,
        now_ts: int | None = None,
    ) -> None:
        self.art = art
        self.client = client
        self.user_features = user_features
        self.items = items
        self.now_ts = now_ts if now_ts is not None else art.cutoff_ts
        self._profile_index = {int(u): i for i, u in enumerate(art.profile_users)}

    def _profile(self, user_id: int) -> tuple[float, FloatArray] | None:
        i = self._profile_index.get(user_id)
        if i is None:
            return None
        return float(self.art.profile_year[i]), self.art.profile_genres[i]

    def _result(self, rows: IntArray, scores: FloatArray, source: str) -> Recommendation:
        return Recommendation(
            item_ids=[int(x) for x in self.art.item_ids[rows]],
            titles=[str(x) for x in self.art.titles[rows]],
            scores=[float(x) for x in scores],
            source=source,
        )

    def recommend(self, user_id: int, k: int = 10) -> Recommendation:
        """Top-k items for a user."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str) -> None:
            nonlocal t0
            now = time.perf_counter()
            timings[name] = round((now - t0) * 1000, 3)
            t0 = now

        state = read(self.client, user_id)
        lap("state")
        if not state.known:
            rows = self.art.fallback[:k]
            result = self._result(rows, np.zeros(rows.size, dtype=np.float32), "popular")
            result.timings_ms = timings
            return result
        user_vec = self.art.user_tower.run(None, left_pad(state, self.art.max_len))[0]
        lap("user_tower")
        rows, scores = retrieve(self.art.index, user_vec, state.seen, self.art.item_ids.size)
        lap("retrieval")
        stats = self.user_features(user_id)
        lap("user_features")
        feats = request_features(
            rows, scores, stats, self._profile(user_id), items=self.items, now_ts=self.now_ts
        )
        ranked = np.asarray(self.art.ranker.run(None, {"features": feats})[0]).ravel()
        order = np.argsort(-ranked, kind="stable")[:k]
        lap("ranking")
        result = self._result(rows[order], ranked[order].astype(np.float32), "personalized")
        result.timings_ms = timings
        return result
