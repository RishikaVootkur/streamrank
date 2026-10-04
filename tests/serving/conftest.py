"""Tiny serving artifacts built without training (ONNX graphs written with onnx.helper)."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import polars as pl
import pytest
import redis
from onnx import TensorProto, helper, numpy_helper

from streamrank.common.config import get_settings
from streamrank.data.columns import GENRES
from streamrank.eval.dataset import build_train
from streamrank.models.sequences import build_sequences
from streamrank.ranking.features import FEATURES
from streamrank.retrieval.index import IndexSpec, build, save
from streamrank.serving.state import bulk_load

N_ITEMS, DIM, MAX_LEN = 50, 8, 6


def _user_tower(emb: np.ndarray, path: Path) -> None:
    """Mean of the item embeddings of the input tokens (padding row 0 is zeros)."""
    inputs = [
        helper.make_tensor_value_info(n, TensorProto.INT64, ["batch", MAX_LEN])
        for n in ("tokens", "positive", "gaps")
    ]
    out = helper.make_tensor_value_info("user_vector", TensorProto.FLOAT, ["batch", DIM])
    nodes = [
        helper.make_node("Gather", ["table", "tokens"], ["g"]),
        helper.make_node("ReduceMean", ["g"], ["user_vector"], axes=[1], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes, "user_tower", inputs, [out], [numpy_helper.from_array(emb, "table")]
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10),
        str(path),
    )


def _ranker(weights: np.ndarray, path: Path) -> None:
    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", len(FEATURES)])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["n", 1])
    node = helper.make_node("MatMul", ["features", "w"], ["score"])
    graph = helper.make_graph([node], "ranker", [x], [y], [numpy_helper.from_array(weights, "w")])
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10),
        str(path),
    )


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    s = get_settings()
    c = redis.Redis(host=s.redis_host, port=s.redis_port, db=15)
    c.flushdb()
    yield c
    c.flushdb()


@pytest.fixture
def serving_dir(tmp_path: Path, client: redis.Redis) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "serving"
    root.mkdir()
    item_vecs = rng.normal(size=(N_ITEMS, DIM)).astype(np.float32)
    item_vecs /= np.linalg.norm(item_vecs, axis=1, keepdims=True)
    table = np.vstack([np.zeros((1, DIM), np.float32), item_vecs])  # token = row + 1
    _user_tower(table, root / "user_tower.onnx")
    w = np.zeros((len(FEATURES), 1), dtype=np.float32)
    w[FEATURES.index("retrieval_score")] = 1.0  # ranker keeps retrieval order
    _ranker(w, root / "ranker.onnx")
    save(build(item_vecs, IndexSpec("flat")), root / "items.faiss")
    item_ids = np.arange(1, N_ITEMS + 1)
    genres = (rng.random((N_ITEMS, len(GENRES))) < 0.2).astype(np.float32)
    np.savez(
        root / "items.npz",
        item_ids=item_ids,
        has_movie=np.ones(N_ITEMS, bool),
        year=np.full(N_ITEMS, 2000.0),
        genres=genres,
        titles=np.array([f"Movie {i}" for i in item_ids], dtype=object),
    )
    np.savez(
        root / "profiles.npz",
        user_ids=np.array([1, 2]),
        mean_year=np.array([1999.0, 2001.0]),
        genres=rng.random((2, len(GENRES))).astype(np.float32),
    )
    np.save(root / "fallback.npy", np.arange(20, 40))
    (root / "serving.json").write_text(json.dumps({"cutoff_ts": 1000, "max_len": MAX_LEN}))
    (root / "loadtest_users.json").write_text(json.dumps([1, 2]))
    history = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "item_id": [1, 2, 3, 10, 11],
            "rating": [5.0, 4.0, 2.0, 5.0, 5.0],
            "ts": [10, 20, 30, 40, 50],
        }
    )
    train = build_train(history, cutoff_ts=1000)
    # Map training catalog tokens onto the serving catalog (both are item_id order here).
    seqs = build_sequences(train)
    tokens = train.item_ids[seqs.tokens - 1]
    seqs = type(seqs)(seqs.indptr, tokens, seqs.positive, seqs.gaps, seqs.ts)
    bulk_load(client, train.user_ids, seqs, max_len=MAX_LEN)
    return root
