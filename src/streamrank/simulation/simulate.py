"""Replay future events and interleave two rankers, as an online test would.

For each sampled user, events after the training cutoff are replayed in time order.
Before each positive event, both rankers (A: retrieval order; B: retrieval + LambdaMART)
produce top-10 lists from the user's state at that moment; the lists are team-draft
interleaved, and the event's item counts as a click for the team that contributed it.
Every event (positive or not) then advances the user's serving state, exactly as the
streaming job would. Only users the ranker never trained on are replayed.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import redis

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.eval.dataset import load_setup
from streamrank.eval.metrics import bootstrap_mean
from streamrank.features.definitions import POSITIVE_RATING, SECONDS_PER_DAY
from streamrank.models.sequences import build_sequences
from streamrank.ranking.features import ITEM_STATS, USER_STATS
from streamrank.ranking.online import ItemTable
from streamrank.serving.engine import Artifacts, Engine
from streamrank.serving.state import append_event, bulk_load
from streamrank.simulation.interleave import credit, team_draft
from streamrank.streaming.events import replay_log

log = logging.getLogger(__name__)
SIM_DB = 10  # Redis database for the simulation, so serving state is untouched


class DailyFeatures:
    """Batch features refreshed every simulated day from the point-in-time snapshot tables.

    Production runs the batch pipeline daily; frozen cutoff features go stale over a
    replay that spans months. A snapshot on day d covers events before d, so the values
    used for an event on day d never include that day's events.
    """

    def __init__(
        self,
        items: ItemTable,
        item_snaps: pl.DataFrame,
        user_snaps: pl.DataFrame,
        item_ids: npt.NDArray[np.int64],
    ) -> None:
        self.items = items
        for name in ITEM_STATS:  # writable copies: they are updated in place each day
            items.stats[name] = np.array(items.stats[name], dtype=np.float64)
        index = {int(v): i for i, v in enumerate(item_ids)}
        snaps = item_snaps.filter(pl.col("item_id").is_in(list(index))).sort("snapshot_day")
        self._item_rows = snaps.with_columns(
            row=pl.col("item_id").replace_strict(index, return_dtype=pl.Int64)
        )
        self._users = user_snaps.sort("snapshot_day")
        self._user_values: dict[int, dict[str, float | None]] = {}
        self._day = -1
        self._i = self._u = 0

    def advance(self, day: int) -> None:
        """Apply every snapshot up to and including `day`."""
        if day <= self._day:
            return
        days = self._item_rows["snapshot_day"].to_numpy()
        j = int(np.searchsorted(days, day, side="right"))
        if j > self._i:
            chunk = self._item_rows.slice(self._i, j - self._i)
            rows = chunk["row"].to_numpy()
            for name in ITEM_STATS:
                self.items.stats[name][rows] = (
                    chunk[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
                )
            self._i = j
        udays = self._users["snapshot_day"].to_numpy()
        k = int(np.searchsorted(udays, day, side="right"))
        for row in self._users.slice(self._u, k - self._u).iter_rows(named=True):
            self._user_values[int(row["user_id"])] = {c: row[c] for c in USER_STATS}
        self._u = k
        self._day = day

    def user_features(self, user_id: int) -> dict[str, float | None]:
        return self._user_values.get(user_id, dict.fromkeys(USER_STATS))


def run(
    engine: Engine,
    events: pl.DataFrame,
    users: npt.NDArray[np.int64],
    *,
    k: int = 10,
    max_impressions: int = 20,
    seed: int = 0,
    daily: DailyFeatures | None = None,
    pair: tuple[str, str] = ("retrieval", "ranker"),
) -> dict[str, Any]:
    """Replay `events` for `users`, interleaving at each positive event.

    With `daily`, batch features advance to each event's day and feature times are taken
    relative to the event (a daily batch refresh); otherwise they stay at the cutoff.
    """
    rng = np.random.default_rng(seed)
    token = {int(item): i + 1 for i, item in enumerate(engine.art.item_ids)}
    keep = set(users.tolist())
    impressions: list[dict[str, Any]] = []
    shown: dict[int, int] = {}
    t0 = time.perf_counter()
    for row in events.filter(pl.col("user_id").is_in(list(keep))).iter_rows(named=True):
        user, item = int(row["user_id"]), int(row["item_id"])
        positive = float(row["rating"]) >= POSITIVE_RATING
        if daily is not None:
            daily.advance(int(row["ts"]) // SECONDS_PER_DAY)
            engine.now_ts = int(row["ts"])
        if positive and shown.get(user, 0) < max_impressions:
            lists = engine.variants(user, k)
            if lists is not None:
                a, b = lists[pair[0]], lists[pair[1]]
                merged = team_draft(a, b, k, rng)
                won = credit(merged, {item})
                impressions.append(
                    {
                        "user_id": user,
                        "A": won["A"],
                        "B": won["B"],
                        "hit_A": int(item in a),
                        "hit_B": int(item in b),
                    }
                )
                shown[user] = shown.get(user, 0) + 1
        if item in token:
            append_event(
                engine.client,
                user,
                token=token[item],
                positive=positive,
                ts=int(row["ts"]),
                max_len=engine.art.max_len,
            )
    result = summarize(pl.DataFrame(impressions), time.perf_counter() - t0, seed)
    result["pair"] = {"A": pair[0], "B": pair[1]}
    return result


def summarize(imp: pl.DataFrame, seconds: float, seed: int) -> dict[str, Any]:
    """Interleaving outcome with a bootstrap interval over users."""
    per_user = (
        imp.group_by("user_id")
        .agg(
            pl.col("A").sum(),
            pl.col("B").sum(),
            pl.col("hit_A").sum(),
            pl.col("hit_B").sum(),
            pl.len().alias("n"),
        )
        .sort("user_id")
    )  # fixed order: the bootstrap is then reproducible
    clicked = per_user.filter((pl.col("A") + pl.col("B")) > 0)
    # Per-user preference in [-1, 1]: (wins B - wins A) / clicks; the standard interleaving
    # "delta" averaged over users, so heavy users do not dominate.
    delta = ((clicked["B"] - clicked["A"]) / (clicked["B"] + clicked["A"])).to_numpy()
    ci = bootstrap_mean(delta.astype(np.float64), n_resamples=2000, seed=seed)
    total_a, total_b = int(imp["A"].sum()), int(imp["B"].sum())
    return {
        "impressions": imp.height,
        "users": per_user.height,
        "users_with_clicks": clicked.height,
        "clicks": {"A": total_a, "B": total_b},
        "b_click_share": total_b / max(1, total_a + total_b),
        "delta": {"mean": ci.mean, "low": ci.low, "high": ci.high},
        "winner": "B" if ci.low > 0 else "A" if ci.high < 0 else "tie",
        "hit_rate@10": {
            "A": float(imp["hit_A"].to_numpy().mean()),
            "B": float(imp["hit_B"].to_numpy().mean()),
        },
        "seconds": round(seconds, 1),
    }


def main(argv: list[str] | None = None) -> None:
    """Simulate an interleaving test between retrieval order and the ranker."""
    from streamrank.serving.app import (  # noqa: PLC0415
        _feast_store,
        load_item_table,
        user_feature_fn,
    )

    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--serving-dir", type=Path, default=s.artifacts_dir / "serving")
    p.add_argument("--ranker-dir", type=Path, default=s.artifacts_dir / "ranker")
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--max-impressions", type=int, default=20)
    p.add_argument("--features-dir", type=Path, default=s.data_dir / "features")
    p.add_argument(
        "--daily-features",
        action="store_true",
        help="refresh batch features each simulated day (default: frozen at cutoff)",
    )
    p.add_argument(
        "--pair", default="retrieval,ranker", help="two of popular, retrieval, ranker: A,B"
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    configure_logging(s.log_level)
    setup = load_setup(args.split_dir, "val")
    client = redis.Redis(host=s.redis_host, port=s.redis_port, db=SIM_DB)
    client.flushdb()
    art = Artifacts.load(args.serving_dir)
    bulk_load(client, setup.train.user_ids, build_sequences(setup.train), max_len=art.max_len)
    store = _feast_store()
    items = load_item_table(store, art)
    daily = None
    if args.daily_features:
        daily = DailyFeatures(
            items,
            pl.read_parquet(args.features_dir / "item_features.parquet"),
            pl.read_parquet(args.features_dir / "user_features.parquet"),
            art.item_ids,
        )
        engine = Engine(art, client, daily.user_features, items)
    else:
        engine = Engine(art, client, user_feature_fn(store), items)
    eval_users = np.load(args.ranker_dir / "eval_users.npy")
    rng = np.random.default_rng(s.seed)
    users = np.sort(rng.choice(eval_users, size=min(args.users, eval_users.size), replace=False))
    t1 = int(setup.train.cutoff_ts)
    events = replay_log(pl.read_parquet(args.split_dir / "val.parquet"), t1)
    a, b = args.pair.split(",")
    result = run(
        engine,
        events,
        users,
        max_impressions=args.max_impressions,
        seed=s.seed,
        daily=daily,
        pair=(a, b),
    )
    result["features"] = "daily" if args.daily_features else "frozen at cutoff"
    mode = "daily" if args.daily_features else "frozen"
    out = args.out or s.artifacts_dir / "simulation" / (
        f"interleaving_{args.pair.replace(',', '_vs_')}_{mode}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
