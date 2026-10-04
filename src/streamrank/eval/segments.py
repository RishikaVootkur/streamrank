"""Where the two-tower model wins or loses against EASE: per-user comparison by segment.

Uses exported two-tower vectors (no PyTorch in this process) and a fresh EASE fit, scores
both on the same validation users with the shared full-catalog path, and breaks the paired
Recall@100 difference down by history length and by days since the user's last activity.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from streamrank.common.config import get_settings
from streamrank.eval import metrics
from streamrank.eval.dataset import load_setup
from streamrank.eval.evaluate import recommend
from streamrank.features.definitions import SECONDS_PER_DAY
from streamrank.models.baselines import EASE

FloatArray = npt.NDArray[np.float32]
HISTORY_EDGES = (20, 100, 500)
HISTORY_LABELS = ["1-20", "21-100", "101-500", ">500"]
RECENCY_EDGES = (1, 30, 365)
RECENCY_LABELS = ["<=1 day", "2-30 days", "31-365 days", ">1 year"]
MIN_USERS = 30  # smaller segments get no row: their intervals would be too wide to read


def bucket(
    values: npt.NDArray[np.float64], edges: tuple[int, ...], labels: list[str]
) -> npt.NDArray[np.str_]:
    """Label each value by the first edge it does not exceed (the last label above all)."""
    out: npt.NDArray[np.str_] = np.asarray(labels)[np.searchsorted(edges, values, side="left")]
    return out


class VectorScorer:
    """Scores users against items with precomputed vectors (dot product)."""

    def __init__(self, users: FloatArray, items: FloatArray, user_rows: npt.NDArray[np.int64]):
        self.users = users
        self.items = items
        self.row_of = {int(r): i for i, r in enumerate(user_rows)}

    def score(self, user_rows: npt.NDArray[np.int64]) -> FloatArray:
        idx = [self.row_of[int(r)] for r in user_rows]
        out: FloatArray = self.users[idx] @ self.items.T
        return out


def per_user_recall(model: Any, setup: Any, k: int = 100) -> npt.NDArray[np.float64]:
    t = setup.targets
    recs = recommend(model, t.user_rows, t.exclude, 200)
    hits = metrics.hit_matrix(recs, t.relevant)
    n_rel = np.diff(t.relevant.indptr).astype(np.int64)
    out: npt.NDArray[np.float64] = metrics.recall_at_k(hits, n_rel, k).astype(np.float64)
    return out


def segment_table(
    *,
    diff: npt.NDArray[np.float64],
    tt: npt.NDArray[np.float64],
    ease: npt.NDArray[np.float64],
    labels: npt.NDArray[np.str_],
    order: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for label in order:
        m = labels == label
        if m.sum() < MIN_USERS:
            continue
        iv = metrics.bootstrap_mean(diff[m], n_resamples=1000, seed=seed)
        rows.append(
            {
                "segment": label,
                "users": int(m.sum()),
                "two_tower": float(tt[m].mean()),
                "ease": float(ease[m].mean()),
                "diff": iv,
            }
        )
    return rows


def run(model_dir: Path, split_dir: Path, early_stop_users: Path, seed: int) -> str:
    setup = load_setup(split_dir, "val")
    vec = np.load(model_dir / "vectors_val.npz")
    tt = per_user_recall(VectorScorer(vec["users"], vec["items"], vec["user_rows"]), setup)
    ease_model = EASE(l2=2000.0, signal="positive")
    ease_model.fit(setup.train)
    ease = per_user_recall(ease_model, setup)
    users = setup.train.user_ids[setup.targets.user_rows]
    held = ~np.isin(users, np.load(early_stop_users))
    tt, ease, users = tt[held], ease[held], users[held]
    rows = setup.targets.user_rows[held]
    n_hist = np.diff(setup.train.interactions.indptr)[rows]
    last_ts = np.asarray(setup.train.timestamps[rows].max(axis=1).todense()).ravel()
    days = (setup.train.cutoff_ts - last_ts) / SECONDS_PER_DAY
    diff = tt - ease
    hist_labels = bucket(n_hist.astype(np.float64), HISTORY_EDGES, HISTORY_LABELS)
    rec_labels = bucket(days, RECENCY_EDGES, RECENCY_LABELS)
    lines = [
        "# Two-tower versus EASE by user segment",
        "",
        f"Validation users outside the retrieval early-stopping set ({held.sum():,} users). "
        "Recall@100 on the full catalog; the difference is two-tower minus EASE per user, with "
        "a 95% bootstrap interval.",
        "",
    ]
    overall = metrics.bootstrap_mean(diff, n_resamples=1000, seed=seed)
    lines += [
        f"Overall: two-tower {tt.mean():.4f}, EASE {ease.mean():.4f}, difference {overall}.",
        "",
    ]
    for title, labels, order in (
        ("Training history length (ratings)", hist_labels, HISTORY_LABELS),
        ("Days since last activity before the cutoff", rec_labels, RECENCY_LABELS),
    ):
        lines += [
            f"## {title}",
            "",
            "| Segment | Users | Two-tower | EASE | Difference [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in segment_table(diff=diff, tt=tt, ease=ease, labels=labels, order=order, seed=seed):
            lines.append(
                f"| {r['segment']} | {r['users']:,} | {r['two_tower']:.4f} | "
                f"{r['ease']:.4f} | {r['diff']} |"
            )
        lines.append("")
    pl.DataFrame(
        {
            "user_id": users,
            "two_tower": tt,
            "ease": ease,
            "n_history": n_hist,
            "days_since_last": days,
        }
    ).write_parquet(model_dir / "segments.parquet")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Write the per-segment comparison of the two-tower model and EASE."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--model-dir", type=Path, default=s.artifacts_dir / "retrieval" / "two_tower_full"
    )
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--doc", type=Path, default=Path("docs/two-tower-segments.md"))
    args = p.parse_args(argv)
    text = run(args.model_dir, args.split_dir, args.model_dir / "early_stop_users.npy", s.seed)
    args.doc.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
