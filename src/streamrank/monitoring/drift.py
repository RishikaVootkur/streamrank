"""Data drift report: rating events in a recent window against a reference window.

Each event gets the features the recommender relies on, computed point-in-time: the
rating, hour and weekday, the item's 30-day popularity and age from the snapshot taken
before the event's day, the item's release year, and its first genre. Evidently compares
the current window with the reference window (by default the 90 days before the training
cutoff against the last 90 days of the validation period) and writes HTML and JSON reports.
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import verify_outputs
from streamrank.features.definitions import POSITIVE_RATING, SECONDS_PER_DAY

log = logging.getLogger(__name__)
NUMERICAL = ["rating", "hour_utc", "item_log_n_ratings_30d", "item_age_days", "item_year"]
CATEGORICAL = ["positive", "weekday", "item_genre"]
WINDOW_DAYS = 90


def event_features(
    ratings: pl.DataFrame, item_snaps: pl.DataFrame, movies: pl.DataFrame, start: int, end: int
) -> pl.DataFrame:
    """Point-in-time features for events with start <= ts < end."""
    events = ratings.filter((pl.col("ts") >= start) & (pl.col("ts") < end)).with_columns(
        day_start=(pl.col("ts") // SECONDS_PER_DAY) * SECONDS_PER_DAY
    )
    snaps = item_snaps.select(
        "item_id",
        snap_ts=pl.col("snapshot_day") * SECONDS_PER_DAY,
        item_n_ratings_30d=pl.col("item_n_ratings_30d"),
        item_first_ts=pl.col("item_first_ts"),
    ).sort("snap_ts")
    joined = events.sort("day_start").join_asof(
        snaps, left_on="day_start", right_on="snap_ts", by="item_id", strategy="backward"
    )
    when = pl.from_epoch("ts", time_unit="s")
    return joined.join(movies.select("item_id", "year", "genres"), on="item_id", how="left").select(
        pl.col("rating").cast(pl.Float64),
        positive=(pl.col("rating") >= POSITIVE_RATING).cast(pl.Utf8),
        hour_utc=when.dt.hour().cast(pl.Float64),
        weekday=when.dt.weekday().cast(pl.Utf8),
        item_log_n_ratings_30d=pl.col("item_n_ratings_30d").fill_null(0).cast(pl.Float64).log1p(),
        item_age_days=((pl.col("ts") - pl.col("item_first_ts")) / SECONDS_PER_DAY).cast(pl.Float64),
        item_year=pl.col("year").cast(pl.Float64),
        item_genre=pl.col("genres").list.first().fill_null("(unknown)"),
    )


def run_report(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Run Evidently's data drift preset and return a compact summary."""
    from evidently import DataDefinition, Dataset, Report  # noqa: PLC0415
    from evidently.presets import DataDriftPreset  # noqa: PLC0415

    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    cur = Dataset.from_pandas(current, data_definition=definition)
    ref = Dataset.from_pandas(reference, data_definition=definition)
    # Distance methods at any sample size: Evidently's small-sample defaults are p-value tests,
    # where drift means a score below the threshold, so the comparison below would flip.
    preset = DataDriftPreset(num_method="wasserstein", cat_method="jensenshannon")
    snapshot = Report([preset]).run(cur, ref)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_dir / "drift_report.html"))
    snapshot.save_json(str(out_dir / "drift_report.json"))
    return summarize(json.loads((out_dir / "drift_report.json").read_text()))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Drift share and per-column scores from Evidently's saved JSON report."""
    summary: dict[str, Any] = {"columns": {}}
    for metric in report.get("metrics", []):
        config = metric.get("config", {})
        kind = str(config.get("type", ""))
        value = metric.get("value")
        if kind.endswith("DriftedColumnsCount") and isinstance(value, dict):
            summary["drifted_columns"] = int(value["count"])
            summary["drift_share"] = float(value["share"])
        elif kind.endswith("ValueDrift") and value is not None:
            threshold = float(config.get("threshold", 0.1))
            summary["columns"][config["column"]] = {
                "score": float(value),
                "method": config.get("method"),
                "threshold": threshold,
                "drift": float(value) >= threshold,
            }
    return summary


def window(end: int, days: int) -> tuple[int, int]:
    return end - days * SECONDS_PER_DAY, end


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).date().isoformat()


def main(argv: list[str] | None = None) -> None:
    """Write a data drift report for recent events against the reference window."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--features-dir", type=Path, default=s.data_dir / "features")
    p.add_argument("--out-dir", type=Path, default=s.artifacts_dir / "drift")
    p.add_argument("--doc", type=Path, default=Path("docs/drift.md"))
    p.add_argument("--sample", type=int, default=50_000)
    p.add_argument("--current-end", default=None, help="UTC date; default: end of validation")
    args = p.parse_args(argv)
    configure_logging(s.log_level)
    manifest = verify_outputs(args.split_dir)
    t1, t2 = int(manifest["t1"]), int(manifest["t2"])
    cur_end = (
        t2
        if args.current_end is None
        else int(datetime.fromisoformat(args.current_end).replace(tzinfo=UTC).timestamp())
    )
    ratings = pl.read_parquet(s.data_dir / "processed" / "ratings.parquet")
    snaps = pl.read_parquet(args.features_dir / "item_features.parquet")
    movies = pl.read_parquet(s.data_dir / "processed" / "movies.parquet")
    ref_win, cur_win = window(t1, WINDOW_DAYS), window(cur_end, WINDOW_DAYS)
    frames, totals = [], []
    for start, end in (ref_win, cur_win):
        df = event_features(ratings, snaps, movies, start, end)
        totals.append(df.height)
        df = df.sort(df.columns)  # sample by position from a fixed row order
        frames.append(df.sample(min(args.sample, df.height), seed=s.seed).to_pandas())
    summary = run_report(frames[0], frames[1], args.out_dir)
    summary |= {
        "reference": [_iso(ref_win[0]), _iso(ref_win[1])],
        "current": [_iso(cur_win[0]), _iso(cur_win[1])],
        "rows": {"reference": len(frames[0]), "current": len(frames[1])},
        "events": {"reference": totals[0], "current": totals[1]},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    args.doc.write_text(render(summary))
    log.info("drift", extra={"drift_share": summary.get("drift_share")})
    print(json.dumps(summary, indent=2))


def render(summary: dict[str, Any]) -> str:
    """Markdown summary of the drift report."""
    lines = [
        "# Data drift report",
        "",
        f"Reference: rating events from {summary['reference'][0]} to {summary['reference'][1]} "
        f"(the 90 days before the training cutoff). Current: {summary['current'][0]} to "
        f"{summary['current'][1]}. Sampled events: {summary['rows']['reference']:,} reference, "
        f"{summary['rows']['current']:,} current. Evidently `DataDriftPreset` with Wasserstein "
        "distance for numeric columns and Jensen-Shannon distance for categorical ones. "
        "Full report: `make drift` writes "
        "`artifacts/drift/drift_report.html`.",
        "",
        f"Drifted columns: {summary.get('drifted_columns')} of "
        f"{len(summary['columns'])} (share {summary.get('drift_share')}).",
        "",
        "| Column | Method | Score | Threshold | Drift |",
        "| --- | --- | ---: | ---: | :---: |",
    ]
    for column, info in summary["columns"].items():
        flag = "yes" if info["drift"] else "no"
        score, threshold = f"{info['score']:.4f}", f"{info['threshold']:g}"
        lines.append(f"| {column} | {info['method']} | {score} | {threshold} | {flag} |")
    events = summary.get("events")
    if events:
        lines += [
            "",
            f"Event volume: {events['reference']:,} reference and {events['current']:,} current "
            "events. Item popularity is an absolute 30-day count, so a change in overall volume "
            "shifts its whole distribution; check volume before reading popularity drift as a "
            "change in what people watch.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
