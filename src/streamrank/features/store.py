"""Access to the Feast feature store defined in `feature_repo/`."""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from feast import FeatureStore

from streamrank.common.config import get_settings
from streamrank.features.definitions import ITEM_FEATURES, USER_FEATURES, FeatureSet

# The repo checkout's feature_repo; containers set FEATURE_REPO because an installed
# package does not live next to it.
DEFAULT_REPO = Path(
    os.environ.get("FEATURE_REPO", Path(__file__).resolve().parents[3] / "feature_repo")
)
ITEM_CONTENT = ("item_year", "item_genres")
STATS_VIEWS = [f"{fs.entity}_stats" for fs in (USER_FEATURES, ITEM_FEATURES)]
STATIC_VIEWS = ["item_content"]  # stamped at the epoch, valid at any time
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def feature_refs(fs: FeatureSet) -> list[str]:
    """Feast references (`view:feature`) for every feature in a feature set."""
    return [f"{fs.entity}_stats:{name}" for name in fs.names]


USER_REFS = feature_refs(USER_FEATURES)
ITEM_REFS = feature_refs(ITEM_FEATURES) + [f"item_content:{n}" for n in ITEM_CONTENT]


def open_store(repo_path: Path = DEFAULT_REPO) -> FeatureStore:
    """Open the feature store.

    The Redis address comes from REDIS_CONNECTION_STRING, defaulting to the configured
    Redis host and port.
    """
    settings = get_settings()
    os.environ.setdefault("REDIS_CONNECTION_STRING", f"{settings.redis_host}:{settings.redis_port}")
    return FeatureStore(repo_path=str(repo_path))


def online_features(
    store: FeatureStore, refs: list[str], key: str, ids: list[int]
) -> dict[str, list[Any]]:
    """Latest materialized feature values for `ids`, as columns keyed by feature name."""
    response = store.get_online_features(features=refs, entity_rows=[{key: i} for i in ids])
    out: dict[str, list[Any]] = response.to_dict()
    return out


def historical_features(
    store: FeatureStore, entity_df: pd.DataFrame, refs: list[str]
) -> pd.DataFrame:
    """Point-in-time join: each row gets the latest snapshot at or before its event_timestamp."""
    if "event_timestamp" not in entity_df.columns:
        raise ValueError("entity_df needs an event_timestamp column")
    out: pd.DataFrame = store.get_historical_features(entity_df=entity_df, features=refs).to_df()
    return out


def materialize(store: FeatureStore, start: datetime, end: datetime) -> None:
    """Load the latest snapshot per entity into the online store.

    Snapshot views use rows in [start, end]; static views always start at the epoch.
    """
    store.materialize(start_date=start, end_date=end, feature_views=STATS_VIEWS)
    store.materialize(start_date=EPOCH, end_date=end, feature_views=STATIC_VIEWS)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> None:
    """Materialize feature snapshots into the online store."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--start", type=_utc, required=True, help="UTC date, e.g. 2019-11-01")
    parser.add_argument("--end", type=_utc, default=None, help="UTC date (default: now)")
    parser.add_argument(
        "--end-at-cutoff",
        type=Path,
        default=None,
        help="split directory: end at its training cutoff t1, the serving clock",
    )
    args = parser.parse_args(argv)
    end = args.end or datetime.now(UTC)
    if args.end_at_cutoff is not None:
        from streamrank.common.provenance import verify_outputs  # noqa: PLC0415

        end = datetime.fromtimestamp(int(verify_outputs(args.end_at_cutoff)["t1"]), tz=UTC)
    materialize(open_store(args.repo), args.start, end)


if __name__ == "__main__":
    main()
