"""Declarative feature definitions and a reference implementation.

Each feature is an aggregate over an entity's past rating events. The Spark job turns these
specs into window expressions and the streaming job updates them incrementally; both must
match `reference_snapshots`, which tests use as ground truth.

Point-in-time semantics: a snapshot with timestamp `s` (always midnight UTC) summarizes the
events from days strictly before `s`. A snapshot is emitted the day after each active day,
and again when each windowed count expires, so for any query time `t` the latest snapshot
at or before `t` holds the exact aggregates over all days before `t`'s day. Events on the
query day itself are never included, which keeps a rating out of its own features.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

SECONDS_PER_DAY: Final = 86_400
POSITIVE_RATING: Final = 4.0

Kind = Literal["count", "sum", "max", "min"]
Entity = Literal["user", "item"]


@dataclass(frozen=True)
class Aggregate:
    """An aggregate over past events, optionally limited to recent days or positives."""

    name: str
    kind: Kind
    column: Literal["rating", "ts"] | None = None  # value column; None for count
    window_days: int | None = None  # None means all history
    positives_only: bool = False

    def __post_init__(self) -> None:
        if (self.kind == "count") != (self.column is None):
            raise ValueError(f"{self.name}: count takes no column; other kinds need one")
        if self.window_days is not None and self.window_days < 1:
            raise ValueError(f"{self.name}: window_days must be at least 1")


def value_type(spec: Aggregate) -> Literal["int64", "float64"]:
    """Storage type of an aggregate: counts and timestamps are integers, the rest floats."""
    return "int64" if spec.kind == "count" or spec.column == "ts" else "float64"


@dataclass(frozen=True)
class SmoothedRatio:
    """(numerator + prior * weight) / (denominator + weight): a shrunk mean or rate."""

    name: str
    numerator: str
    denominator: str
    prior: float
    weight: float


@dataclass(frozen=True)
class FeatureSet:
    """All features for one entity type."""

    entity: Entity
    key: str
    aggregates: tuple[Aggregate, ...]
    ratios: tuple[SmoothedRatio, ...]

    @property
    def windows(self) -> tuple[int, ...]:
        return tuple(sorted({a.window_days for a in self.aggregates if a.window_days}))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.aggregates) + tuple(r.name for r in self.ratios)


def _common(prefix: str) -> tuple[tuple[Aggregate, ...], tuple[SmoothedRatio, ...]]:
    aggregates = (
        Aggregate(f"{prefix}_n_ratings", "count"),
        Aggregate(f"{prefix}_n_positive", "count", positives_only=True),
        Aggregate(f"{prefix}_rating_sum", "sum", "rating"),
        Aggregate(f"{prefix}_n_ratings_7d", "count", window_days=7),
        Aggregate(f"{prefix}_n_ratings_30d", "count", window_days=30),
        Aggregate(f"{prefix}_first_ts", "min", "ts"),
        Aggregate(f"{prefix}_last_ts", "max", "ts"),
    )
    ratios = (
        SmoothedRatio(
            f"{prefix}_mean_rating", f"{prefix}_rating_sum", f"{prefix}_n_ratings", 3.5, 5.0
        ),
        SmoothedRatio(
            f"{prefix}_positive_rate", f"{prefix}_n_positive", f"{prefix}_n_ratings", 0.5, 5.0
        ),
    )
    return aggregates, ratios


USER_FEATURES: Final = FeatureSet("user", "user_id", *_common("user"))
ITEM_FEATURES: Final = FeatureSet("item", "item_id", *_common("item"))
FEATURE_SETS: Final = (USER_FEATURES, ITEM_FEATURES)


@dataclass(frozen=True)
class Event:
    """One rating event."""

    user_id: int
    item_id: int
    rating: float
    ts: int


def day_of(ts: int) -> int:
    """Days since the Unix epoch (UTC) for a timestamp in seconds."""
    return ts // SECONDS_PER_DAY


def snapshot_days(
    active_days: Iterable[int], windows: Sequence[int], from_day: int | None = None
) -> list[int]:
    """Days on which an entity gets a snapshot: after each active day and at each expiry.

    With `from_day`, earlier snapshot days are dropped and an entity active before
    `from_day` gets a boundary snapshot on `from_day`, so lookups from that day on still
    find a row holding its full history.
    """
    days = list(active_days)
    out: set[int] = set()
    for d in days:
        out.add(d + 1)
        out.update(d + 1 + w for w in windows)
    if from_day is not None:
        out = {d for d in out if d >= from_day}
        if any(d < from_day for d in days):
            out.add(from_day)
    return sorted(out)


def _aggregate(spec: Aggregate, events: Sequence[Event], day: int) -> float | None:
    lo = day - spec.window_days if spec.window_days else None
    picked = [
        e
        for e in events
        if day_of(e.ts) < day
        and (lo is None or day_of(e.ts) >= lo)
        and (not spec.positives_only or e.rating >= POSITIVE_RATING)
    ]
    if spec.kind == "count":
        return float(len(picked))
    if not picked:
        return 0.0 if spec.kind == "sum" else None
    values = [float(getattr(e, str(spec.column))) for e in picked]
    if spec.kind == "sum":
        return sum(values)
    return max(values) if spec.kind == "max" else min(values)


def ratio_value(spec: SmoothedRatio, values: dict[str, float | None]) -> float:
    """Compute a smoothed ratio from already computed aggregates."""
    num = values[spec.numerator] or 0.0
    den = values[spec.denominator] or 0.0
    return (num + spec.prior * spec.weight) / (den + spec.weight)


def snapshot(fs: FeatureSet, events: Sequence[Event], day: int) -> dict[str, float | None]:
    """Feature values for one entity as of midnight starting `day` (events before it only)."""
    values = {a.name: _aggregate(a, events, day) for a in fs.aggregates}
    for r in fs.ratios:
        values[r.name] = ratio_value(r, values)
    return values


def reference_snapshots(
    fs: FeatureSet, events: Sequence[Event], from_day: int | None = None
) -> list[dict[str, float | int | None]]:
    """Every snapshot row for every entity, computed directly from the definitions.

    Quadratic in events per entity: meant for tests and small data, not production.
    With `from_day`, rows start there (see `snapshot_days`); aggregates use all history.
    """
    by_key: dict[int, list[Event]] = {}
    for e in events:
        by_key.setdefault(getattr(e, fs.key), []).append(e)
    rows: list[dict[str, float | int | None]] = []
    for key in sorted(by_key):
        evs = by_key[key]
        for day in snapshot_days({day_of(e.ts) for e in evs}, fs.windows, from_day):
            rows.append({fs.key: key, "snapshot_day": day, **snapshot(fs, evs, day)})
    return rows


# Session features: what the user did in the last few minutes, updated by the stream.

SESSION_WINDOW_SECONDS: Final = 30 * 60
SESSION_FEATURES: Final = (
    "session_n_events",
    "session_n_positive",
    "session_mean_rating",
    "session_last_item_id",
    "session_last_ts",
)


def session_snapshot(events: Sequence[Event], as_of: int) -> dict[str, float]:
    """Session features as of `as_of`: events with as_of - window < ts <= as_of.

    `events` are one user's events in arrival order; the last qualifying event defines
    `session_last_item_id` and `session_last_ts`.
    """
    window = [e for e in events if as_of - SESSION_WINDOW_SECONDS < e.ts <= as_of]
    if not window:
        return {
            "session_n_events": 0.0,
            "session_n_positive": 0.0,
            "session_mean_rating": float("nan"),
            "session_last_item_id": float("nan"),
            "session_last_ts": float("nan"),
        }
    return {
        "session_n_events": float(len(window)),
        "session_n_positive": float(sum(e.rating >= POSITIVE_RATING for e in window)),
        "session_mean_rating": sum(e.rating for e in window) / len(window),
        "session_last_item_id": float(window[-1].item_id),
        "session_last_ts": float(window[-1].ts),
    }


@dataclass
class SessionState:
    """Incremental session features for one user (the streaming implementation).

    Events must arrive in non-decreasing time order; `update` returns the features as of
    the new event, equal to `session_snapshot(all_events_so_far, event.ts)`.
    """

    events: list[tuple[int, int, float]]  # (ts, item_id, rating) inside the window

    def update(self, event: Event) -> dict[str, float]:
        self.events.append((event.ts, event.item_id, event.rating))
        cutoff = event.ts - SESSION_WINDOW_SECONDS
        self.events = [e for e in self.events if e[0] > cutoff]
        ratings = [r for _, _, r in self.events]
        return {
            "session_n_events": float(len(self.events)),
            "session_n_positive": float(sum(r >= POSITIVE_RATING for r in ratings)),
            "session_mean_rating": sum(ratings) / len(ratings),
            "session_last_item_id": float(event.item_id),
            "session_last_ts": float(event.ts),
        }
