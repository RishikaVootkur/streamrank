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


def snapshot_days(active_days: Iterable[int], windows: Sequence[int]) -> list[int]:
    """Days on which an entity gets a snapshot: after each active day and at each expiry."""
    out: set[int] = set()
    for d in active_days:
        out.add(d + 1)
        out.update(d + 1 + w for w in windows)
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
    Rows before `from_day` are dropped (their aggregates still use all history).
    """
    by_key: dict[int, list[Event]] = {}
    for e in events:
        by_key.setdefault(getattr(e, fs.key), []).append(e)
    rows: list[dict[str, float | int | None]] = []
    for key in sorted(by_key):
        evs = by_key[key]
        for day in snapshot_days({day_of(e.ts) for e in evs}, fs.windows):
            if from_day is not None and day < from_day:
                continue
            rows.append({fs.key: key, "snapshot_day": day, **snapshot(fs, evs, day)})
    return rows
