import pytest

from streamrank.features.definitions import (
    ITEM_FEATURES,
    SECONDS_PER_DAY,
    USER_FEATURES,
    Aggregate,
    Event,
    day_of,
    reference_snapshots,
    snapshot,
    snapshot_days,
)

D = SECONDS_PER_DAY


def _ev(user: int, item: int, rating: float, day: int, second: int = 0) -> Event:
    return Event(user, item, rating, day * D + second)


EVENTS = [
    _ev(1, 10, 5.0, 100, 10),
    _ev(1, 11, 2.0, 100, 50),
    _ev(1, 12, 4.0, 105),
    _ev(2, 10, 4.5, 105, 7),
    _ev(1, 13, 3.0, 140),
]


def test_day_of_uses_utc_days() -> None:
    assert day_of(0) == 0
    assert day_of(D - 1) == 0
    assert day_of(D) == 1


def test_snapshot_days_cover_activity_and_expiry() -> None:
    assert snapshot_days([100, 105], [7, 30]) == [101, 106, 108, 113, 131, 136]


def test_count_requires_no_column() -> None:
    with pytest.raises(ValueError, match="count"):
        Aggregate("bad", "count", "rating")
    with pytest.raises(ValueError, match="column"):
        Aggregate("bad", "sum")


def test_snapshot_excludes_the_query_day() -> None:
    user1 = [e for e in EVENTS if e.user_id == 1]
    # Day 100 itself: nothing before it.
    v = snapshot(USER_FEATURES, user1, 100)
    assert v["user_n_ratings"] == 0
    assert v["user_first_ts"] is None
    assert v["user_mean_rating"] == pytest.approx(3.5)
    # Day 101 sees both day-100 events.
    v = snapshot(USER_FEATURES, user1, 101)
    assert v["user_n_ratings"] == 2
    assert v["user_n_positive"] == 1
    assert v["user_rating_sum"] == 7.0
    assert v["user_first_ts"] == 100 * D + 10
    assert v["user_last_ts"] == 100 * D + 50
    assert v["user_mean_rating"] == pytest.approx((7.0 + 3.5 * 5) / 7)
    assert v["user_positive_rate"] == pytest.approx((1 + 0.5 * 5) / 7)


def test_windows_expire() -> None:
    user1 = [e for e in EVENTS if e.user_id == 1]
    assert snapshot(USER_FEATURES, user1, 106)["user_n_ratings_7d"] == 3
    assert snapshot(USER_FEATURES, user1, 108)["user_n_ratings_7d"] == 1  # day 100 left
    assert snapshot(USER_FEATURES, user1, 113)["user_n_ratings_7d"] == 0
    assert snapshot(USER_FEATURES, user1, 113)["user_n_ratings_30d"] == 3
    assert snapshot(USER_FEATURES, user1, 131)["user_n_ratings_30d"] == 1
    assert snapshot(USER_FEATURES, user1, 131)["user_n_ratings"] == 3


def test_reference_rows_per_entity() -> None:
    rows = reference_snapshots(ITEM_FEATURES, EVENTS)
    item10 = [r for r in rows if r["item_id"] == 10]
    assert [r["snapshot_day"] for r in item10] == [101, 106, 108, 113, 131, 136]
    assert item10[1]["item_n_ratings"] == 2
    assert item10[-1]["item_n_ratings_30d"] == 0


def test_reference_from_day_keeps_full_history() -> None:
    rows = reference_snapshots(USER_FEATURES, EVENTS, from_day=140)
    assert all(r["snapshot_day"] >= 140 for r in rows)
    last = next(r for r in rows if r["user_id"] == 1 and r["snapshot_day"] == 141)
    assert last["user_n_ratings"] == 4


@pytest.mark.parametrize("from_day", [101, 120, 140, 150, 200])
def test_from_day_boundary_rows_answer_lookups(from_day: int) -> None:
    rows = reference_snapshots(USER_FEATURES, EVENTS, from_day=from_day)
    for user in (1, 2):
        evs = [e for e in EVENTS if e.user_id == user]
        for day in range(from_day, from_day + 40):
            seen = [r for r in rows if r["user_id"] == user and r["snapshot_day"] <= day]
            if not any(day_of(e.ts) < day for e in evs):
                assert not seen
                continue
            latest = max(seen, key=lambda r: r["snapshot_day"])
            expected = snapshot(USER_FEATURES, evs, day)
            assert {k: latest[k] for k in expected} == expected, (user, day)


def test_snapshot_days_with_from_day() -> None:
    assert snapshot_days([100, 105], [7], from_day=107) == [107, 108, 113]
    assert snapshot_days([110], [7], from_day=107) == [111, 118]


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window_days"):
        Aggregate("bad", "count", window_days=0)


def test_names_are_unique_and_prefixed() -> None:
    for fs in (USER_FEATURES, ITEM_FEATURES):
        assert len(set(fs.names)) == len(fs.names)
        assert all(n.startswith(fs.entity + "_") for n in fs.names)
    assert USER_FEATURES.windows == (7, 30)
