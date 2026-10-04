import numpy as np
import polars as pl

from streamrank.simulation.interleave import credit, team_draft
from streamrank.simulation.simulate import summarize


def test_team_draft_alternates_and_never_repeats() -> None:
    rng = np.random.default_rng(0)
    a, b = [1, 2, 3, 4, 5, 6], [2, 7, 1, 8, 9, 10]
    for _ in range(50):
        merged = team_draft(a, b, 6, rng)
        assert len(merged.items) == 6 and len(set(merged.items)) == 6
        assert abs(merged.teams.count("A") - merged.teams.count("B")) <= 1
        for item, team in zip(merged.items, merged.teams, strict=True):
            assert item in (a if team == "A" else b)


def test_identical_lists_split_credit_evenly_over_many_draws() -> None:
    rng = np.random.default_rng(1)
    same = list(range(10))
    wins = {"A": 0, "B": 0}
    for _ in range(4000):
        merged = team_draft(same, same, 10, rng)
        for team, n in credit(merged, {0}).items():
            wins[team] += n
    share = wins["B"] / (wins["A"] + wins["B"])
    assert 0.46 < share < 0.54  # no bias toward either ranker


def test_better_ranker_wins() -> None:
    rng = np.random.default_rng(2)
    good, bad = [1, 2, 3], [9, 8, 7]
    merged = team_draft(bad, good, 3, rng)
    assert credit(merged, {1}) == {"A": 0, "B": 1}


def test_short_lists_and_summary() -> None:
    merged = team_draft([1], [2, 3, 4], 4, np.random.default_rng(3))
    assert sorted(merged.items) == [1, 2, 3, 4]
    imp = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "A": [0, 0, 1, 0],
            "B": [1, 1, 0, 0],
            "hit_A": [0, 1, 1, 0],
            "hit_B": [1, 1, 0, 0],
        }
    )
    out = summarize(imp, 1.0, seed=0)
    assert out["clicks"] == {"A": 1, "B": 2}
    assert out["users_with_clicks"] == 2
    assert out["delta"]["mean"] == 0.0  # user 1 prefers B, user 2 prefers A
    assert out["winner"] == "tie"
