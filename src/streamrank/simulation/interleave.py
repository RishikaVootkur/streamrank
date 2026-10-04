"""Team-draft interleaving (Radlinski, Kurup, Joachims 2008).

Two rankers take turns picking their best not-yet-shown item; a coin flip decides who
picks first in each round. A click credits the team that contributed the clicked item.
Over many impressions, the ranker with more credited clicks is preferred by users.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interleaved:
    """The merged list and which ranker ("A" or "B") contributed each position."""

    items: list[int]
    teams: list[str]


def team_draft(a: list[int], b: list[int], k: int, rng: np.random.Generator) -> Interleaved:
    """Interleave ranked lists `a` and `b` into k unique items."""
    items: list[int] = []
    teams: list[str] = []
    shown: set[int] = set()
    ia = ib = 0
    count = {"A": 0, "B": 0}
    while len(items) < k and (ia < len(a) or ib < len(b)):
        # The team with fewer picks goes next; ties are broken by a coin flip.
        if count["A"] < count["B"] or (count["A"] == count["B"] and rng.integers(2) == 0):
            order = ("A", "B")
        else:
            order = ("B", "A")
        for team in order:
            if len(items) >= k:
                break
            source, idx = (a, ia) if team == "A" else (b, ib)
            while idx < len(source) and source[idx] in shown:
                idx += 1
            if idx < len(source):
                items.append(source[idx])
                teams.append(team)
                shown.add(source[idx])
                count[team] += 1
                idx += 1
            if team == "A":
                ia = idx
            else:
                ib = idx
    return Interleaved(items, teams)


def credit(merged: Interleaved, clicked: set[int]) -> dict[str, int]:
    """Clicks credited to each team."""
    out = {"A": 0, "B": 0}
    for item, team in zip(merged.items, merged.teams, strict=True):
        if item in clicked:
            out[team] += 1
    return out
