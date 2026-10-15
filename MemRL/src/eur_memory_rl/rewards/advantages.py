from __future__ import annotations

import math
from dataclasses import dataclass

from .outcome import return_to_go


def mean(values):
    if not values:
        raise ValueError("Cannot average an empty planned group")
    return math.fsum(values) / len(values)


def loo(values, debug_zero_baseline=False):
    if len(values) < 2:
        if len(values) == 1 and debug_zero_baseline:
            return list(values)
        raise ValueError("LOO requires independent siblings >=2")
    total = math.fsum(values)
    return [v - (total - v) / (len(values) - 1) for v in values]


def assign_main_advantages(episodes, debug_zero_baseline=False):
    if not episodes:
        raise ValueError("No complete main group")
    first = episodes[0].plan
    if any(e.plan != first or e.positions != episodes[0].positions for e in episodes):
        raise ValueError("Replica plans/positions must align independently of missing policy rows")
    for p, episode in enumerate(episodes):
        for row in episode.rows:
            task_only = (
                row.metadata.get("task_id")
                if row.role == "R" and first.readonly_independent_queries
                else None
            )
            values = [
                return_to_go(other.rewards, other.positions[row.position], task_only) for other in episodes
            ]
            row.advantage = loo(values, debug_zero_baseline)[p]


@dataclass(frozen=True)
class NestedAdvantages:
    q: tuple[tuple[float, ...], ...]
    v: tuple[float, ...]
    e: tuple[float, ...]
    u: tuple[tuple[float, ...], ...]
    r: dict[tuple[int, int, int], tuple[float, ...]]


def nested_advantages(values, I, J, K, L, invalid_score=-1.0):
    """values[(i,j)] is K x L mean-over-B quality, or None for invalid parent/leaf."""
    if min(I, J, L) < 2 or K < 1:
        raise ValueError("I/J/L >=2, K>=1 required")
    if set(values) != {(i, j) for i in range(I) for j in range(J)}:
        raise ValueError("A tree must include all planned leaves, including invalid ones")
    q, ar = [], {}
    for i in range(I):
        qi = []
        for j in range(J):
            leaf = values[i, j]
            if leaf is None:
                qi.append(invalid_score)
                continue
            if len(leaf) != K or any(len(slates) != L for slates in leaf):
                raise ValueError("Incomplete local tree: exclude group, do not re-normalize survivors")
            qi.append(mean([score for slates in leaf for score in slates]))
            for k, slates in enumerate(leaf):
                ar[i, j, k] = tuple(loo(slates))
        q.append(tuple(qi))
    v = tuple(mean(qi) for qi in q)
    return NestedAdvantages(tuple(q), v, tuple(loo(v)), tuple(tuple(loo(qi)) for qi in q), ar)
