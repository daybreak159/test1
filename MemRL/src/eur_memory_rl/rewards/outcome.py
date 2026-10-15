from __future__ import annotations

import math

from ..contracts import RewardEvent


class RewardLedger:
    def __init__(self, scored_count, write_count, cost_lambda=0.01, bad_write_lambda=1.0, cost_budget=1.0):
        if scored_count < 1 or write_count < 0 or cost_budget <= 0:
            raise ValueError("Invalid fixed reward normalization")
        self.n, self.t = scored_count, write_count
        self.cost_lambda, self.bad_lambda, self.cost_budget = cost_lambda, bad_write_lambda, cost_budget
        self.events = []
        self._identities = set()

    def _append(self, event):
        key = (event.kind, event.identity)
        if key in self._identities:
            raise ValueError("Duplicate reward: multiple turns cannot duplicate task success")
        self._identities.add(key)
        self.events.append(event)

    def outcome(self, task_id, time, quality, raw_cost):
        if not math.isfinite(quality) or not 0 <= quality <= 1 or raw_cost < 0 or not math.isfinite(raw_cost):
            raise ValueError("Invalid task grade/cost")
        cost = min(raw_cost / self.cost_budget, 1.0)
        self._append(
            RewardEvent(
                time, (quality - self.cost_lambda * cost) / self.n, "outcome", task_id, quality, raw_cost
            )
        )

    def bad_write(self, position, time):
        self._append(RewardEvent(time, -self.bad_lambda / max(self.t, 1), "bad_write", position))


def return_to_go(events, time, task_only=None):
    return sum(
        e.value
        for e in events
        if e.time > time and (task_only is None or (e.kind == "outcome" and e.identity == task_only))
    )
