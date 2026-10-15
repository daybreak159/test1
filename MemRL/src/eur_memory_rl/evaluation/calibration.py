from __future__ import annotations

import math
from dataclasses import dataclass

from .metrics import calibration_metrics


@dataclass
class BinaryCalibrator:
    """Optional separate binary event model. Never rewrites selector behavior probabilities."""

    event_definition: str
    slope: float = 1.0
    intercept: float = 0.0
    fitting_groups: tuple[str, ...] = ()

    def predict(self, score):
        logit = max(-40.0, min(40.0, self.slope * score + self.intercept))
        return 1 / (1 + math.exp(-logit))

    def fit(self, records, split, steps=1000, learning_rate=0.01):
        if split not in {"train", "dev"} or any(r["split"] != split for r in records):
            raise ValueError("Calibration cannot fit test labels or silently mix partitions")
        if not self.event_definition or not records or any(r["label"] not in (0, 1) for r in records):
            raise ValueError("Provide independently annotated binary event labels")
        for _ in range(steps):
            errors = [self.predict(r["score"]) - r["label"] for r in records]
            self.slope -= learning_rate * sum(e * r["score"] for e, r in zip(errors, records)) / len(records)
            self.intercept -= learning_rate * sum(errors) / len(records)
        self.fitting_groups = tuple(sorted({r["group"] for r in records}))
        return self

    def evaluate(self, records):
        if {r["group"] for r in records} & set(self.fitting_groups):
            raise ValueError("Report calibration on independent source groups")
        return {
            **calibration_metrics([self.predict(r["score"]) for r in records], [r["label"] for r in records]),
            "event_definition": self.event_definition,
            "policy_temperature_changed": False,
        }
