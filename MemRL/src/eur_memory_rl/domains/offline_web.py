from __future__ import annotations

import json

from ..contracts import Grade
from .qa import QAAdapter, token_f1


class OfflineWebGrader:
    def __init__(self, labels, version):
        self.labels, self.version = labels, version

    def __call__(self, task, execution):
        gold = self.labels[task.id]
        try:
            pred = json.loads(execution.answer)
            element = float(str(pred["element_id"]) in {str(e) for e in gold["element_ids"]})
            operation = float(pred["operation"].lower() == gold["operation"].lower())
            value = token_f1(pred.get("value", ""), gold.get("value", ""))
            exact_value = float(pred.get("value", "").strip() == gold.get("value", "").strip())
        except (ValueError, KeyError, TypeError):
            element = operation = value = exact_value = 0.0
        metrics = {
            "element_accuracy": element,
            "action_f1": operation * value,
            "step_success": element * operation * exact_value,
        }
        return Grade(metrics["step_success"], metrics, self.version)


class OfflineWebAdapter(QAAdapter):
    domain = "offline_web"
    training_enabled = False
    # Task.turn.public_history holds ONLY the recorded current page. No gold suffix rollout.
