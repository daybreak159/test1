from __future__ import annotations

import time
from collections import Counter
import re
import string

from ..contracts import Execution, Grade
from ..util import ConfigurationError, digest
from .base import SourceThenQueryAdapter


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    p, r = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    if (
        normalize_answer(prediction) in {"yes", "no", "noanswer"}
        or normalize_answer(reference) in {"yes", "no", "noanswer"}
    ) and p != r:
        return 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    return 2 * overlap / (len(p) + len(r)) if overlap else 0.0


class AnswerGrader:
    """Explicit custom EM/F1 profile; official evaluators use a pinned callable/service."""

    def __init__(self, private_labels, version, quality_metric="f1"):
        if quality_metric not in {"f1", "em"}:
            raise ValueError("Use a separate official evaluator for non-EM/F1 profiles")
        self.labels, self.version, self.quality_metric = private_labels, version, quality_metric

    def __call__(self, task, execution):
        record = self.labels[task.id]
        aliases = record.get("answers", [])
        if not aliases:
            raise ConfigurationError(f"Missing private answer aliases for {task.id}")
        metrics = {
            "em": max(exact_match(execution.answer, a) for a in aliases),
            "f1": max(token_f1(execution.answer, a) for a in aliases),
        }
        if task.category == "5":
            acceptable = record.get("abstention_aliases")
            if not acceptable:
                raise ConfigurationError("Category 5 requires a separate declared abstention profile")
            metrics["abstention_accuracy"] = max(exact_match(execution.answer, a) for a in acceptable)
            return Grade(metrics["abstention_accuracy"], metrics, self.version)
        return Grade(metrics[self.quality_metric], metrics, self.version)


class QAAdapter(SourceThenQueryAdapter):
    domain = "qa"

    def clone_or_reset(self, task, execution_id, seed):
        return task.identity.history_or_environment_hash

    def run_consumer(self, task, execution_id, seed, read, budget=None):
        if len(task.turns) != 1:
            raise ConfigurationError("Independent QA profile has one external query per scored unit")
        start = time.perf_counter()
        turn = task.turns[0]
        slate = read(turn)
        budget = budget or self.profile.budget
        messages = [
            {
                "role": "system",
                "content": "Answer the question from applicable evidence. If insufficient, "
                "state that you do not know.\n" + slate.memory_block,
            },
            {"role": "user", "content": turn.public_history + "\n" + turn.public_time + "\n" + turn.text},
        ]
        completion = self.consumer.complete(messages, budget.output_tokens, seed, execution_id + ":qa")
        state = self.clone_or_reset(task, execution_id, seed)
        trace = (
            {"event_origin": "external_request", "text": turn.text},
            {"event_origin": "assistant_output", "text": completion.text},
        )
        return Execution(
            execution_id,
            task.id,
            completion.text,
            trace,
            state,
            state,
            self.profile.hash,
            digest([messages, budget, self.profile.hash]),
            seed,
            "answer",
            1,
            completion.input_tokens,
            completion.output_tokens,
            float(completion.input_tokens + completion.output_tokens + 1),
            time.perf_counter() - start,
            completion.monetary_cost,
        )

    def validate_probe_visibility(self, task, source):
        if not super().validate_probe_visibility(task, source):
            return False
        certificate = task.probe_certificate
        allowed = set(source.prefix_evidence_ids or source.identity.evidence_ids)
        if not set(certificate.get("evidence_ids", ())) <= allowed:
            return False
        # Time-sensitive questions need explicit full-prefix validation, not just max evidence session.
        if certificate.get("time_sensitive") and not certificate.get("complete_prefix_checked"):
            return False
        return bool(certificate.get("evidence_ids")) or certificate.get("abstention_sample", False)
