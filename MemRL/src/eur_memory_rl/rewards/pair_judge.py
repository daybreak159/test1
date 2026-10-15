from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..util import InfrastructureError, canonical, digest, seed_for

DIMENSIONS = ("verified_information", "supported_decision", "observable_progress")
RUBRIC = """Compare two anonymous executions from the same initial state and task.
For each dimension return -1 if A is worse than B, 0 if tied, +1 if A is better,
or null when evidence is insufficient. Dimensions: verified_information (verifiable facts),
supported_decision (actions supported by available evidence), observable_progress (actual
state changes, executable validation). Do not reward length, tone, citing memory, or agreement.
A window cutoff is not task failure. Self-reported completion is not verified success.
Return JSON with exactly these three keys and an evidence object mapping each key to a short
reference to observed tool/action evidence. Null is uncertainty, not a loss."""


def judge_evidence(value):
    """Copy public evidence, eliding only runs of eight or more blank lines.

    Never edit the execution itself or parse/recover invalid model actions.
    The explicit marker retains the size and hash of each removed run.
    """
    if isinstance(value, str):
        def elide(match):
            text = match.group()
            return ("\n[redundant blank lines elided: chars=" + str(len(text))
                    + "; sha256=" + digest(text) + "]\n")
        return re.sub(r"(?:[ \t]*\r?\n){8,}", elide, value)
    if isinstance(value, dict):
        return {key: judge_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [judge_evidence(item) for item in value]
    return value


@dataclass(frozen=True)
class PairScore:
    value: float
    dimensions: dict[str, float]
    uncertain_dimensions: int
    orders: tuple[dict, ...]
    judge_version: str


class PairJudge:
    def __init__(self, model, rubric_version, resources=None, evidence_profile="raw-v1"):
        self.model, self.version, self.resources = model, rubric_version, resources
        if evidence_profile not in {"raw-v1", "blank-line-runs-v1"}:
            raise ValueError("Unknown paired-judge evidence profile")
        self.evidence_profile = evidence_profile

    def compare(self, task, candidate, reference, seed):
        if (
            candidate.initial_state_hash != reference.initial_state_hash
            or candidate.consumer_profile_hash != reference.consumer_profile_hash
        ):
            raise InfrastructureError("Pair environment/Consumer identity mismatch")
        if candidate.input_hash == reference.input_hash:
            return PairScore(0.0, {k: 0.0 for k in DIMENSIONS}, 0, (), self.version)
        scores, uncertain, orders = {k: [] for k in DIMENSIONS}, 0, []
        for order, (a, b, sign) in enumerate(((candidate, reference, 1), (reference, candidate, -1))):
            body = {
                "task": task.turns[0].text,
                "A": {"trace": a.public_trace, "answer": a.answer, "finish_reason": a.finish_reason},
                "B": {"trace": b.public_trace, "answer": b.answer, "finish_reason": b.finish_reason},
            }
            if self.evidence_profile == "blank-line-runs-v1":
                body = judge_evidence(body)
            messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": canonical(body)}]
            completion = self.model.complete(
                messages,
                1024,
                seed_for(seed, order),
                digest([candidate.execution_id, reference.execution_id, order, self.version]),
            )
            if self.resources:
                self.resources.add("judge_calls", 1)
                self.resources.add("judge_tokens", completion.input_tokens + completion.output_tokens)
                self.resources.add("monetary_cost", completion.monetary_cost)
            try:
                result = json.loads(completion.text)
                if set(result) != {*DIMENSIONS, "evidence"} or not isinstance(result["evidence"], dict):
                    raise ValueError("Wrong judge schema")
                for key in DIMENSIONS:
                    value = result[key]
                    if value is not None and (type(value) is not int or value not in (-1, 0, 1)):
                        raise ValueError("Wrong judge score")
                    if value is not None and value != 0 and not result["evidence"].get(key):
                        raise ValueError("Nonzero judgement lacks evidence")
                    uncertain += int(value is None)
                    scores[key].append(sign * (value or 0))
                orders.append(result)
            except (TypeError, ValueError, KeyError) as exc:
                raise InfrastructureError(
                    "Judge schema failure; exclude/retry group, never assign a loss"
                ) from exc
        dimensions = {k: sum(values) / 2 for k, values in scores.items()}
        return PairScore(
            sum(dimensions.values()) / len(DIMENSIONS), dimensions, uncertain, tuple(orders), self.version
        )


class PairedQuality:
    def __init__(self, adapter, judge=None):
        self.adapter, self.judge = adapter, judge

    def __call__(self, task, candidate, reference, seed):
        if (
            candidate.initial_state_hash != reference.initial_state_hash
            or candidate.consumer_profile_hash != reference.consumer_profile_hash
        ):
            raise InfrastructureError("Unmatched paired executions")
        if candidate.input_hash == reference.input_hash:
            return PairScore(0.0, {}, 0, (), "identical-exposure-v1")
        if self.adapter.domain in {"qa", "offline_web"}:
            c, r = self.adapter.grade(task, candidate), self.adapter.grade(task, reference)
            return PairScore(
                c.quality - r.quality,
                {"candidate_quality": c.quality, "reference_quality": r.quality},
                0,
                (),
                c.grader_version,
            )
        if self.judge is None:
            raise ValueError("Executable short probes require a fixed paired judge")
        return self.judge.compare(task, candidate, reference, seed)
