from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..util import InfrastructureError, canonical, digest, seed_for

DIMENSIONS = ("verified_information", "supported_decision", "observable_progress")
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema", "json_schema": {
        "name": "paired_public_evidence", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {
                       **{k: {"type": ["integer", "null"], "enum": [-1, 0, 1, None]} for k in DIMENSIONS},
                       "evidence": {"type": "object", "additionalProperties": False,
                                    "properties": {k: {"type": "string"} for k in DIMENSIONS},
                                    "required": list(DIMENSIONS)}},
                   "required": [*DIMENSIONS, "evidence"]}}}


class JudgeFormatError(InfrastructureError):
    """Format repair already exhausted; repeating the same comparison cannot help."""
    retryable = False


def validate_judgement(text):
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {*DIMENSIONS, "evidence"}:
        raise ValueError("Expected exactly three score keys and evidence")
    evidence = result['evidence']
    if not isinstance(evidence, dict) or set(evidence) != set(DIMENSIONS):
        raise ValueError('evidence must contain exactly all three dimension keys')
    for key in DIMENSIONS:
        if not isinstance(evidence[key], str):
            raise ValueError('Evidence must be a string: ' + key)
        value = result[key]
        if value is not None and (type(value) is not int or value not in (-1, 0, 1)):
            raise ValueError('Invalid score: ' + key)
        if value not in (None, 0) and (not isinstance(evidence.get(key), str) or not evidence[key].strip()):
            raise ValueError('Nonzero score needs observed tool/action evidence: ' + key)
    return result
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
    def __init__(self, model, rubric_version, resources=None, evidence_profile="raw-v1",
                 repair_attempts=0, audit_path=None):
        self.model, self.version, self.resources = model, rubric_version, resources
        if evidence_profile not in {"raw-v1", "blank-line-runs-v1"}:
            raise ValueError("Unknown paired-judge evidence profile")
        self.evidence_profile = evidence_profile
        if not 0 <= repair_attempts <= 2:
            raise ValueError('Judge format repair attempts must be between 0 and 2')
        self.repair_attempts, self.audit_path = repair_attempts, audit_path

    def _evaluate(self, messages, seed, identity):
        history = list(messages)
        for attempt in range(self.repair_attempts + 1):
            completion = self.model.complete(history, 1024, seed_for(seed, attempt),
                                             digest([identity, 'format', attempt]))
            if self.resources:
                self.resources.add('judge_calls', 1)
                self.resources.add('judge_tokens', completion.input_tokens + completion.output_tokens)
                self.resources.add('monetary_cost', completion.monetary_cost)
            error, result = '', None
            try:
                result = validate_judgement(completion.text)
            except (TypeError, ValueError, KeyError) as exc:
                error = str(exc)
            if self.audit_path:
                path = Path(self.audit_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as handle:
                    handle.write(canonical({'comparison': identity, 'attempt': attempt,
                                            'response': completion.text, 'error': error,
                                            'valid': result is not None}) + '\n')
            if result is not None:
                return result
            if attempt == self.repair_attempts:
                raise JudgeFormatError('Judge schema failure after bounded repair: ' + error)
            history = [*history, {'role': 'assistant', 'content': completion.text},
                        {'role': 'user', 'content': 'Your previous response failed validation: ' + error
                         + '. Return the complete corrected JSON using only the original public A/B evidence. '
                         'Include all three score keys and all three evidence keys. '
                         'Do not invent evidence; use null for a score when evidence is insufficient.'}]

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
            result = self._evaluate(messages, seed_for(seed, order),
                                    [candidate.execution_id, reference.execution_id, order, self.version])
            for key in DIMENSIONS:
                value = result[key]
                uncertain += int(value is None)
                scores[key].append(sign * (value or 0))
            orders.append(result)
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
