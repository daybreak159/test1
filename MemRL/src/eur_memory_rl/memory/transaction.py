from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from ..contracts import Bank, MemoryCandidate, MemoryEntry, Source
from .render import MemoryRenderer, count_tokens

KINDS = {"fact", "event", "preference", "procedure", "failure_lesson"}
UNCLASSIFIED_KIND = "unclassified"  # Internal placeholder, not a model classification.


class InvalidWrite(ValueError):
    pass


def strict_json(text: str):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise InvalidWrite(f"Duplicate JSON key: {k}")
            out[k] = v
        return out

    try:
        return json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda x: (_ for _ in ()).throw(InvalidWrite(f"Invalid number {x}")),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidWrite(f"Invalid JSON: {exc}") from exc


def fields(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise InvalidWrite(f"Expected fields {sorted(required)} with optional {sorted(optional)}")


def text_field(value, name, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise InvalidWrite(f"{name} must be a {'nonempty ' if nonempty else ''}string")
    return value


def evidence(value, allowed):
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v in allowed for v in value)
    ):
        raise InvalidWrite("Evidence IDs must reference visible public evidence")
    if len(value) != len(set(value)):
        raise InvalidWrite("Duplicate evidence IDs")
    return tuple(value)


def parse_text_extraction(text: str, source: Source, max_candidates=3) -> tuple[MemoryCandidate, ...]:
    """Wrap text memories with trusted task-level provenance, not invented evidence links.

    Missing conditions remain empty. Without a marker the whole response is one
    memory; only standalone markers outside code fences split multiple memories.
    The original GenerationRecord is never rewritten or used as generated JSON.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidWrite("Empty E text response; use NO_MEMORY explicitly")
    if text.strip() == "NO_MEMORY":
        return ()
    ids = tuple(dict.fromkeys(source.identity.source_ids))
    if not ids:
        raise InvalidWrite("Text E requires trusted task-level source IDs")
    blocks, current, fence = [], [], None
    for line in text.strip().splitlines():
        stripped = line.strip()
        match = re.match(r"^(`{3,}|~{3,})", stripped)
        if match:
            token = match[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        if fence is None and stripped == "NO_MEMORY":
            raise InvalidWrite("NO_MEMORY must stand alone, not follow memory content")
        if fence is None and stripped == "[MEMORY]":
            if current and any(part.strip() for part in current):
                blocks.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if current and any(part.strip() for part in current):
        blocks.append("\n".join(current).strip())
    if not blocks or len(blocks) > max_candidates:
        raise InvalidWrite("Text E requires 1..3 memories or NO_MEMORY")
    candidates = []
    label = re.compile(r"^\s*(?:正文|Content)\s*[:：]\s*", re.I)
    conditions_label = re.compile(r"^\s*(?:使用条件|适用条件|Conditions|Applicability)\s*[:：]\s*", re.I)
    for block in blocks:
        content, conditions, in_conditions = [], [], False
        fence = None
        for line in block.splitlines():
            match = re.match(r"^\s*(`{3,}|~{3,})", line)
            if match:
                token = match[1]
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
            if fence is None and not in_conditions and conditions_label.match(line):
                in_conditions = True
                conditions.append(conditions_label.sub("", line, count=1))
            elif in_conditions:
                conditions.append(line)
            else:
                content.append(label.sub("", line, count=1) if not content else line)
        body = "\n".join(content).strip()
        candidates.append(MemoryCandidate(
            UNCLASSIFIED_KIND, "", text_field(body, "content"),
            "\n".join(conditions).strip(), ids,
        ))
    return tuple(candidates)


def parse_extraction(text: str, source: Source, max_candidates=3,
                     profile="e-text-v1") -> tuple[MemoryCandidate, ...]:
    if profile == "e-text-v1":
        return parse_text_extraction(text, source, max_candidates)
    obj = strict_json(text)
    fields(obj, {"candidates"})
    if not isinstance(obj["candidates"], list) or len(obj["candidates"]) > max_candidates:
        raise InvalidWrite("Too many E candidates")
    candidates = []
    allowed = set(source.identity.evidence_ids) | set(source.identity.source_ids)
    for value in obj["candidates"]:
        fields(value, {"kind", "content", "source_ids", "conditions"}, {"title", "subject", "event_time"})
        if not isinstance(value["kind"], str) or value["kind"] not in KINDS:
            raise InvalidWrite("Unknown memory kind")
        content = text_field(value["content"], "content")
        event_time = value.get("event_time")
        if event_time is not None:
            text_field(event_time, "event_time")
        candidates.append(
            MemoryCandidate(
                value["kind"],
                text_field(value.get("title", ""), "title", False),
                content,
                text_field(value["conditions"], "conditions", False),
                evidence(value["source_ids"], allowed),
                text_field(value.get("subject", ""), "subject", False),
                event_time,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True)
class ApplyResult:
    bank: Bank
    valid: bool
    error: str = ""
    operations: tuple[dict, ...] = ()


class TransactionExecutor:
    def __init__(
        self,
        counter_tokenizer,
        consumer_tokenizer,
        selector_tokenizer,
        bank_tokens=8192,
        entry_consumer_tokens=1024,
        entry_selector_tokens=1536,
        max_operations=4,
    ):
        self.counter = counter_tokenizer
        self.consumer = consumer_tokenizer
        self.selector = selector_tokenizer
        self.bank_tokens = bank_tokens
        self.entry_consumer_tokens = entry_consumer_tokens
        self.entry_selector_tokens = entry_selector_tokens
        self.max_operations = max_operations
        self.renderer = MemoryRenderer()

    def apply(
        self,
        bank: Bank,
        candidates: tuple[MemoryCandidate, ...],
        text: str,
        visible: tuple[MemoryEntry, ...],
        source: Source,
    ) -> ApplyResult:
        try:
            return self._apply(bank, candidates, text, visible, source)
        except InvalidWrite as exc:
            return ApplyResult(bank, False, str(exc))

    def _apply(self, bank, candidates, text, visible, source):
        if source.identity.scope != bank.scope:
            raise InvalidWrite("Source scope mismatch")
        obj = strict_json(text)
        fields(obj, {"base_revision", "operations"})
        if obj["base_revision"] != bank.version:
            raise InvalidWrite("Stale bank revision")
        ops = obj["operations"]
        if not isinstance(ops, list) or not 1 <= len(ops) <= self.max_operations:
            raise InvalidWrite("U must return 1..max_operations operations")
        entries = {e.id: e for e in bank.entries}
        available = {e.id: e for e in visible if e.scope == bank.scope and e.status == "active"}
        allowed = set(source.identity.evidence_ids) | set(source.identity.source_ids)
        for e in visible:
            if e.scope == bank.scope:
                allowed.update(e.source_ids)
        touched, next_id, changed = set(), bank.next_id, False
        for op in ops:
            if not isinstance(op, dict):
                raise InvalidWrite("Operation must be an object")
            kind = op.get("op")
            if not isinstance(kind, str):
                raise InvalidWrite("Operation name must be a string")
            if kind == "NOOP":
                fields(op, {"op"})
                if len(ops) != 1:
                    raise InvalidWrite("NOOP must be the only operation")
                continue
            if kind == "ADD":
                fields(op, {"op", "candidate_index"})
                i = op["candidate_index"]
                if type(i) is not int or not 0 <= i < len(candidates):
                    raise InvalidWrite("Invalid candidate_index")
                c = candidates[i]
                mid = f"m{next_id:08d}"
                if mid in entries:
                    raise InvalidWrite("ID allocator collision")
                entries[mid] = MemoryEntry(
                    mid,
                    1,
                    bank.scope,
                    c.kind,
                    c.title,
                    c.content,
                    c.conditions,
                    c.source_ids,
                    source.observed_at,
                    c.event_time,
                    c.subject,
                    source_outcomes=(source.official_outcome,) if source.official_outcome else (),
                )
                next_id += 1
            elif kind in {"UPDATE", "DELETE"}:
                required = {"op", "target_id", "expected_revision"}
                optional = {"title", "kind", "conditions", "subject", "event_time", "supersedes"}
                if kind == "UPDATE":
                    required |= {"content", "source_ids"}
                fields(op, required, optional if kind == "UPDATE" else ())
                target = op["target_id"]
                if not isinstance(target, str) or target not in available or target in touched:
                    raise InvalidWrite("Target is invisible, cross-scope, or repeated")
                old = entries.get(target)
                if (
                    old != available[target]
                    or type(op["expected_revision"]) is not int
                    or old.revision != op["expected_revision"]
                ):
                    raise InvalidWrite("Stale entry revision")
                touched.add(target)
                if kind == "DELETE":
                    entries[target] = replace(old, revision=old.revision + 1, status="deleted")
                else:
                    changes = {k: op[k] for k in optional if k in op}
                    for k in ("title", "conditions", "subject"):
                        if k in changes:
                            text_field(changes[k], k, False)
                    if (
                        not isinstance(changes.get("kind", old.kind), str)
                        or changes.get("kind", old.kind) not in KINDS | {UNCLASSIFIED_KIND}
                    ):
                        raise InvalidWrite("Invalid updated kind")
                    if changes.get("event_time") is not None:
                        text_field(changes["event_time"], "event_time")
                    supersedes = changes.get("supersedes", old.supersedes)
                    if not isinstance(supersedes, (list, tuple)) or any(
                        not isinstance(s, str) or s not in available for s in supersedes
                    ):
                        raise InvalidWrite("supersedes must reference visible entries")
                    changes["supersedes"] = tuple(supersedes)
                    entries[target] = replace(
                        old,
                        revision=old.revision + 1,
                        content=text_field(op["content"], "content"),
                        source_ids=evidence(op["source_ids"], allowed),
                        observed_at=source.observed_at,
                        source_outcomes=tuple(dict.fromkeys((*old.source_outcomes,
                            *((source.official_outcome,) if source.official_outcome else ())))),
                        **changes,
                    )
            else:
                raise InvalidWrite("Unknown operation")
            changed = True
        result = Bank(bank.scope, bank.revision + int(changed), next_id, tuple(entries.values()))
        for entry in result.active:
            rendered = self.renderer.render([entry])
            if count_tokens(self.consumer, rendered) > self.entry_consumer_tokens:
                raise InvalidWrite("Entry Consumer token limit exceeded")
            if count_tokens(self.selector, self.renderer.entry(entry)) > self.entry_selector_tokens:
                raise InvalidWrite("Entry selector token limit exceeded")
        if count_tokens(self.counter, self.renderer.render(result.active)) > self.bank_tokens:
            raise InvalidWrite("Bank capacity exceeded; explicit compression/deletion required")
        return ApplyResult(result, True, operations=tuple(ops))
