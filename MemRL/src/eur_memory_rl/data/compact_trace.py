"""Deterministic, reversible formatting of public event text; no semantic filtering."""
from __future__ import annotations

from collections import Counter
import json

from ..util import ConfigurationError


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def compact_document(value):
    counts = Counter(_strings(value))
    bodies, lookup = {}, {}

    def encode(node):
        if isinstance(node, str):
            if len(node) >= 128 and counts[node] > 1:
                if node not in lookup:
                    key = f"b{len(bodies)}"
                    lookup[node] = key
                    bodies[key] = node
                return {"kind": "ref", "id": lookup[node]}
            return {"kind": "text", "value": node}
        if isinstance(node, dict):
            return {"kind": "object", "items": [[k, encode(v)] for k, v in node.items()]}
        if isinstance(node, list):
            return {"kind": "list", "items": [encode(v) for v in node]}
        return {"kind": "scalar", "value": node}

    tree = encode(value)
    return {"version": "compact_exact_v1", "tree": tree, "blobs": bodies}


def expand(document):
    def decode(node):
        kind = node["kind"]
        if kind == "ref":
            return document["blobs"][node["id"]]
        if kind in {"text", "scalar"}:
            return node["value"]
        if kind == "object":
            return {k: decode(v) for k, v in node["items"]}
        return [decode(v) for v in node["items"]]
    return decode(document["tree"])


def render(document):
    lines = ["COMPACT_TRACE_V1", "Events stay in original order. Shared text is in BODY_TABLE."]

    def visit(node, label):
        kind = node["kind"]
        if kind == "object":
            lines.append(f"[{label}]")
            for key, value in node["items"]:
                visit(value, key)
        elif kind == "list":
            lines.append(f'[{label}: {len(node["items"])} entries]')
            for i, value in enumerate(node["items"]):
                visit(value, f"{label}:{i}")
        elif kind == "text":
            lines.extend([f"{label}:", node["value"]])
        elif kind == "ref":
            lines.append(f'{label}: REF {node["id"]}')
        else:
            lines.append(f'{label}: {json.dumps(node["value"], ensure_ascii=False)}')

    visit(document["tree"], "trace")
    if document["blobs"]:
        lines.append("BODY_TABLE")
        for key, value in document["blobs"].items():
            lines.extend([f"{key} TEXT chars={len(value)}", value])
    return "\n".join(lines)


def compact_public_text(text, evidence_ids):
    """Parse native [evidence-id] JSON events, or retain non-native text verbatim."""
    events, excluded = [], {}
    lines = text.splitlines()
    native = bool(lines) and all(line.startswith("[") and "] " in line for line in lines)
    if native:
        for i, line in enumerate(lines):
            eid, raw = line[1:].split("] ", 1)
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                native = False
                break
            if not isinstance(event, dict):
                native = False
                break
            if eid not in evidence_ids or "evidence_id" in event:
                raise ConfigurationError("Invalid native trace evidence identity")
            if "state_hash" in event:
                excluded[f"events[{i}].state_hash"] = event.pop("state_hash")
            events.append({"evidence_id": eid, **event})
    payload = {"events": events} if native else {"public_text": text}
    if not native:
        excluded = {}
    elif tuple(event["evidence_id"] for event in events) != tuple(evidence_ids):
        raise ConfigurationError("Native trace evidence IDs must match the ordered source identity")
    document = compact_document(payload)
    if expand(document) != payload:
        raise ConfigurationError("Compact trace round-trip failed")
    return render(document), document, excluded
