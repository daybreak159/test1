from __future__ import annotations

from ..contracts import Bank, DataIdentity, MemoryEntry, Source, SourceOutcome, Task, Turn


def identity(value):
    return DataIdentity(
        **{**value, "source_ids": tuple(value["source_ids"]), "evidence_ids": tuple(value["evidence_ids"])}
    )


def source(value):
    return Source(
        identity(value["identity"]),
        value["text"],
        value["observed_at"],
        value.get("agent_profile", "dataset"),
        tuple(value.get("public_feedback", ())),
        tuple(value.get("prefix_evidence_ids", ())),
        value.get("text_format", "default"),
        SourceOutcome(**value["official_outcome"]) if value.get("official_outcome") else None,
    )


def task(value):
    return Task(
        identity(value["identity"]),
        tuple(Turn(**turn) for turn in value["turns"]),
        value.get("public_payload", {}),
        str(value.get("category", "")),
        value.get("probe_certificate", {}),
    )


def bank(value):
    entries = tuple(
        MemoryEntry(
            **{**e, "source_ids": tuple(e["source_ids"]), "supersedes": tuple(e.get("supersedes", ())),
               "source_outcomes": tuple(SourceOutcome(**o) for o in e.get("source_outcomes", ()))}
        )
        for e in value["entries"]
    )
    return Bank(value["scope"], value["revision"], value["next_id"], entries)
