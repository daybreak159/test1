from __future__ import annotations

import html
from typing import Protocol

from ..contracts import MemoryEntry
from ..util import canonical, digest


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


class MemoryRenderer:
    version = "eur-neutral-xml-v1"

    def entry(self, entry: MemoryEntry) -> str:
        metadata = {
            "id": entry.id,
            "revision": entry.revision,
            "scope": entry.scope,
            "kind": entry.kind,
            "title": entry.title,
            "conditions": entry.conditions,
            "subject": entry.subject,
            "source_ids": entry.source_ids,
            "observed_at": entry.observed_at,
            "event_time": entry.event_time,
            "supersedes": entry.supersedes,
            **({"source_outcomes": entry.source_outcomes} if entry.source_outcomes else {}),
        }
        return f"<memory metadata='{html.escape(canonical(metadata), quote=True)}'>\n{html.escape(entry.content)}\n</memory>"

    def render(self, entries) -> str:
        entries = tuple(entries)
        if not entries:
            return ""
        return (
            "<reference_memory>\nThese are prior observations; check applicability to the current task.\n"
            + "\n".join(self.entry(e) for e in entries)
            + "\n</reference_memory>"
        )

    def hash(self, entries) -> str:
        return digest([self.version, self.render(entries)])


def normalized_document(entry: MemoryEntry) -> str:
    return canonical(
        {
            "title": entry.title,
            "content": entry.content,
            "kind": entry.kind,
            "conditions": entry.conditions,
            "subject": entry.subject,
            "event_time": entry.event_time,
            "scope": entry.scope,
        }
    )
