from __future__ import annotations

from dataclasses import replace

from ..policies.prompts import encode_messages, extraction_messages
from ..util import ConfigurationError, digest


def pack_source(source, tokenizer, max_input_tokens, profile="reject", *, prompt_profile="e-text-v1"):
    """Explicit deterministic packing before sampling. Complete original text remains in rollout anchors."""
    original = source
    original_hash = digest(source.text)

    def messages(value):
        from ..policies.generator import GeneratorPolicy

        return GeneratorPolicy(None, prompt_profile=prompt_profile).prepare_messages(
            "E", extraction_messages(value))
    if profile == "compact_exact_v1":
        from .compact_trace import compact_public_text

        text, document, excluded = compact_public_text(source.text, source.identity.evidence_ids)
        packed = replace(source, text=text, text_format="compact_exact_v1")
        # Full final input, including metadata and the complete body table. Never truncate.
        ids, _, _ = encode_messages(tokenizer, messages(packed), max_input_tokens)
        return packed, {
            "profile": profile,
            "original_hash": original_hash,
            "visible_hash": digest(text),
            "retained_characters": [0, len(source.text)],
            "visible_evidence_ids": source.identity.evidence_ids,
            "input_tokens": len(ids),
            "shared_bodies": len(document["blobs"]),
            "excluded_fields": excluded,
            "roundtrip_equal": True,
        }
    try:
        encode_messages(tokenizer, messages(source), max_input_tokens)
        return source, {
            "profile": profile,
            "original_hash": original_hash,
            "visible_hash": original_hash,
            "retained_characters": [0, len(source.text)],
        }
    except ConfigurationError:
        if profile == "reject":
            raise
    if profile != "prefix_public_text_v1":
        raise ConfigurationError("Unknown source packing profile")

    def candidate(chars):
        text = original.text[:chars]
        visible_ids = tuple(e for e in original.identity.evidence_ids if f"[{e}]" in text)
        if not visible_ids:
            raise ConfigurationError("Prefix packing requires bracketed public evidence IDs")
        identity = replace(original.identity, evidence_ids=visible_ids)
        return replace(
            original,
            text=text + "\n[Public source truncated by fixed prefix packing profile.]",
            identity=identity,
        )

    low, high, best = 0, len(source.text), None
    while low <= high:
        middle = (low + high) // 2
        try:
            trial = candidate(middle)
            encode_messages(tokenizer, messages(trial), max_input_tokens)
        except ConfigurationError:
            high = middle - 1
        else:
            best = (middle, trial)
            low = middle + 1
    if best is None:
        raise ConfigurationError("No public evidence fits the E prompt budget")
    chars, packed = best
    return packed, {
        "profile": profile,
        "original_hash": original_hash,
        "visible_hash": digest(packed.text),
        "retained_characters": [0, chars],
        "visible_evidence_ids": packed.identity.evidence_ids,
    }
