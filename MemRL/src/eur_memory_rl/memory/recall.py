from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from ..contracts import Bank, MemoryEntry
from ..util import ConfigurationError, digest
from .render import normalized_document


class Embedder(Protocol):
    revision: str
    dimension: int

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class RecallHit:
    entry: MemoryEntry
    score: float
    document_hash: str


class FrozenIndex:
    """No query cache and no writes on search. New banks create new index snapshots."""

    normalization_version = "canonical-memory-v1"

    def __init__(self, bank: Bank, embedder: Embedder, vectors=None, on_embed=None):
        self.bank_hash = bank.hash
        self.scope = bank.scope
        self.embedder = embedder
        self.entries = bank.active
        self.on_embed = on_embed or (lambda count: None)
        cache = dict(vectors or {})
        self.keys = tuple(self.document_key(e) for e in self.entries)
        missing = [
            (key, normalized_document(e)) for key, e in zip(self.keys, self.entries) if key not in cache
        ]
        if missing:
            values = embedder.encode([text for _, text in missing])
            if len(values) != len(missing):
                raise ConfigurationError("Embedding response cardinality mismatch")
            for (key, _), vector in zip(missing, values):
                self._validate_vector(vector)
                cache[key] = tuple(vector)
            self.on_embed(len(missing))
        # Retain only the current snapshot; a caller can reuse this mapping on commit.
        self.vectors = MappingProxyType({key: cache[key] for key in self.keys})

    def document_key(self, entry):
        return digest(
            [
                normalized_document(entry),
                entry.id,
                entry.revision,
                self.embedder.revision,
                self.embedder.dimension,
                "last_valid_token",
                "l2",
                self.normalization_version,
            ]
        )

    def _validate_vector(self, vector):
        if len(vector) != self.embedder.dimension or not all(math.isfinite(x) for x in vector):
            raise ConfigurationError("Wrong embedding dimension or nonfinite embedding")
        if abs(sum(x * x for x in vector) - 1.0) > 0.01:
            raise ConfigurationError("Embedding must be L2 normalized")

    def search(self, query: str, instruction: str, k: int, scope: str) -> tuple[RecallHit, ...]:
        if scope != self.scope:
            raise ConfigurationError("Cross-scope recall")
        if not self.entries:
            return ()
        if not instruction or k < 1:
            raise ConfigurationError("Fixed retrieval instruction and positive k required")
        vector = self.embedder.encode([f"Instruct: {instruction}\nQuery:{query}"])[0]
        self._validate_vector(vector)
        self.on_embed(1)
        hits = [
            RecallHit(e, sum(a * b for a, b in zip(vector, self.vectors[key])), key)
            for e, key in zip(self.entries, self.keys)
        ]
        return tuple(sorted(hits, key=lambda hit: (-hit.score, hit.entry.id))[:k])


class QwenEmbedder:
    dimension = 4096

    def __init__(self, source, revision, device="cuda:1", max_tokens=8192, batch_size=1):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not revision:
            raise ConfigurationError("Pin the embedding revision; 0.6B indexes cannot be reused")
        self.revision = revision
        self.device, self.max_tokens, self.batch_size = device, max_tokens, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
        self.model = (
            AutoModel.from_pretrained(
                source, revision=revision, trust_remote_code=False, dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != self.dimension:
            raise ConfigurationError("EUR v3 requires the full 4096-dimensional 8B embeddings")

    def encode(self, texts):
        import torch
        import torch.nn.functional as F

        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            chunk = texts[offset : offset + self.batch_size]
            if any(len(self.tokenizer.encode(t)) > self.max_tokens for t in chunk):
                raise ConfigurationError("Embedding input exceeds profile; no silent truncation")
            batch = self.tokenizer(chunk, padding=True, truncation=False, return_tensors="pt").to(self.device)
            resources = getattr(self, "resources", None)
            timer = resources.gpu_timer(self.device, "embedding") if resources else nullcontext()
            with timer, torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
                positions = torch.arange(batch.attention_mask.shape[1], device=self.device).expand_as(
                    batch.attention_mask
                )
                last = positions.masked_fill(~batch.attention_mask.bool(), -1).max(dim=1).values
                pooled = hidden[torch.arange(hidden.shape[0], device=self.device), last].float()
                vectors.extend(tuple(v) for v in F.normalize(pooled, p=2, dim=-1).cpu().tolist())
        return vectors


class RemoteQwenEmbedder:
    """Transport adapter for the same pinned LAST-pooled 8B retrieval model.

    Sends texts unchanged, rejects overlength inputs, normalizes all returned
    vectors. No query instruction, document format or ranking rule is changed.
    """
    dimension = 4096

    def __init__(self, source, revision, base_url, model, api_key_env=None,
                 max_tokens=8192, batch_size=1, timeout=120):
        from transformers import AutoTokenizer
        from ..domains.clients import JSONClient

        if not revision or batch_size != 1:
            raise ConfigurationError("Remote embedding requires pinned revision and serial requests")
        self.revision, self.max_tokens = revision, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
        self.client = JSONClient(base_url, api_key_env, timeout)
        self.model = model

    def encode(self, texts):
        import time
        vectors = []
        for text in texts:
            if len(self.tokenizer.encode(text)) > self.max_tokens:
                raise ConfigurationError("Embedding input exceeds profile; no silent truncation")
            started = time.monotonic()
            result = self.client.post('/embeddings', {'model':self.model,'input':[text],
                                                     'encoding_format':'float'})
            data = result.get('data', [])
            if len(data) != 1 or data[0].get('index') != 0:
                raise ConfigurationError("Invalid embedding response cardinality/index")
            v = data[0]['embedding']
            if len(v) != self.dimension or not all(math.isfinite(x) for x in v):
                raise ConfigurationError("Wrong embedding dimension or nonfinite embedding")
            norm = math.sqrt(sum(x*x for x in v))
            if norm == 0:
                raise ConfigurationError("Zero embedding vector")
            vectors.append(tuple(x/norm for x in v))
            resources = getattr(self, 'resources', None)
            if resources:
                resources.add('embedding_api_calls', 1)
                resources.add('embedding_api_seconds', time.monotonic()-started)
        return vectors
