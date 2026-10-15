from __future__ import annotations

from dataclasses import replace

from ..contracts import PolicyRow
from ..memory.recall import FrozenIndex
from ..memory.render import count_tokens
from ..memory.transaction import InvalidWrite, parse_extraction
from ..policies.prompts import encode_messages, extraction_messages, transaction_messages
from ..util import ConfigurationError, canonical, digest


class MemoryEngine:
    def __init__(
        self,
        generator,
        selector,
        transactions,
        embedder,
        instructions,
        management_k=8,
        consumer_k=12,
        management_query_tokens=2048,
        resource_log=None,
        source_packing="reject",
    ):
        self.generator, self.selector, self.transactions = generator, selector, transactions
        self.embedder, self.instructions = embedder, instructions
        self.management_k, self.consumer_k = management_k, consumer_k
        self.management_query_tokens = management_query_tokens
        self.resource_log = resource_log
        self.source_packing = source_packing
        self._indexes = {}

    def count(self, name, amount=1):
        if self.resource_log:
            self.resource_log.add(name, amount)

    def index(self, bank):
        if bank.hash not in self._indexes:
            # Reuse content/revision vectors, never cache a query or modify a bank.
            cache = {}
            for index in self._indexes.values():
                cache.update(index.vectors)
            self._indexes[bank.hash] = FrozenIndex(
                bank, self.embedder, cache, lambda n: self.count("embeddings", n)
            )
        return self._indexes[bank.hash]

    def clear_indexes(self):
        self._indexes.clear()

    def extract(self, source, seed, deterministic=False):
        from ..data.packing import pack_source

        source, mapping = pack_source(
            source, self.generator.shared.tokenizer, self.generator.input_limit("E"), self.source_packing,
            prompt_profile=getattr(self.generator, "prompt_profile", "e-text-v1"),
        )
        record = self.generator.generate("E", extraction_messages(source), seed, deterministic)
        record = replace(record, prompt_mapping=mapping)
        self.count("E_input_tokens", len(record.input_ids))
        self.count("E_output_tokens", len(record.response_token_ids))
        try:
            candidates = parse_extraction(record.text, source,
                                          profile=getattr(self.generator, "prompt_profile", "e-text-v1"))
            return replace(record, schema_valid=True), candidates
        except InvalidWrite as exc:
            return replace(record, schema_valid=False, schema_error=str(exc)), None

    def update(self, source, bank, candidates, seed, deterministic=False):
        query = "\n".join(f"{c.title}\n{c.content}" for c in candidates)
        tokenizer = self.generator.shared.tokenizer
        original_ids = tokenizer.encode(query, add_special_tokens=False)
        packed = original_ids[: self.management_query_tokens]
        query = tokenizer.decode(packed)
        recalled = tuple(
            hit.entry
            for hit in self.index(bank).search(
                query, self.instructions["management"], self.management_k, bank.scope
            )
        )
        capacity = {
            "active_bank_token_limit": self.transactions.bank_tokens,
            "current_active_tokens": count_tokens(
                self.transactions.counter, self.transactions.renderer.render(bank.active)
            ),
            "entry_consumer_token_limit": self.transactions.entry_consumer_tokens,
            "entry_selector_token_limit": self.transactions.entry_selector_tokens,
        }
        visible = []
        encode_messages(
            tokenizer,
            transaction_messages(source, bank, candidates, (), capacity),
            self.generator.input_limit("U"),
        )
        for entry in recalled:
            try:
                encode_messages(
                    tokenizer,
                    transaction_messages(source, bank, candidates, [*visible, entry], capacity),
                    self.generator.input_limit("U"),
                )
            except ConfigurationError:
                continue
            visible.append(entry)
        visible = tuple(visible)
        messages = transaction_messages(source, bank, candidates, visible, capacity)
        record = self.generator.generate("U", messages, seed, deterministic)
        self.count("U_input_tokens", len(record.input_ids))
        self.count("U_output_tokens", len(record.response_token_ids))
        result = self.transactions.apply(bank, candidates, record.text, visible, source)
        metadata = {
            "visible_ids_revisions": [(e.id, e.revision) for e in visible],
            "recalled_ids": [e.id for e in recalled],
            "packing_profile": "whole-entry-recall-order-v1",
            "management_query_mapping": {
                "original_hash": digest(original_ids),
                "visible_hash": digest(packed),
                "retained_token_range": [0, len(packed)],
                "profile": "fixed-prefix-v1",
            },
            "operations": result.operations,
        }
        return replace(record, schema_valid=result.valid, schema_error=result.error), result, metadata

    def prepare_read(self, bank, turn, fixed=False):
        query = canonical(
            {"request": turn.text, "public_history": turn.public_history, "time": turn.public_time}
        )
        hits = self.index(bank).search(query, self.instructions["consumer"], self.consumer_k, bank.scope)
        prepared = self.selector.prepare(turn, hits, bank.scope, fixed=fixed)
        self.count("R_rollout_forwards", prepared[0].forward_count)
        self.count("R_input_tokens", len(prepared[0].input_ids) * prepared[0].forward_count)
        return prepared


def policy_row(
    record,
    role,
    *,
    row_id,
    episode_id,
    group_id,
    domain,
    split,
    position,
    time,
    bank_before,
    bank_after=None,
    source="outcome",
    anchor_id="",
    metadata=None,
):
    return PolicyRow(
        row_id,
        role,
        episode_id,
        group_id,
        domain,
        split,
        position,
        time,
        record,
        bank_before.hash,
        (bank_after or bank_before).hash,
        source,
        anchor_id,
        metadata=metadata or {},
    )
