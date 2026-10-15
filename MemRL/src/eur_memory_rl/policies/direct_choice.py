from __future__ import annotations

import math
import uuid

from ..contracts import ChoiceForward, Slate
from ..memory.render import MemoryRenderer, count_tokens
from ..util import ConfigurationError, digest
from .prompts import choice_messages, encode_messages, slot_ids


def probabilities(logits, temperature=1.0):
    if not logits:
        return ()
    if temperature <= 0 or not all(math.isfinite(z) for z in logits):
        raise ValueError("Invalid logits/temperature")
    maximum = max(logits)
    exp = [math.exp((z - maximum) / temperature) for z in logits]
    return tuple(v / sum(exp) for v in exp)


def legal_mask(candidates, selected, scope, renderer, tokenizer, token_budget):
    chosen = [candidates[i] for i in selected]
    return tuple(
        i not in selected
        and e.scope == scope
        and e.status == "active"
        and count_tokens(tokenizer, renderer.render([*chosen, e])) <= token_budget
        for i, e in enumerate(candidates)
    )


def forward_option_logits(model, input_ids, attention_mask, slots):
    """Differentiable FP32 gather. No inference_mode/detach in the training path."""
    import torch

    if input_ids.shape != attention_mask.shape or not attention_mask.bool().any(dim=1).all():
        raise ValueError("Invalid selector attention mask")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
    last = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    kept, inverse = torch.unique(last, sorted=True, return_inverse=True)
    # Only requested time positions need a full vocabulary projection.
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=kept
    ).logits
    last_logits = logits[torch.arange(input_ids.shape[0], device=input_ids.device), inverse].float()
    indices = torch.as_tensor(slots, dtype=torch.long, device=input_ids.device)
    return last_logits.index_select(-1, indices)


def project_option_logits(hidden, lm_head, slots):
    """Audit/optimized readout for heads verified to be an untransformed linear layer."""
    import torch
    import torch.nn.functional as F

    indices = torch.as_tensor(slots, dtype=torch.long, device=hidden.device)
    weight = lm_head.weight.index_select(0, indices)
    bias = None if lm_head.bias is None else lm_head.bias.index_select(0, indices)
    return F.linear(hidden, weight, bias).float()


def replay_slate_logprob(logits, slate, tokenizer, renderer=None):
    """Rebuild and check every conditional action set against the actual exposure."""
    import torch

    renderer = renderer or MemoryRenderer()
    f = slate.forward
    if renderer.version != f.renderer_version or len(logits) != len(f.candidates):
        raise ValueError("Selector input/renderer drift")
    selected, parts = [], []
    for h, action in enumerate(slate.selected_indices):
        mask = legal_mask(f.candidates, selected, f.scope, renderer, tokenizer, f.token_budget)
        if mask != slate.legal_masks[h] or not mask[action] or len(selected) >= f.max_selected:
            raise ValueError("Behavior mask/action mismatch")
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=logits.device)
        values = (logits.float() / f.temperature).masked_fill(~mask_tensor, -torch.inf)
        parts.append(torch.log_softmax(values, dim=-1)[action])
        selected.append(action)
    terminal = "load_limit" if len(selected) == f.max_selected else "no_legal_candidate"
    if terminal == "no_legal_candidate" and any(
        legal_mask(f.candidates, selected, f.scope, renderer, tokenizer, f.token_budget)
    ):
        raise ValueError("Premature stopping is not an EUR action")
    if (
        slate.termination != terminal
        or renderer.render([f.candidates[i] for i in selected]) != slate.memory_block
    ):
        raise ValueError("Behavior termination/rendering mismatch")
    return torch.stack(parts).sum() if parts else logits.sum() * 0


def make_slate(forward, tokenizer, *, logits=None, rng=None, greedy=False, fixed_order=False):
    """One logits tensor, repeated masked categorical draws; no additional model calls."""
    renderer = MemoryRenderer()
    selected, masks, logps = [], [], []
    f = forward
    while len(selected) < f.max_selected:
        mask = legal_mask(f.candidates, selected, f.scope, renderer, tokenizer, f.token_budget)
        legal = [i for i, allowed in enumerate(mask) if allowed]
        if not legal:
            break
        if fixed_order:
            action, logp = legal[0], 0.0
        elif greedy:
            action = min(legal, key=lambda i: (-f.old_logits[i], f.candidates[i].id))
            dist = probabilities([f.old_logits[i] for i in legal], f.temperature)
            logp = math.log(dist[legal.index(action)])
        elif len(legal) == 1:
            action, logp = legal[0], 0.0
        else:
            import torch

            if logits is None or rng is None:
                raise ValueError("Training sampling requires the shared logits tensor and independent RNG")
            valid = torch.tensor(mask, device=logits.device, dtype=torch.bool)
            log_probs = torch.log_softmax(
                (logits.float() / f.temperature).masked_fill(~valid, -torch.inf), -1
            )
            action = int(torch.multinomial(log_probs.exp(), 1, generator=rng))
            logp = float(log_probs[action].detach())
        selected.append(action)
        masks.append(mask)
        logps.append(logp)
    memory = renderer.render([f.candidates[i] for i in selected])
    return Slate(
        f,
        tuple(selected),
        tuple(masks),
        tuple(logps),
        sum(logps),
        "load_limit" if len(selected) == f.max_selected else "no_legal_candidate",
        memory,
        count_tokens(tokenizer, memory),
    )


class DirectChoicePolicy:
    def __init__(
        self,
        shared,
        consumer_tokenizer,
        max_input_tokens=6144,
        token_budget=2048,
        max_selected=4,
        candidate_slots=12,
        temperature=1.0,
    ):
        if not 1 <= candidate_slots <= 16 or max_selected < 1 or temperature <= 0:
            raise ConfigurationError("Invalid direct selector profile")
        self.shared, self.tokenizer = shared, consumer_tokenizer
        self.max_input_tokens, self.token_budget = max_input_tokens, token_budget
        self.max_selected, self.candidate_slots, self.temperature = max_selected, candidate_slots, temperature

    def pack(self, turn, hits):
        # Greedily pack whole entries in recall order before action sampling. No clipping.
        encode_messages(
            self.shared.tokenizer,
            choice_messages(turn, (), self.token_budget, self.max_selected),
            self.max_input_tokens,
        )
        selected = []
        for hit in hits[: self.candidate_slots]:
            candidate = hit.entry
            if count_tokens(self.tokenizer, MemoryRenderer().render([candidate])) > self.token_budget:
                continue
            try:
                encode_messages(
                    self.shared.tokenizer,
                    choice_messages(turn, [*selected, candidate], self.token_budget, self.max_selected),
                    self.max_input_tokens,
                )
            except ConfigurationError:
                continue
            selected.append(candidate)
        return tuple(selected)

    def prepare(self, turn, hits, scope, fixed=False):
        import torch

        candidates = self.pack(turn, hits)
        ids, prompt_hash, prompt = encode_messages(
            self.shared.tokenizer,
            choice_messages(turn, candidates, self.token_budget, self.max_selected),
            self.max_input_tokens,
        )
        slots = slot_ids(self.shared.tokenizer, prompt, ids, len(candidates))
        mask = legal_mask(candidates, (), scope, MemoryRenderer(), self.tokenizer, self.token_budget)
        do_forward = sum(mask) > 1 and not fixed
        if do_forward:
            with self.shared.activate("R"), torch.no_grad():
                inputs = torch.tensor([ids], dtype=torch.long, device=self.shared.device)
                logits = forward_option_logits(self.shared.model, inputs, torch.ones_like(inputs), slots)[0]
        else:
            logits = torch.zeros(len(candidates), dtype=torch.float32, device=self.shared.device)
        old = tuple(logits.cpu().tolist())
        record = ChoiceForward(
            str(uuid.uuid4()),
            ids,
            prompt_hash,
            slots,
            candidates,
            old,
            probabilities(old, self.temperature),
            self.temperature,
            self.shared.version,
            int(do_forward),
            scope,
            MemoryRenderer.version,
            self.token_budget,
            self.max_selected,
            tuple(hit.entry for hit in hits),
            {
                "profile": "whole-entry-recall-order-v1",
                "source_hashes": {hit.entry.id: digest(hit.entry) for hit in hits},
                "visible_hashes": {e.id: digest(e) for e in candidates},
            },
        )
        return record, logits

    def sample(self, prepared, seed, deterministic=False, fixed=False):
        import torch

        forward, logits = prepared
        rng = torch.Generator(device=logits.device).manual_seed(seed)
        return make_slate(
            forward, self.tokenizer, logits=logits, rng=rng, greedy=deterministic, fixed_order=fixed
        )

    def recompute_logits(self, forward):
        import torch

        inputs = torch.tensor([forward.input_ids], dtype=torch.long, device=self.shared.device)
        if forward.forward_count == 0:
            return torch.zeros(len(forward.candidates), device=self.shared.device)
        return forward_option_logits(
            self.shared.model, inputs, torch.ones_like(inputs), forward.slot_token_ids
        )[0]
