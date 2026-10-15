from __future__ import annotations

from ..contracts import GenerationRecord
from .prompts import encode_messages


def response_logprobs(logits, input_ids, response_mask, attention_mask, temperature=1.0):
    """Shift once. Include generated EOS; exclude prompt, padding and observations."""
    import torch

    logp = torch.log_softmax(logits[:, :-1].float() / temperature, dim=-1)
    values = logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    mask = response_mask[:, 1:].bool() & attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    return values, mask


def gathered_logprobs(logits, targets, temperature=1.0, chunk_tokens=64):
    """Same full-vocabulary normalization, with bounded FP32 temporary tensors."""
    import torch
    from torch.utils.checkpoint import checkpoint

    if chunk_tokens <= 0 or logits.shape[:-1] != targets.shape:
        raise ValueError("Invalid logprob projection chunks or target shape")

    def score(part, chosen):
        logp = torch.log_softmax(part.float() / temperature, dim=-1)
        return logp.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)

    chunks = []
    for offset in range(0, logits.shape[-2], chunk_tokens):
        part = logits[:, offset : offset + chunk_tokens]
        chosen = targets[:, offset : offset + chunk_tokens]
        if torch.is_grad_enabled() and part.requires_grad:
            chunks.append(checkpoint(score, part, chosen, use_reentrant=False))
        else:
            chunks.append(score(part, chosen))
    return torch.cat(chunks, dim=-1)


class GeneratorPolicy:
    def __init__(self, shared, max_input_tokens=8192, limits=None, temperature=1.0, use_generation_cache=False,
                 prompt_profile="e-text-v1", release_unused_cache=False, generation_progress=None,
                 input_limits=None):
        if temperature != 1.0:
            raise ValueError("v3 E/U use temperature=1, no top-k/top-p/grammar transforms")
        self.shared = shared
        self.max_input_tokens = max_input_tokens
        self.input_limits = dict(input_limits or {})
        if set(self.input_limits) - {"E", "U"} or any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in self.input_limits.values()
        ):
            raise ValueError("input_limits must contain positive E/U token limits")
        self.limits = limits or {"E": 384, "U": 768}
        self.temperature = temperature
        self.use_generation_cache = use_generation_cache
        self.release_unused_cache = release_unused_cache
        self.generation_progress = generation_progress
        if prompt_profile not in {"default", "compact-json-v1", "strict-json-v2", "raw-json-v3", "swe-json-v4", "e-text-v1"}:
            raise ValueError("Unknown memory prompt profile")
        self.prompt_profile = prompt_profile

    def input_limit(self, role):
        return self.input_limits.get(role, self.max_input_tokens)

    def prepare_messages(self, role, messages):
        """Apply the same profile for inference and supervised initialization."""
        if self.prompt_profile == "e-text-v1":
            if role == "E":
                from .prompts import text_extraction_messages

                return text_extraction_messages(messages)
            # U still issues explicit atomic JSON transactions.
            return GeneratorPolicy(None, prompt_profile="raw-json-v3").prepare_messages(role, messages)
        if self.prompt_profile in {"raw-json-v3", "swe-json-v4"}:
            messages = [dict(message) for message in messages]
            messages[0]["content"] += (
                '\nOutput raw JSON only, starting with { and ending with }. '
                'No markdown fences, analysis or text outside JSON. '
                'Keep each candidate concise and cite only the necessary visible evidence IDs. '
                'Close all objects and arrays within the output budget.'
            )
            # Repeat the existing output contract after long source evidence;
            # do not add a grammar mask or transform sampled token probabilities.
            reminder = (
                '\n\nEND OF INPUT DATA. Return ONE complete raw JSON object only. '
                'No markdown fences, analysis or text outside JSON. '
            )
            if role == "E":
                reminder += (
                    'The top-level key is "candidates": an array of ZERO TO THREE objects, '
                    'NEVER more than three. Each object uses kind, title, content, conditions '
                    'and source_ids as specified above; cite exact visible evidence IDs. '
                    'Keep content concise and close every array and object.'
                )
            else:
                reminder += (
                    'Use the exact input base_revision and ONE TO FOUR legal operations. '
                    'ADD refers to candidate_index and does not repeat candidate content. '
                    'NOOP must stand alone. Close every array and object.'
                )
            for message in reversed(messages):
                if message["role"] == "user":
                    if self.prompt_profile == "swe-json-v4" and role == "E":
                        import json
                        content = message["content"]
                        if content.startswith("PUBLIC_METADATA\n"):
                            metadata = json.loads(content.split("\nPUBLIC_SOURCE",1)[0].split("\n",1)[1])
                        else:
                            metadata = json.loads(content)
                        outcome = metadata.get("official_outcome")
                        if outcome is not None:
                            reminder += (
                                '\nFINAL OFFICIAL OUTCOME: resolved=' +
                                ("true" if outcome["resolved"] else "false") +
                                '. This verdict overrides solver claims and self-tests. '
                                'If false, attempted changes are unverified; summarize observed facts '
                                'or conditional failure lessons, never a proven fix. '
                                'If true, the submitted patch passed this task; do not infer failure '
                                'from exploratory self-tests or claim broader coverage. '
                            )
                        reminder += (
                            '\nFIELD TYPES: kind, title, content and conditions are JSON STRINGS. '
                            'source_ids is an array of distinct exact visible ID strings; cite only '
                            'the smallest evidence set that supports each candidate. '
                            'Exactly ONE top-level candidates key, no duplicate keys or literal ... . '
                            'Do not repeat candidates. Close the object once and stop.'
                        )
                    message["content"] += reminder
                    break
        elif self.prompt_profile == "compact-json-v1":
            messages = [dict(message) for message in messages]
            reminder = (
                '\nOutput raw JSON starting with { and ending with }. No markdown fences or explanations. '
                'Keep content concise. For E, use at most ONE candidate, with only 1 or 2 exact '
                'evidence IDs needed to support it; do not copy the whole evidence list. '
                'Example E shape: {"candidates":[{"kind":"procedure","title":"Short title",'
                '"content":"Short reusable observation","conditions":"When applicable",'
                '"source_ids":["COPY_AN_ACTUAL_VISIBLE_ID"]}]}. '
                'For U, ADD copies the candidate; do not repeat its text. '
                'Example U shape: {"base_revision":"COPY_ACTUAL_BANK_REVISION",'
                '"operations":[{"op":"ADD","candidate_index":0}]}. '
                'These are shape examples, not evidence or values to copy.'
            )
            messages[0]["content"] += reminder
        elif self.prompt_profile == "strict-json-v2":
            from .prompts import STRICT_E_SYSTEM, STRICT_U_SYSTEM

            messages = [dict(message) for message in messages]
            system = STRICT_E_SYSTEM if role == "E" else STRICT_U_SYSTEM
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = system
            else:
                messages.insert(0, {"role": "system", "content": system})
        return messages

    def generate(self, role, messages, seed, deterministic=False):
        import torch

        messages = self.prepare_messages(role, messages)
        ids, prompt_hash, _ = encode_messages(self.shared.tokenizer, messages, self.input_limit(role))
        rng = torch.Generator(device=self.shared.device).manual_seed(seed)
        sequence, response, probabilities = list(ids), [], []
        eos = self.shared.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        reason = "length"
        cache = None
        with self.shared.activate(role), torch.no_grad():
            for _ in range(self.limits[role]):
                if self.release_unused_cache and cache is None and self.shared.device.type == "cuda":
                    torch.cuda.empty_cache()
                tokens = sequence[-1:] if self.use_generation_cache and cache is not None else sequence
                inputs = torch.tensor([tokens], dtype=torch.long, device=self.shared.device)
                # Native HF forward, no generate() logits processors or synthetic EOS.
                output = self.shared.model(
                    input_ids=inputs,
                    attention_mask=torch.ones((1, len(sequence)), dtype=torch.long, device=inputs.device),
                    use_cache=self.use_generation_cache,
                    past_key_values=cache,
                    logits_to_keep=1,
                )
                if self.use_generation_cache:
                    cache = output.past_key_values
                logp = torch.log_softmax(output.logits[0, -1].float() / self.temperature, dim=-1)
                token = (
                    int(logp.argmax())
                    if deterministic
                    else int(torch.multinomial(logp.exp(), 1, generator=rng))
                )
                response.append(token)
                probabilities.append(float(logp[token]))
                sequence.append(token)
                del output, inputs, logp
                if self.generation_progress is not None:
                    self.generation_progress(role, len(response), token in eos)
                if token in eos:
                    reason = "eos"
                    break
        # Cache belongs only to this sample and adapter version, never another role/request/update.
        del cache
        text = self.shared.tokenizer.decode(response, skip_special_tokens=True)
        return GenerationRecord(
            role,
            ids,
            prompt_hash,
            tuple(response),
            (True,) * len(response),
            tuple(probabilities),
            text,
            self.temperature,
            reason,
            self.shared.version,
        )

    def recompute(self, records):
        """Batched teacher forcing; callers hold the role lock through backward."""
        import torch

        if len({r.role for r in records}) != 1:
            raise ValueError("One active role per batch")
        if self.release_unused_cache and self.shared.device.type == "cuda":
            torch.cuda.empty_cache()
        sequences = [r.input_ids + r.response_token_ids for r in records]
        width = max(map(len, sequences))
        pad = self.shared.tokenizer.pad_token_id
        if pad is None:
            pad = self.shared.tokenizer.eos_token_id
        ids = torch.full((len(records), width), pad, dtype=torch.long, device=self.shared.device)
        attention = torch.zeros_like(ids)
        response_mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (record, seq) in enumerate(zip(records, sequences)):
            ids[i, : len(seq)] = torch.tensor(seq, device=ids.device)
            attention[i, : len(seq)] = 1
            response_mask[i, len(record.input_ids) : len(seq)] = torch.tensor(
                record.response_mask, device=ids.device
            )
        # Avoid materializing the 248K vocabulary at all 8192 prompt positions.
        keep = torch.tensor(
            sorted(
                {
                    j
                    for r in records
                    for j in range(len(r.input_ids) - 1, len(r.input_ids) + len(r.response_token_ids) - 1)
                }
            ),
            device=ids.device,
        )
        logits = self.shared.model(
            input_ids=ids, attention_mask=attention, use_cache=False, logits_to_keep=keep
        ).logits
        targets = ids[:, keep + 1]
        selected = gathered_logprobs(logits, targets, self.temperature)
        offsets = {int(position): i for i, position in enumerate(keep.tolist())}
        result, masks = [], []
        for i, r in enumerate(records):
            indices = [offsets[j] for j in range(len(r.input_ids) - 1, len(sequences[i]) - 1)]
            result.append(selected[i, indices])
            masks.append(torch.tensor(r.response_mask, dtype=torch.bool, device=ids.device))
        return result, masks
