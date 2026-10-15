from __future__ import annotations

from ..contracts import GenerationRecord, PolicyRow
from ..memory.transaction import parse_extraction
from ..policies.prompts import encode_messages
from ..rewards.advantages import loo
from ..util import digest


def validate_sft_demonstration(example, source, executor, bank=None, candidates=(), visible=(),
                               prompt_profile="e-text-v1"):
    if source.identity.split != "train_source" or example["provenance"]["split"] != "train_source":
        raise ValueError("SFT accepts only documented train_source demonstrations")
    if not example["provenance"].get("teacher_revision") or "cost" not in example["provenance"]:
        raise ValueError("SFT provenance needs a teacher/human version and cost")
    if example["role"] == "E":
        parse_extraction(example["response"], source, profile=prompt_profile)
    elif example["role"] == "U":
        if bank is None or not executor.apply(bank, candidates, example["response"], visible, source).valid:
            raise ValueError("Invalid SFT transaction")
    else:
        raise ValueError("R learns classification, not JSON generation")


def sft_record(shared, role, messages, response, max_input=8192, max_output=768):
    ids, prompt_hash, _ = encode_messages(shared.tokenizer, messages, max_input)
    tokens = list(shared.tokenizer.encode(response, add_special_tokens=False))
    eos = shared.tokenizer.eos_token_id
    if eos is not None:
        tokens.append(eos)
    if len(tokens) > max_output:
        raise ValueError("SFT response exceeds role budget")
    return GenerationRecord(
        role,
        ids,
        prompt_hash,
        tuple(tokens),
        (True,) * len(tokens),
        (0.0,) * len(tokens),
        response,
        1.0,
        "eos",
        shared.version,
        True,
    )


def train_sft(shared, generator, records, optimizers, epochs=1, max_grad_norm=1, progress=None,
              start_step=0, initial_metrics=None, diagnostic=None, skip_indices=()):
    import torch

    if not 0 <= epochs <= 1 or any(r.role not in {"E", "U"} for r in records):
        raise ValueError("v3 SFT is at most one E/U-only epoch")
    if not 0 <= start_step <= len(records):
        raise ValueError("Invalid SFT resume cursor")
    metrics = dict(initial_metrics or {"steps": 0, "token_nll": 0.0})
    if metrics["steps"] != start_step:
        raise ValueError("SFT metrics and resume cursor disagree")
    skipped = frozenset(skip_indices)
    for _ in range(epochs):
        for record_index, record in enumerate(records[start_step:], start=start_step):
            if record_index in skipped:
                metrics["steps"] += 1
                if progress:
                    progress({"step": metrics["steps"], "role": record.role,
                              "loss": None, "skipped": True,
                              "skip_reason": "input_exceeds_measured_hf_sft_capacity",
                              "input_tokens": len(record.input_ids),
                              "token_nll": metrics["token_nll"]})
                continue
            if diagnostic:
                diagnostic("begin", metrics["steps"] + 1, record)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            with shared.activate(record.role, train=True):
                logp, masks = generator.recompute([record])
                loss = -logp[0][masks[0]].mean()  # SFT deliberately uses conventional token mean.
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite SFT loss; no optimizer step")
                loss.backward()
                owner = record.role if shared.adapter_layout == "independent" else "shared"
                torch.nn.utils.clip_grad_norm_([p for _, p in shared.role_parameters[owner]], max_grad_norm,
                                              error_if_nonfinite=True)
                optimizers[owner].step()
            metrics["steps"] += 1
            metrics["token_nll"] += float(loss.detach())
            if diagnostic:
                diagnostic("end", metrics["steps"], record)
            if progress:
                progress({"step": metrics["steps"], "role": record.role,
                          "loss": float(loss.detach()), "prompt_hash": record.input_hash,
                          "token_nll": metrics["token_nll"]})
    shared.assert_base_unchanged()
    return metrics


def warmup_rows(planned_queries, L, execute, selector, domain, seed):
    """Freeze E/U; each input owns L actions, B execution repetitions never create rows."""
    from ..util import seed_for

    if not planned_queries or L < 2:
        raise ValueError("Warmup needs planned bank-query pairs and L>=2")
    rows = []
    for index, (bank, task, prepared) in enumerate(planned_queries):
        slates = [selector.sample(prepared, seed_for(seed, index, ell)) for ell in range(L)]
        values = [
            execute(task, slate, seed_for(seed, "execute", index, ell)) for ell, slate in enumerate(slates)
        ]
        for ell, (slate, advantage) in enumerate(zip(slates, loo(values))):
            rows.append(
                PolicyRow(
                    digest([seed, index, ell]),
                    "R",
                    f"warmup-{index}",
                    f"warmup-{index}",
                    domain,
                    "train",
                    str(index),
                    index,
                    slate,
                    bank.hash,
                    bank.hash,
                    "warmup",
                    advantage=advantage,
                    row_weight=1 / (len(planned_queries) * L),
                )
            )
    return rows
