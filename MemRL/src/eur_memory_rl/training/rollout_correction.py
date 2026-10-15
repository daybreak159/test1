"""Opt-in mixed-engine loss primitives used by Coordinator's explicit mode.

These are conditional-action surrogate corrections, not an exact correction of
the distribution of entire memory-bank/environment trajectories.
"""
from __future__ import annotations

import math


def corrected_surrogate(new, reference, behavior, advantage, *, epsilon=0.2,
                        correction_cap=2.0, mode="decoupled", mask=None,
                        correction_level="token"):
    """Return masked per-action surrogate terms and detached diagnostic weights.

    `reference` is a frozen HF logprob from BEFORE any optimizer step. `behavior`
    is the actual rollout logprob. For R pass a scalar ordered-slate logprob;
    for E/U pass one response's token logprobs. All transforms/masks must match.
    Sequence weighting is a joint-response surrogate experiment, not token-wise
    exact importance sampling. An upper cap introduces deliberate bias.
    """
    import torch

    if mode not in {"decoupled", "bypass", "hf_reference"}:
        raise ValueError("Unknown correction mode")
    if correction_level not in {"token", "sequence"} or not 0 <= epsilon < 1:
        raise ValueError("Invalid correction level or epsilon")
    if correction_cap is not None and (not math.isfinite(correction_cap) or correction_cap < 1):
        raise ValueError("Correction cap must be finite and >= 1")
    new = new.float()
    reference = torch.as_tensor(reference, device=new.device, dtype=new.dtype).detach()
    behavior = torch.as_tensor(behavior, device=new.device, dtype=new.dtype).detach()
    if new.shape != reference.shape or new.shape != behavior.shape or new.ndim > 1:
        raise ValueError("Pass one scalar action or one unpadded response at a time")
    valid = torch.ones_like(new, dtype=torch.bool) if mask is None else mask.to(new.device).bool()
    if valid.shape != new.shape:
        raise ValueError("Mask shape mismatch")
    for tensor in (new, reference, behavior):
        if not torch.isfinite(tensor[valid]).all():
            raise ValueError("Nonfinite valid logprob")
    new = torch.where(valid, new, torch.zeros_like(new))
    reference = torch.where(valid, reference, torch.zeros_like(reference))
    behavior = torch.where(valid, behavior, torch.zeros_like(behavior))
    log_c = reference - behavior
    if correction_level == "sequence":
        log_c = log_c.sum().expand_as(new)
    bounded = log_c if correction_cap is None else log_c.clamp(max=math.log(correction_cap))
    correction = bounded.exp().detach()
    ratio = (new - (behavior if mode == "bypass" else reference)).exp()
    adv = torch.as_tensor(advantage, device=new.device, dtype=new.dtype).detach()
    if not torch.isfinite(adv).all():
        raise ValueError("Nonfinite advantage")
    terms = torch.minimum(ratio * adv, ratio.clamp(1 - epsilon, 1 + epsilon) * adv)
    if mode == "decoupled":
        terms = correction * terms
    terms = torch.where(valid, terms, torch.zeros_like(terms))
    if not torch.isfinite(terms).all():
        raise ValueError("Nonfinite surrogate (inspect log-ratio / support)")
    return terms, {
        "log_correction": log_c.detach(),
        "correction": correction,
        "learning_ratio": (new - reference).exp().detach(),
        "behavior_ratio": (new - behavior).exp().detach(),
    }
