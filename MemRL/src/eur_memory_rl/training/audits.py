from __future__ import annotations

from ..config import protocol_hash
from ..policies.direct_choice import forward_option_logits, project_option_logits
from ..policies.prompts import encode_messages, slot_ids
from ..util import atomic_json
from .checkpoint import adapter_state, random_state, restore_random_state


def inspect_model(config, output):
    """Inspect actual native module names before choosing LoRA targets. Explicit CLI only."""
    import torch
    import transformers

    a = config["architecture"]
    hf_config = transformers.AutoConfig.from_pretrained(
        a["base_model"], revision=a["base_revision"], trust_remote_code=False
    )
    cls = (
        transformers.Qwen3_5ForCausalLM
        if hf_config.model_type in {"qwen3_5", "qwen3_5_text"}
        else transformers.AutoModelForCausalLM
    )
    hf_config = hf_config.get_text_config()
    # Construct shapes on meta: no checkpoint download or server/GPU execution.
    with torch.device("meta"):
        model = cls(hf_config) if cls is not transformers.AutoModelForCausalLM else cls.from_config(hf_config)
    heads = {id(model.get_input_embeddings()), id(model.get_output_embeddings())}
    modules = [
        {"name": name, "shape": list(module.weight.shape), "eligible": id(module) not in heads}
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    atomic_json(
        output,
        {
            "base_revision": a["base_revision"],
            "model_type": hf_config.model_type,
            "linear_modules": modules,
            "weight_load_verified": False,
            "note": "Select reviewed attention AND feed-forward linear names; shape inspection is not a gradient audit.",
        },
    )


def model_audit(shared, generator, cfg, output):
    """Explicit deployment gate. Not called by installation, offline preflight or unit tests."""
    import torch

    state = random_state()
    weights = adapter_state(shared)
    messages = [{"role": "user", "content": "Choose A or B for this gradient audit. Answer:"}]
    ids, _, prompt = encode_messages(shared.tokenizer, messages, 6144)
    slots = slot_ids(shared.tokenizer, prompt, ids, 2)
    inputs = torch.tensor([ids], device=shared.device)
    checks = {}
    try:
        for role in ("E", "U", "R"):
            for _, parameter in shared.model.named_parameters():
                parameter.grad = None
            with shared.activate(role, train=True):
                logits = forward_option_logits(shared.model, inputs, torch.ones_like(inputs), slots)[0]
                loss = -torch.log_softmax(logits, -1)[0]
                loss.backward()
                norms = {
                    r: sum(float(p.grad.float().square().sum()) for _, p in group if p.grad is not None)
                    ** 0.5
                    for r, group in shared.role_parameters.items()
                }
                owner = role if shared.adapter_layout == "independent" else "shared"
                if not norms[owner] > 0 or any(value != 0 for r, value in norms.items() if r != owner):
                    raise RuntimeError(f"Role gradient isolation failed: {norms}")
                checks[f"{role}_gradient_norms"] = norms
        for role in ("E", "U"):
            old_limits = generator.limits
            generator.limits = {"E": 4, "U": 4}
            try:
                record = generator.generate(role, messages, 17)
            finally:
                generator.limits = old_limits
            with shared.activate(role, train=True):
                values, masks = generator.recompute([record])
                delta = float(
                    (values[0].detach() - torch.tensor(record.old_logprobs, device=shared.device)).abs().max()
                )
                if delta > cfg["optimization"]["behavior_tolerance"]:
                    raise RuntimeError(f"Generation behavior drift: {delta}")
                checks[f"{role}_logprob_max_error"] = delta
        captured = []
        hook = shared.model.get_output_embeddings().register_forward_pre_hook(
            lambda module, args: captured.append(args[0])
        )
        try:
            with shared.activate("R", train=True):
                gathered = forward_option_logits(shared.model, inputs, torch.ones_like(inputs), slots)
                projected = project_option_logits(
                    captured[-1][:, -1], shared.model.get_output_embeddings(), slots
                )
                error = float((gathered - projected).abs().max())
                if error > 0.05:
                    raise RuntimeError("Full-vocabulary gather differs from option-row projection")
                checks["readout_max_error"] = error
        finally:
            hook.remove()
        shared.assert_base_unchanged()
        for role, group in shared.role_parameters.items():
            if any(not torch.equal(p.detach().cpu(), weights[role][name]) for name, p in group):
                raise RuntimeError("Audit mutated an adapter")
        report = {
            "passed": True,
            "protocol_hash": protocol_hash(cfg),
            "base_hash": shared.base_hash,
            "checks": checks,
            "model": shared.manifest(),
            "benchmark_validated": False,
        }
        atomic_json(output, report)
        return report
    finally:
        for _, parameter in shared.model.named_parameters():
            parameter.grad = None
        restore_random_state(state)
