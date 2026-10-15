from __future__ import annotations

import hashlib
import re
import threading
from contextlib import contextmanager, nullcontext

from ..util import ConfigurationError

ROLES = ("E", "U", "R")


def chunk_mlp_forward(module, chunk_tokens):
    """Exact token-independent MLP in checkpointed chunks, same parameters."""
    import torch
    from torch.utils.checkpoint import checkpoint
    original = module.forward
    def forward(x):
        if x.shape[-2] <= chunk_tokens:
            return original(x)
        parts = x.split(chunk_tokens, dim=-2)
        if module.training and torch.is_grad_enabled():
            return torch.cat([checkpoint(original, part, use_reentrant=False) for part in parts],dim=-2)
        return torch.cat([original(part) for part in parts],dim=-2)
    module.forward = forward


def chunk_pointwise_forward(module, chunk_tokens, axis=-2):
    """Checkpoint token-independent projections/norms without changing their arithmetic."""
    import torch
    from torch.utils.checkpoint import checkpoint

    original = module.forward
    def forward(*args, **kwargs):
        if not args or not torch.is_tensor(args[0]) or args[0].shape[axis] <= chunk_tokens:
            return original(*args, **kwargs)
        length = args[0].shape[axis]
        if any(not torch.is_tensor(x) or x.shape[axis] != length for x in args):
            raise ValueError("Pointwise chunks require equal input lengths")
        chunks = [x.split(chunk_tokens, dim=axis) for x in args]
        def apply(*parts):
            return original(*parts, **kwargs)
        outputs = []
        for parts in zip(*chunks):
            outputs.append(checkpoint(apply,*parts,use_reentrant=False)
                           if module.training and torch.is_grad_enabled() else apply(*parts))
        return torch.cat(outputs,dim=axis)
    module.forward = forward


def tensor_hash(named_parameters):
    h = hashlib.sha256()
    for name, p in sorted(named_parameters, key=lambda item: item[0]):
        h.update(name.encode())
        h.update(str((tuple(p.shape), p.dtype)).encode())
        h.update(p.detach().contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes())
    return h.hexdigest()


class SharedQwen:
    def __init__(self, model, tokenizer, base_revision: str, adapter_layout="independent"):
        self.model, self.tokenizer, self.base_revision = model, tokenizer, base_revision
        self.lock = threading.RLock()
        self.version = "iteration-0"
        self.adapter_layout = adapter_layout
        names = ROLES if adapter_layout == "independent" else ("shared",)
        self.role_parameters = {
            role: [
                (name, p) for name, p in model.named_parameters() if f".{role}." in name and "lora_" in name
            ]
            for role in names
        }
        if any(not group for group in self.role_parameters.values()):
            raise ConfigurationError("Each role must own an independent LoRA parameter set")
        all_ids = [id(p) for group in self.role_parameters.values() for _, p in group]
        if len(all_ids) != len(set(all_ids)):
            raise ConfigurationError("Role parameter sets overlap")
        self.base_parameters = [(n, p) for n, p in model.named_parameters() if id(p) not in set(all_ids)]
        for _, p in self.base_parameters:
            p.requires_grad_(False)
        self.base_hash = tensor_hash(self.base_parameters)
        # model.eval() disables dropout but does not disable autograd.
        self.model.eval()
        self.activation_offload = False
        self.mlp_chunk_tokens = None
        self.pointwise_chunk_tokens = None

    @classmethod
    def load(cls, config):
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model

        source, revision = config["base_model"], config["base_revision"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
            raise ConfigurationError("Base revision must be a pinned commit")
        common = dict(revision=revision, trust_remote_code=False)
        model_config = transformers.AutoConfig.from_pretrained(source, **common)
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
        model_cls = transformers.AutoModelForCausalLM
        if model_config.model_type in {"qwen3_5", "qwen3_5_text"}:
            model_cls = transformers.Qwen3_5ForCausalLM
            model_config = model_config.get_text_config()
        model, info = model_cls.from_pretrained(
            source,
            config=model_config,
            dtype=getattr(torch, config.get("dtype", "bfloat16")),
            device_map={"": config.get("device", "cuda:0")},
            output_loading_info=True,
            **common,
        )
        if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ConfigurationError(f"Incomplete base load: {info}")
        targets = config.get("target_modules")
        if not targets or len(targets) != len(set(targets)):
            raise ConfigurationError("Provide the inspected, exact unique target module names")
        modules = dict(model.named_modules())
        for name in targets:
            if name not in modules or not isinstance(modules[name], torch.nn.Linear):
                raise ConfigurationError(f"Target must be an actual linear layer: {name}")
            if (
                modules[name] is model.get_output_embeddings()
                or modules[name] is model.get_input_embeddings()
            ):
                raise ConfigurationError("Embedding and LM head must remain frozen")
        model.requires_grad_(False)
        target_regex = "^(?:" + "|".join(re.escape(name) for name in targets) + ")$"

        def lora():
            return LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_regex,
                task_type=TaskType.CAUSAL_LM,
            )

        layout = config.get("adapter_layout", "independent")
        if layout not in {"independent", "shared"}:
            raise ConfigurationError("Unknown adapter layout")
        model = get_peft_model(model, lora(), adapter_name="E" if layout == "independent" else "shared")
        if layout == "independent":
            model.add_adapter("U", lora())
            model.add_adapter("R", lora())
        if config.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        shared = cls(model, tokenizer, revision, layout)
        shared.activation_offload = config.get("activation_offload", False)
        chunk = config.get("mlp_chunk_tokens")
        if chunk is not None:
            if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0:
                raise ConfigurationError("mlp_chunk_tokens must be a positive integer")
            modules = [m for m in model.modules() if type(m).__name__ == "Qwen3_5MLP"]
            if not modules:raise ConfigurationError("Expected Qwen3.5 MLP modules for chunking")
            for module in modules:chunk_mlp_forward(module,chunk)
            shared.mlp_chunk_tokens = chunk
        pointwise_chunk = config.get("pointwise_chunk_tokens")
        if pointwise_chunk is not None:
            if not isinstance(pointwise_chunk,int) or isinstance(pointwise_chunk,bool) or pointwise_chunk<=0:
                raise ConfigurationError("pointwise_chunk_tokens must be positive")
            from peft.tuners.lora.layer import Linear as LoRALinear
            for module in model.modules():
                if isinstance(module,LoRALinear):
                    chunk_pointwise_forward(module,pointwise_chunk)
                elif type(module).__name__ == "Qwen3_5RMSNormGated":
                    chunk_pointwise_forward(module,pointwise_chunk,axis=0)
                elif type(module).__name__ == "Qwen3_5RMSNorm":
                    chunk_pointwise_forward(module,pointwise_chunk,axis=1)
            shared.pointwise_chunk_tokens = pointwise_chunk
        return shared

    @property
    def device(self):
        return self.model.get_input_embeddings().weight.device

    @contextmanager
    def activate(self, role, train=False):
        if role not in ROLES:
            raise ValueError(role)
        with self.lock:
            adapter = role if self.adapter_layout == "independent" else "shared"
            self.model.set_adapter(adapter)
            for name, group in self.role_parameters.items():
                for _, p in group:
                    p.requires_grad_(train and name == adapter)
            for _, p in self.base_parameters:
                p.requires_grad_(False)
            # Enable checkpointing in training while keeping every dropout disabled.
            self.model.train(train)
            import torch

            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                    module.eval()
            resources = getattr(self, "resources", None)
            timer = (
                resources.gpu_timer(self.device, "training_policy" if train else "rollout_policy")
                if resources
                else nullcontext()
            )
            # Exact saved-tensor transfer, not quantization or shorter inputs.
            # Non-reentrant checkpointing can save its boundary tensors on CPU.
            offload = torch.autograd.graph.save_on_cpu(pin_memory=False) if train and self.activation_offload else nullcontext()
            with timer, offload:
                yield self.model

    def manifest(self):
        return {
            "base_revision": self.base_revision,
            "base_hash": self.base_hash,
            "adapter_layout": self.adapter_layout,
            "activation_offload": self.activation_offload,
            "mlp_chunk_tokens": self.mlp_chunk_tokens,
            "pointwise_chunk_tokens": self.pointwise_chunk_tokens,
            "roles": {
                r: [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in group]
                for r, group in self.role_parameters.items()
            },
        }

    def assert_base_unchanged(self):
        if any(p.grad is not None for _, p in self.base_parameters):
            raise RuntimeError("Frozen base received gradients")
        if tensor_hash(self.base_parameters) != self.base_hash:
            raise RuntimeError("Frozen base weights changed")
