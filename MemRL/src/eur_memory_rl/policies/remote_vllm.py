"""Exact token-ID vLLM rollout RPC; HF stays responsible for gradients."""
from __future__ import annotations

import hashlib
import json
import tarfile
import time
import uuid
from pathlib import Path

from ..contracts import ChoiceForward, GenerationRecord
from ..memory.render import MemoryRenderer
from ..util import InfrastructureError, digest
from .direct_choice import DirectChoicePolicy, legal_mask, probabilities
from .prompts import choice_messages, encode_messages, slot_ids


def export_rollout_adapters(shared, root, qkv_layout="split_qkv"):
    """Preserve every HF target in the serving engine's physical layout."""
    import torch
    from safetensors.torch import load_file, save_file

    if qkv_layout not in {"split_qkv", "native_qkv"}:
        raise ValueError("Unknown vLLM QKV layout")
    root = Path(root)
    source = root / "hf"
    destination = root / "vllm"
    shared.model.save_pretrained(source, selected_adapters=["E", "U", "R"])
    cfg = shared.model.config
    key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    manifest = {"version": shared.version, "base_revision": shared.base_revision,
                "base_hash": shared.base_hash, "qkv_layout": qkv_layout, "roles": {}}
    for role in ("E", "U", "R"):
        src = source / role / "adapter_model.safetensors"
        tensors, converted = load_file(src), {}
        for original, tensor in tensors.items():
            name = original.replace("base_model.model.model.layers.",
                                    "base_model.model.model.language_model.layers.")
            if ".in_proj_qkv." in name and qkv_layout == "split_qkv":
                if ".lora_A." in name:
                    for part in "qkv":
                        converted[name.replace(".in_proj_qkv.", f".in_proj_{part}.")] = tensor
                else:
                    if tensor.shape[0] != key * 2 + value:
                        raise ValueError("Unexpected QKV LoRA shape")
                    for part, chunk in zip("qkv", tensor.split([key, key, value], dim=0)):
                        converted[name.replace(".in_proj_qkv.", f".in_proj_{part}.")] = chunk.contiguous()
            else:
                converted[name] = tensor
        if not any(".in_proj_z." in n for n in converted):
            raise ValueError("Export omitted existing Z targets")
        target = destination / role
        target.mkdir(parents=True, exist_ok=False)
        # Explicit rollout precision; HF master adapter parameters stay FP32.
        save_file({n: t.to(torch.bfloat16).contiguous() for n, t in converted.items()},
                  target / "adapter_model.safetensors", metadata={"format": "pt"})
        (target / "adapter_config.json").write_bytes((source / role / "adapter_config.json").read_bytes())
        manifest["roles"][role] = {
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "rollout_sha256": hashlib.sha256((target / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "source_tensors": len(tensors), "rollout_tensors": len(converted), "dtype": "bfloat16"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive = root / "adapters.tar"
    with tarfile.open(archive, "w") as tar:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(destination)))
    return archive, manifest


class RolloutClient:
    def __init__(self, shared, base_url, export_dir, timeout=900):
        self.shared, self.base_url = shared, base_url.rstrip("/")
        self.export_dir = Path(export_dir)
        self.timeout, self.published = timeout, None

    def request(self, path, payload):
        import httpx
        try:
            with httpx.Client(trust_env=False, timeout=self.timeout) as client:
                response = client.post(self.base_url + path, json=payload)
                response.raise_for_status()
                value = response.json()
            if value.get("version") != self.shared.version:
                raise InfrastructureError("Rollout adapter version mismatch")
            return value
        except (httpx.HTTPError, ValueError) as exc:
            raise InfrastructureError(f"vLLM RPC {path} failed: {type(exc).__name__}") from exc

    def publish(self):
        import httpx
        if self.published == self.shared.version:
            return
        started = time.monotonic()
        # A readiness wait is safe before any policy action is sampled. Do not
        # automatically retry generation requests with unknown completion.
        while True:
            try:
                with httpx.Client(trust_env=False, timeout=5) as client:
                    health = client.get(self.base_url + "/health")
                    if health.status_code == 200 and health.json().get("ready"):
                        break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() - started > self.timeout:
                raise InfrastructureError("vLLM service readiness timed out before rollout")
            time.sleep(5)
        qkv_layout = health.json().get("qkv_layout")
        archive, manifest = export_rollout_adapters(
            self.shared, self.export_dir / self.shared.version, qkv_layout=qkv_layout)
        with httpx.Client(trust_env=False, timeout=self.timeout) as client, archive.open("rb") as stream:
            response = client.post(self.base_url + "/publish", content=stream,
                                   headers={"Content-Length": str(archive.stat().st_size),
                                            "Content-Type": "application/x-tar"})
            response.raise_for_status()
            result = response.json()
        if result.get("manifest") != manifest:
            raise InfrastructureError("Remote adapter transfer/hash verification failed")
        self.published = self.shared.version
        log = {"version": self.shared.version, "bytes": archive.stat().st_size,
               "seconds": time.monotonic() - started, "manifest": manifest}
        (archive.parent / "publication.json").write_text(json.dumps(log, indent=2))


class RemoteVLLMGenerator:
    mixed_backend = True

    def __init__(self, local, client):
        self.local, self.client, self.shared = local, client, local.shared
        self.max_input_tokens, self.limits = local.max_input_tokens, local.limits
        self.temperature, self.prompt_profile = local.temperature, local.prompt_profile
        self.generation_progress = None

    def input_limit(self, role):
        return self.local.input_limit(role)

    def prepare_messages(self, role, messages):
        return self.local.prepare_messages(role, messages)

    def recompute(self, records):
        return self.local.recompute(records)

    def generate(self, role, messages, seed, deterministic=False):
        ids, prompt_hash, _ = encode_messages(self.shared.tokenizer,
            self.prepare_messages(role, messages), self.input_limit(role))
        self.client.publish()
        result = self.client.request("/generate", {"role": role, "version": self.shared.version,
            "input_ids": list(ids), "max_tokens": self.limits[role],
            "seed": int(seed) % (2**31), "deterministic": deterministic})
        tokens, logprobs = tuple(result["token_ids"]), tuple(result["logprobs"])
        if not tokens or len(tokens) != len(logprobs):
            raise InfrastructureError("Malformed rollout token/logprob alignment")
        text = self.shared.tokenizer.decode(tokens, skip_special_tokens=True)
        return GenerationRecord(role, ids, prompt_hash, tokens, (True,) * len(tokens),
            logprobs, text, 1.0, "eos" if result["finish_reason"] == "stop" else "length",
            self.shared.version)


class RemoteDirectChoicePolicy(DirectChoicePolicy):
    def __init__(self, local, client):
        self.__dict__.update(local.__dict__)
        self.client = client

    def prepare(self, turn, hits, scope, fixed=False):
        import torch
        candidates = self.pack(turn, hits)
        ids, prompt_hash, prompt = encode_messages(self.shared.tokenizer,
            choice_messages(turn, candidates, self.token_budget, self.max_selected), self.max_input_tokens)
        slots = slot_ids(self.shared.tokenizer, prompt, ids, len(candidates))
        mask = legal_mask(candidates, (), scope, MemoryRenderer(), self.tokenizer, self.token_budget)
        do_forward = sum(mask) > 1 and not fixed
        if do_forward:
            self.client.publish()
            result = self.client.request("/scores", {"role": "R", "version": self.shared.version,
                "input_ids": list(ids), "slot_ids": list(slots)})
            old = tuple(result["scores"])
            if len(old) != len(candidates):
                raise InfrastructureError("Missing R candidate probability")
        else:
            old = (0.0,) * len(candidates)
        logits = torch.tensor(old, dtype=torch.float32, device=self.shared.device)
        record = ChoiceForward(str(uuid.uuid4()), ids, prompt_hash, slots, candidates, old,
            probabilities(old, self.temperature), self.temperature, self.shared.version, int(do_forward),
            scope, MemoryRenderer.version, self.token_budget, self.max_selected,
            tuple(hit.entry for hit in hits), {"profile": "whole-entry-recall-order-v1",
                "source_hashes": {hit.entry.id: digest(hit.entry) for hit in hits},
                "visible_hashes": {e.id: digest(e) for e in candidates}})
        return record, logits
