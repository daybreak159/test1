from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

from .util import ConfigurationError, atomic_json, digest, file_hash, read_json

DOMAINS = {"swe", "web", "qa", "alf", "offline_web"}


def starter_config(domain):
    if domain not in DOMAINS:
        raise ValueError(domain)
    return {
        "schema_version": "eur-runtime-v3.2",
        "domain": domain,
        "seed": 11,
        "output_dir": f"runs/{domain}/seed-11",
        "checkpoint_dir": "checkpoints",
        "architecture": {
            "base_model": "Qwen/Qwen3.5-4B",
            "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "target_modules": None,
            "gradient_checkpointing": True,
        },
        "retrieval": {
            "model": "Qwen/Qwen3-Embedding-8B",
            "revision": None,
            "device": "cuda:1",
            "max_tokens": 8192,
            "batch_size": 1,
            "dimension": 4096,
            "management_k": 8,
            "consumer_k": 12,
            "management_query_tokens": 2048,
            "instructions": {
                "management": f"Retrieve existing {domain} memories to consolidate with these new observations.",
                "consumer": f"Retrieve applicable {domain} observations that help solve the current task.",
            },
        },
        "memory": {
            "bank_tokens": 8192,
            "entry_consumer_tokens": 1024,
            "entry_selector_tokens": 1536,
            "max_operations": 4,
        },
        "generation": {
            "prompt_profile": "e-text-v1",
            "max_input_tokens": 8192,
            "input_limits": {"E": 65536, "U": 8192} if domain == "swe" else {},
            "limits": {"E": 384, "U": 768}, "temperature": 1.0,
            # Single-step PG accepts finite cached/recompute drift as a diagnostic.
            "use_generation_cache": True,
        },
        "source_packing": "compact_exact_v1" if domain == "swe" else "reject",
        "rollout": {"backend": "hf"},
        "selector": {
            "max_input_tokens": 6144,
            "token_budget": 2048,
            "max_selected": 4,
            "candidate_slots": 12,
            "temperature": 1.0,
        },
        "global": {
            "replicas": 2,
            "cost_lambda": 0.01,
            "bad_write_lambda": 1.0,
            "cost_budget": 100000,
            "debug_zero_baseline": False,
        },
        "local": {
            "anchors_per_replica": 1,
            "I": 2,
            "J": 2,
            "K": 2,
            "L": 2,
            "B": 1,
            "invalid_score": -1.0,
            "beta": 0.1,
            "budget": {"model_calls": 8, "output_tokens": 8192, "command_timeout_seconds": 120},
        },
        "optimization": {
            "lr": {"E": 1e-5, "U": 1e-5, "R": 1e-5},
            "microbatch_size": 1,
            "divisor": 768,
            "epsilon": 0.2,
            "max_grad_norm": 1.0,
            "behavior_tolerance": 0.03,
            "policy_loss": "single_step_pg",
        },
        "schedule": {
            "batches": 100,
            "plans_per_batch": 1,
            "dev_interval": 10,
            "max_excluded_groups": 20,
            "seeds": [11, 23, 37, 51, 71],
            "sft_epochs_max": 1,
            "R_warmup_batches": 0,
        },
        "data": {
            "public_manifest": None,
            "private_labels": None,
            "duplicate_manifest": None,
            "public_manifest_sha256": None,
            "private_labels_sha256": None,
            "duplicate_manifest_sha256": None,
            "answer_and_media_profile": None,
        },
        "consumer": {
            "base_url": None,
            "api_key_env": None,
            "pricing": {"input_per_million": 0, "output_per_million": 0},
            "profile": {
                "backend": "openai_compatible",
                "model": None,
                "model_revision": None,
                "harness_revision": "eur-tool-loop-v1",
                "tools": [] if domain in {"qa", "offline_web"} else [domain],
                "memory_injection": "replace_active_system_reference_block",
                "decoding": {"temperature": 0},
                "budget": {
                    "model_calls": 64 if domain not in {"qa", "offline_web"} else 1,
                    "output_tokens": 32768 if domain not in {"qa", "offline_web"} else 512,
                    "command_timeout_seconds": 120,
                },
                "reset_method": "independent_readonly_query"
                if domain in {"qa", "offline_web"}
                else "fresh_full_environment",
                "tokenizer": None,
                "tokenizer_revision": None,
            },
        },
        "environment": {"base_url": None, "api_key_env": None, "image_manifest": None, "reset_audit": None},
        "grader": {
            "kind": "locomo_compatible"
            if domain == "qa"
            else "offline_action"
            if domain == "offline_web"
            else "remote",
            "version": None,
            "quality_metric": "f1",
            "base_url": None,
            "api_key_env": None,
        },
        "judge": {
            "base_url": None,
            "model": None,
            "revision": None,
            "rubric_version": None,
            "api_key_env": None,
            "decoding": {"temperature": 0},
        },
        "runtime": {
            "dependency_lock": None,
            "adapter_gradient_audit": None,
            "old_new_logprob_audit": None,
            "retry_attempts": 3,
        },
        "experiment": {
            "name": "full_EUR",
            "adapter_layout": "independent",
            "objective": "dual",
            "train_roles": ["E", "U", "R"],
            "memory_enabled": True,
            "selector_mode": "learned",
            "credit": "conditional",
            "local_score": "paired",
            "old_reference": True,
        },
    }


def require_hf_backend(cfg):
    """The current runtime uses HF for both sampling and differentiable updates."""
    if cfg.get("rollout", {}).get("backend", "hf") != "hf":
        raise ConfigurationError("Current runtime requires rollout.backend=hf; vLLM rollout is retired")
    if cfg.get("optimization", {}).get("rollout_correction"):
        raise ConfigurationError("HF-only runtime does not use HF/vLLM rollout_correction")
    if cfg.get("optimization", {}).get("policy_loss", "single_step_pg") not in {"single_step_pg", "ppo"}:
        raise ConfigurationError("optimization.policy_loss must be single_step_pg or ppo")


def load_config(path):
    cfg = read_json(path)
    if cfg.get("schema_version") != "eur-runtime-v3.2":
        raise ConfigurationError(
            "Use an executable runtime config; the reference Pilot JSON is a design document"
        )
    if cfg.get("domain") not in DOMAINS:
        raise ConfigurationError("Exactly one known domain per run")
    cfg.setdefault("rollout", {}).setdefault("backend", "hf")
    cfg.setdefault("optimization", {}).setdefault("policy_loss", "single_step_pg")
    cfg.setdefault("generation", {}).setdefault("use_generation_cache", True)
    require_hf_backend(cfg)
    profile = cfg.setdefault("generation", {}).setdefault("prompt_profile", "e-text-v1")
    if cfg.get("e_preprocessing", {}).get("profile") == "swe-evidence-hybrid-v1":
        cfg["e_preprocessing"].setdefault("config", {}).setdefault("prompt_profile", profile)
    cfg["_config_dir"] = str(Path(path).resolve().parent)
    return cfg


def resolved(cfg, value):
    path = Path(value)
    return path if path.is_absolute() else Path(cfg.get("_config_dir", ".")) / path


def protocol_hash(cfg):
    # Identity includes data, models, templates, costs and experimental behavior. Paths/cursors do not alter it.
    return digest(
        {
            k: v
            for k, v in cfg.items()
            if k not in {"_config_dir", "output_dir", "checkpoint_dir", "schedule", "runtime", "sft"}
        }
    )


def preflight(cfg, mode="train", check_installed=False):
    errors, checks = [], []

    def require(path, predicate=bool, explanation="must be configured"):
        value = cfg
        for key in path.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if not predicate(value):
            errors.append(f"{path}: {explanation}")
        else:
            checks.append(path)
        return value

    require(
        "architecture.target_modules",
        lambda v: isinstance(v, list) and bool(v) and len(v) == len(set(v)),
        "needs exact audited module names, not assumed q_proj suffixes",
    )
    for path in ("architecture.base_revision", "retrieval.revision"):
        require(
            path,
            lambda v: isinstance(v, str) and bool(re.fullmatch(r"[a-f0-9]{40}", v)),
            "needs pinned commit",
        )
    require("retrieval.dimension", lambda v: v == 4096, "must use full Qwen3-Embedding-8B dimension")
    require(
        "selector.candidate_slots",
        lambda v: type(v) is int and 1 <= v <= 16,
        "direct choice has at most 16 slots",
    )
    require("selector.max_selected", lambda v: type(v) is int and 1 <= v <= 16)
    require("optimization.divisor", lambda v: v == 768, "E/U/R share c=768")
    require("generation.temperature", lambda v: v == 1)
    for path in (
        "consumer.base_url",
        "consumer.profile.model",
        "consumer.profile.model_revision",
        "consumer.profile.harness_revision",
        "consumer.profile.tokenizer",
        "consumer.profile.tokenizer_revision",
        "consumer.profile.memory_injection",
        "consumer.profile.reset_method",
        "grader.version",
        "data.answer_and_media_profile",
    ):
        require(path)
    for field in ("public_manifest", "private_labels", "duplicate_manifest"):
        path = require("data." + field)
        expected = require("data." + field + "_sha256")
        if path:
            actual = resolved(cfg, path)
            if not actual.is_file():
                errors.append(f"Missing {field}: {actual}")
            elif expected and file_hash(actual) != expected:
                errors.append(f"Manifest hash changed: {field}")
    lock_path = require("runtime.dependency_lock")
    if lock_path:
        path = resolved(cfg, lock_path)
        if not path.is_file():
            errors.append("Dependency lock file is missing")
        elif check_installed:
            lock = read_json(path)
            for name, expected in lock.get("packages", {}).items():
                try:
                    actual = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    actual = "not-installed"
                if actual != expected:
                    errors.append(f"Dependency drift: {name}={actual}, expected {expected}")
            if not {"torch", "transformers", "peft", "accelerate"} <= lock.get("packages", {}).keys():
                errors.append("Dependency lock lacks the training backend")
    executable = cfg["domain"] in {"swe", "web", "alf"}
    if executable:
        for path in ("environment.base_url", "environment.image_manifest", "environment.reset_audit"):
            require(path)
        for key in ("image_manifest", "reset_audit"):
            value = cfg["environment"].get(key)
            if value:
                path = resolved(cfg, value)
                if not path.is_file():
                    errors.append(f"Missing environment artifact: {key}")
                elif key == "reset_audit":
                    audit = read_json(path)
                    if not audit.get("passed") or not audit.get("complete_state_verified"):
                        errors.append("Real complete-state paired reset audit has not passed")
    if cfg["grader"]["kind"] == "remote":
        require("grader.base_url")
    if mode == "train":
        if cfg["domain"] == "offline_web":
            errors.append("Mind2Web is transfer-only under the primary protocol")
        require(
            "global.replicas",
            lambda v: type(v) is int and (v >= 2 or (v == 1 and cfg["global"].get("debug_zero_baseline"))),
        )
        if cfg["experiment"].get("objective") != "outcome_only" and cfg["local"]["anchors_per_replica"]:
            for key in ("I", "J", "L"):
                require("local." + key, lambda v: type(v) is int and v >= 2)
            if (
                executable
                and cfg["experiment"].get("local_score") == "paired"
                and cfg["experiment"].get("old_reference", True)
            ):
                for key in ("base_url", "model", "revision", "rubric_version"):
                    require("judge." + key)
        for name in ("adapter_gradient_audit", "old_new_logprob_audit"):
            audit_path = require("runtime." + name)
            if audit_path:
                path = resolved(cfg, audit_path)
                if not path.is_file():
                    errors.append(f"Missing audit artifact: {name}")
                else:
                    audit = read_json(path)
                    if not audit.get("passed") or audit.get("protocol_hash") != protocol_hash(cfg):
                        errors.append(f"Invalid/stale audit artifact: {name}")
    return {
        "ready": not errors,
        "mode": mode,
        "errors": errors,
        "checks": checks,
        "protocol_hash": protocol_hash(cfg),
        "server_validation_performed": False,
    }


def require_preflight(cfg, mode="train"):
    report = preflight(cfg, mode, check_installed=True)
    if not report["ready"]:
        raise ConfigurationError("Preflight blocked:\n" + "\n".join(report["errors"]))
    return report


def lock_environment(output):
    import platform

    names = {
        d.metadata["Name"].lower().replace("_", "-")
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    # A dependency fallback path can contain another version of the same distribution.
    # Pin the version actually selected by import resolution, not the last enumerated copy.
    packages = {name: importlib.metadata.version(name) for name in names}
    atomic_json(
        output, {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
    )
