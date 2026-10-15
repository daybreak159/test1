from __future__ import annotations

from .config import protocol_hash, require_hf_backend, resolved
from .contracts import Budget, ConsumerProfile
from .data.manifest import audit_manifests, load_manifest
from .domains.alf import ALFAdapter
from .domains.clients import ChatModel, JSONClient, RemoteGrader
from .domains.offline_web import OfflineWebAdapter, OfflineWebGrader
from .domains.qa import AnswerGrader, QAAdapter
from .domains.swe import SWEAdapter
from .domains.web import WebAdapter
from .memory.recall import QwenEmbedder, RemoteQwenEmbedder
from .memory.transaction import TransactionExecutor
from .policies.direct_choice import DirectChoicePolicy
from .policies.generator import GeneratorPolicy
from .policies.shared_qwen import SharedQwen
from .rewards.pair_judge import JUDGE_RESPONSE_FORMAT, PairJudge, PairedQuality
from .rollout.engine import MemoryEngine
from .rollout.main_stream import MainStream
from .rollout.paired_probe import PairedProbe
from .storage import ResourceLog
from .training.checkpoint import CheckpointStore
from .training.coordinator import Coordinator
from .util import read_json


def make_adapter(cfg, resources=None):
    c = cfg["consumer"]
    p = c["profile"]
    profile = ConsumerProfile(**{**p, "budget": Budget(**p["budget"]), "tools": tuple(p["tools"])})
    consumer = ChatModel(
        JSONClient(c["base_url"], c.get("api_key_env")),
        p["model"],
        p["model_revision"],
        p["decoding"],
        c.get("pricing"),
        resources=resources,
    )
    grader_cfg = cfg["grader"]
    if grader_cfg["kind"] == "remote":
        grader = RemoteGrader(
            JSONClient(grader_cfg["base_url"], grader_cfg.get("api_key_env"), grader_cfg.get("timeout",180)), grader_cfg["version"]
        )
    else:
        labels = read_json(resolved(cfg, cfg["data"]["private_labels"]))["labels"]
        if grader_cfg["kind"] == "custom_em_f1":
            grader = AnswerGrader(labels, grader_cfg["version"], grader_cfg["quality_metric"])
        elif grader_cfg["kind"] == "offline_action":
            grader = OfflineWebGrader(labels, grader_cfg["version"])
        elif grader_cfg["kind"] == "locomo_compatible":
            from .domains.answer_profiles import LoCoMoGrader

            grader = LoCoMoGrader(labels, grader_cfg["version"], grader_cfg["quality_metric"])
        else:
            raise ValueError("Unknown grader; official benchmark profiles use a versioned grader service")
    env = cfg["environment"]
    environment = JSONClient(env["base_url"], env.get("api_key_env")) if env["base_url"] else None
    cls = {
        "swe": SWEAdapter,
        "web": WebAdapter,
        "qa": QAAdapter,
        "alf": ALFAdapter,
        "offline_web": OfflineWebAdapter,
    }[cfg["domain"]]
    adapter = cls(consumer, profile, grader, environment)
    adapter.official_outcome_feedback = cfg.get("memory_feedback", {}).get("protocol") == "swe-official-outcome-v1"
    if adapter.official_outcome_feedback and cfg["domain"] != "swe":
        raise ValueError("Scalar official outcome feedback is SWE-specific")
    return adapter


class Runtime:
    def __init__(self, cfg, with_optimizer=True):
        require_hf_backend(cfg)
        import random
        import torch
        from transformers import AutoTokenizer

        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        import numpy as np

        np.random.seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
        self.cfg = cfg
        self.resources = ResourceLog()
        architecture = dict(cfg["architecture"])
        architecture["adapter_layout"] = cfg["experiment"].get("adapter_layout", "independent")
        if cfg["runtime"].get("hf_pointwise_chunk_tokens") is not None:
            architecture["pointwise_chunk_tokens"] = cfg["runtime"]["hf_pointwise_chunk_tokens"]
        self.shared = SharedQwen.load(architecture)
        self.shared.resources = self.resources
        if with_optimizer:
            for key in ("adapter_gradient_audit", "old_new_logprob_audit"):
                if cfg["runtime"].get(key):
                    audit = read_json(resolved(cfg, cfg["runtime"][key]))
                    if audit.get("base_hash") != self.shared.base_hash:
                        raise ValueError("Model audit was produced with different frozen base weights")
        self.generator = GeneratorPolicy(self.shared, **{
            **cfg["generation"],
            "release_unused_cache":cfg["runtime"].get("release_unused_cache",False),
        })
        profile = cfg["consumer"]["profile"]
        consumer_tokenizer = AutoTokenizer.from_pretrained(
            profile["tokenizer"], revision=profile["tokenizer_revision"], trust_remote_code=False
        )
        selector_cfg = dict(cfg["selector"])
        for key in ("max_selected", "candidate_slots"):
            if key in cfg["experiment"]:
                selector_cfg[key] = cfg["experiment"][key]
        self.selector = DirectChoicePolicy(self.shared, consumer_tokenizer, **selector_cfg)
        self.transactions = TransactionExecutor(
            self.shared.tokenizer, consumer_tokenizer, self.shared.tokenizer, **cfg["memory"]
        )
        r = cfg["retrieval"]
        if r.get("backend") == "openai_compatible":
            self.embedder = RemoteQwenEmbedder(
                r["model"], r["revision"], r["base_url"], r["served_model"],
                r.get("api_key_env"), r["max_tokens"], r["batch_size"], r.get("timeout",120))
        else:
            self.embedder = QwenEmbedder(r["model"], r["revision"], r["device"], r["max_tokens"], r["batch_size"])
        self.embedder.resources = self.resources
        self.engine = MemoryEngine(
            self.generator,
            self.selector,
            self.transactions,
            self.embedder,
            r["instructions"],
            r["management_k"],
            cfg["experiment"].get("candidate_slots", r["consumer_k"]),
            r["management_query_tokens"],
            self.resources,
            cfg.get("source_packing", "reject"),
        )
        self.adapter = make_adapter(cfg, self.resources)
        if cfg.get("e_preprocessing", {}).get("profile") == "swe-evidence-hybrid-v1":
            if cfg["domain"] != "swe":
                raise ValueError("Hybrid E preprocessing is SWE-specific")
            from .data.runtime_hybrid import HybridSourceProcessor
            hybrid_config = dict(cfg["e_preprocessing"]["config"])
            generation_profile = cfg["generation"].get("prompt_profile", "e-text-v1")
            hybrid_config.setdefault("prompt_profile", generation_profile)
            if hybrid_config["prompt_profile"] != generation_profile:
                raise ValueError("E preprocessing and generation prompt profiles must match")
            self.hybrid_processor = HybridSourceProcessor(self.shared.tokenizer,
                hybrid_config, resolved(cfg, cfg["e_preprocessing"]["output_dir"]))
            original_memory_source = self.adapter.memory_source
            def hybrid_memory_source(task, execution, grade):
                source = original_memory_source(task, execution, grade)
                return self.hybrid_processor(task, execution, source)
            self.adapter.memory_source = hybrid_memory_source
        self.data = load_manifest(resolved(cfg, cfg["data"]["public_manifest"]), self.adapter)
        self.data_audit = audit_manifests([self.data], resolved(cfg, cfg["data"]["duplicate_manifest"]))
        global_args = {k: v for k, v in cfg["global"].items() if k not in {"replicas", "debug_zero_baseline"}}
        retry = cfg["runtime"]["retry_attempts"]
        self.stream = MainStream(self.engine, self.adapter, **global_args, retry_attempts=retry)
        j = cfg["judge"]
        judge = (
            PairJudge(
                ChatModel(
                    JSONClient(j["base_url"], j.get("api_key_env")),
                    j["model"],
                    j["revision"],
                    {**j["decoding"], **({'response_format': JUDGE_RESPONSE_FORMAT}
                       if j.get('structured_output', False) else {})},
                    pricing=j.get("pricing"),
                    resources=self.resources,
                    usage_role="judge",
                ),
                j["rubric_version"],
                self.resources,
                evidence_profile=j.get("evidence_profile", "raw-v1"),
                repair_attempts=j.get('repair_attempts', 0),
                audit_path=resolved(cfg, cfg['output_dir']) / 'judge_format_audit.jsonl',
            )
            if j["base_url"]
            else None
        )
        quality = PairedQuality(self.adapter, judge)
        # Same frozen profile/weights/decoding; only the transport address differs.
        from copy import deepcopy
        self.local_adapters = [self.adapter]
        for url in cfg.get('local_solver_endpoints', []):
            if url.rstrip('/') == cfg['consumer']['base_url'].rstrip('/'):
                raise ValueError('Duplicate local Solver endpoint')
            worker_cfg = deepcopy(cfg)
            worker_cfg['consumer']['base_url'] = url
            self.local_adapters.append(make_adapter(worker_cfg, self.resources))
        local = {k: v for k, v in cfg["local"].items()
                 if k not in {"beta", "anchors_per_replica", "budget", "min_legal_candidates"}}
        reference = cfg["experiment"].get("old_reference", True)
        self.probe = PairedProbe(
            self.engine,
            self.adapter,
            quality,
            **local,
            budget=Budget(**cfg["local"]["budget"]),
            retry_attempts=retry,
            credit=cfg["experiment"].get("credit", "conditional"),
            reference=reference,
            selector_mode=cfg["experiment"].get("selector_mode", "learned"),
            scoring=cfg["experiment"].get("local_score", "paired"),
            execution_adapters=self.local_adapters,
        )
        checkpoint_root = (
            resolved(cfg, cfg["checkpoint_dir"]) / cfg["experiment"]["name"] / f"seed-{cfg['seed']}"
        )
        self.checkpoints = CheckpointStore(
            checkpoint_root, cfg["domain"], cfg["architecture"]["base_revision"], protocol_hash(cfg)
        )
        self.coordinator = (
            Coordinator(
                self.shared,
                self.generator,
                self.selector,
                cfg["domain"],
                self.checkpoints,
                **cfg["optimization"],
                probability_diagnostics_path=resolved(cfg, cfg["output_dir"]) / "probability_diagnostics.jsonl",
                train_roles=cfg["experiment"]["train_roles"],
            )
            if with_optimizer
            else None
        )
