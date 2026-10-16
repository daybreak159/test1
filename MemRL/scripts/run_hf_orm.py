"""HF-only task-flow runner; emits startup, training and frozen-test evidence."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from threading import RLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eur_memory_rl.util import atomic_json, plain
from eur_memory_rl.config import load_config, preflight

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--full-suite", action="store_true", help="Train declared batches and rebuild a frozen main-task bank")
p.add_argument('--resume-checkpoint', type=Path, help='Restore same-protocol RL weights, optimizers, RNG and cursor')
p.add_argument("--sft-checkpoint", type=Path, help="Continue the same E/U/R weights from a trusted SWE SFT checkpoint")
p.add_argument("--sft-weights-only", action="store_true", help="Import SWE adapter weights into a new protocol, resetting optimizer and cursor")
p.add_argument("--startup-only", action="store_true", help="Check model, runtime, 32K backward and HF sampling, then stop before task rollouts or optimizer updates")
p.add_argument("--max-updates", type=int, help="Stop after this cumulative update count; defer tests until all groups finish")
a = p.parse_args()
if a.resume_checkpoint and (a.sft_checkpoint or a.sft_weights_only or a.startup_only):
    p.error('RL resume cannot be combined with SFT initialization or startup-only')
root = a.root
started = time.monotonic()
summary = {"status": "running", "completed_updates": 0}
log_lock = RLock()


def stage(name, **kw):
    with log_lock:
        summary.update(stage=name, wall_seconds=time.monotonic() - started, **kw)
        atomic_json(root / "summary.json", summary)
        print("STAGE", name, flush=True)


def record(name, value):
    with log_lock:
        with (root / (name + ".jsonl")).open("a") as f:
            f.write(json.dumps(plain(value), ensure_ascii=False) + "\n")


try:
    import torch

    from eur_memory_rl.training.attention import hf_attention_backend

    cfg = load_config(root / "config.json")
    attention_backend = cfg.get("runtime", {}).get("hf_attention_backend", "efficient")
    atomic_json(root / "hf_attention_backend.json", {"backend": attention_backend})
    stage("model_audit")
    from eur_memory_rl.policies.shared_qwen import SharedQwen
    from eur_memory_rl.policies.generator import GeneratorPolicy
    from eur_memory_rl.training.audits import model_audit
    from eur_memory_rl.runtime import Runtime
    from eur_memory_rl.training.trainer import Trainer

    architecture = dict(cfg["architecture"], adapter_layout=cfg["experiment"].get("adapter_layout", "independent"))
    if cfg["runtime"].get("hf_pointwise_chunk_tokens") is not None:
        architecture["pointwise_chunk_tokens"] = cfg["runtime"]["hf_pointwise_chunk_tokens"]
    shared = SharedQwen.load(architecture)
    generator = GeneratorPolicy(shared, **cfg["generation"])
    with hf_attention_backend(attention_backend):
        model_audit(shared, generator, cfg, root / "model_audit.json")
    import gc

    del shared, generator
    gc.collect()
    torch.cuda.empty_cache()
    report = preflight(cfg, "train", check_installed=True)
    atomic_json(root / "preflight.json", report)
    if not report["ready"]:
        raise ValueError(report["errors"])
    stage("runtime_loading")
    rt = Runtime(cfg)
    if a.resume_checkpoint:
        initial = a.resume_checkpoint
    elif a.sft_checkpoint:
        # Strict protocol/base/data checks are performed by Trainer.resume below.
        initial = a.sft_checkpoint
        manifest = json.loads((initial / "manifest.json").read_text())
        if manifest["domain"] != "swe":
            raise ValueError("SWE RL cannot inherit an ALF checkpoint")
        if a.sft_weights_only:
            rt.checkpoints.load(rt.shared, checkpoint=initial, restore_rng=False, strict_protocol=False)
            initial = rt.checkpoints.publish(rt.shared, rt.coordinator.optimizers, 0,
                {"cursor": 0, "main_batches": 0, "manifest_hash": rt.data.manifest_hash, "stage": "sft"},
                {"initialization": "SWE SFT weights only; new protocol; fresh optimizer, cursor and bank",
                 "source_checkpoint": str(a.sft_checkpoint)})
    else:
        initial = rt.checkpoints.publish(
            rt.shared, rt.coordinator.optimizers, 0,
            {"cursor": 0, "main_batches": 0, "manifest_hash": rt.data.manifest_hash, "stage": "rl"},
            {"initialization": "fresh E/U/R LoRAs; no reused solver rollout or reward"},
        )
    if not a.resume_checkpoint:
        atomic_json(
            root / "initialization.json", {"initial_checkpoint": str(initial), "sft": bool(a.sft_checkpoint)}
        )
    trainer = Trainer(rt)
    trainer.resume(str(initial))
    stage("loaded_sft_model_audit")
    with hf_attention_backend(attention_backend):
        model_audit(rt.shared, rt.generator, cfg, root / "loaded_sft_model_audit.json")
    stage("hf_policy_ready", rollout_backend="hf", training_backend="hf")
    models = [("solver", rt.adapter.consumer), ("pair_judge", rt.probe.quality.judge.model)]
    models.extend((f"solver_worker_{i}", a.consumer) for i, a in enumerate(rt.local_adapters[1:], 1))
    for label, model in models:
        original = model.client.post

        def posted(path, payload, request_id=None, _post=original, _label=label):
            record("api_requests", {"kind": _label, "payload": payload, "request_id": request_id})
            t = time.monotonic()
            result = _post(path, payload, request_id)
            record("api_responses", {"kind": _label, "response": result, "seconds": time.monotonic() - t})
            return result

        model.client.post = posted
        model.client.timeout = 900
    original_generate = rt.generator.generate

    def generated(role, messages, seed, deterministic=False):
        stage("generation", role=role)
        t = time.monotonic()
        result = original_generate(role, messages, seed, deterministic)
        record("generations", {"record": result, "seconds": time.monotonic() - t})
        return result

    rt.generator.generate = generated
    original_extract = rt.engine.extract

    def extracted(*args, **kwargs):
        result = original_extract(*args, **kwargs)
        record("memory_validation", {"role": "E", "record": result[0], "candidates": result[1]})
        return result

    rt.engine.extract = extracted
    original_update = rt.engine.update

    def managed(*args, **kwargs):
        result = original_update(*args, **kwargs)
        record(
            "memory_validation",
            {"role": "U", "record": result[0], "transaction": result[1], "metadata": result[2]},
        )
        return result

    rt.engine.update = managed
    def replayed(task, execution_id, seed, read, budget=None):
        replay = cfg.get("first_task_replay_by_task", {}).get(task.id) or cfg.get("first_task_replay")
        if replay and task.id == replay["task_id"] and task.identity.split == "train":
            from eur_memory_rl.data.replay_swe import replay_first_execution
            result = replay_first_execution(rt, task, execution_id, seed, read, replay)
            record("reused_executions", {"execution_id": execution_id, "source": replay,
                "solver_calls_this_run": 0, "reward_source": "new official evaluation of actual historical patch",
                "acceptance_only": False, "not_on_policy_solver_rollout": True})
            return result
        return None

    from eur_memory_rl.rollout.solver_audit import install_solver_audit

    install_solver_audit(rt.local_adapters, record, stage, replay=replayed)
    atomic_json(root / "solver_parallelism.json", {
        "main_parallel_workers": rt.stream.main_parallel_workers,
        "endpoints": [adapter.consumer.client.base_url for adapter in rt.local_adapters],
        "hf_policy_calls_serialized": True,
        "update_barrier": "all main replicas and selected local trees completed",
    })
    stage("long_input_precheck")
    # Only a scoring/backward capacity check; no synthetic optimizer update.
    from eur_memory_rl.contracts import GenerationRecord
    from eur_memory_rl.util import digest

    raw = rt.shared.tokenizer.encode("Public code observation. Check empty query behavior. ")
    ids = tuple((raw * (32768 // len(raw) + 1))[:32768])
    token = rt.shared.tokenizer.encode("{")[0]
    rec = GenerationRecord(
        "E", ids, digest(ids), (token,), (True,), (0.0,), "{", 1.0, "length", rt.shared.version
    )
    t = time.monotonic()
    torch.cuda.reset_peak_memory_stats(rt.shared.device)
    with rt.shared.activate("E", train=True), hf_attention_backend(attention_backend):
        vals, _ = rt.generator.recompute([rec])
        loss = -vals[0].mean()
        loss.backward()
    norms = {
        r: sum(float(p.grad.float().square().sum()) for _, p in g if p.grad is not None) ** 0.5
        for r, g in rt.shared.role_parameters.items()
    }
    for o in rt.coordinator.optimizers.values():
        o.zero_grad(set_to_none=True)
    atomic_json(
        root / "long_input_precheck.json",
        {
            "input_tokens": len(ids),
            "seconds": time.monotonic() - t,
            "gradient_norms": norms,
            "peak_bytes": torch.cuda.max_memory_allocated(rt.shared.device),
            "activation_offload": rt.shared.activation_offload,
            "optimizer_steps": 0,
        },
    )
    stage("rollout_smoke")
    from eur_memory_rl.policies.prompts import encode_messages, slot_ids

    smoke_messages = [
        {"role": "system", "content": "Choose a declared option letter. No explanations."},
        {"role": "user", "content": "Select a reusable debugging tip from options A through P."},
    ]
    smoke_ids, _, smoke_prompt = encode_messages(rt.shared.tokenizer, smoke_messages, 6144)
    smoke_slots = slot_ids(rt.shared.tokenizer, smoke_prompt, smoke_ids, 16)
    from eur_memory_rl.policies.direct_choice import forward_option_logits

    def probe_scores():
        inputs = torch.tensor([smoke_ids], dtype=torch.long, device=rt.shared.device)
        with rt.shared.activate("R"), torch.no_grad(), hf_attention_backend(attention_backend):
            logits = forward_option_logits(rt.shared.model, inputs, torch.ones_like(inputs), smoke_slots)
        return {"backend": "hf", "version": rt.shared.version, "logits": logits[0].cpu().tolist()}

    initial_scores = probe_scores()
    smoke = {"R": initial_scores}
    probe_generator = GeneratorPolicy(rt.shared, **{**cfg["generation"], "limits": {"E": 8, "U": 8}})
    for role in ("E", "U"):
        with hf_attention_backend(attention_backend):
            result = probe_generator.generate(role, smoke_messages, 123)
        smoke[role] = plain(result)
        if len(result.response_token_ids) != len(result.old_logprobs):
            raise ValueError("Rollout smoke probability alignment failed")
    atomic_json(root / "rollout_smoke.json", smoke)
    if a.startup_only:
        stage("startup_passed", status="startup_passed", completed_updates=0,
              task_rollouts=0, optimizer_steps=0, rollout_backend="hf", training_backend="hf")
        raise SystemExit(0)
    stage("training_batch")
    with hf_attention_backend(attention_backend):
        trainer.train(max_updates=a.max_updates)
    completed_updates = trainer.iteration
    update = json.loads((root / f"rl/update-{completed_updates:06d}.json").read_text())
    stage("hf_policy_updated", completed_updates=completed_updates, checkpoint=update["checkpoint"])
    updated_scores = probe_scores()
    atomic_json(root / "updated_rollout_probe.json", {"initial": initial_scores, "updated": updated_scores})
    all_updates = [json.loads((root / f'rl/update-{n:06d}.json').read_text())
                   for n in range(1, completed_updates + 1)]
    changed = sorted({role for item in all_updates
                      for role, norm in item['parameter_update_norms'].items() if norm > 0})
    atomic_json(
        root / "learning_signal.json",
        {
            "updated_roles": changed,
            "all_roles_updated": set(changed) == {"E", "U", "R"},
            "gradient_norms": update["gradient_norms"],
            "parameter_update_norms": update["parameter_update_norms"],
            "norms_scope": "last update; updated_roles aggregates all updates",
            "per_update": [{'iteration': n, 'gradient_norms': item['gradient_norms'],
                            'parameter_update_norms': item['parameter_update_norms']}
                           for n, item in enumerate(all_updates, 1)],
        },
    )
    if a.max_updates is not None and completed_updates < cfg["schedule"]["batches"]:
        stage("paused_after_update", status="paused_after_update", completed_updates=completed_updates,
              next_resume_update=completed_updates + 1, checkpoint=update["checkpoint"],
              frozen_tests_deferred=True)
        raise SystemExit(0)
    test_count = 0
    test_plans = [plan for plan in rt.data.plans if plan.split == "test"]
    if test_plans:
        if cfg['environment'].get('readiness_mode') == 'per-group-v1':
            from eur_memory_rl.data.environment_readiness import ensure_environments
            from eur_memory_rl.config import resolved
            environment = json.loads(resolved(cfg, cfg['environment']['image_manifest']).read_text())['swe']
            ensure_environments([event.task for plan in test_plans for event in plan.events if event.task],
                                environment, root / 'rl/environment_readiness.json')
        from eur_memory_rl.data.serde import bank as parse_bank
        from eur_memory_rl.training.checkpoint import adapter_state
        from eur_memory_rl.util import seed_for

        if a.full_suite:
            from eur_memory_rl.contracts import Bank, Execution
            from eur_memory_rl.data.serde import task as parse_task
            from eur_memory_rl.util import digest

            # Rebuild in fixed main-task order, using public replica-0 traces
            # and the final frozen policy. Only the enabled scalar outcome
            # profile can expose a saved official verdict, never private tests.
            bank = Bank(rt.data.plans[0].scope)
            rebuild = []
            for number in range(1, completed_updates + 1):
                rollout = json.loads((root / f"rl/rollout-{number:06d}.json").read_text())
                episode = rollout["main"][0]
                for event in episode["plan"]["events"]:
                    if not event["task"]:
                        continue
                    task = parse_task(event["task"])
                    saved = episode["executions"][task.id]
                    execution = Execution(**{**saved, "public_trace": tuple(saved["public_trace"])})
                    from eur_memory_rl.contracts import Grade
                    from eur_memory_rl.data.hybrid_trace import CompactionSkip
                    try:
                        source = rt.adapter.memory_source(task, execution, Grade(**episode["private_grades"][task.id]))
                    except CompactionSkip as exc:
                        rebuild.append({"task_id": task.id, "E": None, "U": None,
                                        "valid_transaction": False, "bank_hash": bank.hash,
                                        "preprocessing_skip": str(exc)})
                        continue
                    stage("rebuild_frozen_bank", task_id=task.id)
                    with hf_attention_backend(attention_backend):
                        e_record, candidates = rt.engine.extract(
                            source, seed_for(cfg["seed"], "rebuild", task.id, "E"), deterministic=True
                        )
                    if candidates is not None:
                        with hf_attention_backend(attention_backend):
                            u_record, transaction, _ = rt.engine.update(
                                source, bank, candidates,
                                seed_for(cfg["seed"], "rebuild", task.id, "U"), deterministic=True
                            )
                        bank = transaction.bank
                        rebuild.append({"task_id": task.id, "E": e_record, "U": u_record,
                                        "valid_transaction": transaction.valid, "bank_hash": bank.hash})
                    else:
                        rebuild.append({"task_id": task.id, "E": e_record, "U": None,
                                        "valid_transaction": False, "bank_hash": bank.hash})
            declared = [event.task.id for plan in rt.data.plans if plan.split == "train"
                        for event in plan.events if event.task]
            if [item["task_id"] for item in rebuild] != declared:
                raise ValueError("Frozen bank rebuilding changed declared training order")
            atomic_json(root / "bank_rebuild.json", {
                "records": rebuild, "task_order_hash": digest(declared),
                "source": "public main replica-0 traces in original order",
                "policy_version": rt.shared.version, "private_test_material_used": False,
                "scalar_official_feedback": getattr(rt.adapter, "official_outcome_feedback", False),
            })
        else:
            rollout = json.loads((root / "rl/rollout-000001.json").read_text())
            bank = parse_bank(rollout["main"][0]["bank"])
        atomic_json(root / "frozen_test_bank.json", bank)
        before = adapter_state(rt.shared)
        version = rt.shared.version
        reports = []
        for plan in test_plans:
            for event in plan.events:
                if not event.task:
                    raise ValueError("This frozen-bank pilot must not contain test writes")
                task = event.task
                stage("frozen_test", task_id=task.id)
                reads = []
                bank_hash = bank.hash

                def read_test(turn):
                    with hf_attention_backend(attention_backend):
                        prepared = rt.engine.prepare_read(bank, turn)
                        slate = rt.selector.sample(
                            prepared, seed_for(cfg["seed"], "test", task.id), deterministic=True
                        )
                    reads.append({"turn": turn, "slate": slate, "bank_hash": bank.hash})
                    return slate

                execution = rt.adapter.run_consumer(
                    task, "frozen-test/" + task.id, seed_for(cfg["seed"], "test-consumer", task.id), read_test
                )
                grade = rt.adapter.grade(task, execution)
                reports.append({"task_id": task.id, "execution": execution, "grade": grade, "reads": reads})
                if bank.hash != bank_hash:
                    raise ValueError("Frozen test changed the memory bank")
                test_count += 1
                atomic_json(root / 'frozen_test_progress.json', {
                    'status': 'in_progress', 'reports': reports, 'bank_hash': bank.hash,
                    'policy_version': version, 'parameters_updated': False,
                    'test_memory_writes': 0, 'completed_tasks': test_count,
                })
        after = adapter_state(rt.shared)
        if version != rt.shared.version or any(
            not torch.equal(before[role][key], after[role][key]) for role in before for key in before[role]
        ):
            raise ValueError("Frozen test unexpectedly changed adapters")
        atomic_json(
            root / "frozen_test.json",
            {
                "reports": reports,
                "bank_hash": bank.hash,
                "bank_source": "final-policy public-trace rebuild" if a.full_suite else "training main replica-0",
                "parameters_updated": False,
                "test_memory_writes": 0,
                "policy_version": version,
            },
        )
    stage(
        "completed",
        status="completed",
        completed_updates=completed_updates,
        gradient_norms=update["gradient_norms"],
        parameter_update_norms=update["parameter_update_norms"],
        checkpoint=update["checkpoint"],
        test_tasks_executed=test_count,
        updated_roles=changed,
        all_roles_updated=set(changed) == {"E", "U", "R"},
    )
except Exception as exc:
    stage("failed", status="failed", error_type=type(exc).__name__, error=str(exc))
    (root / "traceback.log").write_text(traceback.format_exc())
    raise
