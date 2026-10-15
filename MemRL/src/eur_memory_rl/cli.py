from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import (
    DOMAINS,
    load_config,
    lock_environment,
    preflight,
    protocol_hash,
    require_preflight,
    resolved,
    starter_config,
)
from .util import ConfigurationError, atomic_json, plain, read_json, seed_for


def parser():
    root = argparse.ArgumentParser(prog="eur", description="EUR v3.2 memory RL; no service starts implicitly")
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config")
    init.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    init.add_argument("--output", required=True)
    flight = sub.add_parser("preflight")
    flight.add_argument("config")
    flight.add_argument("--mode", choices=["train", "evaluate", "model-audit"], default="train")
    flight.add_argument("--check-installed", action="store_true")
    flight.add_argument("--output")
    lock = sub.add_parser("lock-environment")
    lock.add_argument("--output", required=True)
    for name in ("inspect-model", "audit-model", "ablation-plan", "audit-data"):
        command = sub.add_parser(name)
        command.add_argument("config")
        command.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("config")
    train.add_argument("--resume", nargs="?", const="latest")
    initialize = sub.add_parser("initialize")
    initialize.add_argument("config")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("config")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--allow-profile-change",
        action="store_true",
        help="Frozen same-domain transfer/evaluation with a separately logged dataset/Consumer profile",
    )
    sft = sub.add_parser("sft")
    sft.add_argument("config")
    sft.add_argument("--examples", required=True)
    warmup = sub.add_parser("warmup")
    warmup.add_argument("config")
    warmup.add_argument("--batches", type=int, default=20)
    warmup.add_argument("--resume", nargs="?", const="latest")
    importer = sub.add_parser("import-locomo")
    importer.add_argument("raw")
    importer.add_argument("--public", required=True)
    importer.add_argument("--private", required=True)
    importer.add_argument("--media", required=True, choices=["text_only", "text_and_caption"])
    importer.add_argument("--query-assignments")
    transfer = sub.add_parser("import-qa-transfer")
    transfer.add_argument("raw")
    transfer.add_argument(
        "--dataset",
        required=True,
        choices=["longmemeval", "hotpotqa_long_context", "msc_self_instruct", "membench"],
    )
    transfer.add_argument("--variant", required=True)
    transfer.add_argument("--media", required=True)
    transfer.add_argument("--public", required=True)
    transfer.add_argument("--private", required=True)
    transfer.add_argument("--field-profile")
    trajectories = sub.add_parser("import-trajectories")
    trajectories.add_argument("raw")
    trajectories.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    trajectories.add_argument("--output", required=True)
    tasks = sub.add_parser("import-tasks")
    tasks.add_argument("raw")
    tasks.add_argument("--dataset", required=True, choices=["swe_bench", "webarena", "alfworld", "mind2web"])
    tasks.add_argument("--assignments", required=True)
    tasks.add_argument("--support")
    tasks.add_argument("--public", required=True)
    tasks.add_argument("--private", required=True)
    serve = sub.add_parser("serve-environments")
    serve.add_argument("config")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    crossed = sub.add_parser("crossed-audit")
    crossed.add_argument("config")
    crossed.add_argument("--checkpoint", required=True)
    crossed.add_argument("--old-bank", required=True)
    crossed.add_argument("--new-bank", required=True)
    crossed.add_argument("--tasks", required=True)
    crossed.add_argument("--repeats", type=int, default=1)
    crossed.add_argument("--output", required=True)
    experiments = sub.add_parser("run-experiments")
    experiments.add_argument("plan")
    experiments.add_argument("--output-root", required=True)
    experiments.add_argument("--methods", nargs="+")
    sub.add_parser("estimate-budget")
    return root


def _audit_data(cfg):
    from .data.manifest import audit_manifests, load_manifest
    from .domains.alf import ALFAdapter
    from .domains.offline_web import OfflineWebAdapter
    from .domains.qa import QAAdapter
    from .domains.swe import SWEAdapter
    from .domains.web import WebAdapter

    adapter = {
        "swe": SWEAdapter,
        "web": WebAdapter,
        "qa": QAAdapter,
        "alf": ALFAdapter,
        "offline_web": OfflineWebAdapter,
    }[cfg["domain"]](None, None, None)
    bundle = load_manifest(resolved(cfg, cfg["data"]["public_manifest"]), adapter)
    return audit_manifests([bundle], resolved(cfg, cfg["data"]["duplicate_manifest"]))


def _sft(cfg, examples_path, runtime=None, progress=None, start_step=0,
         initial_metrics=None, diagnostic=None):
    from .data.serde import bank as parse_bank, source as parse_source
    from .memory.transaction import parse_extraction
    from .policies.prompts import extraction_messages, transaction_messages
    from .runtime import Runtime
    from .training.stages import sft_record, train_sft, validate_sft_demonstration

    rt, records = runtime or Runtime(cfg), []
    sft_lr = cfg.get("sft", {}).get("lr")
    if sft_lr is not None:
        if not isinstance(sft_lr, (int, float)) or not 0 < sft_lr < 1:
            raise ValueError("SFT learning rate must be positive and finite")
        for optimizer in rt.coordinator.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = sft_lr
    for example in read_json(examples_path):
        source = parse_source(example["source"])
        if example["role"] == "E":
            from .data.packing import pack_source

            source, _ = pack_source(source, rt.shared.tokenizer, rt.generator.input_limit("E"),
                                    cfg.get("source_packing", "reject"),
                                    prompt_profile=rt.generator.prompt_profile)
        bank = parse_bank(example["bank"]) if "bank" in example else None
        extraction = example.get("extraction", {"candidates": []})
        candidates = parse_extraction(
            extraction if isinstance(extraction, str) else json.dumps(extraction), source,
            profile=rt.generator.prompt_profile if isinstance(extraction, str) else "default")
        visible = tuple(e for e in (bank.active if bank else ()) if e.id in example.get("visible_ids", []))
        validate_sft_demonstration(example, source, rt.transactions, bank, candidates, visible,
                                   prompt_profile=rt.generator.prompt_profile)
        messages = (
            extraction_messages(source)
            if example["role"] == "E"
            else transaction_messages(
                source, bank, candidates, visible, {
                    "active_bank_token_limit": rt.transactions.bank_tokens,
                    "current_active_tokens": len(rt.shared.tokenizer.encode(
                        rt.transactions.renderer.render(bank.active), add_special_tokens=False)),
                    "entry_consumer_token_limit": rt.transactions.entry_consumer_tokens,
                    "entry_selector_token_limit": rt.transactions.entry_selector_tokens,
                }
            )
        )
        records.append(
            sft_record(
                rt.shared,
                example["role"],
                rt.generator.prepare_messages(example["role"], messages),
                example["response"],
                max_output=cfg["generation"]["limits"][example["role"]],
                max_input=rt.generator.input_limit(example["role"]),
            )
        )
    skip_limit = cfg.get("sft", {}).get("skip_input_over_tokens")
    skip_indices = tuple(i for i, record in enumerate(records)
                         if skip_limit is not None and len(record.input_ids) > skip_limit)
    logs = train_sft(rt.shared, rt.generator, records, rt.coordinator.optimizers, progress=progress,
                     start_step=start_step, initial_metrics=initial_metrics, diagnostic=diagnostic,
                     skip_indices=skip_indices)
    logs["skipped_indices"] = list(skip_indices)
    logs["skip_input_over_tokens"] = skip_limit
    checkpoint = rt.checkpoints.publish(
        rt.shared,
        rt.coordinator.optimizers,
        0,
        {"cursor": 0, "main_batches": 0, "manifest_hash": rt.data.manifest_hash, "stage": "sft"},
        logs,
    )
    return {"checkpoint": str(checkpoint), **logs}


def _warmup(cfg, args, runtime=None):
    from .rewards.advantages import mean
    from .runtime import Runtime
    from .training.stages import warmup_rows

    rt = runtime or Runtime(cfg)
    iteration = int(rt.shared.version.split("-")[-1])
    if args.resume:
        bundle = rt.checkpoints.load(
            rt.shared, rt.coordinator.optimizers, None if args.resume == "latest" else args.resume
        )
        iteration = bundle["iteration"]
        rt.coordinator.initialize_from_stage(bundle["data_state"].get("stage"))
    previous_train_roles = set(rt.coordinator.train_roles)
    previous_budget = rt.probe.budget
    rt.coordinator.train_roles = {"R"}
    plans = [p for p in rt.data.plans if p.split == "train"]
    if not plans:
        raise ValueError("No warmup training data")
    rt.probe.budget = rt.adapter.profile.budget
    initial_iteration = iteration
    for batch in range(initial_iteration, initial_iteration + args.batches):
        plan = replace(plans[batch % len(plans)], id=f"warmup/{iteration + 1}/{batch}")
        main = rt.stream.run_episode(plan, plan.id, seed_for(cfg["seed"], "warmup", batch))
        pairs = []
        for anchor in main.anchors:
            try:
                probes = rt.probe.select_probes(anchor, rt.data.probes, seed_for(cfg["seed"], anchor.id))
            except ValueError:
                continue
            for task in probes:
                pairs.append((anchor.bank, task, rt.engine.prepare_read(anchor.bank, task.turns[0])))
            break
        if not pairs:
            raise ValueError("Warmup needs prevalidated training bank/query pairs")
        counter = [0]

        def execute(task, slate, seed):
            counter[0] += 1
            values = []
            for b in range(cfg["local"]["B"]):
                execution = rt.probe._execute(
                    task, slate, f"warmup/{iteration + 1}/{counter[0]}/{b}", seed_for(seed, b)
                )
                values.append(rt.adapter.grade(task, execution).quality)
            return mean(values)

        rows = warmup_rows(
            pairs, cfg["local"]["L"], execute, rt.selector, cfg["domain"], seed_for(cfg["seed"], batch)
        )
        iteration += 1
        rt.coordinator.update(
            rows,
            iteration,
            {"cursor": 0, "main_batches": 0, "stage": "warmup", "manifest_hash": rt.data.manifest_hash},
        )
    rt.coordinator.train_roles = previous_train_roles
    rt.probe.budget = previous_budget
    return {"warmup_batches": args.batches, "joint_iteration": iteration}


def run(args):
    if args.command == "estimate-budget":
        from .evaluation.budgets import planned_costs

        return planned_costs()
    if args.command == "run-experiments":
        from .evaluation.experiments import execute_experiment_plan

        return execute_experiment_plan(args.plan, args.output_root, args.methods)
    if args.command == "init-config":
        atomic_json(args.output, starter_config(args.domain))
        return {"config": str(Path(args.output).resolve()), "status": "deployment_fields_required"}
    if args.command == "lock-environment":
        lock_environment(args.output)
        return {"lock": args.output}
    if args.command == "import-locomo":
        from .data.importers import import_locomo

        import_locomo(
            args.raw,
            args.public,
            args.private,
            args.media,
            read_json(args.query_assignments) if args.query_assignments else None,
        )
        return {"public": args.public, "private": args.private}
    if args.command == "import-qa-transfer":
        from .data.importers import import_qa_transfer

        import_qa_transfer(
            args.raw,
            args.dataset,
            args.variant,
            args.media,
            args.public,
            args.private,
            read_json(args.field_profile) if args.field_profile else None,
        )
        return {"public": args.public, "private": args.private}
    if args.command == "import-trajectories":
        from .data.importers import import_public_trajectories

        import_public_trajectories(read_json(args.raw), args.output, args.domain)
        return {"output": args.output}
    if args.command == "import-tasks":
        from .data.task_importers import import_task_domain

        import_task_domain(args.raw, args.dataset, args.assignments, args.public, args.private, args.support)
        return {"public": args.public, "private": args.private}
    if args.command == "serve-environments":
        import uvicorn
        from .services.server import create_app

        uvicorn.run(create_app(read_json(args.config)), host=args.host, port=args.port)
        return {"stopped": True}
    cfg = load_config(args.config)
    if args.command == "preflight":
        result = preflight(cfg, args.mode, args.check_installed)
        if args.output:
            atomic_json(args.output, result)
        return result
    if args.command == "ablation-plan":
        from .evaluation.experiments import write_experiment_plan

        write_experiment_plan(cfg, args.output)
        return {"output": args.output}
    if args.command == "audit-data":
        result = _audit_data(cfg)
        atomic_json(args.output, result)
        return result
    if args.command == "inspect-model":
        from .training.audits import inspect_model

        inspect_model(cfg, args.output)
        return {"output": args.output}
    if args.command == "audit-model":
        from .policies.generator import GeneratorPolicy
        from .policies.shared_qwen import SharedQwen
        from .training.audits import model_audit

        shared = SharedQwen.load(
            {**cfg["architecture"], "adapter_layout": cfg["experiment"]["adapter_layout"]}
        )
        return model_audit(shared, GeneratorPolicy(shared, **cfg["generation"]), cfg, args.output)
    require_preflight(cfg, "train" if args.command in {"train", "warmup"} else "evaluate")
    if args.command == "sft":
        return _sft(cfg, args.examples)
    if args.command == "warmup":
        return _warmup(cfg, args)
    from .runtime import Runtime

    rt = Runtime(cfg, with_optimizer=args.command in {"train", "initialize"})
    if args.command == "initialize":
        path = rt.checkpoints.publish(
            rt.shared,
            rt.coordinator.optimizers,
            0,
            {"cursor": 0, "main_batches": 0, "manifest_hash": rt.data.manifest_hash, "stage": "initial"},
        )
        return {"checkpoint": str(path), "parameters_trained": False}
    if args.command == "train":
        from .training.trainer import Trainer

        trainer = Trainer(rt)
        if args.resume:
            trainer.resume(None if args.resume == "latest" else args.resume)
        remaining_warmup = cfg["schedule"].get("R_warmup_batches", 0) - trainer.iteration
        if trainer.main_batches == 0 and remaining_warmup > 0:
            from types import SimpleNamespace

            _warmup(cfg, SimpleNamespace(batches=remaining_warmup, resume=None), runtime=rt)
            trainer.resume()
        return trainer.train()
    loaded = rt.checkpoints.load(
        rt.shared,
        checkpoint=args.checkpoint,
        restore_rng=False,
        strict_protocol=not getattr(args, "allow_profile_change", False),
    )
    if args.command == "evaluate":
        from .evaluation.protocols import evaluate

        report, _ = evaluate(
            rt.stream,
            [p for p in rt.data.plans if p.split == args.split],
            cfg["seed"],
            cfg["experiment"]["selector_mode"],
            cfg["experiment"]["memory_enabled"],
        )
        report["resources"] = rt.resources.snapshot()
        report["protocol_hash"] = protocol_hash(cfg)
        report["training_protocol_hash"] = loaded["protocol_hash"]
        atomic_json(args.output, report)
        return report
    if args.command == "crossed-audit":
        from .data.serde import bank, task
        from .evaluation.crossed_audit import crossed_audit

        tasks = [task(t) for t in read_json(args.tasks)]
        if any(t.identity.split not in {"train_probe", "dev"} for t in tasks):
            raise ValueError("Crossed audit uses a fixed training/development audit set")
        rt.probe.budget = rt.adapter.profile.budget
        result = crossed_audit(
            rt.engine,
            rt.probe,
            bank(read_json(args.old_bank)),
            bank(read_json(args.new_bank)),
            tasks,
            args.repeats,
            cfg["seed"],
        )
        atomic_json(args.output, result)
        return result
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        result = run(args)
        print(json.dumps(plain(result), ensure_ascii=False, indent=2))
        return 2 if isinstance(result, dict) and result.get("ready") is False else 0
    except (ConfigurationError, ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc), "command": args.command}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
