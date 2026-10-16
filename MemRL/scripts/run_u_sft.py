"""Isolated U-only SFT on reviewed demonstrations and explicit retention cases.

Use scripts/memrl.sh --u-sft. Preparation does not load a GPU model; --train
is required for updates. No Solver, embedding or environment calls are made.
"""
import argparse
import copy
from dataclasses import replace
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eur_memory_rl.data.serde import bank as parse_bank, source as parse_source
from eur_memory_rl.memory.transaction import TransactionExecutor, parse_extraction
from eur_memory_rl.policies.generator import GeneratorPolicy
from eur_memory_rl.policies.prompts import encode_messages, transaction_messages
from eur_memory_rl.training.stages import validate_sft_demonstration
from eur_memory_rl.util import atomic_json, digest, file_hash, plain, read_json, seed_for


def decode(example):
    source = parse_source(example["source"])
    bank = parse_bank(example["bank"])
    raw = example["extraction"]
    candidates = parse_extraction(raw if isinstance(raw, str) else json.dumps(raw), source,
                                 profile="e-text-v1" if isinstance(raw, str) else "default")
    visible = tuple(e for e in bank.active if e.id in example["visible_ids"])
    return source, bank, candidates, visible


def messages_for(example, executor, generator):
    source, bank, candidates, visible = decode(example)
    capacity = dict(active_bank_token_limit=executor.bank_tokens,
                    current_active_tokens=len(executor.counter.encode(
                        executor.renderer.render(bank.active), add_special_tokens=False)),
                    entry_consumer_token_limit=executor.entry_consumer_tokens,
                    entry_selector_token_limit=executor.entry_selector_tokens)
    return generator.prepare_messages("U", transaction_messages(source, bank, candidates, visible, capacity))


def prepare_examples(rows):
    """Retain old labels; add distinct-task same-repository retention exercises.

    Donors stay inside the same source partition. Their original evidence and
    applicability are kept. Rule labels say only to preserve an independent old
    entry and ADD new candidates; they do not invent a repair or edit content.
    These are explicitly synthetic bank states, not online-policy observations.
    """
    old = [copy.deepcopy(r) for r in rows if r["role"] == "U"]
    empty = [r for r in old if r["provenance"]["context_scenario"] == "empty"]
    supplements = []
    for row in empty:
        sid = row["source"]["identity"]["sample_id"]
        repo = sid.rsplit("-", 1)[0]
        donors = [r for r in empty if r["source"]["identity"]["sample_id"] != sid
                  and r["source"]["identity"]["sample_id"].rsplit("-", 1)[0] == repo]
        if not donors:
            continue
        donor = donors[0]
        ds, _, _, _ = decode(donor)
        # Construct the old entry by the same ADD path as runtime (below in prepare).
        for scenario in ("independent_nonempty", "duplicate_plus_independent"):
            new = copy.deepcopy(row)
            new["retention_donor"] = copy.deepcopy(donor)
            new["retention_scenario"] = scenario
            new["provenance"] = dict(
                split="train_source", teacher_revision="deterministic-retention-v1",
                cost={"monetary_cost": 0, "external_model_calls": 0},
                original_teacher_revision=row["provenance"]["teacher_revision"],
                original_target_hash=digest(row["response"]),
                donor_task_id=ds.identity.sample_id,
                partition=row["provenance"]["partition"], synthetic_bank=True,
                context_scenario=scenario,
                label_rule="preserve different-task same-repository memory; ADD only nonduplicate candidates",
            )
            supplements.append(new)
    return old + supplements


def materialize(example, executor):
    if "retention_donor" not in example:
        return example
    from eur_memory_rl.contracts import Bank
    example = copy.deepcopy(example)
    donor = example.pop("retention_donor")
    ds, db, dc, dv = decode(donor)
    result = executor.apply(db, dc, donor["response"], dv, ds)
    if not result.valid or not result.bank.active:
        raise ValueError("Invalid retention donor")
    source, _, candidates, _ = decode(example)
    if ds.identity.scope != source.identity.scope:
        raise ValueError("Cross-scope retention donor")
    old = replace(result.bank.active[0], id="m00000001", revision=1,
                  kind="unclassified", title="")
    # Exercise the real text-E parser and its deterministic provenance wrapping.
    text_extraction = "\n\n".join(
        f"[MEMORY]\n正文：{c.content}\n使用条件：{c.conditions}" for c in candidates)
    candidates = parse_extraction(text_extraction, source, profile="e-text-v1")
    entries = [old]
    start = 0
    if example["retention_scenario"] == "duplicate_plus_independent":
        c = candidates[0]
        entries.append(replace(old, id="m00000002", content=c.content, conditions=c.conditions,
                               source_ids=c.source_ids, subject=c.subject, event_time=c.event_time,
                               observed_at=source.observed_at,
                               source_outcomes=(source.official_outcome,) if source.official_outcome else ()))
        start = 1
    bank = Bank(source.identity.scope, 3, len(entries) + 1, tuple(entries))
    example["bank"] = plain(bank)
    example["visible_ids"] = [e.id for e in entries]
    example["extraction"] = text_extraction
    ops = [{"op": "ADD", "candidate_index": i} for i in range(start, len(candidates))]
    example["response"] = json.dumps(dict(base_revision=bank.version,
                                         operations=ops or [{"op": "NOOP"}]))
    return example


def prepare(args, cfg, tokenizer):
    executor = TransactionExecutor(tokenizer, tokenizer, tokenizer, **cfg["memory"])
    generator = GeneratorPolicy(None, prompt_profile="e-text-v1")
    review = read_json(args.teacher / "semantic_review.json")
    if not review.get("train_ready"):
        raise ValueError("Original teacher review is not ready")
    parts = {}
    audits = {}
    for part in ("train", "dev"):
        original = read_json(args.teacher / (part + ".json"))
        rows = [materialize(r, executor) for r in prepare_examples(original)]
        lengths = []
        for r in rows:
            s, b, cs, visible = decode(r)
            validate_sft_demonstration(r, s, executor, b, cs, visible, prompt_profile="e-text-v1")
            prepared = messages_for(r, executor, generator)
            ids, phash, _ = encode_messages(tokenizer, prepared, 8192)
            output_count = len(tokenizer.encode(r["response"], add_special_tokens=False)) + 1
            if output_count > 2048:
                raise ValueError("U target exceeds output budget")
            r["provenance"]["student_prompt_hash"] = phash
            lengths.append(len(ids))
            if r.get("retention_scenario"):
                after = executor.apply(b, cs, r["response"], visible, s).bank
                if after.active[0] != b.active[0] or not all(
                    any(e.content == c.content and e.conditions == c.conditions for e in after.active) for c in cs
                ):
                    raise ValueError("Retention target loses old or incoming knowledge")
        parts[part] = rows
        audits[part] = dict(count=len(rows), input_min=min(lengths), input_max=max(lengths),
                           task_ids=sorted({r["source"]["identity"]["sample_id"] for r in rows}),
                           scenarios={k: sum(r["provenance"]["context_scenario"] == k for r in rows)
                                      for k in sorted({r["provenance"]["context_scenario"] for r in rows})},
                           original_file_sha256=file_hash(args.teacher / (part + ".json")))
    if set(audits["train"]["task_ids"]) & set(audits["dev"]["task_ids"]):
        raise ValueError("Train/dev task overlap")
    # Protect all currently declared RL task/probe/test IDs, including nested manifests.
    excluded = set()
    def walk(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k in {"sample_id", "query_id", "instance_id", "task_id"} and isinstance(v, str):
                    excluded.add(v)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(read_json(args.config.parent / "public_manifest.json"))
    if excluded & (set(audits["train"]["task_ids"]) | set(audits["dev"]["task_ids"])):
        raise ValueError("SFT sources overlap current RL manifest")
    for part, rows in parts.items():
        atomic_json(args.root / (part + ".json"), rows)
    audits.update(train_sha256=file_hash(args.root / "train.json"),
                  dev_sha256=file_hash(args.root / "dev.json"),
                  original_review_sha256=file_hash(args.teacher / "semantic_review.json"),
                  excluded_current_manifest_ids=sorted(excluded),
                  synthetic_labels="deterministic-retention-v1; expert-review required beyond structural validity",
                  train_ready=True)
    atomic_json(args.root / "data_audit.json", audits)
    return parts, executor


def signature(text):
    value = json.loads(text)
    return sorted((op["op"], str(op.get("candidate_index", "")), str(op.get("target_id", "")))
                  for op in value["operations"])


def main(args):
    import torch
    from transformers import AutoTokenizer
    from eur_memory_rl.config import protocol_hash
    from eur_memory_rl.policies.shared_qwen import SharedQwen, tensor_hash
    from eur_memory_rl.training.attention import hf_attention_backend
    from eur_memory_rl.training.checkpoint import CheckpointStore
    from eur_memory_rl.training.stages import sft_record, train_sft
    args.root.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    state = dict(status="preparing", optimizer_steps=0)
    def stage(name, **values):
        state.update(stage=name, seconds=time.monotonic() - begun, **values)
        atomic_json(args.root / "summary.json", state)
        print(json.dumps(state), flush=True)
    try:
        cfg = read_json(args.config)
        cfg["architecture"]["adapter_layout"] = "independent"
        cfg["architecture"]["pointwise_chunk_tokens"] = 512
        cfg["experiment"]["name"] = args.root.name
        cfg["sft"] = dict(role="U", lr=args.lr, epochs=1,
                          initialization="fresh U LoRA on original base; accepted pre-RL E/R retained; fresh optimizer")
        cfg["generation"].update(prompt_profile="e-text-v1", use_generation_cache=True)
        atomic_json(args.root / "config.json", cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg["architecture"]["base_model"], local_files_only=True)
        parts, executor = prepare(args, cfg, tokenizer)
        if not args.train:
            stage("prepared", status="prepared")
            return
        stage("loading", status="running")
        torch.manual_seed(cfg["seed"])
        shared = SharedQwen.load(cfg["architecture"])
        fresh_u = {name: p.detach().cpu().clone() for name, p in shared.role_parameters["U"]}
        generator = GeneratorPolicy(shared, max_input_tokens=8192, limits={"E": 2048, "U": 2048},
                                    prompt_profile="e-text-v1", use_generation_cache=True,
                                    release_unused_cache=True)
        store = CheckpointStore(args.root / "checkpoints", "swe", cfg["architecture"]["base_revision"], protocol_hash(cfg))
        store.load(shared, checkpoint=args.checkpoint, strict_protocol=False, restore_rng=False)
        with torch.no_grad():
            for name, parameter in shared.role_parameters["U"]:
                parameter.copy_(fresh_u[name].to(parameter.device, parameter.dtype))
        if not all(torch.equal(parameter.detach().cpu(), fresh_u[name])
                   for name, parameter in shared.role_parameters["U"]):
            raise RuntimeError("Fresh U initialization was not preserved")
        if any(torch.count_nonzero(parameter).item() for name, parameter in shared.role_parameters["U"]
               if "lora_B" in name):
            raise RuntimeError("Fresh U must have zero initial LoRA delta")
        del fresh_u
        before_hash = {r: tensor_hash(p) for r, p in shared.role_parameters.items()}
        source_root = Path(__file__).resolve().parents[1]
        atomic_json(args.root / "provenance.json", dict(
            checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint / "state.pt"),
            source_config_sha256=file_hash(args.config), script_sha256=file_hash(__file__),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip(),
            git_diff=subprocess.check_output(["git", "diff"], cwd=source_root, text=True),
            core_hash=digest({str(p.relative_to(source_root)): file_hash(p) for p in sorted((source_root / "src").rglob("*.py"))}),
            command=sys.argv, before_hash=before_hash, roles_trained=["U"], new_optimizer=True,
            U_initialization="fresh adapter; no SFT or RL U weights inherited",
            checkpoint_imported_roles=["E", "R"],
            model_manifest=shared.manifest()))
        def evaluate(label):
            stage(label, evaluated=0)
            shared.model.gradient_checkpointing_disable()
            results = []
            try:
                for i, r in enumerate(parts["dev"]):
                    source, bank, cs, vis = decode(r)
                    begin = time.monotonic()
                    with hf_attention_backend("flash-gqa"):
                        record = generator.generate("U", transaction_messages(source, bank, cs, vis, dict(
                            active_bank_token_limit=8192,
                            current_active_tokens=len(tokenizer.encode(executor.renderer.render(bank.active), add_special_tokens=False)),
                            entry_consumer_token_limit=1024, entry_selector_token_limit=1536)),
                            seed_for(cfg["seed"], "u-dev", i), deterministic=True)
                    outcome = executor.apply(bank, cs, record.text, vis, source)
                    match = outcome.valid and signature(record.text) == signature(r["response"])
                    retention = None
                    if r.get("retention_scenario"):
                        retention = outcome.valid and outcome.bank.active[0] == bank.active[0] and all(
                            any(e.content == c.content and e.conditions == c.conditions for e in outcome.bank.active) for c in cs)
                    results.append(dict(task_id=source.identity.sample_id, scenario=r["provenance"]["context_scenario"],
                                        valid=outcome.valid, error=outcome.error, operation_signature_match=match,
                                        retention_passed=retention, response=record.text, target=r["response"],
                                        finish_reason=record.finish_reason, seconds=time.monotonic() - begin,
                                        input_tokens=len(record.input_ids), output_tokens=len(record.response_token_ids)))
                    atomic_json(args.root / (label + ".json"), results)
                    stage(label, evaluated=len(results), valid=sum(x["valid"] for x in results))
            finally:
                shared.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            summary = dict(total=len(results), valid=sum(x["valid"] for x in results),
                           operation_signature_match=sum(x["operation_signature_match"] for x in results),
                           retention_total=sum(x["retention_passed"] is not None for x in results),
                           retention_passed=sum(x["retention_passed"] is True for x in results))
            atomic_json(args.root / (label + "_summary.json"), summary)
            return summary
        baseline = evaluate("before_sft")
        records = []
        rows = list(parts["train"])
        random.Random(cfg["seed"]).shuffle(rows)
        for r in rows:
            rec = sft_record(shared, "U", messages_for(r, executor, generator), r["response"], max_input=8192, max_output=2048)
            if rec.input_hash != r["provenance"]["student_prompt_hash"]:
                raise ValueError("Teacher/student prompt mismatch")
            records.append(rec)
        atomic_json(args.root / "training_order.json", [r.input_hash for r in records])
        optimizer = torch.optim.AdamW([p for _, p in shared.role_parameters["U"]], lr=args.lr, weight_decay=0)
        torch.cuda.reset_peak_memory_stats(shared.device)
        def progress(row):
            with (args.root / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if row["step"] == 1 or row["step"] % 10 == 0 or row["step"] == len(records):
                checkpoint = store.publish(shared, {"U": optimizer}, row["step"],
                                           dict(stage="sft", cursor=row["step"], main_batches=0),
                                           dict(roles_trained=["U"], source_checkpoint=str(args.checkpoint), metrics=row,
                                                train_sha256=file_hash(args.root / "train.json")))
                state["checkpoint"] = str(checkpoint)
            stage("sft", optimizer_steps=row["step"], loss=row["loss"])
            gc.collect()
            torch.cuda.empty_cache()
        with hf_attention_backend("flash-gqa"):
            metrics = train_sft(shared, generator, records, {"U": optimizer}, progress=progress)
        hashes = {r: tensor_hash(p) for r, p in shared.role_parameters.items()}
        if hashes["E"] != before_hash["E"] or hashes["R"] != before_hash["R"] or hashes["U"] == before_hash["U"]:
            raise RuntimeError("U-only adapter isolation failed")
        shared.assert_base_unchanged()
        atomic_json(args.root / "training_result.json", dict(metrics=metrics, before_hash=before_hash,
                    after_hash=hashes, E_changed=False, U_changed=True, R_changed=False, base_unchanged=True,
                    peak_gib=torch.cuda.max_memory_allocated(shared.device) / 2**30,
                    checkpoint=state["checkpoint"]))
        after = evaluate("after_sft")
        stage("completed", status="completed", before=baseline, after=after,
              solver_accuracy_claim=False, default_checkpoint_replaced=False)
    except Exception as error:
        stage("failed", status="failed", error_type=type(error).__name__, error=str(error))
        (args.root / "traceback.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    for name in ("root", "teacher", "checkpoint", "config"):
        setattr(args, name, getattr(args, name).resolve())
    if not 0 < args.lr <= 1e-3:
        parser.error("Learning rate must be in (0, 1e-3]")
    lock_dir = Path(__file__).resolve().parents[2] / "03_实验结果/runs/.locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ".launcher-cuda-1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args)
