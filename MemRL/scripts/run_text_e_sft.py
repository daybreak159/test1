"""One isolated text E SFT epoch; HF evaluation and updates, no Solver runs."""

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eur_memory_rl.util import atomic_json, file_hash, plain, read_json, seed_for


def main(a):
    import torch
    from eur_memory_rl.config import protocol_hash
    from eur_memory_rl.data.serde import source as parse_source
    from eur_memory_rl.memory.transaction import TransactionExecutor, parse_extraction
    from eur_memory_rl.policies.generator import GeneratorPolicy
    from eur_memory_rl.policies.prompts import extraction_messages
    from eur_memory_rl.policies.shared_qwen import SharedQwen, tensor_hash
    from eur_memory_rl.training.attention import hf_attention_backend
    from eur_memory_rl.training.checkpoint import CheckpointStore
    from eur_memory_rl.training.stages import sft_record, train_sft, validate_sft_demonstration

    a.root.mkdir(parents=True, exist_ok=False)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    started = time.monotonic()
    state = dict(status="running", optimizer_steps=0)

    def stage(name, **values):
        state.update(stage=name, seconds=time.monotonic() - started, **values)
        atomic_json(a.root / "summary.json", state)
        print(json.dumps(state), flush=True)

    try:
        review = read_json(a.teacher / "semantic_review.json")
        if not review["train_ready"] or any(
            file_hash(a.teacher / (part + ".json")) != review[part + "_sha256"] for part in ("train", "dev")
        ):
            raise ValueError("Teacher review missing or changed")
        train, dev = (read_json(a.teacher / (part + ".json")) for part in ("train", "dev"))
        if len(train) != 24 or len(dev) != 6:
            raise ValueError("Expected 24 train and 6 held-out tasks")
        ids = [{e["source"]["identity"]["sample_id"] for e in part} for part in (train, dev)]
        if (
            len(ids[0]) != 24
            or len(ids[1]) != 6
            or ids[0] & ids[1]
            or (ids[0] | ids[1]) & set(review["excluded_ids"])
        ):
            raise ValueError("Task contamination")
        cfg = read_json(a.config)
        cfg["experiment"]["name"] = a.root.name
        cfg["generation"]["prompt_profile"] = "e-text-v1"
        cfg["generation"]["input_limits"]["E"] = 32768
        cfg["generation"]["limits"]["E"] = 2048
        cfg["generation"]["use_generation_cache"] = False
        cfg["e_preprocessing"]["config"]["prompt_profile"] = "e-text-v1"
        cfg["e_preprocessing"]["output_dir"] = str(a.root / "e_preprocessing")
        cfg["output_dir"], cfg["checkpoint_dir"] = str(a.root / "rl"), str(a.root / "checkpoints")
        cfg["rollout"] = {"backend": "hf"}
        cfg["optimization"].pop("rollout_correction", None)
        cfg["memory_feedback"].update(
            teacher_train_sha256=file_hash(a.teacher / "train.json"),
            teacher_dev_sha256=file_hash(a.teacher / "dev.json"),
        )
        cfg["sft"].update(lr=1e-4, prompt_profile="e-text-v1", train_tasks=24, dev_tasks=6, epochs=1)
        cfg.pop("_config_dir", None)
        atomic_json(a.root / "config.json", cfg)
        atomic_json(
            a.root / "process.json", dict(pid=os.getpid(), code=str(Path(__file__).resolve().parents[1]))
        )
        torch.manual_seed(cfg["seed"])
        arch = dict(cfg["architecture"], adapter_layout="independent", pointwise_chunk_tokens=512)
        stage("loading")
        shared = SharedQwen.load(arch)
        generator = GeneratorPolicy(
            shared,
            max_input_tokens=32768,
            limits={"E": 2048, "U": 2048},
            prompt_profile="e-text-v1",
            use_generation_cache=True,
            release_unused_cache=True,
        )
        store = CheckpointStore(a.root / "checkpoints", "swe", arch["base_revision"], protocol_hash(cfg))
        store.load(shared, checkpoint=a.checkpoint, strict_protocol=False, restore_rng=False)
        before_hash = {role: tensor_hash(group) for role, group in shared.role_parameters.items()}
        optimizers = {
            "E": torch.optim.AdamW(
                [p for _, p in shared.role_parameters["E"]], lr=1e-4, weight_decay=0
            )
        }
        executor = TransactionExecutor(shared.tokenizer, shared.tokenizer, shared.tokenizer)
        records = []
        for example in train:
            source = parse_source(example["source"])
            validate_sft_demonstration(example, source, executor, prompt_profile="e-text-v1")
            record = sft_record(
                shared,
                "E",
                generator.prepare_messages("E", extraction_messages(source)),
                example["response"],
                max_input=32768,
                max_output=2048,
            )
            if record.input_hash != example["provenance"]["student_prompt_hash"]:
                raise ValueError("Teacher/student prompt mismatch")
            records.append(record)
        atomic_json(
            a.root / "initialization.json",
            dict(
                checkpoint=str(a.checkpoint),
                checkpoint_sha256=file_hash(a.checkpoint / "state.pt"),
                before_hash=before_hash,
                optimizer="fresh E AdamW",
                roles_trained=["E"],
                train_input_tokens=[len(r.input_ids) for r in records],
            ),
        )

        def evaluate(label):
            stage(label)
            shared.model.gradient_checkpointing_disable()
            result = []
            try:
                for index, example in enumerate(dev):
                    source = parse_source(example["source"])
                    begin = time.monotonic()
                    with hf_attention_backend("flash-gqa"):
                        record = generator.generate(
                            "E",
                            extraction_messages(source),
                            seed_for(11, index, "text-e-dev"),
                            deterministic=True,
                        )
                    valid, error = True, ""
                    try:
                        candidates = parse_extraction(record.text, source, profile="e-text-v1")
                    except ValueError as exc:
                        valid, error, candidates = False, str(exc), ()
                    row = dict(
                        task_id=source.identity.sample_id,
                        valid=valid,
                        error=error,
                        candidates=plain(candidates),
                        record=plain(record),
                        seconds=time.monotonic() - begin,
                    )
                    result.append(row)
                    atomic_json(a.root / (label + ".json"), result)
                    stage(label, evaluated=len(result), valid=sum(x["valid"] for x in result))
            finally:
                shared.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            return dict(
                valid=sum(r["valid"] for r in result),
                total=len(result),
                decoding="greedy",
                semantic_review_required=True,
            )

        baseline = evaluate("before_sft")
        torch.cuda.reset_peak_memory_stats(shared.device)

        def progress(row):
            with (a.root / "steps.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            recovery = store.publish(
                shared,
                optimizers,
                row["step"],
                dict(stage="sft", cursor=row["step"], main_batches=0),
                dict(
                    stage="sft",
                    prompt_profile="e-text-v1",
                    roles_trained=["E"],
                    teacher_sha256=file_hash(a.teacher / "train.json"),
                    source_checkpoint=str(a.checkpoint),
                    metrics=row,
                ),
            )
            stage("sft", optimizer_steps=row["step"], loss=row["loss"], recovery_checkpoint=str(recovery))
            gc.collect()
            torch.cuda.empty_cache()

        def diagnostic(event, step, record):
            stage("sft_" + event, current_step=step, input_tokens=len(record.input_ids))

        with hf_attention_backend("flash-gqa"):
            metrics = train_sft(
                shared, generator, records, optimizers, progress=progress, diagnostic=diagnostic
            )
        peak = torch.cuda.max_memory_allocated(shared.device) / 2**30
        after_hash = {role: tensor_hash(group) for role, group in shared.role_parameters.items()}
        if any(before_hash[r] != after_hash[r] for r in ("U", "R")) or before_hash["E"] == after_hash["E"]:
            raise RuntimeError("Adapter change audit failed")
        shared.assert_base_unchanged()
        checkpoint = store.root / "iteration-000024"
        atomic_json(
            a.root / "training_result.json",
            dict(
                checkpoint=str(checkpoint),
                metrics=metrics,
                peak_gib=peak,
                before_hash=before_hash,
                after_hash=after_hash,
                E_changed=True,
                U_changed=False,
                R_changed=False,
            ),
        )
        after = evaluate("after_sft")
        stage(
            "completed",
            status="completed",
            checkpoint=str(checkpoint),
            optimizer_steps=24,
            peak_gib=peak,
            baseline=baseline,
            after=after,
            E_changed=True,
            U_changed=False,
            R_changed=False,
            semantic_or_solver_accuracy_claim=False,
        )
    except Exception as exc:
        stage("failed", status="failed", error_type=type(exc).__name__, error=str(exc))
        (a.root / "traceback.log").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    main(p.parse_args())
