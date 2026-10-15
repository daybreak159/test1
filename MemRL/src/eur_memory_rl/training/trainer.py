from __future__ import annotations

import random
import time
from dataclasses import replace

from ..config import resolved
from ..evaluation.protocols import evaluate
from ..storage import save_rollout
from ..util import GroupExcluded, atomic_json, read_json, seed_for
from .rows import assign_weights


class Trainer:
    def __init__(self, runtime):
        self.runtime, self.cfg = runtime, runtime.cfg
        self.output = resolved(self.cfg, self.cfg["output_dir"])
        self.cursor, self.iteration, self.main_batches = 0, 0, 0
        self.best_dev = None

    def resume(self, checkpoint=None):
        state = self.runtime.checkpoints.load(
            self.runtime.shared, self.runtime.coordinator.optimizers, checkpoint
        )
        self.iteration = state["iteration"]
        self.cursor = state["data_state"]["cursor"]
        self.main_batches = state["data_state"].get("main_batches", state["iteration"])
        self.best_dev = state["data_state"].get("best_dev")
        best_path = self.output / "best_checkpoint.json"
        if best_path.is_file():
            best = read_json(best_path)
            if best["iteration"] <= self.iteration:
                self.best_dev = best["score"]
        if state["data_state"]["manifest_hash"] != self.runtime.data.manifest_hash:
            raise ValueError("Cannot resume with changed training data")
        reset = self.runtime.coordinator.initialize_from_stage(state["data_state"].get("stage"))
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / "resume_initialization.json", {
            "checkpoint": str(checkpoint) if checkpoint else "latest",
            "source_stage": state["data_state"].get("stage", "rl"),
            "iteration": self.iteration,
            "optimizer_state_reset": reset,
        })

    def train(self):
        rt, cfg = self.runtime, self.cfg
        plans = [p for p in rt.data.plans if p.split == "train"]
        dev = [p for p in rt.data.plans if p.split == "dev"]
        if not plans:
            raise ValueError("No train plans; do not fill batches with dev/test")
        if not cfg["experiment"].get("train_roles"):
            raise ValueError("Fixed/no-memory experiments are evaluation-only")
        self.output.mkdir(parents=True, exist_ok=True)
        atomic_json(self.output / "model_manifest.json", rt.shared.manifest())
        while self.main_batches < cfg["schedule"]["batches"]:
            next_iteration = self.iteration + 1
            batch_seed = seed_for(cfg["seed"], "batch", next_iteration)
            main, local, exclusions = [], [], []
            while len(main) < cfg["schedule"]["plans_per_batch"] * cfg["global"]["replicas"]:
                base_plan = plans[self.cursor % len(plans)]
                self.cursor += 1
                plan = replace(base_plan, id=f"batch-{next_iteration}/draw-{self.cursor}/{base_plan.id}")
                if "stream_length" in cfg["experiment"]:
                    if cfg["domain"] not in {"swe", "web"}:
                        raise ValueError("Stream-length ablation applies to SWE/Web task streams")
                    tasks = [e.task for e in plan.events if e.task][: cfg["experiment"]["stream_length"]]
                    plan = rt.adapter.build_plan(plan.id, plan.scope, (), tasks, plan.split)
                try:
                    group = rt.stream.run_group(
                        plan,
                        cfg["global"]["replicas"],
                        seed_for(batch_seed, self.cursor),
                        cfg["global"]["debug_zero_baseline"],
                        cfg["experiment"].get("selector_mode", "learned"),
                    )
                except GroupExcluded as exc:
                    exclusions.append({"group": plan.id, "reason": str(exc)})
                    if len(exclusions) >= cfg["schedule"]["max_excluded_groups"]:
                        atomic_json(self.output / f"blocked-batch-{next_iteration}.json", exclusions)
                        raise RuntimeError("Exceeded predeclared infrastructure exclusion budget") from exc
                    continue
                main.extend(group)
            if cfg["experiment"]["objective"] != "outcome_only" and cfg["local"]["anchors_per_replica"] > 0:
                for episode in main:
                    # Anchor locations use only a seeded declared rule and prefix eligibility, never rewards.
                    anchors = list(episode.anchors)
                    random.Random(seed_for(batch_seed, episode.execution_id, "anchors")).shuffle(anchors)
                    selected = []
                    for anchor in anchors:
                        try:
                            probes = rt.probe.select_probes(
                                anchor, rt.data.probes, seed_for(batch_seed, anchor.id, "probes")
                            )
                        except ValueError:
                            continue  # Pre-candidate fixed fallback to the next eligible anchor.
                        selected.append((anchor, probes))
                        if len(selected) == cfg["local"]["anchors_per_replica"]:
                            break
                    if len(selected) < cfg["local"]["anchors_per_replica"]:
                        raise ValueError("Insufficient prefix-validated anchors; fix the train probe plan")
                    for anchor, probes in selected:
                        try:
                            local.append(
                                rt.probe.run(anchor, probes, seed_for(batch_seed, anchor.id, "tree"))
                            )
                        except GroupExcluded as exc:
                            exclusions.append({"group": anchor.id, "reason": str(exc)})
            rows = assign_weights(
                main,
                local,
                **{k: cfg["local"][k] for k in ("I", "J", "K", "L", "beta")},
                objective=cfg["experiment"]["objective"],
            )
            save_rollout(
                self.output / f"rollout-{next_iteration:06d}.json",
                main,
                local,
                rows,
                rt.resources.snapshot(),
                exclusions,
            )
            started = time.perf_counter()
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(rt.shared.device)
                torch.cuda.reset_peak_memory_stats(rt.shared.device)
            logs = rt.coordinator.update(
                rows,
                next_iteration,
                {
                    "cursor": self.cursor,
                    "main_batches": self.main_batches + 1,
                    "manifest_hash": rt.data.manifest_hash,
                    "best_dev": self.best_dev,
                },
                {"exclusions": exclusions, "resources": rt.resources.snapshot()},
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(rt.shared.device)
                logs["peak_training_memory_bytes"] = torch.cuda.max_memory_allocated(rt.shared.device)
                rt.resources.add("update_wall_seconds", time.perf_counter() - started)
            rt.resources.add(
                "R_recompute_forwards",
                len(
                    {
                        r.record.forward.id
                        for r in rows
                        if r.role == "R" and r.row_weight and r.record.forward.forward_count
                    }
                ),
            )
            self.iteration = next_iteration
            self.main_batches += 1
            if dev and self.iteration % cfg["schedule"]["dev_interval"] == 0:
                report, _ = evaluate(
                    rt.stream,
                    dev,
                    seed_for(cfg["seed"], "dev"),
                    selector_mode=cfg["experiment"].get("selector_mode", "learned"),
                )
                atomic_json(self.output / f"dev-{self.iteration:06d}.json", report)
                score = report["metrics"].get("quality")
                if score is not None and (self.best_dev is None or score > self.best_dev):
                    self.best_dev = score
                    atomic_json(
                        self.output / "best_checkpoint.json",
                        {"iteration": self.iteration, "score": score, "checkpoint": logs["checkpoint"]},
                    )
            logs["resources"] = rt.resources.snapshot()
            logs["accepted_episodes"], logs["accepted_anchors"] = len(main), len(local)
            logs["exclusions"] = exclusions
            atomic_json(self.output / f"update-{self.iteration:06d}.json", logs)
            rt.engine.clear_indexes()
        return {"iterations": self.iteration, "output": str(self.output), "best_dev": self.best_dev}
