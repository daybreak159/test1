from __future__ import annotations

from dataclasses import replace

from ..util import GroupExcluded, InfrastructureError, seed_for
from .metrics import grouped_bootstrap, summarize_episodes


def evaluate(stream, plans, seed, selector_mode="learned", memory_enabled=True):
    if not plans:
        raise ValueError("No plans for this evaluation split")
    results, excluded, records = [], [], []
    for index, plan in enumerate(plans):
        if plan.split not in {"dev", "test"}:
            raise ValueError("Evaluation accepts locked dev/test plans only")
        plan = replace(plan, id=f"eval/{stream.engine.generator.shared.version}/{seed}/{plan.id}")
        try:
            episode = stream.run_episode(
                plan,
                plan.id,
                seed_for(seed, index),
                deterministic=True,
                selector_mode=selector_mode,
                memory_enabled=memory_enabled,
            )
        except (InfrastructureError, GroupExcluded) as exc:
            excluded.append({"plan": plan.id, "reason": str(exc)})
            continue
        results.append(episode)
        for task_id, grade in episode.grades.items():
            task = next(e.task for e in plan.events if e.task and e.task.id == task_id)
            if task.identity.dataset == "locomo" and task.category == "5":
                continue
            records.append({"group": plan.scope, "value": grade.quality})
    report = summarize_episodes(results)
    report.update(
        {
            "excluded": excluded,
            "excluded_fraction": len(excluded) / max(len(plans), 1),
            "inference": "deterministic_budgeted_top_k",
            "training_sampling": "masked_plackett_luce",
            "seed": seed,
            "parameters_updated": False,
            "reward_probes_enabled": False,
        }
    )
    if records:
        report["confidence_interval"] = grouped_bootstrap(records, seed=seed)
    return report, results
