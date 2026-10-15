from __future__ import annotations

import math
import random
from collections import defaultdict

from ..rewards.advantages import mean


def grouped_bootstrap(records, group_key="group", value_key="value", samples=2000, seed=11, confidence=0.95):
    groups = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(float(record[value_key]))
    if not groups:
        raise ValueError("Bootstrap needs independent top-level groups")
    # Equal highest-level unit weighting: each conversation/history/stream is one unit.
    values = [mean(v) for _, v in sorted(groups.items())]
    rng = random.Random(seed)
    draws = sorted(mean([rng.choice(values) for _ in values]) for _ in range(samples))
    tail = (1 - confidence) / 2
    return {
        "estimate": mean(values),
        "low": draws[int(tail * (samples - 1))],
        "high": draws[int((1 - tail) * (samples - 1))],
        "independent_units": len(values),
        "unit": group_key,
        "samples": samples,
        "seed": seed,
        "single_group_degenerate": len(values) == 1,
    }


def evidence_coverage(required_ids, bank, candidates, selected):
    expected = set(required_ids)

    def rate(entries):
        available = {source for entry in entries for source in entry.source_ids}
        return len(expected & available) / len(expected) if expected else None

    return {
        "bank_evidence_coverage": rate(bank.active),
        "recall_evidence_coverage": rate(candidates),
        "exposure_evidence_coverage": rate(selected),
    }


def paired_regressions(before, after):
    if set(before) != set(after):
        raise ValueError("Regression audit must use identical task identities")
    gains = {key: after[key] - before[key] for key in before}
    return {
        "mean_gain": mean(list(gains.values())),
        "harmful_rate": mean([float(v < 0) for v in gains.values()]),
        "forgetting_rate": mean([float(before[k] > 0 and after[k] == 0) for k in gains]),
        "gains": gains,
    }


def first_benefit_distance(write_position, paired_use_gains):
    positions = [
        position - write_position
        for position, gain in paired_use_gains
        if position > write_position and gain > 0
    ]
    return min(positions) if positions else None


def calibration_metrics(probabilities, labels):
    if len(probabilities) != len(labels) or not labels or any(y not in (0, 1) for y in labels):
        raise ValueError("Calibration requires independently defined binary terminal labels")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Not probabilities")
    brier = mean([(p - y) ** 2 for p, y in zip(probabilities, labels)])
    nll = -mean(
        [
            y * math.log(max(p, 1e-12)) + (1 - y) * math.log(max(1 - p, 1e-12))
            for p, y in zip(probabilities, labels)
        ]
    )
    return {"brier": brier, "nll": nll, "count": len(labels)}


def process_calibration(short_scores, full_gains):
    if len(short_scores) != len(full_gains) or len(short_scores) < 2:
        raise ValueError("Need paired dev short/full runs")
    pairs = [(i, j) for i in range(len(short_scores)) for j in range(i) if full_gains[i] != full_gains[j]]
    concordance = (
        mean(
            [
                float((short_scores[i] - short_scores[j]) * (full_gains[i] - full_gains[j]) > 0)
                for i, j in pairs
            ]
        )
        if pairs
        else None
    )
    return {
        "ranking_consistency": concordance,
        "comparable_pairs": len(pairs),
        "misjudged_indices": [i for i, (s, f) in enumerate(zip(short_scores, full_gains)) if s * f < 0],
    }


def summarize_episodes(episodes):
    metrics, categories = defaultdict(list), defaultdict(list)
    for episode in episodes:
        task_map = {e.task.id: e.task for e in episode.plan.events if e.task}
        for task_id, grade in episode.grades.items():
            task = task_map[task_id]
            category = task.category or "all"
            # LoCoMo adversarial remains visible in a separate table, excluded only from primary.
            primary = not (task.identity.dataset == "locomo" and category == "5")
            categories[f"{task.identity.dataset}/{task.identity.context_variant}/{category}"].append(
                grade.quality
            )
            if primary:
                metrics["quality"].append(grade.quality)
                for key, value in grade.metrics.items():
                    if key != "quality":
                        metrics[key].append(value)
        for row in episode.rows:
            if row.role in {"E", "U"}:
                metrics["format_failure_rate"].append(float(not row.record.schema_valid))
        for read in episode.reads:
            slate = read["slate"]
            metrics["loaded_count"].append(len(slate.selected_ids))
            metrics["empty_bank_rate"].append(float(not read["bank"].active))
            metrics["empty_recall_rate"].append(float(not slate.forward.candidates))
            metrics["no_legal_candidate_rate"].append(float(not slate.selected_ids))
            metrics["memory_context_tokens"].append(slate.token_count)
        metrics["memory_entries"].append(len(episode.bank.active))
        metrics["objective"].append(sum(r.value for r in episode.rewards))
    return {
        "metrics": {key: mean(values) for key, values in metrics.items()},
        "categories": {
            key: {"quality": mean(values), "count": len(values)} for key, values in categories.items()
        },
        "episodes": len(episodes),
        "checkpoint_scope": "one_domain",
    }
