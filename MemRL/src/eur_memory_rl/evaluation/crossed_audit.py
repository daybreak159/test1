from __future__ import annotations

from ..rewards.advantages import mean
from ..util import seed_for
from .metrics import grouped_bootstrap


def decompose(v00, v01, v10, v11):
    return {
        "content": v10 - v00,
        "selection": v01 - v00,
        "interaction": v11 - v10 - v01 + v00,
        "joint_gain": v11 - v00,
    }


def crossed_audit(engine, probe_runner, old_bank, new_bank, tasks, repeats=1, seed=11):
    if old_bank.scope != new_bank.scope or repeats < 1:
        raise ValueError("Invalid crossed audit scope/repeats")
    cells = {key: [] for key in ("00", "01", "10", "11")}
    task_records = []
    for q, task in enumerate(tasks):
        scores = {key: [] for key in cells}
        for content, bank in enumerate((old_bank, new_bank)):
            for selection in (0, 1):
                key = f"{content}{selection}"
                prepared = engine.prepare_read(bank, task.turns[0], fixed=selection == 0)
                for b in range(repeats):
                    slate = engine.selector.sample(
                        prepared, seed_for(seed, "select", q, b), deterministic=True, fixed=selection == 0
                    )
                    execution = probe_runner._execute(
                        task, slate, f"audit/{seed}/{q}/{key}/{b}", seed_for(seed, q, b)
                    )
                    # Full-task audits should provide full rollout budget to probe_runner.
                    if not execution.completed:
                        raise ValueError("Full-result crossed audit was truncated; increase the audit budget")
                    grade = probe_runner.adapter.grade(task, execution)
                    cells[key].append(grade.quality)
                    scores[key].append(grade.quality)
        effect = decompose(*(mean(scores[key]) for key in ("00", "01", "10", "11")))
        task_records.append({"group": task.identity.scope, **effect})
    means = {key: mean(values) for key, values in cells.items()}
    return {
        "cells": means,
        "effects": decompose(*(means[k] for k in ("00", "01", "10", "11"))),
        "confidence_intervals": {
            effect: grouped_bootstrap(task_records, value_key=effect, seed=seed)
            for effect in ("content", "selection", "interaction", "joint_gain")
        },
        "repeats": repeats,
        "seed": seed,
        "tasks": task_records,
    }


def locked_candidate_audit(selector, prepared, execute, seeds):
    """Only same-bank candidates are held fixed; never equate new/old memory identities."""
    return [
        {
            "seed": seed,
            "R0": execute(selector.sample(prepared, seed, fixed=True)),
            "R": execute(selector.sample(prepared, seed, deterministic=True)),
        }
        for seed in seeds
    ]
