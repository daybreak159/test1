from __future__ import annotations

import math

from ..contracts import GenerationRecord, Slate


def assign_weights(main_episodes, local_results, I=2, J=2, K=2, L=2, beta=0.1, objective="dual"):
    if not main_episodes:
        raise ValueError("Local training requires an accepted complete main source group")
    if objective not in {"dual", "outcome_only", "local_only"}:
        raise ValueError("Unknown objective")
    ns, na, rows = len(main_episodes), len(local_results), []
    for episode in main_episodes:
        for row in episode.rows:
            row.row_weight = 0.0 if objective == "local_only" else 1 / ns
            rows.append(row)
    if na:
        denominators = {"E": na * I, "U": na * I * J, "R": na * I * J * K * L}
        for result in local_results:
            for row in result.rows:
                row.row_weight = 0.0 if objective == "outcome_only" else beta / denominators[row.role]
                rows.append(row)
    return rows


def validate_rows(rows, policy_version, domain):
    ids = set()
    for row in rows:
        if row.id in ids:
            raise ValueError("Duplicate policy sample ID (B repetitions must not duplicate actions)")
        ids.add(row.id)
        if row.split not in {"train", "train_source", "train_outcome"} or row.domain != domain:
            raise ValueError("Only one domain's training rows may reach the optimizer")
        if row.row_weight < 0 or not math.isfinite(row.row_weight) or not math.isfinite(row.advantage):
            raise ValueError("Nonfinite advantage or invalid row weight")
        if isinstance(row.record, GenerationRecord):
            rec = row.record
            if row.role not in {"E", "U"} or rec.role != row.role:
                raise ValueError("Wrong generation role")
            if not (len(rec.response_token_ids) == len(rec.old_logprobs) == len(rec.response_mask)):
                raise ValueError("Response/logprob/mask length mismatch")
            if any(not math.isfinite(x) for x in rec.old_logprobs):
                raise ValueError("Nonfinite behavior logprob")
            version = rec.policy_version
        elif isinstance(row.record, Slate):
            rec = row.record
            if row.role != "R" or len(set(rec.selected_indices)) != len(rec.selected_indices):
                raise ValueError("Invalid selector record")
            if abs(sum(rec.conditional_logprobs) - rec.old_slate_logprob) > 1e-6:
                raise ValueError("Joint slate logprob mismatch")
            version = rec.forward.policy_version
        else:
            raise TypeError("Unknown policy record")
        if version != policy_version:
            raise ValueError("On-policy batch mixes checkpoint versions")
