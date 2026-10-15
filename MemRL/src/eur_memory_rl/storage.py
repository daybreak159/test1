from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .util import atomic_json, digest, file_hash, plain, read_json


class ResourceLog:
    def __init__(self):
        self.values = defaultdict(float)
        self.started = time.perf_counter()
        self.lock = Lock()

    def add(self, name, value):
        with self.lock:
            self.values[name] += value

    @contextmanager
    def gpu_timer(self, device, category):
        import torch

        if torch.device(device).type != "cuda":
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1000
            self.add("gpu_seconds", elapsed)
            self.add(category + "_gpu_seconds", elapsed)

    def snapshot(self):
        required = (
            "full_steps",
            "short_steps",
            "external_turns",
            "E_input_tokens",
            "E_output_tokens",
            "U_input_tokens",
            "U_output_tokens",
            "R_input_tokens",
            "R_rollout_forwards",
            "R_recompute_forwards",
            "judge_tokens",
            "embeddings",
            "gpu_seconds",
            "environment_seconds",
            "monetary_cost",
        )
        result = {key: self.values.get(key, 0) for key in required}
        result.update(self.values)
        result["wall_seconds"] = time.perf_counter() - self.started
        return result


@dataclass(frozen=True)
class CacheIdentity:
    dataset_identity: str
    environment_snapshot: str
    source_cutoff: int
    bank_hash: str
    query_hash: str
    ordered_candidates_hash: str
    renderer_revision: str
    consumer_profile_hash: str
    selector_revision: str
    model_revision: str
    prompt_hash: str
    budget_hash: str
    seed: int
    grading_version: str
    reference_trajectory_hash: str = ""


class ArtifactCache:
    def __init__(self, root):
        self.root = Path(root)

    def get(self, identity):
        path = self.root / (digest(identity) + ".json")
        if not path.is_file():
            return None
        record = read_json(path)
        if record["identity"] != plain(identity) or digest(record["value"]) != record["value_hash"]:
            raise ValueError("Cache identity/integrity conflict")
        return record["value"]

    def put(self, identity, value):
        path = self.root / (digest(identity) + ".json")
        if path.exists():
            if self.get(identity) != plain(value):
                raise ValueError("Immutable cache key already holds a different result")
            return
        atomic_json(path, {"identity": identity, "value": value, "value_hash": digest(value)})


class ReferencePool:
    def __init__(self, root, split):
        self.root = Path(root) / split
        self.split = split

    def add(self, iteration, identity, execution):
        atomic_json(
            self.root / f"{iteration:06d}-{digest(identity)}.json",
            {"available_after_iteration": iteration, "identity": identity, "execution": execution},
        )

    def candidates(self, iteration, compatible):
        result = []
        for path in sorted(self.root.glob("*.json")):
            record = read_json(path)
            if record["available_after_iteration"] < iteration and compatible(record["identity"]):
                result.append(record)
        return result


def save_rollout(path, main, local, rows, resources, exclusions):
    # Forward records are keyed once; independent slates retain distinct policy rows.
    forwards, serialized = {}, []
    for row in rows:
        value = plain(row)
        if row.role == "R":
            forward = row.record.forward
            forwards[forward.id] = plain(forward)
            value["record"].pop("forward")
            value["record"]["forward_record_id"] = forward.id
        serialized.append(value)
    atomic_json(
        path,
        {
            "format": "eur-rollout-v1",
            "rows": serialized,
            "forward_records": forwards,
            "main": [
                {
                    "id": e.execution_id,
                    "plan": e.plan,
                    "positions": e.positions,
                    "rewards": e.rewards,
                    "bank": e.bank,
                    "anchors": e.anchors,
                    "reads": e.reads,
                    "executions": e.executions,
                    "private_grades": e.grades,
                }
                for e in main
            ],
            "local": [
                {
                    "anchor": r.anchor,
                    "probes": r.probes,
                    "values": r.values,
                    "advantages": r.advantages,
                    "references": r.references,
                    "executions": r.executions,
                    "pair_scores": r.pair_scores,
                    "banks_after_transactions": r.banks,
                }
                for r in local
            ],
            "resources": resources,
            "exclusions": exclusions,
        },
    )


def load_policy_rows(path, expected_sha256, policy_version, domain):
    """Restore exactly the saved actions for an interrupted update, never resample."""
    from .contracts import ChoiceForward, GenerationRecord, MemoryEntry, PolicyRow, Slate, SourceOutcome
    from .training.rows import validate_rows

    if not expected_sha256 or file_hash(path) != expected_sha256:
        raise ValueError("Saved rollout integrity failure")
    document = read_json(path)
    if document.get("format") != "eur-rollout-v1":
        raise ValueError("Unsupported saved rollout format")

    def entry(value):
        return MemoryEntry(**{
            **value, "source_ids": tuple(value["source_ids"]),
            "supersedes": tuple(value.get("supersedes", ())),
            "source_outcomes": tuple(SourceOutcome(**o) for o in value.get("source_outcomes", ())),
        })

    forwards = {}
    for key, value in document["forward_records"].items():
        converted = dict(value)
        for name in ("input_ids", "slot_token_ids", "old_logits", "initial_probabilities"):
            converted[name] = tuple(value[name])
        for name in ("candidates", "recall_candidates"):
            converted[name] = tuple(entry(e) for e in value.get(name, ()))
        forward = ChoiceForward(**converted)
        if key != forward.id:
            raise ValueError("Saved selector forward identity mismatch")
        forwards[key] = forward

    rows = []
    for value in document["rows"]:
        record = dict(value["record"])
        if value["role"] == "R":
            forward_id = record.pop("forward_record_id")
            record["forward"] = forwards[forward_id]
            for name in ("selected_indices", "conditional_logprobs"):
                record[name] = tuple(record[name])
            record["legal_masks"] = tuple(tuple(mask) for mask in record["legal_masks"])
            typed = Slate(**record)
        else:
            for name in ("input_ids", "response_token_ids", "response_mask", "old_logprobs"):
                record[name] = tuple(record[name])
            typed = GenerationRecord(**record)
        rows.append(PolicyRow(**{**value, "record": typed}))
    validate_rows(rows, policy_version, domain)
    return document, rows
