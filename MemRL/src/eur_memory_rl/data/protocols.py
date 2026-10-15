from __future__ import annotations

from collections import defaultdict

from ..util import ConfigurationError, digest

LOCOMO_SPLITS = {
    "train": ("conv-43", "conv-47"),
    "dev": ("conv-44",),
    "test": ("conv-41", "conv-49", "conv-50", "conv-42", "conv-48", "conv-30", "conv-26"),
}
TRANSFER_ONLY = {"longmemeval", "msc_self_instruct", "membench", "hotpotqa_long_context", "mind2web"}
PRIVATE_KEYS = {
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "grader_result",
    "hidden_feedback",
    "resolved",
    "test_patch",
    "reference_answer",
    "fail_to_pass",
    "pass_to_pass",
}


def reject_private_fields(value, path="public"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in PRIVATE_KEYS:
                raise ConfigurationError(f"Private annotation in public manifest: {path}.{key}")
            reject_private_fields(item, path + "." + key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            reject_private_fields(item, f"{path}[{i}]")


def partition_class(split):
    return "train" if split in {"train", "train_source", "train_outcome", "train_probe"} else split


def validate_identities(identities, duplicate_manifest):
    if not duplicate_manifest.get("reviewed") or not duplicate_manifest.get("version"):
        raise ConfigurationError("A reviewed duplicate-component manifest is required")
    groups = defaultdict(set)
    query_usage = defaultdict(set)
    history_partitions = defaultdict(set)
    detailed_uses = defaultdict(set)
    assigned = duplicate_manifest.get("assignments", {})
    for identity in identities:
        split = partition_class(identity.split)
        if split not in {"train", "dev", "test"}:
            raise ConfigurationError("Unknown data split")
        if identity.dataset in TRANSFER_ONLY and split == "train":
            raise ConfigurationError(f"{identity.dataset} is transfer-only in the v3 primary protocol")
        record_key = f"{identity.dataset}:{identity.query_id or identity.sample_id}"
        if assigned.get(record_key) != identity.duplicate_component:
            raise ConfigurationError(f"Missing/mismatched duplicate component: {record_key}")
        groups[identity.duplicate_component].add(split)
        detailed_uses[(identity.dataset, identity.duplicate_component)].add(identity.split)
        if identity.query_id:
            query_usage[identity.duplicate_component].add(identity.split)
        history_key = identity.history_or_environment_hash
        if identity.dataset in {"swe_bench", "webarena", "alfworld", "mind2web"}:
            # Many independent tasks share one repository/image/site initial state.
            # Deduplicate the task component AND state, not every task on the same website.
            history_key = (history_key, identity.duplicate_component)
        history_partitions[history_key].add(split)
        if identity.dataset == "locomo":
            if identity.split_protocol != "r2_2_1_7" or identity.scope not in LOCOMO_SPLITS[split]:
                raise ConfigurationError("LoCoMo protocol/scope split conflict; do not union archived splits")
        if identity.dataset == "longmemeval" and identity.context_variant not in {
            "S",
            "oracle",
            "cleaned",
            "105_query_subset",
        }:
            raise ConfigurationError("LongMemEval variants require separate explicit manifests")
        if identity.dataset == "hotpotqa_long_context" and identity.context_variant not in {
            "50",
            "100",
            "200",
        }:
            raise ConfigurationError("Hotpot long context must declare 50/100/200 documents")
    if any(len(splits) > 1 for splits in groups.values()) or any(
        len(splits) > 1 for splits in history_partitions.values()
    ):
        raise ConfigurationError("Train/dev/test leakage through duplicate components or shared history")
    if any("train_probe" in uses and ({"train", "train_outcome"} & uses) for uses in query_usage.values()):
        raise ConfigurationError("Semantically duplicate outcome/probe questions overlap")
    if any(
        dataset in {"swe_bench", "webarena", "alfworld"} and "train_probe" in uses and "train_source" in uses
        for (dataset, _), uses in detailed_uses.items()
    ):
        raise ConfigurationError("Task-domain source/probe duplicate components must be disjoint")
    return {
        "identities": len(identities),
        "duplicate_components": len(groups),
        "history_groups": len(history_partitions),
        "duplicate_manifest_hash": digest(duplicate_manifest),
    }


def validate_derived(parent, child):
    if parent.duplicate_component != child.duplicate_component or partition_class(
        parent.split
    ) != partition_class(child.split):
        raise ConfigurationError("Derived examples must inherit their parent partition/component")
    if child.visibility_cutoff > parent.visibility_cutoff:
        raise ConfigurationError("Derived source reads beyond its allowed prefix")
