from __future__ import annotations

from collections import OrderedDict

from ..contracts import Task, Turn
from ..util import ConfigurationError, atomic_json, file_hash, read_json
from .importers import identity
from .serde import source as parse_source


def import_task_domain(raw_path, dataset, assignments_path, public_path, private_path, support_path=None):
    """Convert native task records using reviewed stream/split/environment identities.

    Assignments are mandatory: no random task-level split can establish near-duplicate isolation.
    """
    raw, version = read_json(raw_path), file_hash(raw_path)
    assignments = read_json(assignments_path)
    if not assignments.get("reviewed") or not assignments.get("protocol"):
        raise ConfigurationError("Task import requires a reviewed split/stream assignment")
    domain = {"swe_bench": "swe", "webarena": "web", "alfworld": "alf", "mind2web": "offline_web"}[dataset]
    episodes, probes, labels = OrderedDict(), [], {}
    supports = [parse_source(s) for s in read_json(support_path)["sources"]] if support_path else []
    items = []
    if dataset == "mind2web":
        for sample in raw:
            for step in sample["actions"]:
                items.append((str(step["action_uid"]), sample, step))
    else:
        key = {"swe_bench": "instance_id", "webarena": "task_id", "alfworld": "task_id"}[dataset]
        items = [(str(sample[key]), sample, None) for sample in raw]
    for tid, item, step in items:
        if tid not in assignments["tasks"]:
            raise ConfigurationError(f"No reviewed assignment for task {tid}")
        assignment = assignments["tasks"][tid]
        split, scope = assignment["split"], assignment["scope"]
        if dataset == "mind2web" and split not in {"dev", "test"}:
            raise ConfigurationError("Mind2Web is transfer-only")
        if dataset == "swe_bench":
            query = item["problem_statement"]
            payload = {
                "base_commit": item["base_commit"],
                "repo": item["repo"],
                "image_key": assignment["image_key"],
                "expected_initial_state_hash": assignment["initial_state_hash"],
            }
            label = {k: item[k] for k in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS") if k in item}
            history = ""
        elif dataset == "webarena":
            query, history = item["intent"], ""
            payload = {
                "task_key": assignment["task_key"],
                "scope": scope,
                "expected_initial_state_hash": assignment["initial_state_hash"],
            }
            label = {"evaluation": item["eval"]}
        elif dataset == "alfworld":
            query, history = item["goal"], ""
            payload = {
                "task_key": assignment["task_key"],
                "train_eval": item["train_eval"],
                "expected_initial_state_hash": assignment["initial_state_hash"],
            }
            label = {"task_key": assignment["task_key"]}
        else:
            query = (
                item["confirmed_task"]
                + '\nPredict only this recorded step. Return JSON {"element_id":"backend node id","operation":"...","value":"..."}.'
            )
            history = step.get("cleaned_html", step.get("raw_html", ""))
            if not history:
                raise ConfigurationError("Mind2Web step lacks its recorded page")
            payload = {"recorded_state_only": True}
            label = {
                "element_ids": [str(c["backend_node_id"]) for c in step["pos_candidates"]],
                "operation": step["operation"]["op"],
                "value": step["operation"].get("value", ""),
            }
        ident = identity(
            dataset,
            version,
            assignments["protocol"],
            split,
            tid,
            scope,
            assignment["history_or_environment_hash"],
            assignment.get("source_ids", []),
            assignment.get("evidence_ids", []),
            assignment["visibility_cutoff"],
            tid,
            variant=assignment.get("variant", "default"),
            component=assignment["duplicate_component"],
        )
        turns = [Turn(tid, "initial", query, history, assignment.get("public_time", ""))]
        for followup in assignment.get("external_followups", []):
            turns.append(
                Turn(
                    tid,
                    followup["external_turn_id"],
                    followup["text"],
                    public_time=followup.get("public_time", ""),
                )
            )
        task = Task(
            ident,
            tuple(turns),
            payload,
            str(assignment.get("category", "")),
            assignment.get("probe_certificate", {}),
        )
        labels[tid] = label
        if split == "train_probe":
            probes.append(task)
            continue
        planned_split = "train" if split in {"train_outcome", "train_source", "train"} else split
        eid = assignment["episode_id"]
        support_ids = assignment.get("support_ids")
        eligible_supports = [s for s in supports if s.identity.scope == scope]
        if support_ids is not None:
            if len(support_ids) != len(set(support_ids)):
                raise ConfigurationError("Duplicate episode support IDs")
            by_id = {s.identity.sample_id: s for s in eligible_supports}
            if not set(support_ids) <= by_id.keys():
                raise ConfigurationError("Episode support IDs missing or outside scope")
            eligible_supports = [by_id[sid] for sid in support_ids]
        if eid not in episodes:
            episodes[eid] = {
                "id": eid,
                "scope": scope,
                "split": planned_split,
                "sources": eligible_supports,
                "tasks": [],
            }
        episode = episodes[eid]
        if episode["scope"] != scope or episode["split"] != planned_split:
            raise ConfigurationError("A task stream cannot cross scope/split boundaries")
        if [s.identity.sample_id for s in episode["sources"]] != [
            s.identity.sample_id for s in eligible_supports
        ]:
            raise ConfigurationError("Episode targets disagree on support selection")
        episode["tasks"].append(task)
    atomic_json(
        public_path,
        {
            "format": "eur-public-manifest-v1",
            "domain": domain,
            "dataset": dataset,
            "episodes": list(episodes.values()),
            "probes": probes,
        },
    )
    atomic_json(private_path, {"format": "eur-private-labels-v1", "dataset": dataset, "labels": labels})
