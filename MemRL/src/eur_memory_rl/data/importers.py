from __future__ import annotations

import re
from dataclasses import replace

from ..contracts import DataIdentity, Source, Task, Turn
from ..util import ConfigurationError, atomic_json, digest, file_hash, read_json
from .protocols import LOCOMO_SPLITS


def identity(
    dataset,
    version,
    protocol,
    split,
    sample_id,
    scope,
    content_hash,
    source_ids,
    evidence_ids,
    cutoff,
    query_id="",
    variant="default",
    media="text_only",
    component=None,
):
    return DataIdentity(
        dataset,
        version,
        protocol,
        split,
        sample_id,
        component or f"{dataset}:{sample_id}",
        scope,
        tuple(source_ids),
        tuple(evidence_ids),
        cutoff,
        content_hash,
        query_id=query_id,
        context_variant=str(variant),
        media_profile=media,
    )


def import_locomo(raw_path, public_path, private_path, media_profile, query_assignments=None):
    if media_profile not in {"text_only", "text_and_caption"}:
        raise ConfigurationError("Choose a fixed LoCoMo media profile")
    raw = read_json(raw_path)
    version = file_hash(raw_path)
    episodes, probes, labels = [], [], {}
    assignments = query_assignments or {}
    for conversation in raw:
        cid = conversation["sample_id"]
        split = next((split for split, ids in LOCOMO_SPLITS.items() if cid in ids), None)
        if split is None:
            raise ConfigurationError(f"Unexpected LoCoMo conversation ID: {cid}")
        history = conversation["conversation"]
        history_hash = digest(history)
        session_keys = sorted(
            (k for k in history if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])
        )
        sources, all_evidence = [], []
        for index, key in enumerate(session_keys, 1):
            evidence_ids, lines = [], []
            timestamp = history.get(key + "_date_time", "unknown")
            for turn_index, turn in enumerate(history[key]):
                eid = str(turn.get("dia_id", f"{key}:turn_{turn_index}"))
                evidence_ids.append(eid)
                text = turn["text"]
                if media_profile == "text_and_caption" and turn.get("blip_caption"):
                    text += "\nImage caption: " + turn["blip_caption"]
                lines.append(f"[{eid}] {turn['speaker']}: {text}")
            all_evidence.extend(evidence_ids)
            ident = identity(
                "locomo",
                version,
                "r2_2_1_7",
                "train_source" if split == "train" else split,
                cid,
                cid,
                history_hash,
                [f"{cid}:{key}"],
                evidence_ids,
                index,
                media=media_profile,
            )
            sources.append(
                Source(
                    ident,
                    "DATE: " + timestamp + "\n" + "\n".join(lines),
                    timestamp,
                    prefix_evidence_ids=tuple(all_evidence),
                )
            )
        tasks = []
        for index, question in enumerate(conversation["qa"]):
            qid = f"{cid}:q{index}"
            assignment = assignments.get(qid, {})
            use = assignment.get("split", "train_outcome" if split == "train" else split)
            if split != "train" and use != split:
                raise ConfigurationError("Query assignment cannot move dev/test into training")
            component = assignment.get("duplicate_component", f"locomo-query:{digest(question['question'])}")
            ident = identity(
                "locomo",
                version,
                "r2_2_1_7",
                use,
                cid,
                cid,
                history_hash,
                [f"{cid}:{key}" for key in session_keys],
                all_evidence,
                len(sources),
                qid,
                media=media_profile,
                component=component,
            )
            task = Task(
                ident,
                (Turn(qid, "query", question["question"], public_time=sources[-1].observed_at),),
                category=str(question["category"]),
                probe_certificate=assignment.get("probe_certificate", {}),
            )
            labels[qid] = {
                "answers": [str(question.get("answer", ""))],
                "evidence_ids": question.get("evidence", []),
                "category": question["category"],
            }
            if int(question["category"]) == 5:
                labels[qid]["abstention_aliases"] = ["I don't know", "unknown", "insufficient information"]
            if use == "train_probe":
                probes.append(task)
            else:
                tasks.append(task)
        episodes.append({"id": cid, "scope": cid, "split": split, "sources": sources, "tasks": tasks})
    atomic_json(
        public_path,
        {
            "format": "eur-public-manifest-v1",
            "domain": "qa",
            "dataset": "locomo",
            "raw_sha256": version,
            "media_profile": media_profile,
            "episodes": episodes,
            "probes": probes,
        },
    )
    atomic_json(private_path, {"format": "eur-private-labels-v1", "dataset": "locomo", "labels": labels})


def import_qa_transfer(
    raw_path, dataset, variant, media_profile, public_path, private_path, field_profile=None
):
    """Pinned converters; ambiguous MSC/MemBench subtasks require explicit field mappings."""
    raw, version = read_json(raw_path), file_hash(raw_path)
    profile = field_profile or {}
    episodes, labels = [], {}
    if dataset not in {"longmemeval", "hotpotqa_long_context", "msc_self_instruct", "membench"}:
        raise ConfigurationError("Not a supported transfer dataset")
    if dataset in {"msc_self_instruct", "membench"} and not all(
        k in profile for k in ("id", "history", "question", "answer", "subtask")
    ):
        raise ConfigurationError("MSC/MemBench require subtask and explicit raw field names")
    for index, item in enumerate(raw):
        if dataset == "longmemeval":
            sid = str(item["question_id"])
            sessions = [
                "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)
                for session in item["haystack_sessions"]
            ]
            times = item.get("haystack_dates", ["unknown"] * len(sessions))
            question, answers = item["question"], [str(item["answer"])]
            query_time = item.get("question_date", "")
        elif dataset == "hotpotqa_long_context":
            if str(item["num_docs"]) != str(variant):
                raise ConfigurationError("Do not mix Hotpot long-context document counts")
            sid = str(item.get("index", index))
            context = item["context"]
            sessions = re.split(r"(?=Document \d+:)", context) if isinstance(context, str) else context
            sessions = [str(s).strip() for s in sessions if str(s).strip()]
            times, question, answers, query_time = (
                ["unknown"] * len(sessions),
                item["input"],
                item["answers"],
                "",
            )
        else:
            sid = str(item[profile["id"]])
            history = item[profile["history"]]
            sessions = [history] if isinstance(history, str) else [str(s) for s in history]
            times = ["unknown"] * len(sessions)
            question, answer = item[profile["question"]], item[profile["answer"]]
            answers, query_time = answer if isinstance(answer, list) else [str(answer)], ""
        if not sessions:
            raise ConfigurationError("Empty source history")
        scope, qid, history_hash = f"{dataset}:{sid}", f"{dataset}:{sid}:query", digest(sessions)
        sources = []
        for i, (text, observed) in enumerate(zip(sessions, times)):
            eid = f"{sid}:source:{i}"
            ident = identity(
                dataset,
                version,
                profile.get("protocol", "v3-transfer"),
                "test",
                sid,
                scope,
                history_hash,
                [eid],
                [eid],
                i + 1,
                variant=variant,
                media=media_profile,
                component=f"{dataset}:history:{history_hash}",
            )
            sources.append(Source(ident, f"[{eid}] {text}", observed))
        task_identity = replace(
            sources[-1].identity,
            query_id=qid,
            source_ids=tuple(e for s in sources for e in s.identity.source_ids),
            evidence_ids=tuple(e for s in sources for e in s.identity.evidence_ids),
        )
        task = Task(
            task_identity,
            (Turn(qid, "query", question, public_time=query_time),),
            category=str(item.get("question_type", profile.get("subtask", ""))),
        )
        labels[qid] = {"answers": answers}
        episodes.append({"id": scope, "scope": scope, "split": "test", "sources": sources, "tasks": [task]})
    atomic_json(
        public_path,
        {
            "format": "eur-public-manifest-v1",
            "domain": "qa",
            "dataset": dataset,
            "variant": str(variant),
            "episodes": episodes,
            "probes": [],
        },
    )
    atomic_json(private_path, {"format": "eur-private-labels-v1", "labels": labels})


def import_public_trajectories(records, output, domain):
    """Open or freshly collected agent traces. No hidden reasoning or grader labels."""
    sources = []
    for record in records:
        required = {"identity", "public_trace", "agent_revision", "harness_revision", "observed_at", "origin"}
        if not required <= record.keys() or record["origin"] not in {
            "public_open_trajectories",
            "fresh_rollouts_from_user_selected_agents",
        }:
            raise ConfigurationError("Trajectory provenance is incomplete")
        from .protocols import reject_private_fields
        from .serde import identity as parse_identity

        reject_private_fields(record["public_trace"])
        ident = parse_identity(record["identity"])
        trace = record["public_trace"]
        if len(ident.evidence_ids) != len(trace):
            raise ConfigurationError("Every imported public trace event needs an evidence ID")
        text = "\n".join(f"[{eid}] {event}" for eid, event in zip(ident.evidence_ids, trace))
        sources.append(
            Source(
                ident,
                text,
                record["observed_at"],
                digest([record["agent_revision"], record["harness_revision"]]),
            )
        )
    atomic_json(output, {"format": "eur-public-sources-v1", "domain": domain, "sources": sources})
