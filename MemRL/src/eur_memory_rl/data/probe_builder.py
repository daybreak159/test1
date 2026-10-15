from __future__ import annotations

import json
from dataclasses import replace

from ..contracts import Task, Turn
from ..util import InfrastructureError, canonical, digest, seed_for


def generate_prefix_probes(sources, teacher, verifier, count, seed, validation_version):
    """Offline preparation only. Receives source prefixes, never candidate memories or policies."""
    if not sources or count < 1 or any(s.identity.split != "train_source" for s in sources):
        raise ValueError("Generate probes only from legal training source prefixes")
    scope = sources[0].identity.scope
    if any(s.identity.scope != scope for s in sources):
        raise ValueError("Personal facts cannot cross conversation scopes")
    public = [
        {"text": s.text, "date": s.observed_at, "evidence_ids": s.identity.evidence_ids} for s in sources
    ]
    prompt = [
        {
            "role": "system",
            "content": "Create evidence-grounded questions answerable using only the supplied prefix. "
            "Return JSON {probes:[{question,answer,evidence_ids,time_sensitive,probe_kind}]}. "
            "Use exact evidence IDs. Do not assume later facts. Cover new facts, old knowledge and cross-session facts.",
        },
        {"role": "user", "content": canonical({"prefix": public, "count": count})},
    ]
    generated = teacher.complete(prompt, 4096, seed, digest([scope, "probe-generation", public, seed]))
    try:
        items = json.loads(generated.text)["probes"]
        if len(items) != count:
            raise ValueError("Probe cardinality mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        raise InfrastructureError("Probe-generation schema failure") from exc
    tasks, labels, provenance = [], {}, []
    visible = {e for s in sources for e in s.identity.evidence_ids}
    for index, item in enumerate(items):
        if not item.get("evidence_ids") or not set(item["evidence_ids"]) <= visible:
            raise ValueError("Generated probe cites unavailable evidence")
        verify_prompt = [
            {
                "role": "system",
                "content": "Verify this question and answer against the complete supplied prefix. "
                "Check all evidence, dates, and changes to current/last/total facts. Return JSON "
                "{valid:boolean,complete_prefix_checked:boolean,reason:string}.",
            },
            {"role": "user", "content": canonical({"prefix": public, "probe": item})},
        ]
        verification = verifier.complete(
            verify_prompt, 1024, seed_for(seed, index), digest([public, item, "verify"])
        )
        try:
            verdict = json.loads(verification.text)
            if verdict["valid"] is not True or verdict["complete_prefix_checked"] is not True:
                raise ValueError("Probe prefix validation failed")
        except (ValueError, KeyError, TypeError) as exc:
            raise InfrastructureError(
                "Generated probe is not verified; do not use it for a local tree"
            ) from exc
        qid = "generated:" + digest([scope, sources[-1].identity.visibility_cutoff, item["question"]])
        identity = replace(
            sources[-1].identity,
            split="train_probe",
            query_id=qid,
            duplicate_component="unreviewed:" + digest(item["question"]),
        )
        certificate = {
            "prefix_answerable": True,
            "validated_before_sampling": True,
            "validation_version": validation_version,
            "cutoff": identity.visibility_cutoff,
            "time_sensitive": bool(item["time_sensitive"]),
            "complete_prefix_checked": True,
            "evidence_ids": item["evidence_ids"],
            "probe_kind": item["probe_kind"],
        }
        tasks.append(
            Task(
                identity,
                (Turn(qid, "query", item["question"], public_time=sources[-1].observed_at),),
                probe_certificate=certificate,
            )
        )
        labels[qid] = {"answers": [item["answer"]]}
        provenance.append(
            {
                "query_id": qid,
                "teacher_revision": teacher.revision,
                "verifier_revision": verifier.revision,
                "verification": verdict,
                "teacher_tokens": generated.input_tokens + generated.output_tokens,
                "verifier_tokens": verification.input_tokens + verification.output_tokens,
                "duplicate_review_required": True,
            }
        )
    return tasks, labels, provenance
