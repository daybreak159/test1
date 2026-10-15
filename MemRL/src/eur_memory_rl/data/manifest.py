from __future__ import annotations

from dataclasses import dataclass

from ..util import ConfigurationError, file_hash, read_json
from .protocols import reject_private_fields, validate_identities
from .serde import source, task


@dataclass(frozen=True)
class DatasetBundle:
    plans: tuple
    probes: tuple
    manifest_hash: str
    identities: tuple
    excluded_training_adversarial: int = 0


def load_manifest(path, adapter, split=None):
    document = read_json(path)
    if document.get("format") != "eur-public-manifest-v1" or document.get("domain") != adapter.domain:
        raise ConfigurationError("Wrong manifest schema/domain")
    reject_private_fields(document)
    plans, identities, excluded_training_adversarial = [], [], 0
    for episode in document["episodes"]:
        sources = tuple(source(s) for s in episode["sources"])
        tasks = tuple(task(t) for t in episode["tasks"])
        identities.extend(s.identity for s in sources)
        identities.extend(t.identity for t in tasks)
        if episode["split"] == "train":
            primary = tuple(t for t in tasks if not (t.identity.dataset == "locomo" and t.category == "5"))
            excluded_training_adversarial += len(tasks) - len(primary)
            tasks = primary
        if any(s.identity.scope != episode["scope"] for s in sources) or any(
            t.identity.scope != episode["scope"] for t in tasks
        ):
            raise ConfigurationError("Episode memory scope is inconsistent")
        if split is not None and episode["split"] != split:
            continue
        plans.append(adapter.build_plan(episode["id"], episode["scope"], sources, tasks, episode["split"]))
    probes = tuple(task(t) for t in document.get("probes", []))
    identities.extend(p.identity for p in probes)
    return DatasetBundle(
        tuple(plans), probes, file_hash(path), tuple(identities), excluded_training_adversarial
    )


def audit_manifests(bundles, duplicate_path):
    identities = [identity for bundle in bundles for identity in bundle.identities]
    report = validate_identities(identities, read_json(duplicate_path))
    report["excluded_training_adversarial_queries"] = sum(b.excluded_training_adversarial for b in bundles)
    report["locomo_primary_training_categories"] = [1, 2, 3, 4]
    return report
