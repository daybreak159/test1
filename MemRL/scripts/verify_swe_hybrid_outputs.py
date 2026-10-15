"""Verify saved Source/input_ids/prompt roundtrips without a model call."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eur_memory_rl.data.hybrid_trace import prepared_e_messages  # noqa: E402
from eur_memory_rl.data.serde import source as parse_source  # noqa: E402
from eur_memory_rl.policies.prompts import encode_messages  # noqa: E402
from eur_memory_rl.util import digest, plain, read_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    summary = read_json(args.output / "summary.json")
    for name, sha in summary["tokenizer"]["files"].items():
        if hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Tokenizer fingerprint differs: " + name)
    inventory = [json.loads(s) for s in (args.output / "tasks.jsonl").read_text().splitlines() if s.strip()]
    successes = [r for r in inventory if r["status"] == "success"]
    sources = [json.loads(s) for s in (args.output / "sources.jsonl").read_text().splitlines() if s.strip()]
    if len(sources) != len(successes):
        raise ValueError("Source JSONL count differs")
    namespaces, lengths = [], []
    for row, bundled in zip(successes, sources, strict=True):
        directory = args.output / "trajectories" / row["instance_id"]
        raw_source = read_json(directory / "source.json")
        if bundled != raw_source:
            raise ValueError("Sources JSONL order/content differs")
        source = parse_source(raw_source)
        report = read_json(directory / "report.json")
        prepared = prepared_e_messages(source, report["config"]["prompt_profile"])
        ids, ph, prompt = encode_messages(tokenizer, prepared, report["config"]["max_input_tokens"])
        if (list(ids) != read_json(directory / "input_ids.json") or
                prepared != read_json(directory / "prepared_messages.json") or
                prompt != (directory / "prompt.txt").read_bytes().decode("utf-8") or
                ph != report["prompt_hash"] or len(ids) != row["e_input_tokens"] or
                len(ids) != report["e_input_tokens"] or digest(plain(source)) != report["source_hash"]):
            raise ValueError("Input roundtrip differs: " + row["instance_id"])
        if not all(f"[{eid}]" in source.text for eid in source.identity.evidence_ids):
            raise ValueError("Evidence ID has no visible block")
        namespaces.append(report["evidence_namespace"])
        lengths.append(len(ids))
    if len(namespaces) != len(set(namespaces)):
        raise ValueError("Evidence namespace collision")
    for row in inventory:
        if row["status"] == "skipped":
            saved = read_json(args.output / "skipped" / (row["instance_id"] + ".json"))
            if saved["skip_reason"] != row["skip_reason"]:
                raise ValueError("Skip manifest differs")
    print(json.dumps({"verified": len(successes), "skipped": len(inventory) - len(successes),
                      "all_input_roundtrips_match": True, "evidence_namespaces_unique": True,
                      "max_input_tokens": max(lengths), "model_called": False}, indent=2))


if __name__ == "__main__":
    main()
