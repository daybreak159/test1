"""CPU-only single/batch E compaction. No model generation or training."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eur_memory_rl.data.hybrid_trace import (  # noqa: E402
    CompactionSkip, HybridConfig, compact_trajectory, prepared_e_messages,
)
from eur_memory_rl.data.serde import source as parse_source  # noqa: E402
from eur_memory_rl.policies.prompts import encode_messages  # noqa: E402
from eur_memory_rl.util import atomic_json, read_json  # noqa: E402


def initialize(tokenizer_path, config, output, root, base_sources):
    global TOKENIZER, CONFIG, OUTPUT_DIR, ROOT, BASE_SOURCES
    from transformers import AutoTokenizer

    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    CONFIG, OUTPUT_DIR, ROOT = HybridConfig(**config), Path(output), Path(root) if root else None
    BASE_SOURCES = base_sources


def process(record):
    iid = record["instance_id"]
    path = (ROOT / iid / (iid + ".traj.json") if ROOT else
            Path(record["trajectory_path"]) if "trajectory_path" in record else
            Path(record["artifact_root"]) / (iid + ".traj.json"))
    row = {"instance_id": iid, "dataset": record.get("dataset", "single"), "input_path": str(path)}
    start = time.monotonic()
    try:
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        expected = record.get("source_sha256", record.get("trajectory_sha256"))
        if expected and sha != expected:
            raise ValueError("Source differs from frozen inventory")
        trace = json.loads(raw)
        if trace.get("instance_id") != iid:
            raise ValueError("Trajectory instance_id differs from inventory")
        row["input_sha256"] = sha
        base = parse_source(BASE_SOURCES[iid]) if iid in BASE_SOURCES else None
        source, messages, prompt, report = compact_trajectory(trace, TOKENIZER, config=CONFIG, base_source=base)
        report["input_file_sha256"] = sha
        prepared = prepared_e_messages(source, CONFIG.prompt_profile)
        ids, ph, exact = encode_messages(TOKENIZER, prepared, CONFIG.max_input_tokens)
        if ph != report["prompt_hash"] or exact != prompt:
            raise ValueError("Final prompt verification differs")
        destination = OUTPUT_DIR / "trajectories" / iid
        destination.mkdir(parents=True, exist_ok=False)
        atomic_json(destination / "source.json", source)
        atomic_json(destination / "messages.json", messages)
        atomic_json(destination / "prepared_messages.json", prepared)
        atomic_json(destination / "input_ids.json", ids)
        atomic_json(destination / "report.json", report)
        (destination / "e_input.txt").write_text(source.text + "\n", encoding="utf-8")
        (destination / "prompt.txt").write_text(prompt, encoding="utf-8")
        row.update(status="success", e_input_tokens=report["e_input_tokens"],
                   body_tokens=report["body_tokens"], patch_tokens=report["patch_tokens"],
                   selected_events=report["selected_events"], candidate_events=report["candidate_events"],
                   detail=report["detail"], prompt_hash=report["prompt_hash"],
                   evidence_namespace=report["evidence_namespace"],
                   source_json=str(destination / "source.json"))
    except CompactionSkip as exc:
        report = {**exc.report, **row}
        destination = OUTPUT_DIR / "skipped"
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / (iid + ".json"), report)
        row.update(status="skipped", skip_reason=exc.reason,
                   patch_characters=report.get("patch_characters"), patch_tokens=report.get("patch_tokens"),
                   attempts=report.get("attempts"))
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(time.monotonic() - start, 3)
    return row


def summarize(rows):
    lengths = sorted(r["e_input_tokens"] for r in rows if r["status"] == "success")
    def percentile(p):
        if not lengths:
            return None
        x = (len(lengths) - 1) * p
        return lengths[math.floor(x)] + (lengths[math.ceil(x)] - lengths[math.floor(x)]) * (x % 1)
    return {
        "trajectories": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
        "skip_reasons": dict(Counter(r["skip_reason"] for r in rows if r["status"] == "skipped")),
        "complete_e_input_tokens": {"mean": mean(lengths) if lengths else None,
            "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95),
            "min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None},
        "buckets": {"le_8k": sum(n <= 8192 for n in lengths),
                    "8k_to_16k": sum(8192 < n <= 16384 for n in lengths),
                    "16k_to_24k": sum(16384 < n <= 24576 for n in lengths),
                    "24k_to_32k": sum(24576 < n <= 32768 for n in lengths)},
        "lean_fallback": sum(r.get("detail") == "lean" for r in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trajectory", type=Path)
    group.add_argument("--inventory", type=Path, help="Pinned JSONL trajectory inventory")
    parser.add_argument("--dataset", help="Filter inventory dataset, e.g. no_memory500")
    parser.add_argument("--trajectory-root", type=Path, help="Relocate inventory to ROOT/ID/ID.traj.json")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Local tokenizer directory, no model loaded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sources", type=Path, help="Optional JSONL trusted Sources keyed by identity.sample_id")
    parser.add_argument("--prompt-profile", choices=("default", "compact-json-v1", "strict-json-v2", "raw-json-v3", "swe-json-v4", "e-text-v1"),
                        default="swe-json-v4")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--short-output-tokens", type=int, default=256)
    parser.add_argument("--max-patch-tokens", type=int, default=32768)
    parser.add_argument("--max-patch-chars", type=int, default=131072)
    parser.add_argument("--patch-mode", choices=("bounded-hunks", "full"), default="bounded-hunks")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    config = HybridConfig(max_input_tokens=args.max_input_tokens, short_output_tokens=args.short_output_tokens,
                          max_patch_tokens=args.max_patch_tokens, max_patch_chars=args.max_patch_chars,
                          patch_mode=args.patch_mode,
                          prompt_profile=args.prompt_profile)
    if args.trajectory:
        trace = read_json(args.trajectory)
        records = [{"instance_id": trace["instance_id"], "trajectory_path": str(args.trajectory.resolve())}]
    else:
        records = [json.loads(line) for line in args.inventory.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if args.dataset:
            records = [r for r in records if r["dataset"] == args.dataset]
    if not records or len({r["instance_id"] for r in records}) != len(records):
        parser.error("Inventory is empty or has repeated instance IDs; select a single dataset")
    if any(not re.fullmatch(r"[\w.-]+", r["instance_id"]) for r in records):
        parser.error("Unsafe instance_id")
    bases = {}
    if args.base_sources:
        for line in args.base_sources.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                value = json.loads(line)
                key = value["identity"]["sample_id"]
                if key in bases:
                    parser.error("Repeated base Source identity")
                bases[key] = value
        missing = {r["instance_id"] for r in records} - bases.keys()
        if missing:
            parser.error("base-sources must cover every selected trajectory")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    initargs = (str(args.tokenizer.resolve()), asdict_config(config), str(args.output.resolve()),
                str(args.trajectory_root.resolve()) if args.trajectory_root else None, bases)
    rows = []
    with (args.output / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        if args.workers == 1:
            initialize(*initargs)
            results = map(process, records)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=initargs)
            results = pool.map(process, records)
        try:
            for row in results:
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if len(rows) % 25 == 0 or len(rows) == len(records):
                    print(json.dumps({"processed": len(rows), "total": len(records),
                          "statuses": dict(Counter(r["status"] for r in rows)),
                          "seconds": round(time.monotonic() - start, 1)}), flush=True)
        finally:
            if pool:
                pool.shutdown()
    report = {**summarize(rows), "config": asdict_config(config), "model_called": False,
              "commands_executed": False, "seconds": round(time.monotonic() - start, 3),
              "identity_note": "No data-split review is performed. Defaults are unassigned; supply trusted base Sources.",
              "tokenizer": {"path": str(args.tokenizer), "files": {
                  p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.tokenizer.iterdir() if p.is_file()}},
              "code_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path(__file__).parents[1] / "src/eur_memory_rl/data/hybrid_trace.py",
                  Path(__file__).parents[1] / "src/eur_memory_rl/data/lossy_trace_v3.py",
                  Path(__file__).parents[1] / "src/eur_memory_rl/policies/prompts.py",
                  Path(__file__).parents[1] / "src/eur_memory_rl/policies/generator.py")},
              "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest() if args.inventory else None}
    namespaces = [r["evidence_namespace"] for r in rows if r["status"] == "success"]
    report["evidence_namespaces_unique"] = len(namespaces) == len(set(namespaces))
    if not report["evidence_namespaces_unique"]:
        raise ValueError("Unexpected evidence namespace collision; do not use this output")
    atomic_json(args.output / "summary.json", report)
    # A directly reusable JSONL of successful Source objects. The paths above are diagnostics only.
    with (args.output / "sources.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            if row["status"] == "success":
                stream.write(json.dumps(read_json(Path(row["source_json"])), ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if any(r["status"] == "error" for r in rows) else 0


def asdict_config(config):
    from dataclasses import asdict
    return asdict(config)


if __name__ == "__main__":
    raise SystemExit(main())
