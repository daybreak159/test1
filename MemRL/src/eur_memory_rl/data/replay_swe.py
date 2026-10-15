"""Predeclared first-task replay for infrastructure acceptance, not a fresh rollout."""
import json
from pathlib import Path

from ..contracts import Execution
from ..util import digest, file_hash, read_json


def replay_first_execution(runtime, task, execution_id, seed, read, replay):
    if task.id != replay["task_id"] or task.identity.split != "train":
        raise ValueError("Replay is restricted to the declared first training task")
    path = Path(replay["trajectory"])
    if file_hash(path) != replay["sha256"]:
        raise ValueError("Pinned no-memory trajectory changed")
    trace = read_json(path)
    trace = {"instance_id": task.id, "messages": trace["messages"],
             "info": {"submission": trace["info"]["submission"],
                      "exit_status": trace["info"]["exit_status"]}}
    if trace["info"]["exit_status"] != "Submitted":
        raise ValueError("This acceptance replay requires a submitted source")
    reset = read_json(path.parent / "reset_check.json")
    if reset["base_commit"] != task.public_payload["base_commit"]:
        raise ValueError("No-memory replay base commit mismatch")
    slate = read(task.turns[0])
    if slate.memory_block.strip():
        # An empty renderer may include a fixed empty-bank declaration.
        if slate.selected_ids:
            raise ValueError("Cannot replay a no-memory source with nonempty selected memories")
    patch = trace["info"]["submission"]
    events = tuple({"event_origin": "replayed_message", "role": m["role"], "text": m["content"]}
                   for m in trace["messages"] if m["role"] != "system")
    env_config = read_json(runtime.cfg["environment"]["image_manifest"])
    events += ({"event_origin": "final_artifact", "artifact": {"patch": patch,
               "base_commit": reset["base_commit"], "image": env_config["swe"]["allowed_images"][task.id]}},)
    runtime.hybrid_processor.replayed_traces[execution_id] = trace
    timing = read_json(path.parent / "timing.json")
    calls = [json.loads(line) for line in (path.parent / "solver_calls.jsonl").read_text().splitlines() if line.strip()]
    inputs = sum(row["response"].get("usage", {}).get("prompt_tokens", 0) for row in calls)
    outputs = sum(row["response"].get("usage", {}).get("completion_tokens", 0) for row in calls)
    # Execution records keep historical cost for reward accounting. The separate
    # reused_executions audit declares zero new Solver calls in this run.
    return Execution(execution_id, task.id, "Replayed historical no-memory submission", events,
                     task.identity.history_or_environment_hash, digest([reset["base_commit"], patch]),
                     runtime.adapter.profile.hash, digest(["offline-first-task-replay", replay["sha256"]]),
                     seed, "submitted", len(calls), inputs, outputs, float(inputs+outputs+len(calls)),
                     timing["elapsed_seconds"], 0.0, True)
