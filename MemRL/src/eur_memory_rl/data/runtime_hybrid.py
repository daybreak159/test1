"""Explicit completed-execution conversion for the SWE hybrid E preprocessor."""
from dataclasses import replace
from pathlib import Path
import json

from .hybrid_trace import CompactionSkip, HybridConfig, compact_trajectory
from ..util import atomic_json, digest, plain


def completed_execution_trace(task, execution):
    """Convert public events only; never import grader logs or private tests."""
    messages, origins, patch = [], [], ""
    for i, event in enumerate(execution.public_trace):
        kind = event.get("event_origin")
        if kind == "external_request":
            messages.append({"role": "user", "content": "<pr_description>" + event["text"] + "</pr_description>"})
        elif kind == "assistant_output":
            text = event["text"]
            try:
                value = json.loads(text)
            except (ValueError, TypeError):
                value = None
            if isinstance(value, dict) and isinstance(value.get("action"), dict):
                command = value["action"].get("command")
                if isinstance(command, str):
                    text = "```bash\n" + command + "\n```"
            messages.append({"role": "assistant", "content": text})
        elif kind == "tool_result":
            observation = event["observation"]
            if isinstance(observation, dict) and "exit_code" in observation:
                text = (f'<returncode>{int(observation["exit_code"])}</returncode>\n'
                        + "<output>\n" + str(observation.get("stdout", ""))
                        + "\n" + str(observation.get("stderr", "")) + "\n</output>")
            else:
                text = json.dumps(observation, ensure_ascii=False)
            messages.append({"role": "tool", "content": text})
        elif kind == "internal_control":
            messages.append({"role": "tool", "content": event["text"]})
        elif kind == "final_artifact":
            patch = event["artifact"].get("patch", "")
            continue
        else:
            raise ValueError(f"Unrecognized SWE public event: {kind}")
        origins.append({"message_index": len(messages)-1, "event_index": i})
    return {"instance_id": task.id, "messages": messages,
            "info": {"submission": patch, "exit_status": execution.finish_reason}}, origins


class HybridSourceProcessor:
    def __init__(self, tokenizer, config, output_dir):
        self.tokenizer, self.config = tokenizer, HybridConfig(**config)
        self.output_dir = Path(output_dir)
        self.cache = {}
        self.replayed_traces = {}

    def __call__(self, task, execution, source):
        if not execution.completed:
            raise ValueError("Hybrid E preprocessing requires a completed execution")
        trace = self.replayed_traces.get(execution.execution_id)
        if trace is None:
            trace, origins = completed_execution_trace(task, execution)
        else:
            origins = [{"message_index": i, "original_message_index": i}
                       for i in range(len(trace["messages"]))]
        key = digest([source, trace, self.config])
        if key in self.cache:
            return self.cache[key]
        # The input is a completed task. The original task's cutoff is a
        # task-prefix coordinate, not a mini-message index. Map it explicitly
        # for this local completed-source conversion, then restore it so probe
        # visibility certificates retain their original coordinate system.
        original_cutoff = source.identity.visibility_cutoff
        converted = replace(source, identity=replace(source.identity,
                              visibility_cutoff=len(trace["messages"])-1))
        folder = self.output_dir / digest(execution.execution_id)[:20]
        try:
            compact, _, prompt, report = compact_trajectory(
                trace, self.tokenizer, config=self.config, base_source=converted)
        except CompactionSkip as exc:
            atomic_json(folder / "report.json", {**exc.report, "status": "skipped", "reason": exc.reason,
                        "execution_id": execution.execution_id, "message_origins": origins})
            raise
        compact = replace(compact, identity=replace(compact.identity, visibility_cutoff=original_cutoff))
        # Restoring cutoff metadata can change tokenization: measure the real
        # final request again rather than trusting the conversion's count.
        from .hybrid_trace import prepared_e_messages
        from ..policies.prompts import encode_messages
        ids, prompt_hash, prompt = encode_messages(self.tokenizer,
              prepared_e_messages(compact, self.config.prompt_profile), 10**12)
        report.update(execution_id=execution.execution_id, message_origins=origins,
                      original_visibility_cutoff=original_cutoff,
                      cutoff_mapping="completed execution -> final message; original coordinate restored",
                      e_input_tokens=len(ids), prompt_hash=prompt_hash, source_hash=digest(compact))
        if len(ids) > self.config.max_input_tokens:
            atomic_json(folder / "report.json", {**report, "status": "skipped"})
            raise CompactionSkip("restored_metadata_over_input_budget", report)
        atomic_json(folder / "source.json", plain(compact))
        atomic_json(folder / "input_ids.json", ids)
        atomic_json(folder / "report.json", report)
        self.cache[key] = compact
        return compact
