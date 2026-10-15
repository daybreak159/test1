"""Lossy, deterministic excerpts of saved shell feedback, without model calls.

This is an offline view of a mini-swe-agent trace, not an OpenAI messages payload.
An exit code describes the saved shell command, not patch or test correctness.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re

PROFILE = "swe-tool-feedback-v2"
RETURN_CODE = re.compile(r"\A\s*<returncode>\s*(-?\d+)\s*</returncode>")
EXCEPTION = re.compile(
    r"^(?:E\s+)?(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*(?:Error|Exception|Interrupt|Exit):(?:\s|$)"
)
PYTEST_TOTAL = re.compile(
    r"^(?:=+\s*)?(?:\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected)"
    r"(?:\s*,\s*|\s+in\s+|\s*=|$))"
)
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR)\s+\S+.*(?:::|\s+-\s+)")
SHELL_ERROR = re.compile(
    r"(?:command not found|Permission denied|No such file or directory|"
    r"cannot (?:access|open|stat)|No module named|ModuleNotFoundError|"
    r"Connection refused|Connection timed out|Read timed out|"
    r"Operation timed out|Cannot connect to the Docker daemon|"
    r"Could not find code to replace)", re.I
)
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _notification(text):
    """Only explicit notifications; no inference from missing output."""
    if text.startswith("Please always provide EXACTLY ONE action"):
        return "format_error", text.splitlines()[0]
    lines = text.splitlines()
    explicit = [line.strip() for line in lines if re.search(
        r"(?:Solver stopped because|wall.clock budget.*exhausted|"
        r"command.*timed out|command.*timeout|execution.*timed out)", line, re.I)]
    if explicit:
        return "timeout_or_budget_notification", explicit[-1]
    return "unreported", ""


def summarize_feedback(text, max_lines=8, max_chars=1024):
    if max_lines < 1 or max_chars < 64:
        raise ValueError("Feedback bounds must be positive; max_chars must be at least 64")
    match = RETURN_CODE.match(text)
    code = int(match.group(1)) if match else None
    status, notice = ("exit_reported", "") if match else _notification(text)
    candidates = []
    for number, raw in enumerate(text.splitlines()):
        line = ANSI.sub("", raw).strip()
        if EXCEPTION.match(line):
            candidates.append((number, "exception", line))
        elif PYTEST_TOTAL.match(line):
            candidates.append((number, "pytest_summary", line))
        elif PYTEST_FAILED.match(line):
            candidates.append((number, "failed_test", line))
        elif (SHELL_ERROR.search(line) and not line.startswith(("#", "raise ", "assert ", 'print(', "print "))):
            candidates.append((number, "shell_or_runtime_error", line))
        elif re.match(r"^(?:ERROR|FATAL):\s+", line):
            candidates.append((number, "explicit_error", line))
    # Last exception and last overall pytest summary get priority. Nested pytest
    # failures are retained as observations, never converted into command_ok.
    selected = []
    for kind in ("exception", "pytest_summary", "failed_test", "shell_or_runtime_error", "explicit_error"):
        rows = [row for row in candidates if row[1] == kind]
        selected.extend(rows[-1:] if kind in {"exception", "pytest_summary"} else rows[-4:])
    if notice:
        selected.insert(0, (-1, "notification", notice))
    unique = []
    seen = set()
    for row in selected:
        if row[2] not in seen:
            seen.add(row[2]); unique.append(row)
    clipped_by_lines = len(unique) > max_lines
    unique = unique[:max_lines]
    unique.sort(key=lambda row: row[0])
    feedback = "\n".join(row[2] for row in unique)
    clipped_by_chars = len(feedback) > max_chars
    marker = "\n[feedback clipped by fixed limit]"
    if clipped_by_chars:
        feedback = feedback[:max_chars - len(marker)] + marker
    elif clipped_by_lines:
        feedback = feedback[:max_chars - len(marker)] + marker
    result = {"exit_code": code, "command_ok": code == 0 if code is not None else None,
              "key_feedback": feedback}
    audit = {"execution_status": status, "selected_source_lines": [row[0] for row in unique],
             "matched_kinds": [row[1] for row in unique],
             "feedback_clipped": clipped_by_chars or clipped_by_lines,
             "source_already_elided": "<elided_chars>" in text,
             "original_feedback_characters": len(text)}
    return result, audit


def command_from_message(message):
    if not message or message["role"] != "assistant":
        return None, "no_immediately_preceding_assistant"
    blocks = re.findall(r"```bash\s*\n(.*?)```", message["content"], re.S)
    if len(blocks) != 1:
        return None, "missing_or_ambiguous_bash_block"
    return blocks[0].strip(), "one_bash_block"


def compress_mini_trace(trace, max_lines=8, max_chars=1024, max_command_chars=512):
    if max_command_chars < 128:
        raise ValueError("max_command_chars must be at least 128")
    messages = trace["messages"]
    if any(not isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Only saved text messages are supported")
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
    if first_user is None:
        raise ValueError("No initial task message")
    submission = trace.get("info", {}).get("submission", "")
    final = len(messages) - 1
    has_final_patch = bool(submission.strip()) and submission.lstrip().startswith("diff --git ")
    final_matches = has_final_patch and messages[final]["content"] == submission
    task = trace.get("instance_id", "unknown")
    output, audits = [], []
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "system":
            continue
        row = {"message_index": i, "role": role}
        if i == first_user:
            row.update(event_type="task", content=content)
        elif role == "assistant":
            row.update(event_type="assistant_output", content=content)
        elif role == "user" and final_matches and i == final:
            row.update(event_type="final_artifact", content={"final_patch_ref": "final_patch"})
        elif role == "user":
            command, association = command_from_message(messages[i - 1] if i else None)
            value, audit = summarize_feedback(content, max_lines, max_chars)
            command_clipped = command is not None and len(command) > max_command_chars
            display_command = command
            if command_clipped:
                marker = f"\n[remaining command omitted here; full command in assistant message {i - 1}]"
                display_command = command[:max_command_chars - len(marker)] + marker
            row.update(event_type="tool_feedback", content={"command": display_command, **value})
            audits.append({"message_index": i, "command_association": association,
                           "command_display_clipped": command_clipped,
                           "command_source_message_index": i - 1 if command is not None else None,
                           "full_command_sha256": hashlib.sha256(command.encode()).hexdigest() if command is not None else None,
                           "original_content_sha256": hashlib.sha256(content.encode()).hexdigest(), **audit})
        else:
            raise ValueError("Unsupported role: " + role)
        output.append(row)
    result = {"profile": PROFILE, "instance_id": task, "messages": output,
              "finish_reason": trace.get("info", {}).get("exit_status"),
              "final_patch": submission if has_final_patch else ""}
    audit = {"tool_results_converted": len(audits), "records": audits,
             "original_roles": dict(Counter(m["role"] for m in messages)),
             "assistant_unchanged": all(row["content"] == messages[row["message_index"]]["content"]
                                        for row in output if row["role"] == "assistant"),
             "task_unchanged": next(row for row in output if row["event_type"] == "task")["content"] == messages[first_user]["content"],
             "final_patch_preserved": has_final_patch, "final_message_replaced_by_reference": final_matches}
    return result, audit


def render_lossy_trace(document):
    """Readable full view; each tool feedback is displayed as a JSON object."""
    lines = [document["profile"], "TASK: " + document["instance_id"],
             "command_ok describes shell exit only; omitted output is unavailable."]
    for row in document["messages"]:
        lines.append(f'\n[MESSAGE {row["message_index"]} {row["role"]} {row["event_type"]}]')
        value = row["content"]
        if isinstance(value, str):
            lines.append(value)
        else:
            import json
            lines.append(json.dumps(value, ensure_ascii=False, indent=2))
    lines.extend(["\n[FINAL_PATCH]", document["final_patch"],
                  "\n[FINISH_REASON]", str(document["finish_reason"])])
    return "\n".join(lines)
