"""Offline V3: command references, explicit heredoc omissions, and task-only input."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from .tool_feedback import compress_mini_trace

PROFILE = "swe-lossy-trace-v3"


def _headers(line):
    """Recognize literal shell heredoc operators outside quoted strings/comments.

    Supports simple unquoted, single/double-quoted delimiters and <<-.
    Complex shell expansions are deliberately not interpreted or executed.
    """
    result = []
    index = 0
    quote = None
    while index < len(line):
        char = line[index]
        if char == "\\" and quote != "'":
            index += 2
            continue
        if quote:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        if line.startswith("<<<", index):
            index += 3
            continue
        if line.startswith("<<", index):
            match = re.match(
                r"<<(?P<tabs>-?)[ \t]*(?:(?P<q>['\"])(?P<quoted>[A-Za-z_]\w*)(?P=q)"
                r"|(?P<bare>[A-Za-z_]\w*))(?=$|[\s;&|<>)])",
                line[index:],
            )
            if match:
                result.append((match["quoted"] or match["bare"], bool(match["tabs"])))
                index += match.end()
                continue
        index += 1
    return result


def heredoc_body_spans(command):
    """Return literal body spans; skip unclosed groups instead of guessing."""
    lines = command.splitlines(keepends=True)
    offsets = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    spans, unresolved = [], []
    index = 0
    while index < len(lines):
        headers = _headers(lines[index])
        if not headers:
            index += 1
            continue
        cursor = index + 1
        pending = []
        for delimiter, strip_tabs in headers:
            start = cursor
            while cursor < len(lines):
                candidate = lines[cursor].rstrip("\r\n")
                if strip_tabs:
                    candidate = candidate.lstrip("\t")
                if candidate == delimiter:
                    break
                cursor += 1
            if cursor == len(lines):
                unresolved.append({"header_line": index, "delimiter": delimiter,
                                   "reason": "closing_delimiter_missing; command retained"})
                break
            lo = offsets[start] if start < len(lines) else len(command)
            pending.append((lo, offsets[cursor], delimiter))
            cursor += 1
        else:
            spans.extend(pending)
            index = cursor
            continue
        # A malformed multiline input must not trigger deletion in its remaining text.
        break
    return spans, unresolved


def omit_assistant_heredocs(content):
    edits, unresolved = [], []
    for block in re.finditer(r"```bash[ \t]*\r?\n(.*?)```", content, re.S):
        spans, unknown = heredoc_body_spans(block[1])
        unresolved.extend(unknown)
        for lo, hi, delimiter in spans:
            body = block[1][lo:hi]
            edits.append({"start": block.start(1) + lo, "end": block.start(1) + hi,
                          "delimiter": delimiter, "omitted_characters": len(body),
                          "body_sha256": hashlib.sha256(body.encode()).hexdigest()})
    output = content
    for edit in reversed(edits):
        output = output[:edit["start"]] + "[HEREDOC_BODY_OMITTED]\n" + output[edit["end"]:]
    return output, {"omissions": edits, "unresolved": unresolved}


def clean_task_template_comments(content):
    """Remove complete HTML comments outside Markdown code; never infer semantics."""
    token = re.compile(
        r"(?P<fence>^ {0,3}(?P<marker>`{3,}|~{3,})[^\r\n]*(?:\r?\n|$))"
        r"|(?P<indented>^(?: {4}|\t)[^\r\n]*(?:\r?\n|$))"
        r"|(?P<inline>`+)|(?P<comment><!--)", re.M)
    parts, omissions = [], []

    def append(kind, text):
        if not text:
            return
        if parts and parts[-1][0] == kind:
            parts[-1] = (kind, parts[-1][1] + text)
        else:
            parts.append((kind, text))

    cursor = 0
    while match := token.search(content, cursor):
        append("plain", content[cursor:match.start()])
        if match["fence"]:
            marker = match["marker"]
            closing = re.compile(r"^ {0,3}" + re.escape(marker[0]) +
                "{" + str(len(marker)) + r",}[ \t]*(?:\r?\n|$)", re.M)
            end = closing.search(content, match.end())
            cursor = end.end() if end else len(content)
            append("code", content[match.start():cursor])
        elif match["indented"]:
            cursor = match.end()
            append("code", match[0])
        elif match["inline"]:
            end = re.search(r"(?<!`)" + re.escape(match[0]) + r"(?!`)", content[match.end():])
            cursor = match.end() + end.end() if end else match.end()
            append("code" if end else "plain", content[match.start():cursor])
        else:
            end = content.find("-->", match.end())
            if end < 0:
                # Retain incomplete comments literally; do not guess their boundary.
                append("code", content[match.start():])
                cursor = len(content)
                break
            cursor = end + 3
            body = content[match.start():cursor]
            omissions.append({"start": match.start(), "end": cursor,
                "omitted_characters": len(body), "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "reason": "html_comment_outside_markdown_code"})
    append("plain", content[cursor:])
    cleaned = "".join(re.sub(r"(\r?\n)(?:[ \t]*\r?\n){2,}", lambda m: m[1] * 2, text)
                      if kind == "plain" else text for kind, text in parts)
    return cleaned, {"html_comments_removed": len(omissions), "comment_omissions": omissions,
        "comment_characters_removed": sum(row["omitted_characters"] for row in omissions),
        "blank_line_characters_removed": len(content) - len(cleaned) - sum(row["omitted_characters"] for row in omissions)}


def task_description_only(content):
    matches = list(re.finditer(r"<pr_description>(.*?)</pr_description>", content, re.S))
    if len(matches) != 1:
        return content, {"recognized": False, "removed_characters": 0,
                         "reason": "missing_or_multiple_pr_description; retained original"}
    match = matches[0]
    cleaned, cleanup = clean_task_template_comments(match[1])
    return cleaned, {"recognized": True, "source_description_span": [match.start(1), match.end(1)],
        "removed_characters": len(content) - len(cleaned),
        "instructions_removed_characters": len(content) - len(match[1]),
        "description_cleanup": cleanup}


def build_v3(trace, max_feedback_lines=8, max_feedback_chars=1024):
    v2, v2_audit = compress_mini_trace(trace, max_feedback_lines, max_feedback_chars)
    result = deepcopy(v2)
    result["profile"] = PROFILE
    feedback_audits = {row["message_index"]: row for row in v2_audit["records"]}
    for row in result["messages"]:
        if row["event_type"] == "tool_feedback":
            value = row["content"]
            row["content"] = {"command_ref": feedback_audits[row["message_index"]]["command_source_message_index"],
                              "exit_code": value["exit_code"], "command_ok": value["command_ok"],
                              "key_feedback": value["key_feedback"]}
    references_only = deepcopy(result)
    scripts = []
    for row in result["messages"]:
        if row["role"] == "assistant":
            value, audit = omit_assistant_heredocs(row["content"])
            row["content"] = value
            if audit["omissions"] or audit["unresolved"]:
                scripts.append({"message_index": row["message_index"], **audit})
    scripts_omitted = deepcopy(result)
    task = next(row for row in result["messages"] if row["event_type"] == "task")
    task["content"], task_audit = task_description_only(task["content"])
    audit = {"profile": PROFILE, "v2_feedback_audit": v2_audit,
             "heredoc_records": scripts, "task": task_audit,
             "command_refs_only": True, "model_calls": 0, "lossy": True,
             "message_order_preserved": [row["message_index"] for row in v2["messages"]] ==
                                        [row["message_index"] for row in result["messages"]],
             "final_patch_unchanged": result["final_patch"] == v2["final_patch"],
             "feedback_values_unchanged": all(row["content"][key] == other["content"][key]
                 for row, other in zip(v2["messages"], result["messages"])
                 if row["event_type"] == "tool_feedback"
                 for key in ("exit_code", "command_ok", "key_feedback"))}
    stages = {"v2": v2, "references_only": references_only, "heredocs_omitted": scripts_omitted,
              "task_only": result}
    return result, audit, stages


def render_v3_sections(document):
    """Short labels replace repeated JSON keys in the E-facing readable view."""
    sections = [("structure", document["profile"] + "\nTASK " + document["instance_id"] +
        "\nA=assistant; U=feedback; ref=original assistant message; "
        "exit/ok describe shell exit, not correctness; ?=unknown; bodies explicitly omitted.\n")]
    for row in document["messages"]:
        index, content = row["message_index"], row["content"]
        if row["event_type"] == "task":
            sections.extend([("structure", f"\n[TASK {index}]\n"), ("task", content)])
        elif row["role"] == "assistant":
            sections.extend([("structure", f"\n[A {index}]\n"), ("assistant", content)])
        elif row["event_type"] == "final_artifact":
            sections.append(("structure", f"\n[U {index} FINAL_PATCH]\n"))
        else:
            value = row["content"]
            ref = value["command_ref"] if value["command_ref"] is not None else "?"
            code = value["exit_code"] if value["exit_code"] is not None else "?"
            ok = "?" if value["command_ok"] is None else str(value["command_ok"]).lower()
            sections.append(("structure", f"\n[U {index} ref={ref} exit={code} ok={ok}]\n"))
            if value["key_feedback"]:
                sections.append(("key_feedback", value["key_feedback"]))
    sections.extend([("structure", "\n[FINAL_PATCH]\n"), ("final_patch", document["final_patch"]),
                     ("structure", "\n[FINISH_REASON]\n" + str(document["finish_reason"]))])
    return sections


def render_v3(document):
    return "".join(text for _, text in render_v3_sections(document))
