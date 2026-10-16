"""Run the actual mini-swe-agent loop in its own Python environment.

JSON lines on stdin/stdout are local RPC, not the Solver's output protocol.
This file is executed directly, so the worker need not import the HF package.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path


def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def rpc(kind, **payload):
    send({"kind": kind, **payload})
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("MemRL bridge closed")
    reply = json.loads(line)
    if "stop" in reply:
        from minisweagent.exceptions import LimitsExceeded

        status = reply["stop"]
        raise LimitsExceeded({"role": "exit", "content": status,
                              "extra": {"exit_status": status, "submission": ""}})
    return reply


def load_api():
    # Package startup banners must never enter the RPC stream.
    with contextlib.redirect_stdout(sys.stderr):
        import minisweagent
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
    return (minisweagent, DefaultAgent, DockerEnvironment, FormatError,
            format_observation_messages, parse_regex_actions)


def main():
    mini, DefaultAgent, DockerEnvironment, FormatError, format_observations, parse_actions = load_api()
    if sys.argv[1:] == ["--check"]:
        # Verify the supported interface; no model, Docker or network request.
        if not hasattr(DefaultAgent, "execute_actions"):
            raise RuntimeError("mini-swe-agent 2.x text action API is required")
        send({"mini_version": mini.__version__, "runner": "DefaultAgent", "protocol": "bash-v1"})
        return
    config = json.loads(sys.stdin.readline())

    class BridgeModel:
        def get_template_vars(self):
            return {}

        def serialize(self):
            return {"info": {"config": {"model_type": "MemRLChatModelBridge"}}}

        def format_message(self, *, role, content, **kwargs):
            return {"role": role, "content": content, **kwargs}

        def query(self, messages):
            reply = rpc("model", messages=[{"role": m["role"], "content": m["content"]}
                                           for m in messages if m["role"] != "exit"])
            message = {"role": "assistant", "content": reply["text"],
                       "extra": {"cost": reply["cost"], "finish_reason": reply["finish_reason"],
                                 "input_tokens": reply["input_tokens"],
                                 "output_tokens": reply["output_tokens"], "reasoning": reply.get("reasoning", "")}}
            try:
                # Never execute a command from a truncated response, even if a
                # closed bash block happens to occur before the truncated tail.
                if reply["finish_reason"] == "length":
                    raise FormatError({"role": "user", "content":
                        "Your response reached the token limit. No command was executed. "
                        "Reply more briefly with exactly one complete bash code block.",
                        "extra": {"interrupt_type": "FormatError"}})
                message["extra"]["actions"] = parse_actions(
                    reply["text"], action_regex=r"```bash\s*\n(.*?)\n```",
                    format_error_template=("Provide exactly ONE bash code block; found {{actions|length}}. "
                        "To submit, run: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
                    template_kwargs={"finish_reason": reply["finish_reason"]})
            except FormatError as exc:
                # Keep billed/empty/malformed responses in the native trajectory.
                # DefaultAgent charges FormatError's first message once.
                raise FormatError(message, *exc.messages) from exc
            return message

        def format_observation_messages(self, message, outputs, template_vars=None):
            return format_observations(outputs, observation_template=(
                "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"),
                template_vars=template_vars)

    class BridgeEnvironment:
        def get_template_vars(self):
            return {}

        def serialize(self):
            return {"info": {"config": {"environment_type": "MemRLPinnedRemoteEnvironment"}}}

        def execute(self, action):
            output = rpc("tool", action=action)
            # Use mini's own submission-marker semantics, without creating a
            # second Docker container or invoking its destructor/cleanup.
            DockerEnvironment._check_finished(self, output)
            return output

    class AuditedAgent(DefaultAgent):
        def add_messages(self, *messages):
            for message in messages:
                if message.get("extra", {}).get("interrupt_type") == "FormatError":
                    rpc("control", text=message["content"])
            return super().add_messages(*messages)

    output = config.get("trajectory_path")
    agent = AuditedAgent(
        BridgeModel(), BridgeEnvironment(), system_template="{{ memrl_system }}",
        instance_template="{{ task }}", step_limit=config["model_calls"], cost_limit=0,
        # Exact wall/token/command budgets are enforced by the parent bridge.
        max_consecutive_format_errors=config["max_consecutive_format_errors"],
        output_path=Path(output) if output else None)
    result = agent.run(config["task"], memrl_system=config["system_prompt"])
    # Include provenance and the complete effective budget in the native file.
    agent.save(agent.config.output_path, {"info": {"memrl": config["provenance"]}})
    send({"kind": "result", "result": result, "mini_version": mini.__version__})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Detailed traceback stays in stderr, never in the Solver's context.
        import traceback

        traceback.print_exc(file=sys.stderr)
        send({"kind": "error", "error_type": type(exc).__name__})
        sys.exit(1)
