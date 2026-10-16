"""MemRL contracts around mini-swe-agent's actual DefaultAgent runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time

from ..contracts import Execution
from ..util import ConfigurationError, InfrastructureError, atomic_json, digest, plain, seed_for
from .base import DomainAdapter

HARNESS_REVISION = "mini-swe-agent-bash-v1-memrl-bridge-v1"
DEFAULT_PYTHON = "/home/ubuntu/coding/venvs/swebench/bin/python"
WORKER = Path(__file__).with_name("mini_swe_worker.py")
SYSTEM = """You are a software engineer resolving an issue in a repository.
Inspect the relevant code, reproduce the problem when practical, make a focused
fix to non-test source files, and verify it with relevant checks. Do not modify
tests or evaluation configuration to make a failure disappear.
Each response must contain exactly ONE bash code block with the command to run.
You may briefly explain your next action before the code block. Commands run
in independent shells in /testbed; use cd or environment variables explicitly
when needed. Use the tool observations as evidence, not your own claims.
When finished, submit by running this command in a bash code block:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
Submission ends your execution; the frozen grader evaluates the actual patch.
Reference memories below may help, but must be checked against this repository.
"""


def configure_swe_runner(cfg):
    """Resolve the new SWE protocol before configuration/profile hashing."""
    if cfg.get("domain") != "swe":
        return cfg
    consumer = cfg["consumer"]
    consumer["runner"] = "mini_swe_agent"
    settings = consumer.setdefault("mini_swe", {})
    settings.setdefault("python", DEFAULT_PYTHON)
    settings.setdefault("max_consecutive_format_errors", 0)
    profile = consumer["profile"]
    profile["harness_revision"] = HARNESS_REVISION
    decoding = profile["decoding"]
    for key in ("response_format", "guided_json", "structured_outputs", "tools", "tool_choice"):
        decoding.pop(key, None)
    return cfg


def check_worker(python=DEFAULT_PYTHON):
    try:
        result = subprocess.run([python, str(WORKER), "--check"], capture_output=True,
                                text=True, timeout=30, env=worker_environment(), check=True)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ConfigurationError(
            "mini-swe-agent worker unavailable: configure consumer.mini_swe.python "
            "with a separate Python environment containing mini-swe-agent 2.x") from exc


def worker_environment():
    env = os.environ.copy()
    env["MSWEA_SILENT_STARTUP"] = "1"
    # Do not inject the HF runtime's dependency fallbacks into the mini worker.
    env.pop("PYTHONPATH", None)
    return env


class MiniSWEAdapter(DomainAdapter):
    domain = "swe"

    def __init__(self, *args, worker_python=DEFAULT_PYTHON, trajectory_dir=None,
                 max_consecutive_format_errors=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.worker_python = worker_python
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.max_consecutive_format_errors = max_consecutive_format_errors

    def consumer_system(self):
        return SYSTEM

    def run_consumer(self, task, execution_id, seed, read, budget=None):
        if len(task.turns) != 1:
            raise ConfigurationError("SWE mini-swe-agent requires one external task turn")
        budget = budget or self.profile.budget
        started = time.perf_counter()
        environment = self.clone_or_reset(task, execution_id, seed)
        initial = environment.initial_state_hash
        calls = inputs = outputs = steps = 0
        money = 0.0
        answer, reason = "", "budget_exhausted"
        process = None
        trace = []

        def remaining():
            return None if budget.wall_seconds is None else budget.wall_seconds - (time.perf_counter() - started)

        def reply(value):
            try:
                process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise InfrastructureError("mini-swe-agent worker bridge closed") from exc

        try:
            turn = task.turns[0]
            slate = read(turn)  # Exactly once; tools never call E/U/R.
            exposure = digest([turn, slate.memory_block, environment.observation])
            trace.append({"event_origin": "external_request", "external_turn_id": turn.external_turn_id,
                          "text": turn.text, "observation": environment.observation})
            trajectory_path = None
            if self.trajectory_dir:
                # Never turn an untrusted execution_id into a filesystem path.
                self.trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = str(self.trajectory_dir / (digest(execution_id) + ".traj.json"))
            payload = {"task": turn.public_history + "\n" + turn.text + "\nInitial observation:\n"
                       + str(environment.observation), "system_prompt": SYSTEM + "\n" + slate.memory_block,
                       "model_calls": budget.model_calls, "trajectory_path": trajectory_path,
                       "max_consecutive_format_errors": self.max_consecutive_format_errors,
                       "provenance": {"execution_id": execution_id, "task_id": task.id,
                                      "seed": seed, "budget": plain(budget),
                                      "consumer_profile_hash": self.profile.hash}}
            # The worker injects the system/memory as a template value, so
            # braces/Jinja syntax inside memory are not rendered/evaluated.
            stderr_file = (Path(trajectory_path).with_suffix(".stderr.log").open("w+", encoding="utf-8")
                           if trajectory_path else tempfile.TemporaryFile(mode="w+t", encoding="utf-8"))
            with stderr_file as stderr:
                process = subprocess.Popen([self.worker_python, str(WORKER)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8", bufsize=1,
                    env=worker_environment())
                reply(payload)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        # Worker only computes/parses/saves here; API and tool
                        # waits happen in this parent with bounded timeouts.
                        if not selector.select(timeout=30):
                            raise InfrastructureError("mini-swe-agent worker did not respond within 30 seconds")
                        line = process.stdout.readline()
                        if not line:
                            raise InfrastructureError("mini-swe-agent worker exited without a result")
                        try:
                            event = json.loads(line)
                        except ValueError as exc:
                            raise InfrastructureError("Invalid mini-swe-agent worker response") from exc
                        kind = event.get("kind")
                        if kind == "result":
                            status = event["result"].get("exit_status")
                            reasons = {"Submitted": "submitted", "LimitsExceeded": "budget_exhausted",
                                       "budget_exhausted": "budget_exhausted", "TimeExceeded": "wall_budget_exhausted",
                                       "wall_budget_exhausted": "wall_budget_exhausted",
                                       "RepeatedFormatError": "format_error_exhausted"}
                            if status not in reasons:
                                raise InfrastructureError(f"Unexpected mini-swe-agent exit status: {status}")
                            reason = reasons[status]
                            answer = str(event["result"].get("submission", ""))
                            break
                        if kind == "error":
                            raise InfrastructureError("mini-swe-agent worker failed: " + str(event.get("error_type")))
                        if kind == "control":
                            trace.append({"event_origin": "internal_control", "text": event["text"]})
                            reply({})
                            continue
                        if kind not in {"model", "tool"}:
                            raise InfrastructureError("Unknown mini-swe-agent bridge operation")
                        left = remaining()
                        if left is not None and left <= 0:
                            reply({"stop": "wall_budget_exhausted"})
                            continue
                        if kind == "model":
                            if calls >= budget.model_calls or (budget.output_token_policy == "aggregate"
                                                              and outputs >= budget.output_tokens):
                                reply({"stop": "budget_exhausted"})
                                continue
                            client = getattr(self.consumer, "client", None)
                            previous_timeout = getattr(client, "timeout", None)
                            if previous_timeout is not None and left is not None:
                                client.timeout = min(previous_timeout, left)
                            try:
                                completion = self.consumer.complete(event["messages"],
                                    budget.output_tokens if budget.output_token_policy == "per_response"
                                    else budget.output_tokens - outputs,
                                    seed_for(seed, "consumer", calls), f"{execution_id}:model:{calls}")
                            except InfrastructureError:
                                if remaining() is not None and remaining() <= 0:
                                    reply({"stop": "wall_budget_exhausted"})
                                    continue
                                raise
                            finally:
                                if previous_timeout is not None:
                                    client.timeout = previous_timeout
                            calls += 1
                            inputs += completion.input_tokens
                            outputs += completion.output_tokens
                            money += completion.monetary_cost
                            trace.append({"event_origin": "assistant_output", "text": completion.text})
                            reply({"text": completion.text, "cost": completion.monetary_cost,
                                   "finish_reason": getattr(completion, "finish_reason", "stop"),
                                   "reasoning": getattr(completion, "reasoning", ""),
                                   "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens})
                        else:
                            steps += 1
                            observation, _ = environment.step(event["action"], steps,
                                min(budget.command_timeout_seconds, left)
                                if left is not None else budget.command_timeout_seconds)
                            environment.observation = observation
                            trace.append({"event_origin": "tool_result", "action": event["action"],
                                          "observation": observation})
                            if not isinstance(observation, dict) or "exit_code" not in observation:
                                raise InfrastructureError("SWE mini worker requires structured command observations")
                            reply({"returncode": observation["exit_code"],
                                   "output": str(observation.get("stdout", ""))
                                   + str(observation.get("stderr", ""))})
                process.wait(timeout=5)
                if process.returncode:
                    raise InfrastructureError("mini-swe-agent worker returned a failure exit code")
            if budget != self.profile.budget and reason in {"budget_exhausted", "wall_budget_exhausted"}:
                reason = "probe_cutoff"
            artifact = environment.finalize(answer)
            trace.append({"event_origin": "final_artifact", "artifact": artifact})
            final_state = environment.state_hash
            if trajectory_path:
                native = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
                native["info"]["submission"] = artifact.get("patch", "")
                native["info"]["memrl"].update(finish_reason=reason, final_artifact=artifact,
                                              initial_state_hash=initial, final_state_hash=final_state,
                                              solver_submission=answer)
                atomic_json(trajectory_path, native)
        except OSError as exc:
            raise InfrastructureError("Unable to start mini-swe-agent worker; check consumer.mini_swe.python") from exc
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                process.stdin.close()
                process.stdout.close()
            environment.close()
        return Execution(execution_id, task.id, answer, tuple(trace), initial, final_state,
                         self.profile.hash, digest([[exposure], budget, self.profile.hash]), seed,
                         reason, calls, inputs, outputs, float(inputs + outputs + calls),
                         time.perf_counter() - started, money, reason != "probe_cutoff")
