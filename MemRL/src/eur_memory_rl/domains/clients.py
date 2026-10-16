from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..util import InfrastructureError, canonical


class JSONClient:
    def __init__(self, base_url, api_key_env=None, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.api_key_env, self.timeout = api_key_env, timeout

    def post(self, path, payload, request_id=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise InfrastructureError(f"Missing API key environment variable: {self.api_key_env}")
            headers["Authorization"] = "Bearer " + token
        if request_id:
            headers["Idempotency-Key"] = request_id
        request = urllib.request.Request(
            self.base_url + path, data=canonical(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            # Do not include response bodies or Authorization headers in audit logs.
            raise InfrastructureError(f"JSON service {path} failed: {type(exc).__name__}") from exc


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    monetary_cost: float = 0.0
    finish_reason: str = "stop"
    reasoning: str = ""


class ChatModel:
    """OpenAI-compatible text endpoint; model/harness configuration is frozen by profile."""

    def __init__(
        self,
        client: JSONClient,
        model: str,
        revision: str,
        decoding=None,
        pricing=None,
        resources=None,
        usage_role="consumer",
    ):
        if not revision:
            raise ValueError("Consumer/judge model revisions must be explicit")
        self.client, self.model, self.revision = client, model, revision
        self.decoding = decoding or {"temperature": 0}
        self.pricing = pricing or {"input_per_million": 0, "output_per_million": 0}
        self.resources, self.usage_role = resources, usage_role

    def complete(self, messages, max_tokens, seed, request_id):
        payload = {
            **self.decoding,
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        try:
            result = self.client.post("/chat/completions", payload, request_id)
        except InfrastructureError:
            if self.resources:
                self.resources.add(f"{self.usage_role}_failed_api_calls", 1)
                self.resources.add("failed_calls_with_unknown_usage", 1)
            raise
        try:
            usage = result["usage"]
            choice = result["choices"][0]
            message = choice["message"]
            text = message["content"]
            # A thinking model can terminate before producing a final answer.
            # This is an empty model action, not a transport outage. Preserve
            # usage and let the consumer's existing format/budget loop handle it.
            # Do not reinterpret reasoning text or native tool calls as actions.
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if (text is None and isinstance(reasoning, str) and reasoning
                    and not message.get("tool_calls") and not message.get("function_call")
                    and choice.get("finish_reason") in {"stop", "length"}):
                text = ""
            if not isinstance(text, str):
                raise ValueError("Text completion required")
            ni, no = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
            if ni < 0 or no < 0:
                raise ValueError("Negative usage")
            cost = (ni * self.pricing["input_per_million"] + no * self.pricing["output_per_million"]) / 1e6
            if self.resources:
                # Include successful calls from retried/excluded executions as well.
                self.resources.add(f"{self.usage_role}_api_calls", 1)
                self.resources.add(f"{self.usage_role}_api_input_tokens", ni)
                self.resources.add(f"{self.usage_role}_api_output_tokens", no)
                self.resources.add("observed_api_monetary_cost", cost)
            return Completion(text, ni, no, cost, choice.get("finish_reason", "stop"), reasoning or "")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InfrastructureError("Malformed completion or missing token usage") from exc


class RemoteEnvironment:
    def __init__(self, client, task, execution_id, seed, domain):
        self.client, self.task, self.execution_id = client, task, execution_id
        self.domain = domain
        self.closed = False
        result = client.post(
            "/reset",
            {
                "domain": domain,
                "execution_id": execution_id,
                "task_id": task.id,
                "seed": seed,
                "public_payload": task.public_payload,
            },
            execution_id + ":reset",
        )
        try:
            self.session_id = result["session_id"]
            self.initial_state_hash = result["state_hash"]
            self.observation = result["observation"]
            self.state_hash = self.initial_state_hash
            expected = task.public_payload["expected_initial_state_hash"]
            if self.initial_state_hash != expected:
                self.close()
                raise InfrastructureError("Environment reset hash differs from the locked initial snapshot")
            if domain == "web" and not result.get("full_backend_reset"):
                self.close()
                raise InfrastructureError("Web probes require backend/account reset, not URL navigation")
        except KeyError as exc:
            if hasattr(self, "session_id"):
                self.close()
            raise InfrastructureError("Incomplete reset contract") from exc

    def step(self, action, step_id, timeout):
        result = self.client.post(
            "/step",
            {"session_id": self.session_id, "action": action, "timeout_seconds": timeout},
            self.execution_id + f":step:{step_id}",
        )
        try:
            # SWE intentionally omits per-step state hashes. Keep the reset
            # identity until finalize() returns the final state hash.
            if "state_hash" in result:
                self.state_hash = result["state_hash"]
            elif self.domain != "swe":
                raise KeyError("state_hash")
            if type(result["done"]) is not bool:
                raise ValueError("done must be boolean")
            return result["observation"], result["done"]
        except (KeyError, ValueError) as exc:
            raise InfrastructureError("Incomplete environment observation") from exc

    def close(self):
        if not self.closed:
            self.client.post("/close", {"session_id": self.session_id}, self.execution_id + ":close")
            self.closed = True

    def finalize(self, answer=""):
        result = self.client.post(
            "/finalize", {"session_id": self.session_id, "answer": answer}, self.execution_id + ":finalize"
        )
        try:
            self.state_hash = result["state_hash"]
            return result["artifact"]
        except KeyError as exc:
            raise InfrastructureError("Incomplete public final artifact contract") from exc


class RemoteGrader:
    def __init__(self, client, version):
        self.client, self.version = client, version

    def __call__(self, task, execution):
        from ..contracts import Grade

        result = self.client.post(
            "/grade",
            {"task_id": task.id, "execution": execution, "grader_version": self.version},
            execution.execution_id + ":grade",
        )
        try:
            quality = float(result["quality"])
            if not 0 <= quality <= 1 or result["grader_version"] != self.version:
                raise ValueError("Invalid grade/version")
            return Grade(quality, result.get("metrics", {}), self.version)
        except (ValueError, KeyError, TypeError) as exc:
            raise InfrastructureError("Malformed grader response") from exc
