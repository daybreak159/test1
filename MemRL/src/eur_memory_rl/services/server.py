from __future__ import annotations

import threading
import uuid
from contextlib import nullcontext

from ..util import InfrastructureError, digest
from .backends import ALFWorldEnvironment, CommandGrader, SWEEnvironment, WebArenaEnvironment, swebench_grade

CONCURRENCY_PROTOCOL = "per-execution-v1"


class EnvironmentHub:
    def __init__(self, config):
        self.config, self.sessions, self.final_grades, self.requests = config, {}, {}, {}
        self.lock = threading.RLock()
        self.request_locks, self.execution_locks, self.session_locks = {}, {}, {}
        self.web_lock = threading.RLock()

    def _operation_lock(self, method, payload):
        """Only registry access holds self.lock; slow work holds an execution lock."""
        with self.lock:
            if method == "reset":
                eid = payload["execution_id"]
                default = self.web_lock if payload["domain"] == "web" else threading.RLock()
            elif method == "grade":
                eid, default = payload["execution"]["execution_id"], threading.RLock()
            else:
                # Retain the lock after close so delayed retries cannot race
                # finalize/close or change the ordering of an old session.
                return self.session_locks.setdefault(payload["session_id"], threading.RLock())
            return self.execution_locks.setdefault(eid, default)

    def dispatch(self, method, payload, request_id):
        key, fingerprint = (method, request_id), digest(payload)
        with self.lock:
            request_lock = self.request_locks.setdefault(key, threading.RLock()) if request_id else nullcontext()
        # A duplicate request waits for its first caller and reuses the result.
        # Distinct request IDs for one container still share its execution lock.
        with request_lock, self._operation_lock(method, payload):
            with self.lock:
                if request_id and key in self.requests:
                    previous, response = self.requests[key]
                    if previous != fingerprint:
                        raise ValueError("Idempotency key reused with different input")
                    if method != "reset" or response["session_id"] in self.sessions:
                        return response
                    prefix = payload["execution_id"] + ":"
                    self.requests = {
                        k: v for k, v in self.requests.items() if not (k[1] or "").startswith(prefix)
                    }
                    self.final_grades.pop(payload["execution_id"], None)
            result = getattr(self, method)(payload)
            with self.lock:
                if request_id:
                    self.requests[key] = (fingerprint, result)
            return result

    def reset(self, payload):
        domain = payload["domain"]
        with self.lock:
            if domain == "web" and any(session["domain"] == "web" for session in self.sessions.values()):
                raise InfrastructureError("Shared WebArena backend requires sequential complete reset/execution")
        cls = {"swe": SWEEnvironment, "web": WebArenaEnvironment, "alf": ALFWorldEnvironment}[domain]
        env = cls(payload["public_payload"], payload["seed"], self.config[domain])
        session_id = uuid.uuid4().hex
        with self.lock:
            self.session_locks[session_id] = self._operation_lock("reset", payload)
            self.sessions[session_id] = {
                "env": env,
                "domain": domain,
                "execution_id": payload["execution_id"],
                "task_id": payload["task_id"],
            }
        return {
            "session_id": session_id,
            "state_hash": env.state_hash,
            "observation": env.observation,
            "full_backend_reset": env.full_backend_reset,
        }

    def step(self, payload):
        with self.lock:
            session = self.sessions[payload["session_id"]]
        env = session["env"]
        observation, done = env.step(payload["action"], payload["timeout_seconds"])
        # SWE computes its final patch/state only at finalize(); other
        # domains may still expose a per-step state hash.
        response = {"observation": observation, "done": done}
        if session["domain"] != "swe":
            response["state_hash"] = env.state_hash
        return response

    def finalize(self, payload):
        with self.lock:
            session = self.sessions[payload["session_id"]]
        env = session["env"]
        artifact = env.finalize(payload.get("answer", ""))
        if hasattr(env, "grade_live"):
            # Private store: never include this grade in the public finalize response.
            value = {"quality": env.grade_live(), "domain": session["domain"]}
            with self.lock:
                self.final_grades[session["execution_id"]] = value
        return {"artifact": artifact, "state_hash": env.state_hash}

    def close(self, payload):
        with self.lock:
            session = self.sessions.pop(payload["session_id"], None)
        if session:
            session["env"].close()
        return {"closed": True}

    def grade(self, payload):
        cfg = self.config["grader"]
        if payload["grader_version"] != cfg["revision"]:
            raise ValueError("Grader revision mismatch")
        eid = payload["execution"]["execution_id"]
        with self.lock:
            value = self.final_grades.get(eid)
        if value is not None:
            return {
                "quality": value["quality"],
                "metrics": {"success": value["quality"]},
                "grader_version": cfg["revision"],
            }
        if cfg["kind"] == "swebench":
            return swebench_grade(payload, cfg)
        if cfg["kind"] == "command":
            return CommandGrader(cfg["argv"], cfg["revision"], cfg.get("timeout", 1800), cfg.get("cwd"))(
                payload
            )
        raise InfrastructureError("No private terminal grader for execution")


def create_app(config):
    from fastapi import FastAPI, Header, HTTPException

    app, hub = FastAPI(title="EUR isolated environment/grader bridge"), EnvironmentHub(config)

    @app.get("/health")
    def health():
        return {"service": "eur-environment", "config_fingerprint": digest(config),
                "grader_revision": config.get("grader", {}).get("revision"),
                "concurrency_protocol": CONCURRENCY_PROTOCOL}

    def endpoint(method):
        def call(
            payload: dict,
            idempotency_key: str | None = Header(default=None),
            authorization: str | None = Header(default=None),
        ):
            import os

            secret_env = config.get("api_key_env")
            if secret_env and (
                not os.environ.get(secret_env) or authorization != "Bearer " + os.environ[secret_env]
            ):
                raise HTTPException(401, "Invalid service key")
            try:
                return hub.dispatch(method, payload, idempotency_key)
            except Exception as exc:
                raise HTTPException(503, f"{type(exc).__name__}: {exc}") from exc

        return call

    for method in ("reset", "step", "finalize", "close", "grade"):
        app.post("/" + method)(endpoint(method))
    app.state.hub = hub
    return app
