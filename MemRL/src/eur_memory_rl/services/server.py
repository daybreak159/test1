from __future__ import annotations

import threading
import uuid

from ..util import InfrastructureError, digest
from .backends import ALFWorldEnvironment, CommandGrader, SWEEnvironment, WebArenaEnvironment, swebench_grade


class EnvironmentHub:
    def __init__(self, config):
        self.config, self.sessions, self.final_grades, self.requests = config, {}, {}, {}
        self.lock = threading.RLock()

    def dispatch(self, method, payload, request_id):
        with self.lock:
            key, fingerprint = (method, request_id), digest(payload)
            if method == "reset" and request_id and key in self.requests:
                _, previous_reset = self.requests[key]
                if previous_reset["session_id"] not in self.sessions:
                    prefix = payload["execution_id"] + ":"
                    self.requests = {
                        k: v for k, v in self.requests.items() if not (k[1] or "").startswith(prefix)
                    }
                    self.final_grades.pop(payload["execution_id"], None)
            if request_id and key in self.requests:
                previous, response = self.requests[key]
                if previous != fingerprint:
                    raise ValueError("Idempotency key reused with different input")
                return response
            result = getattr(self, method)(payload)
            if request_id:
                self.requests[key] = (fingerprint, result)
            return result

    def reset(self, payload):
        domain = payload["domain"]
        if domain == "web" and any(session["domain"] == "web" for session in self.sessions.values()):
            raise InfrastructureError("Shared WebArena backend requires sequential complete reset/execution")
        cls = {"swe": SWEEnvironment, "web": WebArenaEnvironment, "alf": ALFWorldEnvironment}[domain]
        env = cls(payload["public_payload"], payload["seed"], self.config[domain])
        session_id = uuid.uuid4().hex
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
        env = self.sessions[payload["session_id"]]["env"]
        observation, done = env.step(payload["action"], payload["timeout_seconds"])
        # SWE computes its final patch/state only at finalize(); other
        # domains may still expose a per-step state hash.
        response = {"observation": observation, "done": done}
        if self.sessions[payload["session_id"]]["domain"] != "swe":
            response["state_hash"] = env.state_hash
        return response

    def finalize(self, payload):
        session = self.sessions[payload["session_id"]]
        env = session["env"]
        artifact = env.finalize(payload.get("answer", ""))
        if hasattr(env, "grade_live"):
            # Private store: never include this grade in the public finalize response.
            self.final_grades[session["execution_id"]] = {
                "quality": env.grade_live(),
                "domain": session["domain"],
            }
        return {"artifact": artifact, "state_hash": env.state_hash}

    def close(self, payload):
        session = self.sessions.pop(payload["session_id"], None)
        if session:
            session["env"].close()
        return {"closed": True}

    def grade(self, payload):
        cfg = self.config["grader"]
        if payload["grader_version"] != cfg["revision"]:
            raise ValueError("Grader revision mismatch")
        eid = payload["execution"]["execution_id"]
        if eid in self.final_grades:
            value = self.final_grades[eid]
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
                "grader_revision": config.get("grader", {}).get("revision")}

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
