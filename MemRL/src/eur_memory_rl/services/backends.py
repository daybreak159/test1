from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import uuid
from contextlib import nullcontext
from pathlib import Path

from ..util import ConfigurationError, InfrastructureError, digest


def command(argv, *, cwd=None, timeout=120, input_text=None, check=True):
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            shell=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InfrastructureError(f"Environment command failed: {type(exc).__name__}") from exc
    if check and result.returncode:
        raise InfrastructureError(
            f"Environment command returned {result.returncode}: {result.stderr[-2000:]}"
        )
    return result


class SWEEnvironment:
    """Disposable container. Never invokes ReasoningBank's memory-writing finally block."""

    full_backend_reset = True

    def __init__(self, payload, seed, service_config):
        images = service_config["allowed_images"]
        image_key = payload["image_key"]
        if image_key not in images or "@sha256:" not in images[image_key]:
            raise ConfigurationError("SWE images must use an allowlisted content digest")
        self.image = images[image_key]
        self.workdir = service_config.get("repository_path", "/testbed")
        self.base_commit = payload["base_commit"]
        self.container = "eur-" + uuid.uuid4().hex
        self.closed = False
        self.output_limit_chars = service_config.get("output_limit_chars")
        self.output_artifact_root = service_config.get("output_artifact_root")
        self.command_count = 0
        if self.output_limit_chars is not None and (type(self.output_limit_chars) is not int or self.output_limit_chars<128):
            raise ConfigurationError("output_limit_chars must be an integer >=128")
        if self.output_limit_chars is not None and not self.output_artifact_root:
            raise ConfigurationError("Bounded tool output requires full-output archival")
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container,
                "--network",
                service_config.get("network", "none"),
                "--cpus",
                str(service_config.get("cpus", 4)),
                "--memory",
                service_config.get("memory", "8g"),
                self.image,
                "sleep",
                "infinity",
            ],
            timeout=service_config.get("reset_timeout", 180),
        )
        try:
            self._exec(["git", "checkout", "--detach", self.base_commit])
            actual = self._exec(["git", "rev-parse", "HEAD"]).stdout.strip()
            status = self._exec(["git", "status", "--porcelain"]).stdout
            if status or actual != self.base_commit:
                raise InfrastructureError("Image repository must be clean at the pinned base commit")
            self.state_hash = digest(
                {
                    "image": self.image,
                    "base_commit": actual,
                    "workdir": self.workdir,
                    "reset_profile": service_config["revision"],
                }
            )
            self.observation = self._exec(["pwd"]).stdout + self._exec(["ls", "-la"]).stdout
        except Exception:
            self.close()
            raise

    def _exec(self, argv, timeout=120, check=True):
        return command(
            ["docker", "exec", "-w", self.workdir, self.container, *argv], timeout=timeout, check=check
        )

    def step(self, action, timeout):
        if (
            not isinstance(action, dict)
            or set(action) != {"command"}
            or not isinstance(action["command"], str)
        ):
            return {"error": "Expected a command string"}, False
        try:
            result = self._exec(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=5s",
                    f"{timeout}s",
                    "bash",
                    "-lc",
                    action["command"],
                ],
                timeout=timeout + 15,
                check=False,
            )
            observation = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except InfrastructureError as exc:
            # A command timing out is a policy outcome, not an infrastructure exclusion.
            if isinstance(exc.__cause__, subprocess.TimeoutExpired):
                observation = {
                    "exit_code": 124,
                    "stdout": str(exc.__cause__.stdout or ""),
                    "stderr": "Command timeout",
                }
            else:
                raise
        # SWE branches are isolated for the whole task.  The initial reset
        # identity is checked once; inspecting the whole diff after every
        # command is unnecessary for the action loop and bloats public traces.
        # The final diff/state is computed once in finalize().
        self.command_count += 1
        limit = self.output_limit_chars
        if limit is not None and any(len(observation.get(k,""))>limit for k in ("stdout","stderr")):
            from ..util import atomic_json
            original = dict(observation)
            output_hash = digest(original)
            path = Path(self.output_artifact_root)/self.container/f"command-{self.command_count:06d}.json"
            atomic_json(path, original)
            for key in ("stdout","stderr"):
                text = observation.get(key,"")
                if len(text)>limit:
                    half=limit//2
                    observation[key]=(text[:half]+f"\n[Tool {key} display limited: {len(text)} characters; "
                        f"full output archived, sha256 {output_hash}. Narrow your command to inspect the omitted part.]\n"+text[-half:])
            observation.update(output_truncated=True,full_output_hash=output_hash)
        return observation, False

    def finalize(self, answer=""):
        self._exec(["git", "add", "-N", "."])
        patch = self._exec(["git", "diff", "--binary", self.base_commit]).stdout
        status = self._exec(["git", "status", "--porcelain"], check=False).stdout
        self.state_hash = digest([self.image, self.base_commit, patch, status])
        return {"patch": patch, "base_commit": self.base_commit, "image": self.image}

    def close(self):
        if not self.closed:
            command(["docker", "rm", "-f", self.container], check=False)
            self.closed = True


class WebArenaEnvironment:
    """Real WebArena browser; the deployment supplies a full backend/account reset command."""

    full_backend_reset = True

    def __init__(self, payload, seed, service_config):
        from browser_env import ScriptBrowserEnv

        reset = service_config.get("reset_command")
        if not reset or not service_config.get("reset_covers_database_login_and_accounts"):
            raise ConfigurationError("WebArena must explicitly reset database, login and account state")
        # The reset command returns a verified snapshot hash after checking all task-related services.
        result = command(
            reset,
            input_text=json.dumps({"seed": seed, "scope": payload["scope"]}),
            timeout=service_config.get("reset_timeout", 600),
        )
        reset_record = json.loads(result.stdout)
        if not reset_record.get("verified") or not reset_record.get("state_hash"):
            raise InfrastructureError("Backend reset was not verified")
        self.env = ScriptBrowserEnv(
            headless=True, observation_type="accessibility_tree", current_viewport_only=False
        )
        task_config = service_config["task_configs"][payload["task_key"]]
        observation, info = self.env.reset(options={"config_file": task_config})
        self.observation = observation["text"]
        self.state_hash = digest(
            [reset_record["state_hash"], self.observation, service_config["revision"], payload["task_key"]]
        )
        self.trace = [{"observation": observation, "info": info}]
        self.task_config = task_config

    def step(self, action, timeout):
        from browser_env import ActionTypes, create_id_based_action

        try:
            parsed = create_id_based_action(action)
        except (ValueError, AssertionError) as exc:
            return {"error": str(exc)}, False
        self.trace.append(parsed)
        if parsed["action_type"] == ActionTypes.STOP:
            return {"text": self.observation, "submitted_answer": parsed.get("answer", "")}, True
        self.env.page.set_default_timeout(timeout * 1000)
        self.env.page.set_default_navigation_timeout(timeout * 1000)
        observation, reward, terminated, truncated, info = self.env.step(parsed)
        self.observation = observation["text"]
        self.trace.append({"observation": observation, "info": info})
        self.state_hash = digest([self.observation, self.env.page.url])
        # Raw simulator rewards are withheld; official grading is a separate call.
        return {"text": self.observation, "url": self.env.page.url}, bool(terminated or truncated)

    def finalize(self, answer=""):
        from browser_env import ActionTypes, create_stop_action

        if self.trace[-1].get("action_type") != ActionTypes.STOP:
            self.trace.append(create_stop_action(answer))
        return {
            "final_url": self.env.page.url,
            "public_page": self.observation,
            "task_config_key": Path(self.task_config).name,
        }

    def grade_live(self):
        from evaluation_harness import evaluator_router

        evaluator = evaluator_router(self.task_config)
        return float(
            evaluator(self.trace, self.task_config, self.env.page, self.env.get_page_client(self.env.page))
        )

    def close(self):
        self.env.close()


class ALFWorldEnvironment:
    full_backend_reset = True

    def __init__(self, payload, seed, service_config):
        if service_config.get("backend") == "direct_textworld":
            self._init_direct_textworld(payload, seed, service_config)
            return
        from alfworld.agents.environment import get_environment

        cfg = json.loads(json.dumps(service_config["alfworld_config"]))
        env_type = cfg["env"]["type"]
        manager = get_environment(env_type)(cfg, train_eval=payload["train_eval"])
        target = service_config["gamefiles"][payload["task_key"]]
        manager.game_files = [target]
        manager.num_games = 1
        self.env = manager.init_env(batch_size=1)
        self.env.seed(seed % (2**32))
        observations, info = self.env.reset()
        self.observation = observations[0]
        self.won = False
        self.state_hash = digest([payload["task_key"], self.observation, service_config["revision"]])

    def _init_direct_textworld(self, payload, seed, config):
        """Target/probe execution without a training-mode expert or dataset-wide scan."""
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self._direct_config = config
        self._task_key = payload["task_key"]
        target = config["gamefiles"][self._task_key]
        self._game_sha = hashlib.sha256(Path(target).read_bytes()).hexdigest()
        request = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                                     extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [target], request, batch_size=1, asynchronous=False, auto_reset=False,
            max_episode_steps=config.get("max_episode_steps", 150),
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self.env = textworld.gym.make(env_id)
        try:
            self.env.seed(seed % (2**32))
            observations, info = self.env.reset()
            self._public_direct(observations, info)
            self.won = bool(info.get("won", [False])[0])
            self._state_direct(info)
        except Exception:
            self.env.close()
            raise

    def _public_direct(self, observations, info):
        self.observation = {
            "text": observations[0],
            "admissible_commands": info.get("admissible_commands", [[]])[0],
        }

    def _state_direct(self, info):
        # Simulator facts are hashed privately; neither facts nor expert plans are exposed.
        if "facts" not in info or not info["facts"] or not info["facts"][0]:
            raise InfrastructureError("ALF reset/step did not provide simulator facts for state identity")
        self.reset_evidence = {
            "task_key": self._task_key, "game_sha256": self._game_sha,
            "facts_sha256": digest(sorted(str(fact) for fact in info["facts"][0])),
            "revision": self._direct_config["revision"],
            "max_episode_steps": self._direct_config.get("max_episode_steps", 150),
            "public_observation": self.observation,
        }
        self.state_hash = digest(self.reset_evidence)

    def step(self, action, timeout):
        if not isinstance(action, str):
            return {"error": "Expected textual ALFWorld command"}, False
        observations, rewards, dones, infos = self.env.step([action])
        if hasattr(self, "_direct_config"):
            self._public_direct(observations, infos)
            self.won = bool(infos.get("won", [False])[0])
            self._state_direct(infos)
            return self.observation, bool(dones[0])
        self.observation = observations[0]
        self.won = bool(infos.get("won", [False])[0])
        self.state_hash = digest([self.state_hash, action, self.observation])
        return {
            "text": self.observation,
            "admissible_commands": infos.get("admissible_commands", [[]])[0],
        }, bool(dones[0])

    def finalize(self, answer=""):
        return {"last_observation": self.observation}

    def grade_live(self):
        return float(self.won)

    def close(self):
        self.env.close()


class CommandGrader:
    """Isolated official evaluator bridge; JSON stdin/out, no grade enters the writer graph."""

    def __init__(self, argv, revision, timeout=1800, cwd=None):
        self.argv, self.revision, self.timeout, self.cwd = argv, revision, timeout, cwd

    def __call__(self, payload):
        if payload["grader_version"] != self.revision:
            raise ConfigurationError("Grader revision mismatch")
        result = command(self.argv, cwd=self.cwd, timeout=self.timeout, input_text=json.dumps(payload))
        try:
            grade = json.loads(result.stdout)
            if not 0 <= float(grade["quality"]) <= 1:
                raise ValueError("Invalid grade")
        except (ValueError, KeyError) as exc:
            raise InfrastructureError("Official grader bridge did not return its JSON contract") from exc
        return {**grade, "grader_version": self.revision}


def swebench_grade(payload, config):
    """Runs the official SWE-bench CLI on an actual patch in its own working directory."""
    execution = payload["execution"]
    artifacts = [
        event["artifact"]
        for event in execution["public_trace"]
        if event.get("event_origin") == "final_artifact"
    ]
    patch = next((a["patch"] for a in artifacts if "patch" in a), "")
    task_id = payload["task_id"]
    if not patch.strip():
        return {"quality": 0.0, "metrics": {"resolved": 0.0, "empty_patch_gate": 1.0}, "grader_version": config["revision"]}
    run_id, model_name = uuid.uuid4().hex, "eur"
    persistent = config.get("artifact_root")
    if persistent:
        saved = Path(persistent) / run_id
        saved.mkdir(parents=True, exist_ok=False)
    context = nullcontext(str(saved)) if persistent else tempfile.TemporaryDirectory(prefix="eur-swe-grade-")
    with context as directory:
        root = Path(directory)
        predictions = root / "predictions.jsonl"
        predictions.write_text(
            json.dumps({"instance_id": task_id, "model_name_or_path": model_name, "model_patch": patch})
            + "\n",
            encoding="utf-8",
        )
        result = command(
            [
                config.get("python", os.sys.executable),
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                config["dataset_name"],
                "--split",
                config.get("split", "test"),
                "--predictions_path",
                str(predictions),
                "--run_id",
                run_id,
                "--max_workers",
                "1",
                "--instance_ids",
                task_id,
                "--timeout",
                str(config.get("test_timeout", 900)),
            ],
            cwd=root,
            timeout=config.get("timeout", 1800),
            check=False,
        )
        (root / "evaluator.stdout.log").write_text(result.stdout)
        (root / "evaluator.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise InfrastructureError(f"Official evaluator returned {result.returncode}; artifacts preserved when configured")
        report_path = root / config["report_path_template"].format(model=model_name, run_id=run_id)
        if not report_path.is_file():
            raise InfrastructureError(
                "SWE-bench report missing; configure report_path_template for pinned harness"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if task_id in report.get("error_ids", []):
            raise InfrastructureError("Official SWE-bench execution infrastructure failed")
        resolved = float(task_id in report["resolved_ids"])
        return {"quality": resolved, "metrics": {"resolved": resolved, "official_report": 1.0}, "grader_version": config["revision"]}
