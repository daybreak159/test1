from __future__ import annotations

import json
import time
from dataclasses import replace

from ..contracts import Budget, EpisodePlan, Execution, PlanEvent, Source, SourceOutcome
from ..util import ConfigurationError, InfrastructureError, digest, seed_for
from .clients import RemoteEnvironment


class DomainAdapter:
    domain = "base"
    independent_queries = False
    training_enabled = True

    def __init__(self, consumer, profile, grader, environment_client=None):
        self.consumer, self.profile, self.grader = consumer, profile, grader
        self.environment_client = environment_client

    def canonical_identity(self, task):
        return digest(task.identity)

    def public_source(self, task, execution):
        ids = tuple(f"{task.id}:trace:{i}" for i in range(len(execution.public_trace)))
        # Deliberately does not receive a Grade or any private annotation.
        identity = replace(task.identity, query_id="", source_ids=(task.id,), evidence_ids=ids)
        lines = [
            f"[{evidence_id}] {json.dumps(event, ensure_ascii=False)}"
            for evidence_id, event in zip(ids, execution.public_trace)
        ]
        return Source(identity, "\n".join(lines), task.turns[-1].public_time, self.profile.hash, ())

    def memory_source(self, task, execution, grade):
        source = self.public_source(task, execution)
        if not getattr(self, "official_outcome_feedback", False):
            return source
        if self.domain != "swe" or not execution.completed:
            raise ConfigurationError("Official memory feedback requires a completed SWE execution")
        verdict = grade.metrics.get("raw_official_resolved", grade.metrics.get("resolved"))
        if verdict not in (0, 1) or not grade.grader_version:
            raise ConfigurationError("Official memory feedback requires an actual binary evaluator result")
        if execution.task_id != task.id:
            raise ConfigurationError("Official feedback task/execution mismatch")
        provenance = grade.grader_version
        if grade.metrics.get("empty_patch_gate"):
            provenance += ":empty-patch-gate:no-test-execution"
        outcome = SourceOutcome(task.id, execution.execution_id, bool(verdict),
                                provenance, execution.finish_reason)
        return replace(source, official_outcome=outcome)

    def build_plan(self, plan_id, scope, sources, tasks, split):
        events = []
        for i, task in enumerate(tasks):
            events.append(PlanEvent(f"task:{i}", "task", task=task))
            if i < len(tasks) - 1:
                events.append(PlanEvent(f"write:{i}", "write", source_from_task=task.id))
        return EpisodePlan(plan_id, self.domain, scope, tuple(events), split, False)

    def validate_probe_visibility(self, task, source):
        certificate = task.probe_certificate
        if task.identity.split != "train_probe" or task.identity.scope != source.identity.scope:
            return False
        declared_sources = certificate.get("source_task_ids")
        if declared_sources is not None and source.identity.sample_id not in declared_sources:
            return False
        if not certificate.get("prefix_answerable") or not certificate.get("validated_before_sampling"):
            return False
        if certificate.get("cutoff", -1) > source.identity.visibility_cutoff:
            return False
        return bool(certificate.get("validation_version"))

    def clone_or_reset(self, task, execution_id, seed):
        if self.environment_client is None:
            raise ConfigurationError(f"{self.domain} needs a real environment service")
        return RemoteEnvironment(self.environment_client, task, execution_id, seed, self.domain)

    def grade(self, task, execution):
        return self.grader(task, execution)

    def run_consumer(self, task, execution_id, seed, read, budget: Budget | None = None):
        """One read per declared external turn. Internal tools are never external turns."""
        budget = budget or self.profile.budget
        started = time.perf_counter()
        environment = self.clone_or_reset(task, execution_id, seed)
        initial = environment.initial_state_hash
        trace, history, exposure_hashes = [], [], []
        calls = inputs = outputs = 0
        money = 0.0
        answer, done, reason = "", False, "budget_exhausted"
        def remaining():
            return None if budget.wall_seconds is None else budget.wall_seconds-(time.perf_counter()-started)
        def tokens_available():
            return budget.output_token_policy == "per_response" or outputs < budget.output_tokens
        try:
            for turn_index, planned_turn in enumerate(task.turns):
                # Preserve public execution history for later turns; replace only the active memory block.
                history_summary = json.dumps(history, ensure_ascii=False) if history else ""
                turn = replace(
                    planned_turn,
                    public_history=planned_turn.public_history
                    + ("\n" + history_summary if history_summary else ""),
                )
                slate = read(turn)
                exposure_hashes.append(digest([turn, slate.memory_block, environment.observation]))
                system = {"role": "system", "content": self.consumer_system() + "\n" + slate.memory_block}
                history.append(
                    {
                        "role": "user",
                        "content": planned_turn.public_history
                        + "\n"
                        + turn.text
                        + "\n"
                        + str(environment.observation),
                    }
                )
                trace.append(
                    {
                        "event_origin": "external_request",
                        "external_turn_id": turn.external_turn_id,
                        "text": turn.text,
                        "observation": environment.observation,
                    }
                )
                turn_done = False
                while calls < budget.model_calls and tokens_available() and not turn_done:
                    left = remaining()
                    if left is not None and left <= 0:
                        reason = "wall_budget_exhausted"
                        break
                    client = getattr(self.consumer,"client",None)
                    previous_timeout = getattr(client,"timeout",None)
                    if previous_timeout is not None and left is not None:
                        client.timeout = min(previous_timeout,left)
                    try:
                        completion = self.consumer.complete(
                        [system, *history],
                        budget.output_tokens if budget.output_token_policy == "per_response" else budget.output_tokens - outputs,
                        seed_for(seed, "consumer", calls),
                        f"{execution_id}:model:{calls}",
                        )
                    except InfrastructureError:
                        if remaining() is not None and remaining() <= 0:
                            reason = "wall_budget_exhausted"
                            break
                        raise
                    finally:
                        if previous_timeout is not None:client.timeout=previous_timeout
                    calls += 1
                    inputs += completion.input_tokens
                    outputs += completion.output_tokens
                    money += completion.monetary_cost
                    history.append({"role": "assistant", "content": completion.text})
                    trace.append({"event_origin": "assistant_output", "text": completion.text})
                    if remaining() is not None and remaining() <= 0:
                        reason = "wall_budget_exhausted"
                        break
                    try:
                        action = json.loads(completion.text)
                        if not isinstance(action, dict) or set(action) not in ({"action"}, {"final"}):
                            raise ValueError("Expected action or final")
                        if "final" in action:
                            answer = str(action["final"])
                            turn_done = True
                            reason = "submitted"
                            continue
                        command_timeout=budget.command_timeout_seconds
                        left=remaining()
                        if left is not None:command_timeout=min(command_timeout,max(0.1,left))
                        observation, env_done = environment.step(action["action"], calls, command_timeout)
                        # Always record the final command result, including a terminating action.
                        history.append({"role": "user", "content": "Tool observation:\n" + str(observation)})
                        event = {
                            "event_origin": "tool_result",
                            "action": action["action"],
                            "observation": observation,
                        }
                        # SWE tasks are reset once at the task boundary. Their
                        # initial identity remains in Execution; exposing a
                        # hash after every command is redundant metadata.
                        if self.domain != "swe":
                            event["state_hash"] = environment.state_hash
                        trace.append(event)
                        environment.observation = observation
                        done = done or env_done
                        if env_done:
                            answer, turn_done, reason = str(observation), True, "environment_done"
                    except (json.JSONDecodeError, ValueError, TypeError) as exc:
                        observation = f"Invalid action format: {exc}"
                        history.append({"role": "user", "content": observation})
                        trace.append({"event_origin": "internal_control", "text": observation})
                if reason == "wall_budget_exhausted":
                    if budget != self.profile.budget:reason="probe_cutoff"
                    break
                if calls >= budget.model_calls or not tokens_available():
                    if not turn_done:
                        reason = "probe_cutoff" if budget != self.profile.budget else "budget_exhausted"
                    break
            completed = reason != "probe_cutoff"
            trace.append({"event_origin": "final_artifact", "artifact": environment.finalize(answer)})
            final_state = environment.state_hash
        finally:
            environment.close()
        return Execution(
            execution_id,
            task.id,
            answer,
            tuple(trace),
            initial,
            final_state,
            self.profile.hash,
            digest([exposure_hashes, budget, self.profile.hash]),
            seed,
            reason,
            calls,
            inputs,
            outputs,
            float(inputs + outputs + calls),
            time.perf_counter() - started,
            money,
            completed,
        )

    def consumer_system(self):
        return 'Solve the task using tools. Return exactly {"action": <tool action>} or {"final": "answer"}.'


class SourceThenQueryAdapter(DomainAdapter):
    independent_queries = True

    def build_plan(self, plan_id, scope, sources, tasks, split):
        events = [PlanEvent(f"write:{i}", "write", source=source) for i, source in enumerate(sources)]
        events += [PlanEvent(f"task:{i}", "task", task=task) for i, task in enumerate(tasks)]
        return EpisodePlan(plan_id, self.domain, scope, tuple(events), split, self.independent_queries)
