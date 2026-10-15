from __future__ import annotations

from ..contracts import Anchor, Bank, EpisodeResult
from ..rewards.advantages import assign_main_advantages
from ..rewards.outcome import RewardLedger
from ..util import GroupExcluded, InfrastructureError, digest, retry_infrastructure, seed_for
from .engine import policy_row
from .events import ExternalTurnRouter


class MainStream:
    def __init__(
        self, engine, adapter, cost_lambda=0.01, bad_write_lambda=1.0, cost_budget=100000, retry_attempts=3
    ):
        self.engine, self.adapter = engine, adapter
        self.cost_lambda, self.bad_write_lambda = cost_lambda, bad_write_lambda
        self.cost_budget, self.retry_attempts = cost_budget, retry_attempts

    def run_episode(
        self, plan, execution_id, seed, deterministic=False, selector_mode="learned", memory_enabled=True
    ):
        bank, router = Bank(plan.scope), ExternalTurnRouter()
        ledger = RewardLedger(
            plan.scored_count, plan.write_count, self.cost_lambda, self.bad_write_lambda, self.cost_budget
        )
        rows, anchors, positions, executions, grades, task_lookup = [], [], {}, {}, {}, {}
        reads = []
        # Align planned positions even if an invalid E prevents the U call.
        for index, event in enumerate(plan.events):
            positions[event.id] = index * 10000
            if event.task:
                for h, turn in enumerate(event.task.turns):
                    positions[f"{event.id}/{turn.external_turn_id}"] = index * 10000 + h + 1
        for event in plan.events:
            at = positions[event.id]
            if event.kind == "write":
                if not memory_enabled:
                    continue
                source = event.source
                if source is None:
                    task = task_lookup[event.source_from_task]
                    provider = getattr(self.adapter, "memory_source", None)
                    from ..data.hybrid_trace import CompactionSkip
                    try:
                        source = (provider(task, executions[event.source_from_task], grades[event.source_from_task])
                                  if provider else self.adapter.public_source(task, executions[event.source_from_task]))
                    except CompactionSkip:
                        # Preprocessing exclusion is neither a sampled invalid
                        # model write nor a model training example.
                        self.engine.count("preprocessing_skips")
                        continue
                anchor_id = f"{execution_id}/{event.id}"
                anchors.append(Anchor(anchor_id, execution_id, source, bank, event.id, at))
                e_record, candidates = self.engine.extract(
                    source, seed_for(seed, event.id, "E"), deterministic
                )
                e_row = policy_row(
                    e_record,
                    "E",
                    row_id=anchor_id + "/E",
                    episode_id=execution_id,
                    group_id=plan.id,
                    domain=plan.domain,
                    split=plan.split,
                    position=event.id,
                    time=at,
                    bank_before=bank,
                )
                rows.append(e_row)
                if candidates is None:
                    ledger.bad_write(event.id, at + 1)
                    continue
                u_record, result, metadata = self.engine.update(
                    source, bank, candidates, seed_for(seed, event.id, "U"), deterministic
                )
                rows.append(
                    policy_row(
                        u_record,
                        "U",
                        row_id=anchor_id + "/U",
                        episode_id=execution_id,
                        group_id=plan.id,
                        domain=plan.domain,
                        split=plan.split,
                        position=event.id,
                        time=at,
                        bank_before=bank,
                        bank_after=result.bank,
                        metadata=metadata,
                    )
                )
                e_row.bank_after_hash = result.bank.hash
                bank = result.bank
                if not result.valid:
                    ledger.bad_write(event.id, at + 1)
            else:
                task = event.task
                task_lookup[task.id] = task
                task_execution = f"{execution_id}/{task.id}"
                read_cost = [0.0]

                def read(turn):
                    position = f"{event.id}/{turn.external_turn_id}"
                    if position not in positions:
                        raise ValueError("Unplanned external input cannot be aligned across replicas")

                    def choose():
                        prepared = self.engine.prepare_read(bank, turn, fixed=selector_mode == "R0")
                        return self.engine.selector.sample(
                            prepared,
                            seed_for(seed, position, "R"),
                            deterministic=deterministic,
                            fixed=selector_mode == "R0",
                        )

                    slate, fresh = router.read(task_execution, turn, choose)
                    if fresh:
                        reads.append(
                            {
                                "task_id": task.id,
                                "turn": turn,
                                "bank": bank,
                                "slate": slate,
                                "selector_mode": selector_mode,
                                "execution_id": task_execution,
                            }
                        )
                        self.engine.count("external_turns")
                        self.engine.count("R_slates")
                        read_cost[0] += (
                            slate.forward.forward_count
                            + len(slate.forward.input_ids) * slate.forward.forward_count
                        )
                        if selector_mode != "R0":
                            rows.append(
                                policy_row(
                                    slate,
                                    "R",
                                    row_id=task_execution + "/" + turn.external_turn_id,
                                    episode_id=execution_id,
                                    group_id=plan.id,
                                    domain=plan.domain,
                                    split=plan.split,
                                    position=position,
                                    time=positions[position],
                                    bank_before=bank,
                                    metadata={
                                        "task_id": task.id,
                                        "execution_id": task_execution,
                                        "external_turn_id": turn.external_turn_id,
                                        "event_origin": turn.event_origin,
                                        "request_id": turn.request_id,
                                        "harness_version": self.adapter.profile.harness_revision,
                                        "active_memory_block_hash": digest(slate.memory_block),
                                    },
                                )
                            )
                    return slate

                result = retry_infrastructure(
                    lambda: self.adapter.run_consumer(
                        task, task_execution, seed_for(seed, event.id, "consumer"), read
                    ),
                    self.retry_attempts,
                )
                grade = retry_infrastructure(lambda: self.adapter.grade(task, result), self.retry_attempts)
                executions[task.id], grades[task.id] = result, grade
                ledger.outcome(task.id, at + 9999, grade.quality, result.raw_cost + read_cost[0])
                self.engine.count("full_executions")
                self.engine.count("full_steps", result.model_calls)
                self.engine.count("consumer_input_tokens", result.input_tokens)
                self.engine.count("consumer_output_tokens", result.output_tokens)
                self.engine.count("environment_seconds", result.environment_seconds)
                self.engine.count("monetary_cost", result.monetary_cost)
        if len(grades) != plan.scored_count:
            raise RuntimeError("Not all planned terminal grades were collected")
        return EpisodeResult(
            plan, execution_id, bank, rows, ledger.events, anchors, positions, executions, grades, reads
        )

    def run_group(self, plan, replicas, seed, debug_zero_baseline=False, selector_mode="learned"):
        if replicas < 2 and not debug_zero_baseline:
            raise ValueError("P=1 is only allowed with an explicit debug zero baseline")
        episodes = []
        try:
            for p in range(replicas):
                episodes.append(
                    self.run_episode(
                        plan,
                        f"{plan.id}/replica-{p}",
                        seed_for(seed, "replica", p),
                        selector_mode=selector_mode,
                    )
                )
        except InfrastructureError as exc:
            raise GroupExcluded(f"Entire main replica group excluded: {exc}") from exc
        assign_main_advantages(episodes, debug_zero_baseline)
        return episodes
