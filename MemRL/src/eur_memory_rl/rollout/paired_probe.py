from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Budget
from ..rewards.advantages import mean, nested_advantages
from ..util import (
    ConfigurationError,
    GroupExcluded,
    InfrastructureError,
    digest,
    retry_infrastructure,
    seed_for,
)
from .engine import policy_row
from .events import ExternalTurnRouter


@dataclass
class ProbeResult:
    anchor: object
    probes: tuple
    rows: list
    values: dict
    advantages: object
    references: dict
    executions: dict
    pair_scores: dict
    banks: dict


class PairedProbe:
    def __init__(
        self,
        engine,
        adapter,
        quality,
        I=2,
        J=2,
        K=2,
        L=2,
        B=1,
        invalid_score=-1.0,
        budget=None,
        retry_attempts=3,
        credit="conditional",
        reference=True,
        selector_mode="learned",
        scoring="paired",
    ):
        if min(I, J, L) < 2 or min(K, B) < 1:
            raise ValueError("Invalid nested sampling plan")
        self.engine, self.adapter, self.quality = engine, adapter, quality
        self.I, self.J, self.K, self.L, self.B = I, J, K, L, B
        self.invalid_score, self.budget = invalid_score, budget or Budget(8, 8192, 120)
        self.retry_attempts, self.credit, self.use_reference = retry_attempts, credit, reference
        self.selector_mode, self.scoring = selector_mode, scoring

    def select_probes(self, anchor, pool, seed):
        # Eligibility and fixed categories/fallback are decided before generating E candidates.
        import random

        eligible = [p for p in pool if self.adapter.validate_probe_visibility(p, anchor.source)]
        if len(eligible) < self.K:
            raise ValueError(f"Anchor {anchor.id} has fewer than {self.K} prevalidated probes")
        eligible.sort(key=lambda p: p.id)
        random.Random(seed).shuffle(eligible)
        if self.adapter.domain in {"swe", "web"} and self.K >= 2:
            related = next((p for p in eligible if p.probe_certificate.get("probe_kind") == "related"), None)
            retention = next(
                (p for p in eligible if p.probe_certificate.get("probe_kind") == "retention"), None
            )
            if related and retention:
                eligible = [related, retention] + [p for p in eligible if p not in (related, retention)]
        return tuple(eligible[: self.K])

    def _execute(self, task, slate, execution_id, seed):
        if len(task.turns) != 1:
            raise ValueError("v3 local probes begin at one fixed external turn boundary")
        router = ExternalTurnRouter()

        def read(turn):
            result, _ = router.read(execution_id, turn, lambda: None, precomputed=slate)
            return result

        execution = retry_infrastructure(
            lambda: self.adapter.run_consumer(task, execution_id, seed, read, self.budget),
            self.retry_attempts,
        )
        self.engine.count("short_executions")
        self.engine.count("short_steps", execution.model_calls)
        self.engine.count("consumer_input_tokens", execution.input_tokens)
        self.engine.count("consumer_output_tokens", execution.output_tokens)
        self.engine.count("environment_seconds", execution.environment_seconds)
        self.engine.count("monetary_cost", execution.monetary_cost)
        return execution

    def run(self, anchor, probes, seed):
        if len(probes) != self.K or any(
            not self.adapter.validate_probe_visibility(p, anchor.source) for p in probes
        ):
            raise ValueError("Probe plans must be finalized and prefix-validated before candidate sampling")
        try:
            return self._run(anchor, probes, seed)
        except InfrastructureError as exc:
            raise GroupExcluded(f"Entire anchor tree excluded: {exc}") from exc

    def _run(self, anchor, probes, seed):
        bank, rows, values, refs, executions, scores = anchor.bank, [], {}, {}, {}, {}
        base_hash = bank.hash
        banks = {}
        if self.use_reference:
            for q, task in enumerate(probes):
                prepared = self.engine.prepare_read(bank, task.turns[0], fixed=True)
                slate = self.engine.selector.sample(prepared, seed_for(seed, q), fixed=True)
                for b in range(self.B):
                    refs[q, b] = self._execute(
                        task, slate, f"{anchor.id}/ref/{q}/{b}", seed_for(seed, "pair", q, b)
                    )
        row_context = dict(
            episode_id=anchor.episode_id,
            group_id=anchor.id,
            domain=self.adapter.domain,
            split="train",
            position=anchor.position,
            time=anchor.time,
            bank_before=bank,
            source="local",
            anchor_id=anchor.id,
        )
        for i in range(self.I):
            e_record, candidates = self.engine.extract(anchor.source, seed_for(seed, "E", i))
            e_row = policy_row(e_record, "E", row_id=f"{anchor.id}/E/{i}", metadata={"i": i}, **row_context)
            rows.append(e_row)
            if candidates is None:
                for j in range(self.J):
                    values[i, j] = None
                continue
            for j in range(self.J):
                u_record, applied, metadata = self.engine.update(
                    anchor.source, bank, candidates, seed_for(seed, "U", i, j)
                )
                u_row = policy_row(
                    u_record,
                    "U",
                    row_id=f"{anchor.id}/U/{i}/{j}",
                    bank_after=applied.bank,
                    metadata={**metadata, "i": i, "j": j},
                    **row_context,
                )
                rows.append(u_row)
                if not applied.valid:
                    values[i, j] = None
                    continue
                banks[i, j] = applied.bank
                leaf = []
                for q, task in enumerate(probes):
                    # Exactly one forward for L independently sampled slates.
                    prepared = self.engine.prepare_read(
                        applied.bank, task.turns[0], fixed=self.selector_mode == "R0"
                    )
                    local_values = []
                    for ell in range(self.L):
                        slate = self.engine.selector.sample(
                            prepared, seed_for(seed, "R", i, j, q, ell), fixed=self.selector_mode == "R0"
                        )
                        self.engine.count("R_slates")
                        key = (i, j, q, ell)
                        r_context = {**row_context, "bank_before": applied.bank}
                        row = policy_row(
                            slate,
                            "R",
                            row_id=f"{anchor.id}/R/{i}/{j}/{q}/{ell}",
                            metadata={
                                "i": i,
                                "j": j,
                                "q": q,
                                "ell": ell,
                                "task_id": task.id,
                                "execution_id": f"{anchor.id}/candidate/{i}/{j}/{q}/{ell}",
                                "external_turn_id": task.turns[0].external_turn_id,
                                "event_origin": "external_request",
                                "request_id": task.turns[0].request_id,
                                "harness_version": self.adapter.profile.harness_revision,
                                "active_memory_block_hash": digest(slate.memory_block),
                            },
                            **r_context,
                        )
                        if self.selector_mode != "R0":
                            rows.append(row)
                        rewards = []
                        for b in range(self.B):
                            execution = self._execute(
                                task, slate, f"{row.id}/execution/{b}", seed_for(seed, "pair", q, b)
                            )
                            executions[(*key, b)] = execution
                            if self.use_reference and self.scoring == "paired":
                                score = retry_infrastructure(
                                    lambda: self.quality(
                                        task, execution, refs[q, b], seed_for(seed, "judge", *key, b)
                                    ),
                                    self.retry_attempts,
                                )
                                scores[(*key, b)] = score
                                rewards.append(score.value)
                            else:
                                if not execution.completed:
                                    raise ConfigurationError(
                                        "Direct terminal quality requires complete rollouts; a probe cutoff is not failure"
                                    )
                                rewards.append(self.adapter.grade(task, execution).quality)
                        local_values.append(mean(rewards))
                    leaf.append(local_values)
                values[i, j] = leaf
        advantages = nested_advantages(values, self.I, self.J, self.K, self.L, self.invalid_score)
        for row in rows:
            m = row.metadata
            if row.role == "E":
                row.advantage = advantages.e[m["i"]]
            elif row.role == "U":
                row.advantage = advantages.u[m["i"]][m["j"]]
            else:
                row.advantage = advantages.r[m["i"], m["j"], m["q"]][m["ell"]]
            if self.credit == "flat":
                # Explicit ablation: one flat group baseline over complete planned leaf values.
                from ..rewards.advantages import loo

                flat = [v for group in advantages.q for v in group]
                flat_a = loo(flat)
                if row.role == "E":
                    row.advantage = mean(flat_a[m["i"] * self.J : (m["i"] + 1) * self.J])
                else:
                    row.advantage = flat_a[m["i"] * self.J + m["j"]]
        if bank.hash != base_hash:
            raise RuntimeError("Local probe mutated the main bank")
        return ProbeResult(anchor, tuple(probes), rows, values, advantages, refs, executions, scores, banks)
