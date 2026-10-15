from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .util import digest

Role = Literal["E", "U", "R"]
Origin = Literal[
    "external_request", "tool_result", "internal_control", "assistant_output", "source_ingestion"
]


@dataclass(frozen=True)
class DataIdentity:
    dataset: str
    version: str
    split_protocol: str
    split: str
    sample_id: str
    duplicate_component: str
    scope: str
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    visibility_cutoff: int
    history_or_environment_hash: str
    query_id: str = ""
    annotation_version: str = "v1"
    media_profile: str = "text_only"
    context_variant: str = "default"
    official_or_custom_protocol: str = "custom"

    def __post_init__(self):
        for name in (
            "dataset",
            "version",
            "split_protocol",
            "split",
            "sample_id",
            "duplicate_component",
            "scope",
            "history_or_environment_hash",
            "annotation_version",
            "media_profile",
        ):
            if not getattr(self, name):
                raise ValueError(f"Missing data identity: {name}")


@dataclass(frozen=True)
class SourceOutcome:
    """Explicit SWE feedback: scalar verdict only, never private test material."""

    task_id: str
    execution_id: str
    resolved: bool
    grader_version: str
    finish_reason: str
    protocol: str = "swe-official-outcome-v1"

    def __post_init__(self):
        if type(self.resolved) is not bool or not all((self.task_id,self.execution_id,self.grader_version,self.finish_reason)):
            raise ValueError("Outcome feedback requires a boolean verdict and complete provenance")
        if self.protocol != "swe-official-outcome-v1":
            raise ValueError("Unknown outcome feedback protocol")


@dataclass(frozen=True)
class Source:
    """Public evidence; optional explicitly enabled scalar SWE outcome feedback."""

    identity: DataIdentity
    text: str
    observed_at: str
    agent_profile: str = "dataset"
    public_feedback: tuple[str, ...] = ()
    prefix_evidence_ids: tuple[str, ...] = ()
    text_format: str = "default"
    official_outcome: SourceOutcome | None = None


@dataclass(frozen=True)
class MemoryCandidate:
    kind: str
    title: str
    content: str
    conditions: str
    source_ids: tuple[str, ...]
    subject: str = ""
    event_time: str | None = None


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    revision: int
    scope: str
    kind: str
    title: str
    content: str
    conditions: str
    source_ids: tuple[str, ...]
    observed_at: str
    event_time: str | None = None
    subject: str = ""
    status: str = "active"
    supersedes: tuple[str, ...] = ()
    source_outcomes: tuple[SourceOutcome, ...] = ()


@dataclass(frozen=True)
class Bank:
    scope: str
    revision: int = 0
    next_id: int = 1
    entries: tuple[MemoryEntry, ...] = ()

    @property
    def version(self) -> str:
        return f"bank-{self.revision}"

    @property
    def hash(self) -> str:
        return digest(self)

    @property
    def active(self) -> tuple[MemoryEntry, ...]:
        return tuple(e for e in self.entries if e.status == "active")


@dataclass(frozen=True)
class Turn:
    task_id: str
    external_turn_id: str
    text: str
    public_history: str = ""
    public_time: str = ""
    event_origin: Origin = "external_request"
    request_id: str = ""


@dataclass(frozen=True)
class Task:
    identity: DataIdentity
    turns: tuple[Turn, ...]
    public_payload: dict[str, Any] = field(default_factory=dict)
    category: str = ""
    # Visibility certificates contain evidence IDs and a cutoff, never answers.
    probe_certificate: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.identity.query_id or self.identity.sample_id

    def __post_init__(self):
        if not self.turns or len(self.turns) >= 9998:
            raise ValueError("Task requires a bounded nonempty external-turn schedule")
        if any(t.task_id != self.id or t.event_origin != "external_request" for t in self.turns):
            raise ValueError("Task schedule contains a wrong task ID or an internal message")
        if len({t.external_turn_id for t in self.turns}) != len(self.turns):
            raise ValueError("External turn IDs must be unique inside a task")


@dataclass(frozen=True)
class PlanEvent:
    id: str
    kind: Literal["write", "task"]
    source: Source | None = None
    task: Task | None = None
    source_from_task: str | None = None


@dataclass(frozen=True)
class EpisodePlan:
    id: str
    domain: str
    scope: str
    events: tuple[PlanEvent, ...]
    split: str
    readonly_independent_queries: bool = False

    @property
    def scored_count(self):
        return sum(e.kind == "task" for e in self.events)

    @property
    def write_count(self):
        return sum(e.kind == "write" for e in self.events)

    def __post_init__(self):
        if self.scored_count < 1 or len({e.id for e in self.events}) != len(self.events):
            raise ValueError("A plan needs scored tasks and unique event IDs")
        task_ids = [e.task.id for e in self.events if e.task]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Each scored task occurs once; turns belong inside a task")
        completed = set()
        for event in self.events:
            if event.kind == "task":
                if event.task is None or event.source is not None or event.source_from_task is not None:
                    raise ValueError("Malformed task event")
                completed.add(event.task.id)
            elif event.kind == "write":
                if (event.source is None) == (event.source_from_task is None):
                    raise ValueError("A write needs exactly one source provider")
                if event.source_from_task and event.source_from_task not in completed:
                    raise ValueError("Cannot write from an unexecuted future task")
            else:
                raise ValueError("Unknown planned event")


@dataclass(frozen=True)
class Budget:
    model_calls: int = 64
    output_tokens: int = 32768
    command_timeout_seconds: int = 120
    output_token_policy: str = "aggregate"
    wall_seconds: float | None = None

    def __post_init__(self):
        if self.output_token_policy not in {"aggregate", "per_response"}:
            raise ValueError("Unknown output token budget policy")
        if min(self.model_calls,self.output_tokens,self.command_timeout_seconds)<=0:
            raise ValueError("Budgets must be positive")
        if self.wall_seconds is not None and self.wall_seconds<=0:
            raise ValueError("Wall budget must be positive")


@dataclass(frozen=True)
class ConsumerProfile:
    backend: str
    model: str
    model_revision: str
    harness_revision: str
    tools: tuple[str, ...]
    memory_injection: str
    decoding: dict[str, Any]
    budget: Budget
    reset_method: str
    tokenizer: str
    tokenizer_revision: str

    @property
    def hash(self):
        return digest(self)


@dataclass(frozen=True)
class Execution:
    execution_id: str
    task_id: str
    answer: str
    public_trace: tuple[dict[str, Any], ...]
    initial_state_hash: str
    final_state_hash: str
    consumer_profile_hash: str
    input_hash: str
    seed: int
    finish_reason: str
    model_calls: int
    input_tokens: int
    output_tokens: int
    raw_cost: float
    environment_seconds: float = 0.0
    monetary_cost: float = 0.0
    completed: bool = True


@dataclass(frozen=True)
class Grade:
    quality: float
    metrics: dict[str, float]
    grader_version: str


@dataclass(frozen=True)
class GenerationRecord:
    role: Role
    input_ids: tuple[int, ...]
    input_hash: str
    response_token_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    old_logprobs: tuple[float, ...]
    text: str
    temperature: float
    finish_reason: str
    policy_version: str
    schema_valid: bool | None = None
    schema_error: str = ""
    prompt_mapping: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChoiceForward:
    id: str
    input_ids: tuple[int, ...]
    input_hash: str
    slot_token_ids: tuple[int, ...]
    candidates: tuple[MemoryEntry, ...]
    old_logits: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    temperature: float
    policy_version: str
    forward_count: int
    scope: str
    renderer_version: str
    token_budget: int
    max_selected: int
    recall_candidates: tuple[MemoryEntry, ...] = ()
    packing_mapping: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Slate:
    forward: ChoiceForward
    selected_indices: tuple[int, ...]
    legal_masks: tuple[tuple[bool, ...], ...]
    conditional_logprobs: tuple[float, ...]
    old_slate_logprob: float
    termination: str
    memory_block: str
    token_count: int

    @property
    def selected_ids(self):
        return tuple(self.forward.candidates[i].id for i in self.selected_indices)

    def selection_output(self):
        import math

        return {
            "candidate_ids": [e.id for e in self.forward.candidates],
            "candidate_revisions": [e.revision for e in self.forward.candidates],
            "initial_probabilities": self.forward.initial_probabilities,
            "slot_probabilities": self.forward.initial_probabilities,
            "selected_ids": self.selected_ids,
            "selected_slots": ["ABCDEFGHIJKLMNOP"[i] for i in self.selected_indices],
            "conditional_selected_probabilities": [math.exp(p) for p in self.conditional_logprobs],
            "selected_log_prob": self.conditional_logprobs,
            "slate_log_prob": self.old_slate_logprob,
            "slate_probability": math.exp(self.old_slate_logprob),
            "H": len(self.selected_indices),
            "R_forward_count": self.forward.forward_count,
            "termination": self.termination,
            "adapter": "R",
            "policy_version": self.forward.policy_version,
        }


@dataclass
class PolicyRow:
    id: str
    role: Role
    episode_id: str
    group_id: str
    domain: str
    split: str
    position: str
    time: int
    record: GenerationRecord | Slate
    bank_before_hash: str
    bank_after_hash: str
    source: Literal["outcome", "local", "warmup"] = "outcome"
    anchor_id: str = ""
    advantage: float = 0.0
    row_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardEvent:
    time: int
    value: float
    kind: str
    identity: str
    raw_quality: float | None = None
    raw_cost: float = 0.0


@dataclass(frozen=True)
class Anchor:
    id: str
    episode_id: str
    source: Source
    bank: Bank
    position: str
    time: int


@dataclass
class EpisodeResult:
    plan: EpisodePlan
    execution_id: str
    bank: Bank
    rows: list[PolicyRow]
    rewards: list[RewardEvent]
    anchors: list[Anchor]
    positions: dict[str, int]
    executions: dict[str, Execution]
    grades: dict[str, Grade]
    reads: list[dict[str, Any]] = field(default_factory=list)
