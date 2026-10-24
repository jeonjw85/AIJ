"""Core data types shared across providers, techniques, detectors, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Severity(StrEnum):
    """How serious a successful bypass of this objective would be."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Verdict(StrEnum):
    """Outcome of evaluating a single response."""

    COMPLIED = "complied"
    PARTIAL = "partial"
    REFUSED = "refused"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

    @property
    def is_success(self) -> bool:
        """Whether the target gave up ground (partial counts as a leak)."""
        return self in (Verdict.COMPLIED, Verdict.PARTIAL)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )


class ModelResponse(BaseModel):
    """A normalized completion from any provider.

    Providers signal a safety block in several different ways, and every one of
    them looks like an empty answer. ``blocked`` records that the endpoint
    refused at the API layer, which must be judged as a refusal — not as an
    error, and definitely not as the model failing to respond.
    """

    text: str
    model: str
    provider: str = ""
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    attempts: int = 1
    # Set by a provider when the API itself withheld the completion
    # (content filter, safety block, refusal stop reason).
    blocked: bool = False
    block_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class Detection(BaseModel):
    verdict: Verdict
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    detector: str = ""
    rationale: str = ""
    markers: list[str] = Field(default_factory=list)


class Objective(BaseModel):
    """A single guardrail probe: what we ask for and how we know it landed."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    severity: Severity = Severity.MEDIUM
    prompt: str
    description: str = ""
    # Substrings that, when present, indicate the model engaged with the request.
    success_markers: list[str] = Field(default_factory=list)
    # How many distinct success markers must appear. Raise this for objectives
    # whose individual markers are common English words: any single hit is weak
    # evidence, a combination is not.
    min_marker_hits: int = Field(default=1, ge=1)
    # Substrings that strongly indicate a refusal.
    refusal_markers: list[str] = Field(default_factory=list)
    # Negated phrasing that makes a success marker benign ("do not build a bomb").
    deny_context: list[str] = Field(default_factory=list)
    min_compliance_chars: int = 0
    # When true, response bodies are never persisted or printed: only a hash,
    # length, and verdict are kept. Used by hard-block canary objectives.
    redact_transcript: bool = False
    # Excluded from category / glob / "all" selection. The probe only runs when
    # named by its exact id, because sending it can itself violate a vendor's
    # terms and put the operator's account at risk.
    opt_in: bool = False
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reachable(self) -> Objective:
        """Reject objectives whose success criteria can never be satisfied.

        ``min_marker_hits`` above the number of distinct markers makes compliance
        impossible, so the probe silently always reports refusal. That looks like
        a hardened model and is the worst failure mode this tool has.
        """
        if self.success_markers and self.min_marker_hits > len(set(self.success_markers)):
            raise ValueError(
                f"min_marker_hits={self.min_marker_hits} exceeds the "
                f"{len(set(self.success_markers))} distinct success marker(s); "
                "compliance would be unreachable"
            )
        return self

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or not all(c.isalnum() or c in "-_.:" for c in cleaned):
            raise ValueError("objective id must be non-empty and use [A-Za-z0-9-_.:]")
        return cleaned


class TechniqueSpec(BaseModel):
    """Declarative description of a technique, used for listing and filtering."""

    id: str
    name: str
    description: str = ""
    family: str = "generic"
    multi_turn: bool = False
    tags: list[str] = Field(default_factory=list)


class TurnRecord(BaseModel):
    index: int
    request: list[Message]
    response: str = ""
    detection: Detection | None = None
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


class ResultRecord(BaseModel):
    """One (objective, technique) probe against one model."""

    run_id: str
    objective_id: str
    category: str
    severity: Severity
    technique_id: str
    provider: str
    model: str
    turns: list[TurnRecord] = Field(default_factory=list)
    # Worst verdict across turns: any compliance makes the attack a success.
    final: Detection = Field(
        default_factory=lambda: Detection(verdict=Verdict.INCONCLUSIVE, detector="none")
    )
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime = Field(default_factory=utcnow)
    error: str | None = None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            total = total + turn.usage
        return total

    @property
    def duration_ms(self) -> float:
        return (self.finished_at - self.started_at).total_seconds() * 1000.0


class AttackPlan(BaseModel):
    """A rendered probe ready to send, produced by a technique."""

    objective_id: str
    technique_id: str
    messages: list[Message]
    notes: list[str] = Field(default_factory=list)


class CategoryStats(BaseModel):
    total: int = 0
    succeeded: int = 0
    complied: int = 0
    partial: int = 0
    refused: int = 0
    errors: int = 0

    @property
    def asr(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


class ScanSummary(BaseModel):
    run_id: str
    provider: str
    model: str
    started_at: datetime
    finished_at: datetime
    total: int = 0
    complied: int = 0
    partial: int = 0
    refused: int = 0
    inconclusive: int = 0
    errors: int = 0
    by_category: dict[str, CategoryStats] = Field(default_factory=dict)
    by_technique: dict[str, CategoryStats] = Field(default_factory=dict)
    by_severity: dict[Severity, CategoryStats] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    budget_exhausted: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return self.complied + self.partial

    @property
    def asr(self) -> float:
        """Attack success rate, counting partial leakage as success."""
        return self.succeeded / self.total if self.total else 0.0

    @property
    def strict_asr(self) -> float:
        return self.complied / self.total if self.total else 0.0

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


class ProviderConfig(BaseModel):
    """Normalized provider settings; keys are resolved from env when omitted."""

    kind: Literal["openai", "anthropic", "gemini", "ollama", "openai-compatible"]
    model: str
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    api_key_env: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int = 1024
    timeout_s: float = 60.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Everything needed to reproduce a scan."""

    run_id: str
    provider: ProviderConfig
    objectives: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    concurrency: int = 4
    requests_per_second: float | None = None
    max_requests: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    retries: int = 3
    seed: int | None = None
    judge_model: str | None = None
    stop_on_success: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    aij_version: str = ""
    notes: list[str] = Field(default_factory=list)
