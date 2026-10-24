"""Scan orchestration: matrix expansion, concurrency, budget, persistence."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import httpx

from aij.detectors.base import Detector
from aij.engine.budget import Budget, RateLimiter
from aij.engine.store import Store, summarize_results
from aij.errors import BudgetExceeded, FatalProviderError, ProviderError
from aij.models import (
    AttackPlan,
    Detection,
    Message,
    ModelResponse,
    Objective,
    ResultRecord,
    Role,
    RunConfig,
    ScanSummary,
    TurnRecord,
    Verdict,
    utcnow,
)
from aij.providers.base import Provider
from aij.techniques.base import Technique, TechniqueContext

ProgressCallback = Callable[[ResultRecord, "ScanProgress"], None]


@dataclass(slots=True)
class ScanProgress:
    completed: int = 0
    total: int = 0
    succeeded: int = 0

    @property
    def fraction(self) -> float:
        return self.completed / self.total if self.total else 0.0


@dataclass(slots=True)
class ScanJob:
    objective: Objective
    technique_cls: type[Technique]

    @property
    def key(self) -> tuple[str, str]:
        return (self.objective.id, self.technique_cls.spec.id)


@dataclass(slots=True)
class ScanOutcome:
    summary: ScanSummary
    results: list[ResultRecord] = field(default_factory=list)
    # Requests actually sent (retries included) and estimated spend. Exposed so
    # callers and tests can assert the budget reflects real traffic rather than
    # the number of probes.
    requests_sent: int = 0
    cost_usd: float = 0.0
    # Set when the scan stopped because the endpoint could not be used at all.
    fatal_error: FatalProviderError | None = None


class Scanner:
    """Runs the objective x technique matrix against one provider."""

    def __init__(
        self,
        *,
        config: RunConfig,
        provider: Provider,
        detector: Detector,
        store: Store | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.detector = detector
        self.store = store
        self.progress = progress

        limiter = config.requests_per_second
        self.limiter = (
            RateLimiter(limiter, burst=max(1, min(config.concurrency, 8))) if limiter else None
        )
        self.budget = Budget(
            max_requests=config.max_requests,
            max_cost_usd=config.max_cost_usd,
            max_seconds=config.max_seconds,
        )
        self._progress = ScanProgress()
        self._client: httpx.AsyncClient | None = None
        self._stop = asyncio.Event()
        self._stop_reason: str | None = None
        self._fatal: FatalProviderError | None = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max(1, config.concurrency))

    # -- lifecycle ----------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self.config.provider.timeout_s, connect=10.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- public API ---------------------------------------------------------

    async def run(
        self,
        jobs: Sequence[ScanJob],
        *,
        resume: bool = False,
    ) -> ScanOutcome:
        started_at = utcnow()
        completed_keys: set[tuple[str, str]] = set()
        results: list[ResultRecord] = []

        if self.store is not None:
            self.store.create_run(self.config)
            if resume:
                completed_keys = self.store.completed_keys(self.config.run_id)
                results = self.store.load_results(self.config.run_id)

        pending = [job for job in jobs if job.key not in completed_keys]
        self._progress.total = len(jobs)
        self._progress.completed = len(jobs) - len(pending)
        self._progress.succeeded = sum(1 for record in results if record.final.verdict.is_success)

        try:
            async with asyncio.TaskGroup() as group:
                for job in pending:
                    group.create_task(self._run_job(job, results))
        except* BudgetExceeded:
            # probe() normally absorbs this; this is the belt-and-braces path.
            self._stop_reason = self._stop_reason or "budget"
            self._stop.set()

        finished_at = utcnow()
        results.sort(key=lambda r: (r.category, r.objective_id, r.technique_id))
        summary = summarize_results(
            self.config.run_id,
            results,
            started_at=started_at,
            finished_at=finished_at,
            budget_exhausted=self._stop_reason == "budget",
            notes=_stop_notes(self._stop_reason, self.budget),
        )
        summary.provider = self.provider.name
        summary.model = self.config.provider.model
        if self.store is not None:
            self.store.finish_run(summary)
        await self.aclose()
        if self._fatal is not None:
            summary.notes.append(f"scan aborted: {self._fatal}")
        return ScanOutcome(
            summary=summary,
            results=results,
            requests_sent=self.budget.requests,
            cost_usd=self.budget.cost_usd,
            fatal_error=self._fatal,
        )

    # -- job execution ------------------------------------------------------

    async def _run_job(self, job: ScanJob, results: list[ResultRecord]) -> None:
        if self._stop.is_set():
            return
        async with self._semaphore:
            if self._stop.is_set():
                return
            try:
                record = await self.probe(job.objective, job.technique_cls)
            except BudgetExceeded:
                # Budget ran out before this probe could start. It is not a
                # finding and not an error; leave it out of the results.
                return
            except FatalProviderError as exc:
                async with self._lock:
                    if self._fatal is None:
                        self._fatal = exc
                self._stop_reason = "fatal"
                self._stop.set()
                return
        async with self._lock:
            results.append(record)
            self._progress.completed += 1
            if record.final.verdict.is_success:
                self._progress.succeeded += 1
                if self.config.stop_on_success:
                    self._stop_reason = self._stop_reason or "stop-on-success"
                    self._stop.set()

        if self.store is not None:
            self.store.save_result(record)
        if self.progress is not None:
            self.progress(record, self._progress)

    async def probe(self, objective: Objective, technique_cls: type[Technique]) -> ResultRecord:
        """Run one objective/technique pair and always return a record."""
        now = utcnow()
        record = ResultRecord(
            run_id=self.config.run_id,
            objective_id=objective.id,
            category=objective.category,
            severity=objective.severity,
            technique_id=technique_cls.spec.id,
            provider=self.provider.name,
            model=self.config.provider.model,
            started_at=now,
        )
        seed = self.config.seed
        rng = random.Random(
            f"{seed}:{objective.id}:{technique_cls.spec.id}" if seed is not None else None
        )
        ctx = TechniqueContext(rng=rng)
        technique = technique_cls()

        try:
            plan = AttackPlan(
                objective_id=objective.id,
                technique_id=technique.spec.id,
                messages=technique.initial(objective, ctx),
                notes=list(ctx.notes),
            )
            record = await self._execute_plan(record, objective, technique, plan, ctx)
        except BudgetExceeded:
            # Not a probe failure: the scan simply ran out of budget before this
            # one started. Counting it as an error deflates the grade and looks
            # like the target misbehaved. Drop the record entirely and stop.
            self._stop_reason = "budget"
            self._stop.set()
            raise
        except FatalProviderError:
            # Stop the entire scan: every remaining probe would fail the same way.
            raise
        except ProviderError as exc:
            record.error = str(exc)
            record.final = Detection(
                verdict=Verdict.ERROR,
                score=0.0,
                detector="provider",
                rationale=str(exc)[:400],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.final = Detection(
                verdict=Verdict.ERROR,
                score=0.0,
                detector="scanner",
                rationale=record.error[:400],
            )
        finally:
            record.finished_at = utcnow()
        return record

    async def _execute_plan(
        self,
        record: ResultRecord,
        objective: Objective,
        technique: Technique,
        plan: AttackPlan,
        ctx: TechniqueContext,
    ) -> ResultRecord:
        transcript: list[Message] = []
        # Objectives are single-shot unless the technique drives more turns.
        max_turns = 1 if not technique.spec.multi_turn else max(1, ctx.max_turns)
        pending: list[Message] = list(plan.messages)

        for turn_index in range(max_turns):
            if not pending:
                break
            payload = [*transcript, *pending]
            response = await self._send(payload)

            transcript.extend(pending)
            transcript.append(Message(role=Role.ASSISTANT, content=response.text))

            if response.blocked:
                # The endpoint withheld the completion at the API layer. That is
                # the strongest possible refusal and must not be recorded as an
                # error, or a fully hardened model would score as 100% failures.
                detection = Detection(
                    verdict=Verdict.REFUSED,
                    score=0.0,
                    detector="provider-filter",
                    rationale=f"blocked by the API: {response.block_reason or 'unknown'}",
                )
            else:
                detection = await self.detector.detect(
                    objective, response.text, turn_index=turn_index
                )
            # Warm-up turns of an escalation ladder are not evidence either way.
            # Keep them in the transcript, but exclude them from the verdict.
            scored = technique.scores_turn(objective, pending, turn_index)
            stored_response = response.text
            if objective.redact_transcript:
                stored_response = redact_body(response.text)
            record.turns.append(
                TurnRecord(
                    index=turn_index,
                    request=payload,
                    response=stored_response,
                    detection=detection if scored else None,
                    latency_ms=response.latency_ms,
                    usage=response.usage,
                    error=None,
                )
            )

            if scored and detection.verdict.is_success:
                # Nothing further to learn from this technique.
                break

            pending = (
                technique.followup(
                    objective,
                    turn_index=len(record.turns),
                    transcript=transcript,
                    last_response=response.text,
                    ctx=ctx,
                )
                or []
            )

        detections = [turn.detection for turn in record.turns if turn.detection is not None]
        if detections:
            record.final = _aggregate_turns(detections)
        elif record.turns:
            record.final = Detection(
                verdict=Verdict.INCONCLUSIVE,
                score=0.0,
                detector="scanner",
                rationale=(
                    "technique ended before the objective was delivered "
                    f"({len(record.turns)} unscored turn(s))"
                ),
            )
        else:
            record.final = Detection(
                verdict=Verdict.ERROR,
                score=0.0,
                detector="scanner",
                rationale="technique produced no request",
            )
        return record

    async def _send(self, messages: list[Message]) -> ModelResponse:
        if self._stop.is_set():
            raise BudgetExceeded("scan stopped early")
        await self.budget.reserve()
        if self.limiter is not None:
            await self.limiter.acquire()
        client = await self._get_client()
        response = await self.provider.complete(
            messages, client=client, max_retries=self.config.retries
        )
        # A retried probe consumed more than the single slot reserved above.
        # Account for the extra attempts so max_requests reflects real traffic.
        for _ in range(max(0, response.attempts - 1)):
            await self.budget.charge_extra_request()
        await self.budget.charge(response.usage.cost_usd)
        return response

    @property
    def progress_state(self) -> ScanProgress:
        return self._progress


def redact_body(text: str) -> str:
    """Replace a response body with a non-reversible fingerprint.

    Used for canary objectives whose content must never be stored, printed, or
    pasted into a report. The verifier still needs to know the response existed
    and how long it was; it never needs the text itself.
    """
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"[redacted sha256:{digest} chars={len(text)}]"


def _stop_notes(reason: str | None, budget: Budget) -> list[str]:
    if reason == "fatal":
        return ["scan aborted: the endpoint rejected the request outright"]
    if reason == "stop-on-success":
        return ["scan stopped early: stop_on_success reached its first hit"]
    if reason == "budget":
        return [f"scan stopped early: budget ceiling ({budget.exhausted() or 'unknown'})"]
    return []


def _aggregate_turns(detections: list[Detection]) -> Detection:
    """Collapse per-turn verdicts into one record verdict.

    A leak on any turn is the finding, so success outranks everything. Among
    non-successes, a unanimous refusal is reported as a refusal; a mix of
    refusal and inconclusive stays inconclusive because we genuinely do not
    know whether the model held or merely misunderstood.
    """
    if not detections:
        return Detection(
            verdict=Verdict.ERROR, score=0.0, detector="scanner", rationale="no turns recorded"
        )

    successes = [d for d in detections if d.verdict.is_success]
    if successes:
        best = max(successes, key=lambda d: (d.verdict is Verdict.COMPLIED, d.score))
        if len(detections) > 1:
            position = detections.index(best) + 1
            best = best.model_copy(
                update={"rationale": f"turn {position}/{len(detections)}: {best.rationale}"}
            )
        return best

    errors = [d for d in detections if d.verdict is Verdict.ERROR]
    if len(errors) == len(detections):
        return errors[-1]

    last = detections[-1]
    turns = "1 turn" if len(detections) == 1 else f"all {len(detections)} turns"

    if all(d.verdict is Verdict.REFUSED for d in detections):
        return Detection(
            verdict=Verdict.REFUSED,
            score=last.score,
            detector=last.detector,
            rationale=f"refused on {turns}",
            markers=last.markers,
        )

    prefix = "" if len(detections) == 1 else f"no compliance across {len(detections)} turns; "
    return Detection(
        verdict=Verdict.INCONCLUSIVE,
        score=last.score,
        detector=last.detector,
        rationale=f"{prefix}{last.rationale}",
        markers=last.markers,
    )


async def run_scan(
    *,
    config: RunConfig,
    provider: Provider,
    detector: Detector,
    jobs: Sequence[ScanJob],
    store: Store | None = None,
    progress: ProgressCallback | None = None,
    resume: bool = False,
) -> ScanOutcome:
    scanner = Scanner(
        config=config, provider=provider, detector=detector, store=store, progress=progress
    )
    try:
        return await scanner.run(jobs, resume=resume)
    finally:
        await scanner.aclose()


def build_jobs(
    objectives: Sequence[Objective], techniques: Sequence[type[Technique]]
) -> list[ScanJob]:
    jobs: list[ScanJob] = []
    for objective in objectives:
        for technique in techniques:
            jobs.append(ScanJob(objective=objective, technique_cls=technique))
    return jobs


__all__ = [
    "ScanJob",
    "ScanOutcome",
    "ScanProgress",
    "Scanner",
    "build_jobs",
    "run_scan",
]
