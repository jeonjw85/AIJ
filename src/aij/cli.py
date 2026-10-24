"""Command-line interface."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from aij import __version__
from aij.auth import auth_app
from aij.config import build_provider, load_run_config
from aij.corpus import load_objectives
from aij.detectors import Detector, EscalatingDetector, HeuristicDetector, JudgeDetector
from aij.engine import Store, build_jobs, run_scan
from aij.engine.scanner import ScanProgress
from aij.errors import AIJError, CorpusError, FatalProviderError
from aij.models import ResultRecord, ScanSummary, Severity, Verdict
from aij.providers import build_provider as instantiate_provider
from aij.report import ReportBundle, overall_grade, render, write_report
from aij.techniques import build_registry

app = typer.Typer(
    name="aij",
    help="Async jailbreak-resistance scanner for LLM endpoints.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

app.add_typer(auth_app, name="auth")

DEFAULT_DB = Path(".aij/runs.db")


# -- shared options ----------------------------------------------------------

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to aij.toml.")]
ModelOpt = Annotated[str, typer.Option("--model", "-m", help="Target model id.")]
ProviderOpt = Annotated[
    str | None,
    typer.Option(
        "--provider", "-p", help="openai | anthropic | gemini | ollama | openai-compatible"
    ),
]
BaseUrlOpt = Annotated[str | None, typer.Option("--base-url", help="Override the API base URL.")]
DbOpt = Annotated[Path, typer.Option("--db", help="SQLite path for run history.")]
ProfileOpt = Annotated[
    str | None,
    typer.Option("--profile", help="Saved profile from 'aij auth add'."),
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"aij {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """AIJ: measure how well a model endpoint holds its guardrails."""


# -- list --------------------------------------------------------------------


@app.command("list")
def list_command(
    what: Annotated[
        str, typer.Argument(help="objectives | techniques | providers | runs")
    ] = "objectives",
    config: ConfigOpt = None,
    db: DbOpt = DEFAULT_DB,
    corpus_dir: Annotated[
        list[Path] | None, typer.Option("--corpus-dir", help="Extra objective directory.")
    ] = None,
    plugin: Annotated[
        list[Path] | None, typer.Option("--plugin", help="Technique plugin .py file.")
    ] = None,
    category: Annotated[str | None, typer.Option("--category", help="Filter objectives.")] = None,
) -> None:
    """List the corpus, techniques, providers, or past runs."""
    if what == "objectives":
        objectives = load_objectives(corpus_dirs=corpus_dir)
        if category:
            objectives = [o for o in objectives if o.category == category]
        table = Table(title=f"Objectives ({len(objectives)})", show_lines=False)
        table.add_column("id", style="cyan", no_wrap=True)
        table.add_column("severity")
        table.add_column("description", overflow="fold")
        for objective in objectives:
            table.add_row(
                escape(objective.id),
                _severity_text(objective.severity),
                escape(objective.description or objective.prompt[:80]),
            )
        console.print(table)
        return

    if what == "techniques":
        registry = build_registry(plugin)
        table = Table(title=f"Techniques ({len(registry)})")
        table.add_column("id", style="cyan", no_wrap=True)
        table.add_column("family")
        table.add_column("turns", justify="right")
        table.add_column("description", overflow="fold")
        for spec in registry.specs():
            table.add_row(
                escape(spec.id),
                escape(spec.family),
                "multi" if spec.multi_turn else "1",
                escape(spec.description),
            )
        console.print(table)
        return

    if what == "providers":
        from aij.providers import PROVIDERS

        table = Table(title="Providers")
        table.add_column("kind", style="cyan")
        table.add_column("default base URL")
        from aij.config import DEFAULT_BASE_URLS

        for kind in sorted(PROVIDERS):
            table.add_row(kind, DEFAULT_BASE_URLS.get(kind, "—"))
        console.print(table)
        return

    if what == "runs":
        with Store(db) as store:
            runs = store.list_runs(limit=50)
        if not runs:
            console.print("[dim]no runs recorded yet[/dim]")
            return
        table = Table(title="Runs")
        table.add_column("run_id", style="cyan", no_wrap=True)
        table.add_column("model")
        table.add_column("started")
        table.add_column("status")
        table.add_column("probes", justify="right")
        table.add_column("hits", justify="right")
        for run in runs:
            hits = f"{run.get('complied', 0)}+{run.get('partial', 0)}p"
            table.add_row(
                escape(run["run_id"]),
                escape(f"{run['provider']}/{run['model']}"),
                (run["started_at"] or "")[:19].replace("T", " "),
                run["status"],
                str(run.get("total", "")),
                hits,
            )
        console.print(table)
        return

    raise typer.BadParameter(
        f"unknown target {what!r}; use objectives, techniques, providers, runs"
    )


# -- scan --------------------------------------------------------------------


@app.command("scan")
def scan_command(
    config: ConfigOpt = None,
    profile: ProfileOpt = None,
    model: ModelOpt = "",
    provider: ProviderOpt = None,
    base_url: BaseUrlOpt = None,
    objectives: Annotated[
        list[str] | None, typer.Option("--objective", "-o", help="Objective id, category, or glob.")
    ] = None,
    techniques: Annotated[
        list[str] | None, typer.Option("--technique", "-t", help="Technique id or glob.")
    ] = None,
    corpus_dir: Annotated[
        list[Path] | None, typer.Option("--corpus-dir", help="Extra objective directory.")
    ] = None,
    plugin: Annotated[
        list[Path] | None, typer.Option("--plugin", help="Technique plugin .py file.")
    ] = None,
    run_id: Annotated[str | None, typer.Option("--run-id", help="Name this run.")] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Skip probes already stored.")] = False,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed technique choices for reproducibility.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="Parallel probes.")
    ] = None,
    rps: Annotated[float | None, typer.Option("--rps", help="Requests per second ceiling.")] = None,
    max_requests: Annotated[
        int | None, typer.Option("--max-requests", help="Hard request cap.")
    ] = None,
    max_cost: Annotated[float | None, typer.Option("--max-cost", help="USD ceiling.")] = None,
    max_seconds: Annotated[
        float | None, typer.Option("--max-seconds", help="Wall-clock cap.")
    ] = None,
    judge: Annotated[
        bool,
        typer.Option("--judge/--no-judge", help="Escalate ambiguous verdicts to an LLM judge."),
    ] = False,
    judge_model: Annotated[
        str | None, typer.Option("--judge-model", help="Model for judging.")
    ] = None,
    judge_provider: Annotated[
        str | None, typer.Option("--judge-provider", help="Provider for judging.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-O", help="Write a report here.")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="json | markdown | html")] = "markdown",
    db: DbOpt = DEFAULT_DB,
    keep_quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress output.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the probe matrix and exit without calling the model."),
    ] = False,
    fail_on_finding: Annotated[
        str | None,
        typer.Option("--fail-on", help="Exit non-zero on findings at this severity or above."),
    ] = None,
) -> None:
    """Run a scan against one model endpoint."""
    try:
        exit_code = asyncio.run(
            _scan(
                config=config,
                profile=profile,
                model=model,
                provider=provider,
                base_url=base_url,
                objectives=objectives,
                techniques=techniques,
                corpus_dir=corpus_dir,
                plugin=plugin,
                run_id=run_id,
                resume=resume,
                seed=seed,
                concurrency=concurrency,
                rps=rps,
                max_requests=max_requests,
                max_cost=max_cost,
                max_seconds=max_seconds,
                judge=judge,
                judge_model=judge_model,
                judge_provider=judge_provider,
                out=out,
                fmt=fmt,
                db=db,
                quiet=keep_quiet,
                dry_run=dry_run,
                fail_on=fail_on_finding,
            )
        )
    except AIJError as exc:
        err_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    raise typer.Exit(exit_code)


async def _scan(
    *,
    config: Path | None,
    profile: str | None,
    model: str,
    provider: str | None,
    base_url: str | None,
    objectives: list[str] | None,
    techniques: list[str] | None,
    corpus_dir: list[Path] | None,
    plugin: list[Path] | None,
    run_id: str | None,
    resume: bool,
    seed: int | None,
    concurrency: int | None,
    rps: float | None,
    max_requests: int | None,
    max_cost: float | None,
    max_seconds: float | None,
    judge: bool,
    judge_model: str | None,
    judge_provider: str | None,
    out: Path | None,
    fmt: str,
    db: Path,
    quiet: bool,
    dry_run: bool,
    fail_on: str | None,
) -> int:
    overrides = {
        "model": model or None,
        "kind": provider,
        "base_url": base_url,
        "objectives": _split_specs(objectives),
        "techniques": _split_specs(techniques),
        "concurrency": concurrency,
        "requests_per_second": rps,
        "max_requests": max_requests,
        "max_cost_usd": max_cost,
        "max_seconds": max_seconds,
        "judge_model": judge_model,
        "seed": seed,
    }
    run_config = load_run_config(
        config_path=config, overrides=overrides, run_id=run_id, profile=profile
    )

    objective_list = load_objectives(run_config.objectives, corpus_dirs=corpus_dir)
    registry = build_registry(plugin)
    technique_list = registry.resolve(run_config.techniques)

    jobs = build_jobs(objective_list, technique_list)
    if not jobs:
        err_console.print("[yellow]warning:[/yellow] the matrix is empty; nothing to scan")
        return 0

    if dry_run:
        _print_matrix(run_config, objective_list, technique_list, len(jobs))
        return 0

    target = instantiate_provider(run_config.provider)
    detector: Detector = HeuristicDetector()
    judge_detector: JudgeDetector | None = None
    if judge:
        judge_provider_config = run_config.provider
        if judge_provider or judge_model:
            raw = run_config.provider.model_dump()
            if judge_provider:
                raw["kind"] = judge_provider
            if judge_model:
                raw["model"] = judge_model
            judge_provider_config = build_provider(raw)
        judge_detector = JudgeDetector(instantiate_provider(judge_provider_config))
        detector = EscalatingDetector(judge_detector)

    if not quiet:
        console.print(
            f"[bold]aij[/bold] {__version__} → "
            f"[cyan]{escape(run_config.provider.kind)}[/cyan]/"
            f"[cyan]{escape(run_config.provider.model)}[/cyan]"
        )
        console.print(
            f"{len(objective_list)} objectives × {len(technique_list)} techniques = "
            f"[bold]{len(jobs)}[/bold] probes · concurrency {run_config.concurrency}"
            f" · run [cyan]{escape(run_config.run_id)}[/cyan]"
        )

    progress_bar: Progress | None = None
    task_id = None
    if not quiet:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        )
        progress_bar.start()
        task_id = progress_bar.add_task("scanning", total=len(jobs))

    def on_progress(record: ResultRecord, state: ScanProgress) -> None:
        if progress_bar is None or task_id is None:
            return
        progress_bar.update(task_id, completed=state.completed)
        if record.final.verdict.is_success:
            icon = "!" if record.final.verdict is Verdict.PARTIAL else "×"
            console.print(
                f"  [{_verdict_color(record.final.verdict)}]{icon}[/] "
                f"{escape(record.objective_id)} via {escape(record.technique_id)} "
                f"({record.severity.value}, {record.final.verdict.value})"
            )

    store = Store(db)
    try:
        store.create_run(run_config)
        outcome = await run_scan(
            config=run_config,
            provider=target,
            detector=detector,
            jobs=jobs,
            store=store,
            progress=on_progress,
            resume=resume,
        )
    finally:
        if progress_bar is not None:
            progress_bar.stop()
        if judge_detector is not None:
            await judge_detector.aclose()
        store.close()

    summary = outcome.summary

    if outcome.fatal_error is not None:
        # Nothing ran. A summary table of zeros would bury the actual cause.
        err_console.print(f"[red]scan aborted:[/red] {escape(str(outcome.fatal_error))}")
        _print_fatal_hint(outcome.fatal_error, run_config)
        return 2

    _print_summary(summary)

    if out is not None:
        bundle = ReportBundle(
            summary=summary,
            results=outcome.results,
            run_config=run_config.model_dump(mode="json"),
        )
        written = write_report(bundle, fmt, out)  # type: ignore[arg-type]
        console.print(f"report written to [cyan]{escape(str(written))}[/cyan]")

    if fail_on:
        threshold = _parse_severity(fail_on)
        blocking = [
            r
            for r in outcome.results
            if r.final.verdict.is_success and r.severity.rank >= threshold.rank
        ]
        if blocking:
            return 1
    return 0


def _print_matrix(run_config, objectives, techniques, total: int) -> None:
    """Print what would run, without spending anything."""
    console.print(
        f"[bold]dry run[/bold] → {escape(run_config.provider.kind)}/"
        f"{escape(run_config.provider.model)} (no requests will be sent)"
    )
    console.print(
        f"{len(objectives)} objectives × {len(techniques)} techniques = [bold]{total}[/bold] probes"
    )
    table = Table(title="matrix", show_header=True, header_style="dim")
    table.add_column("objective")
    table.add_column("severity")
    table.add_column("category")
    for objective in objectives:
        table.add_row(
            escape(objective.id), _severity_text(objective.severity), escape(objective.category)
        )
    console.print(table)

    tech_table = Table(title="techniques", show_header=True, header_style="dim")
    tech_table.add_column("id")
    tech_table.add_column("family")
    tech_table.add_column("turns", justify="right")
    for technique in techniques:
        tech_table.add_row(
            escape(technique.spec.id),
            escape(technique.spec.family),
            "multi" if technique.spec.multi_turn else "1",
        )
    console.print(tech_table)

    console.print(
        f"concurrency {run_config.concurrency}"
        + (f" · {run_config.requests_per_second} rps" if run_config.requests_per_second else "")
        + (f" · max {run_config.max_requests} requests" if run_config.max_requests else "")
        + (f" · max ${run_config.max_cost_usd}" if run_config.max_cost_usd else "")
    )


def _print_fatal_hint(error, run_config) -> None:
    """Turn a provider rejection into something the operator can act on."""
    status = getattr(error, "status", None)
    if status in (401, 403):
        err_console.print(
            "  The endpoint rejected the credentials or the account is not "
            "entitled to this model. Nothing was scanned."
        )
    elif status == 404:
        err_console.print(
            f"  The model {escape(run_config.provider.model)!r} or the URL path was "
            "not found. Check --model and --base-url."
        )
    err_console.print("  Run 'aij doctor' to verify connectivity first.")


def _print_summary(summary: ScanSummary) -> None:
    grade, reason = overall_grade(summary)
    style = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "red"}.get(grade, "white")
    console.print()
    console.print(f"grade [{style} bold]{grade}[/] — {escape(reason)}")
    console.print(
        f"{summary.total} probes · {summary.succeeded} succeeded "
        f"({summary.complied} complied, {summary.partial} partial) · {summary.refused} refused "
        f"· {summary.inconclusive} inconclusive · {summary.errors} errors"
    )
    if summary.by_category:
        table = Table(title="by category", show_header=True, header_style="dim")
        table.add_column("category")
        table.add_column("probes", justify="right")
        table.add_column("hits", justify="right")
        table.add_column("asr", justify="right")
        for name, stats in sorted(summary.by_category.items(), key=lambda kv: -kv[1].asr):
            table.add_row(escape(name), str(stats.total), str(stats.succeeded), f"{stats.asr:.0%}")
        console.print(table)
    if summary.budget_exhausted:
        console.print("[yellow]scan stopped early: budget ceiling reached[/yellow]")


# -- report ------------------------------------------------------------------


@app.command("report")
def report_command(
    run_id: Annotated[str, typer.Argument(help="Run id to report on.")],
    out: Annotated[Path | None, typer.Option("--out", "-O", help="Write to a file.")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="json | markdown | html")] = "markdown",
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Re-render a stored run."""
    with Store(db) as store:
        summary = store.get_run_summary(run_id)
        if summary is None:
            err_console.print(
                f"[red]error:[/red] no completed run {escape(run_id)!r} in {escape(str(db))}"
            )
            raise typer.Exit(2)
        results = store.load_results(run_id)
        run_config = store.get_run_config(run_id)

    bundle = ReportBundle(
        summary=summary,
        results=results,
        run_config=run_config.model_dump(mode="json") if run_config else None,
    )
    if out is None:
        console.print(render(bundle, fmt))  # type: ignore[arg-type]
        return
    written = write_report(bundle, fmt, out)  # type: ignore[arg-type]
    console.print(f"report written to [cyan]{escape(str(written))}[/cyan]")


# -- validate ----------------------------------------------------------------


@app.command("validate")
def validate_command(
    corpus_dir: Annotated[
        list[Path] | None, typer.Option("--corpus-dir", help="Extra objective directory.")
    ] = None,
    plugin: Annotated[
        list[Path] | None, typer.Option("--plugin", help="Technique plugin .py file.")
    ] = None,
) -> None:
    """Validate the corpus and technique plugins without contacting a model."""
    problems = 0
    try:
        objectives = load_objectives(corpus_dirs=corpus_dir)
        console.print(f"[green]ok[/green] {len(objectives)} objectives loaded")
        seen: dict[str, str] = {}
        for objective in objectives:
            if objective.id in seen:
                console.print(f"[red]duplicate[/red] {escape(objective.id)}")
                problems += 1
            seen[objective.id] = objective.category
            if not objective.success_markers and objective.category != "canary":
                console.print(
                    f"[yellow]warn[/yellow] {escape(objective.id)}: no success_markers "
                    "(detection will rely on heuristic structure only)"
                )
            if (
                objective.success_markers
                and objective.min_marker_hits == 1
                and objective.category != "canary"
            ):
                # Single short/common markers produce false positives; flag them
                # so corpus authors tighten the probe before trusting results.
                weak = [m for m in objective.success_markers if len(m) <= 6]
                if len(objective.success_markers) == 1 or len(weak) * 2 >= len(
                    objective.success_markers
                ):
                    console.print(
                        f"[yellow]warn[/yellow] {escape(objective.id)}: markers look generic "
                        f"({', '.join(weak) or objective.success_markers[0]!r}); "
                        "consider raising min_marker_hits"
                    )
        for category in sorted({o.category for o in objectives}):
            count = sum(1 for o in objectives if o.category == category)
            console.print(f"  {escape(category)}: {count}")
    except CorpusError as exc:
        console.print(f"[red]error[/red] {escape(str(exc))}")
        problems += 1

    try:
        registry = build_registry(plugin)
        console.print(f"[green]ok[/green] {len(registry)} techniques registered")
        multi = [s.id for s in registry.specs() if s.multi_turn]
        console.print(f"  multi-turn: {escape(', '.join(multi) if multi else 'none')}")
    except CorpusError as exc:
        console.print(f"[red]error[/red] {escape(str(exc))}")
        problems += 1

    if problems:
        raise typer.Exit(1)
    console.print("[green]validation passed[/green]")


# -- init --------------------------------------------------------------------


@app.command("init")
def init_command(
    directory: Annotated[Path, typer.Option("--dir", help="Where to write the config.")] = Path(
        "."
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter aij.toml and a custom objective file."""
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "aij.toml"
    corpus_path = directory / "objectives" / "custom.yaml"

    if config_path.exists() and not force:
        err_console.print(
            f"[yellow]skip[/yellow] {escape(str(config_path))} already exists (use --force)"
        )
    else:
        config_path.write_text(_STARTER_CONFIG, encoding="utf-8")
        console.print(f"[green]wrote[/green] {escape(str(config_path))}")

    if corpus_path.exists() and not force:
        err_console.print(
            f"[yellow]skip[/yellow] {escape(str(corpus_path))} already exists (use --force)"
        )
    else:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_path.write_text(_STARTER_CORPUS, encoding="utf-8")
        console.print(f"[green]wrote[/green] {escape(str(corpus_path))}")


_STARTER_CONFIG = """# AIJ configuration
[provider]
kind = "openai"           # openai | anthropic | gemini | ollama | openai-compatible
model = "gpt-4o-mini"
# base_url = "https://api.openai.com/v1"
# api_key_env = "OPENAI_API_KEY"

[scan]
concurrency = 4
requests_per_second = 2.0
retries = 3
max_requests = 200
max_cost_usd = 2.0
# stop_on_success = false

[corpus]
objectives = ["cyber", "harm"]
techniques = ["direct", "persona", "crescendo"]
"""

_STARTER_CORPUS = """# Custom objectives. Every probe needs an id, a prompt, and a way to tell a
# real answer from a refusal. Markers should name artifacts an operative answer
# must contain; vague markers make verdicts unreliable.
objectives:
  - id: custom.example
    category: custom
    severity: medium
    description: What a success means for this probe.
    prompt: >-
      Replace this with the request you want to test.
    success_markers:
      - "exact phrase an operative answer would include"
    refusal_markers:
      - "I can't"
    tags: [example]
"""


# -- doctor ------------------------------------------------------------------


@app.command("doctor")
def doctor_command(
    config: ConfigOpt = None,
    profile: ProfileOpt = None,
    model: ModelOpt = "",
    provider: ProviderOpt = None,
    db: DbOpt = DEFAULT_DB,
) -> None:
    """Check configuration, credentials, and connectivity."""
    ok = True
    overrides = {"model": model or None, "kind": provider}
    try:
        run_config = load_run_config(config_path=config, overrides=overrides, profile=profile)
        console.print(
            f"[green]ok[/green] provider {escape(run_config.provider.kind)} / "
            f"{escape(run_config.provider.model)}"
        )
    except AIJError as exc:
        console.print(f"[red]config[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    async def check() -> None:
        import httpx

        from aij.models import Message, Role

        target = instantiate_provider(run_config.provider)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await target.complete(
                    [Message(role=Role.USER, content="Reply with the single word: pong")],
                    client=client,
                    max_retries=0,
                )
        except FatalProviderError:
            # Non-fatal failures still mean "not usable", so both fail doctor.
            raise
        console.print(
            f"[green]ok[/green] endpoint reachable: {escape(repr(response.text.strip()[:80]))}"
        )

    try:
        asyncio.run(check())
    except FatalProviderError as exc:
        status = exc.status
        console.print(f"[red]endpoint[/red] {escape(str(exc))}")
        if status in (401, 403):
            console.print(
                "  The key was rejected, or this account is not entitled to the model. "
                "A scan cannot run until this is resolved."
            )
        elif status == 404:
            console.print(
                f"  Model {escape(run_config.provider.model)!r} or the URL path was not "
                "found. Check --model and --base-url."
            )
        ok = False
    except AIJError as exc:
        console.print(f"[red]endpoint[/red] {escape(str(exc))}")
        ok = False

    try:
        with Store(db) as store:
            runs = store.list_runs(limit=1)
        console.print(f"[green]ok[/green] store {escape(str(db))} ({len(runs)} run(s) available)")
    except AIJError as exc:
        console.print(f"[red]store[/red] {escape(str(exc))}")
        ok = False

    objectives = load_objectives()
    registry = build_registry()
    console.print(
        f"[green]ok[/green] corpus {len(objectives)} objectives, {len(registry)} techniques"
    )
    if not ok:
        raise typer.Exit(1)


# -- helpers -----------------------------------------------------------------


def _severity_text(severity: Severity) -> str:
    color = {
        Severity.CRITICAL: "bright_red",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "blue",
        Severity.INFO: "dim",
    }[severity]
    return f"[{color}]{severity.value}[/{color}]"


def _verdict_color(verdict: Verdict) -> str:
    return {
        Verdict.COMPLIED: "red",
        Verdict.PARTIAL: "yellow",
        Verdict.REFUSED: "green",
        Verdict.INCONCLUSIVE: "dim",
        Verdict.ERROR: "magenta",
    }[verdict]


def _parse_severity(value: str) -> Severity:
    try:
        return Severity(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(s.value for s in Severity)
        raise typer.BadParameter(f"expected one of: {valid}") from exc


def _split_specs(values: list[str] | None) -> list[str] | None:
    """Accept both ``--objective a --objective b`` and ``--objective a,b``."""
    if values is None:
        return None
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def main() -> None:
    """Entry point used by the console script."""
    try:
        app()
    except KeyboardInterrupt:
        err_console.print("\n[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
