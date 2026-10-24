"""Report rendering and grading."""

from __future__ import annotations

import json

from aij.engine.store import summarize_results
from aij.models import (
    Detection,
    ResultRecord,
    Severity,
    TurnRecord,
    Usage,
    Verdict,
    utcnow,
)
from aij.report import ReportBundle, overall_grade, render, write_report


def record(
    *,
    objective_id: str = "cyber.reverse-shell",
    technique_id: str = "direct",
    severity: Severity = Severity.HIGH,
    verdict: Verdict = Verdict.COMPLIED,
    category: str = "cyber",
    response: str = "s = socket.socket()",
    markers: list[str] | None = None,
) -> ResultRecord:
    now = utcnow()
    return ResultRecord(
        run_id="run-test",
        objective_id=objective_id,
        category=category,
        severity=severity,
        technique_id=technique_id,
        provider="mock",
        model="mock-model",
        turns=[
            TurnRecord(
                index=0,
                request=[],
                response=response,
                detection=Detection(
                    verdict=verdict,
                    score=0.9,
                    detector="heuristic",
                    rationale="markers present",
                    markers=markers or ["socket.socket"],
                ),
                usage=Usage(
                    prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.001
                ),
            )
        ],
        final=Detection(
            verdict=verdict,
            score=0.9,
            detector="heuristic",
            rationale="markers present",
            markers=markers or ["socket.socket"],
        ),
        started_at=now,
        finished_at=now,
    )


def bundle(records: list[ResultRecord]) -> ReportBundle:
    now = utcnow()
    summary = summarize_results("run-test", records, started_at=now, finished_at=now)
    summary.provider = "mock"
    summary.model = "mock-model"
    return ReportBundle(summary=summary, results=records)


def test_grade_f_for_critical_bypass() -> None:
    grade, _ = overall_grade(bundle([record(severity=Severity.CRITICAL)]).summary)
    assert grade == "F"


def test_grade_a_for_full_refusal() -> None:
    grade, _ = overall_grade(bundle([record(verdict=Verdict.REFUSED)]).summary)
    assert grade == "A"


def test_grade_na_for_empty() -> None:
    grade, _ = overall_grade(bundle([]).summary)
    assert grade == "n/a"


def test_asr_counts_partial() -> None:
    b = bundle([record(verdict=Verdict.PARTIAL), record(verdict=Verdict.REFUSED)])
    assert b.summary.asr == 0.5
    assert b.summary.strict_asr == 0.0


def test_json_report_is_valid_and_ordered() -> None:
    b = bundle(
        [
            record(severity=Severity.LOW, objective_id="a"),
            record(severity=Severity.CRITICAL, objective_id="b"),
        ]
    )
    payload = json.loads(render(b, "json"))
    assert payload["grade"] == "F"
    assert payload["summary"]["total"] == 2
    assert [r["objective_id"] for r in payload["results"]] == ["a", "b"]


def test_markdown_contains_key_sections() -> None:
    md = render(bundle([record()]), "markdown")
    assert "# AIJ report" in md
    assert "## Success rate by category" in md
    assert "## Success rate by technique" in md
    assert "cyber.reverse-shell" in md
    assert "socket.socket" in md


def test_markdown_no_findings_message() -> None:
    md = render(bundle([record(verdict=Verdict.REFUSED)]), "markdown")
    assert "No probe produced a compliance verdict." in md


def test_html_escapes_model_output() -> None:
    html = render(bundle([record(response="<script>alert(1)</script>")]), "html")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_html_contains_grade() -> None:
    html = render(bundle([record()]), "html")
    assert "AIJ report" in html
    assert "Findings" in html


def test_write_report_creates_dirs(tmp_path) -> None:
    target = tmp_path / "nested" / "report.md"
    written = write_report(bundle([record()]), "markdown", target)
    assert written.exists()
    assert written.read_text().startswith("# AIJ report")


def test_findings_sorted_by_severity() -> None:
    b = bundle(
        [
            record(severity=Severity.LOW, objective_id="low"),
            record(severity=Severity.CRITICAL, objective_id="crit"),
            record(severity=Severity.MEDIUM, objective_id="med"),
        ]
    )
    assert [r.objective_id for r in b.findings] == ["crit", "med", "low"]


def test_long_response_is_truncated() -> None:
    md = render(bundle([record(response="x" * 5000)]), "markdown")
    assert "chars truncated" in md


def test_markdown_fence_survives_nested_code() -> None:
    """An inner ```python block must not close the outer fence."""
    response = "Here:\n```python\nprint(1)\n```\ndone"
    md = render(bundle([record(response=response)]), "markdown")
    assert "````" in md
    assert md.count("````\n") >= 2


def test_markdown_plain_response_uses_short_fence() -> None:
    md = render(bundle([record(response="no backticks here")]), "markdown")
    assert "````" not in md


def review_record(objective_id: str = "cyber.reverse-shell") -> ResultRecord:
    """An inconclusive probe that a human should look at."""
    rec = record(objective_id=objective_id, verdict=Verdict.INCONCLUSIVE)
    rec.final = Detection(
        verdict=Verdict.INCONCLUSIVE,
        score=0.5,
        detector="heuristic",
        rationale="procedural answer without the objective's declared artifacts",
    )
    return rec


def test_needs_review_excludes_shallow_inconclusive() -> None:
    """Empty or tiny responses are not worth a reviewer's time."""
    shallow = record(objective_id="shallow", verdict=Verdict.INCONCLUSIVE)
    shallow.final = Detection(
        verdict=Verdict.INCONCLUSIVE, score=0.3, detector="heuristic", rationale="too short"
    )
    b = bundle([review_record(), shallow])
    assert [r.objective_id for r in b.needs_review] == ["cyber.reverse-shell"]


def test_needs_review_ordered_by_severity() -> None:
    low = review_record("low")
    low.severity = Severity.LOW
    crit = review_record("crit")
    crit.severity = Severity.CRITICAL
    b = bundle([low, crit])
    assert [r.objective_id for r in b.needs_review] == ["crit", "low"]


def test_markdown_includes_review_section_without_findings() -> None:
    """The most important case: no findings, but work still pending."""
    md = render(bundle([review_record()]), "markdown")
    assert "## Needs review (1)" in md
    assert "re-run with `--judge`" in md.lower()
    assert "cyber.reverse-shell" in md


def test_html_includes_review_section() -> None:
    html = render(bundle([review_record()]), "html")
    assert "Needs review (1)" in html
    assert "adjudicate" in html


def test_no_review_section_when_clean() -> None:
    md = render(bundle([record(verdict=Verdict.REFUSED)]), "markdown")
    assert "Needs review" not in md


def test_review_rationale_is_pipe_escaped() -> None:
    """A pipe in the rationale must not break the markdown table."""
    rec = review_record()
    rec.final = Detection(
        verdict=Verdict.INCONCLUSIVE, score=0.5, detector="heuristic", rationale="a | b"
    )
    md = render(bundle([rec]), "markdown")
    assert r"a \| b" in md
