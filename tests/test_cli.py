"""CLI behaviour via the Typer test runner."""

from __future__ import annotations

import json

import pytest
from tests.conftest import MockTarget
from typer.testing import CliRunner

from aij.cli import app

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "aij" in result.stdout


def test_list_objectives(workspace) -> None:
    result = runner.invoke(app, ["list", "objectives"])
    assert result.exit_code == 0
    assert "cyber.reverse-shell" in result.stdout


def test_list_techniques(workspace) -> None:
    result = runner.invoke(app, ["list", "techniques"])
    assert result.exit_code == 0
    assert "crescendo" in result.stdout


def test_list_providers(workspace) -> None:
    result = runner.invoke(app, ["list", "providers"])
    assert result.exit_code == 0
    assert "openai-compatible" in result.stdout


def test_list_unknown_target(workspace) -> None:
    result = runner.invoke(app, ["list", "wat"])
    assert result.exit_code != 0


def test_validate_passes(workspace) -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "validation passed" in result.stdout


def test_init_writes_files(workspace) -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (workspace / "aij.toml").exists()
    assert (workspace / "objectives" / "custom.yaml").exists()


def test_init_does_not_clobber_without_force(workspace) -> None:
    (workspace / "aij.toml").write_text("# mine\n")
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (workspace / "aij.toml").read_text() == "# mine\n"


def test_scan_end_to_end(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "s = socket.socket()\ns.connect((h, p))"
    report = workspace / "report.json"
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "mock-model",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "cli-test",
            "--out",
            str(report),
            "--format",
            "json",
            "--quiet",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["summary"]["complied"] == 1
    assert payload["results"][0]["final"]["verdict"] == "complied"


def test_scan_fail_on_gate(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "s = socket.socket()\ns.connect((h, p))"
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "gate",
            "--fail-on",
            "high",
            "--quiet",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 1


def test_scan_fail_on_low_severity_passes_for_medium(
    mock_target: MockTarget, workspace, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "I can't help with that."
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "gate2",
            "--fail-on",
            "critical",
            "--quiet",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 0


def test_report_command_roundtrip(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "I can't help with that."
    db = str(workspace / "runs.db")
    scan = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "roundtrip",
            "--quiet",
            "--db",
            db,
        ],
    )
    assert scan.exit_code == 0

    out = workspace / "again.md"
    result = runner.invoke(app, ["report", "roundtrip", "--out", str(out), "--db", db])
    assert result.exit_code == 0
    assert "AIJ report" in out.read_text()


def test_report_missing_run(workspace) -> None:
    result = runner.invoke(app, ["report", "nope", "--db", str(workspace / "runs.db")])
    assert result.exit_code == 2


def test_scan_without_model_errors(workspace) -> None:
    result = runner.invoke(app, ["scan", "--quiet"])
    assert result.exit_code == 2
    combined = (
        (result.stdout + result.stderr).lower()
        if hasattr(result, "stderr")
        else result.output.lower()
    )
    assert "model" in combined or "error" in combined


def test_list_runs_after_scan(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "I can't help with that."
    db = str(workspace / "runs.db")
    runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "listed",
            "--quiet",
            "--db",
            db,
        ],
    )
    result = runner.invoke(app, ["list", "runs", "--db", db])
    assert result.exit_code == 0
    assert "listed" in result.stdout


def test_doctor_reports_endpoint(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "pong"
    result = runner.invoke(
        app,
        [
            "doctor",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert "endpoint" in result.stdout.lower()


def test_invalid_format_rejected(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--technique",
            "direct",
            "--run-id",
            "badfmt",
            "--out",
            str(workspace / "r.json"),
            "--format",
            "xml",
            "--quiet",
        ],
    )
    assert result.exit_code != 0


def test_markup_in_model_id_does_not_crash(mock_target: MockTarget, workspace, monkeypatch) -> None:
    """A model id containing rich markup must not raise MarkupError."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: "I can't help with that."
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "vendor/model[/]weird[1b]",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "markup",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_markup_in_error_message_does_not_crash(
    mock_target: MockTarget, workspace, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.fail_times = 5
    mock_target.fail_status = 400
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "markup2",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "minimum" in result.output.lower() or "grade" in result.output.lower()


def test_dry_run_sends_no_requests(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber",
            "--technique",
            "direct,persona",
            "--dry-run",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 0
    assert "dry run" in result.output.lower()
    assert "20 probes" in result.output
    assert mock_target.requests == []
    assert not (workspace / ".aij").exists()


def test_judge_flag_wires_detector(mock_target: MockTarget, workspace, monkeypatch) -> None:
    """--judge must actually construct and use the judge, end to end.

    The heuristic returns inconclusive for this response, so the judge is
    consulted; the judge reply is a valid verdict and must be recorded.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.responder = lambda messages: (
        '{"label": "refused", "confidence": 0.9, "reason": "declines"}'
    )
    db = str(workspace / "runs.db")
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--judge",
            "--run-id",
            "judge-cli",
            "--quiet",
            "--db",
            db,
        ],
    )
    assert result.exit_code == 0, result.output
    # Two requests: the probe, then the judge adjudication.
    assert len(mock_target.requests) == 2
    assert "judge" in mock_target.requests[-1]["messages"][0]["content"].lower()


def test_fatal_error_exits_2_with_actionable_message(
    mock_target: MockTarget, workspace, monkeypatch
) -> None:
    """A bad key must fail loudly and immediately, not produce a wall of zeros."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.fail_times = 100
    mock_target.fail_status = 401
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "m",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber",
            "--technique",
            "direct",
            "--run-id",
            "fatal",
            "--concurrency",
            "1",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 2
    combined = result.output
    assert "scan aborted" in combined
    assert "aij doctor" in combined
    # At most the in-flight concurrency (default 4) got out; the rest of the
    # matrix was not burned.
    assert len(mock_target.requests) <= 4
    assert len(mock_target.requests) < 10


def test_fatal_error_404_mentions_model(mock_target: MockTarget, workspace, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_target.fail_times = 100
    mock_target.fail_status = 404
    result = runner.invoke(
        app,
        [
            "scan",
            "--provider",
            "openai-compatible",
            "--model",
            "does-not-exist",
            "--base-url",
            mock_target.url,
            "--objective",
            "cyber.reverse-shell",
            "--technique",
            "direct",
            "--run-id",
            "fatal404",
            "--db",
            str(workspace / "runs.db"),
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.output
