"""SQLite persistence: resumable scans and result history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aij.errors import StoreError
from aij.models import (
    ResultRecord,
    RunConfig,
    ScanSummary,
    Usage,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    summary_json TEXT,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    objective_id TEXT NOT NULL,
    technique_id TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    result_json TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    PRIMARY KEY (run_id, objective_id, technique_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_run_verdict
    ON results (run_id, verdict);
CREATE INDEX IF NOT EXISTS idx_results_category
    ON results (run_id, category);
"""


class Store:
    """Thin SQLite wrapper. Safe to use from a single event loop."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.path, timeout=30.0)
        except sqlite3.Error as exc:  # pragma: no cover - environmental
            raise StoreError(f"cannot open store {self.path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- runs ---------------------------------------------------------------

    def create_run(self, config: RunConfig) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO runs (run_id, provider, model, started_at, config_json, status)
                VALUES (?, ?, ?, ?, ?, 'running')
                ON CONFLICT(run_id) DO UPDATE SET
                    provider=excluded.provider,
                    model=excluded.model,
                    config_json=excluded.config_json
                """,
                (
                    config.run_id,
                    config.provider.kind,
                    config.provider.model,
                    utcnow().isoformat(),
                    config.model_dump_json(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"cannot create run {config.run_id}: {exc}") from exc

    def finish_run(self, summary: ScanSummary) -> None:
        try:
            self._conn.execute(
                """
                UPDATE runs SET finished_at=?, summary_json=?, status='complete'
                WHERE run_id=?
                """,
                (
                    summary.finished_at.isoformat(),
                    summary.model_dump_json(),
                    summary.run_id,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"cannot finalize run {summary.run_id}: {exc}") from exc

    def get_run_config(self, run_id: str) -> RunConfig | None:
        row = self._conn.execute(
            "SELECT config_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return RunConfig.model_validate_json(row["config_json"])

    def get_run_summary(self, run_id: str) -> ScanSummary | None:
        row = self._conn.execute(
            "SELECT summary_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None or not row["summary_json"]:
            return None
        return ScanSummary.model_validate_json(row["summary_json"])

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT run_id, provider, model, started_at, finished_at, status, summary_json
            FROM runs ORDER BY started_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            summary = row["summary_json"]
            if summary:
                parsed = json.loads(summary)
                item["total"] = parsed.get("total", 0)
                item["complied"] = parsed.get("complied", 0)
                item["partial"] = parsed.get("partial", 0)
            out.append(item)
        return out

    # -- results ------------------------------------------------------------

    def save_result(self, record: ResultRecord) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO results (
                    run_id, objective_id, technique_id, category, severity,
                    verdict, score, result_json, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, objective_id, technique_id) DO UPDATE SET
                    verdict=excluded.verdict,
                    score=excluded.score,
                    result_json=excluded.result_json,
                    finished_at=excluded.finished_at
                """,
                (
                    record.run_id,
                    record.objective_id,
                    record.technique_id,
                    record.category,
                    record.severity.value,
                    record.final.verdict.value,
                    record.final.score,
                    record.model_dump_json(),
                    record.finished_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StoreError(f"cannot save result: {exc}") from exc

    def completed_keys(self, run_id: str) -> set[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT objective_id, technique_id FROM results WHERE run_id=?", (run_id,)
        ).fetchall()
        return {(row["objective_id"], row["technique_id"]) for row in rows}

    def iter_results(self, run_id: str) -> Iterator[ResultRecord]:
        cursor = self._conn.execute(
            "SELECT result_json FROM results WHERE run_id=? "
            "ORDER BY category, objective_id, technique_id",
            (run_id,),
        )
        for row in cursor:
            yield ResultRecord.model_validate_json(row["result_json"])

    def load_results(self, run_id: str) -> list[ResultRecord]:
        return list(self.iter_results(run_id))

    # -- maintenance --------------------------------------------------------

    def delete_run(self, run_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        self._conn.execute("DELETE FROM results WHERE run_id=?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0


def summarize_results(
    run_id: str,
    results: list[ResultRecord],
    *,
    started_at,
    finished_at,
    budget_exhausted: bool = False,
    notes: list[str] | None = None,
) -> ScanSummary:
    from aij.models import CategoryStats, Verdict

    summary = ScanSummary(
        run_id=run_id,
        provider=results[0].provider if results else "",
        model=results[0].model if results else "",
        started_at=started_at,
        finished_at=finished_at,
        budget_exhausted=budget_exhausted,
        notes=list(notes or []),
    )
    usage = Usage()
    for record in results:
        summary.total += 1
        usage = usage + record.usage
        verdict = record.final.verdict
        if verdict is Verdict.COMPLIED:
            summary.complied += 1
        elif verdict is Verdict.PARTIAL:
            summary.partial += 1
        elif verdict is Verdict.REFUSED:
            summary.refused += 1
        elif verdict is Verdict.ERROR:
            summary.errors += 1
        else:
            summary.inconclusive += 1

        for stats in (
            summary.by_category.setdefault(record.category, CategoryStats()),
            summary.by_technique.setdefault(record.technique_id, CategoryStats()),
            summary.by_severity.setdefault(record.severity, CategoryStats()),
        ):
            stats.total += 1
            if verdict is Verdict.COMPLIED:
                stats.complied += 1
                stats.succeeded += 1
            elif verdict is Verdict.PARTIAL:
                stats.partial += 1
                stats.succeeded += 1
            elif verdict is Verdict.REFUSED:
                stats.refused += 1
            elif verdict is Verdict.ERROR:
                stats.errors += 1

    summary.usage = usage
    return summary
