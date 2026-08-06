from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunEvent(BaseModel):
    sequence: int
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class RunTrace(BaseModel):
    id: str
    session_id: str
    prompt: str
    status: str
    final_output: str | None
    error: str | None
    started_at: str
    finished_at: str | None
    events: list[RunEvent] = Field(default_factory=list)


class TraceStore:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL,
                    final_output TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
                """
            )

    def start_run(self, prompt: str, session_id: str | None = None) -> str:
        run_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(id, session_id, prompt, status, started_at)
                VALUES (?, ?, ?, 'running', ?)
                """,
                (run_id, session_id, prompt, self._now()),
            )
        return run_id

    def event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row["sequence"])
            connection.execute(
                """
                INSERT INTO run_events(run_id, sequence, event_type, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    event_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    self._now(),
                ),
            )

    def finish(self, run_id: str, final_output: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'completed', final_output = ?, finished_at = ?
                WHERE id = ?
                """,
                (final_output, self._now(), run_id),
            )

    def fail(self, run_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', error = ?, finished_at = ?
                WHERE id = ?
                """,
                (error, self._now(), run_id),
            )

    def get(self, run_id: str) -> RunTrace | None:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            rows = connection.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
        return RunTrace(
            id=run["id"],
            session_id=run["session_id"],
            prompt=run["prompt"],
            status=run["status"],
            final_output=run["final_output"],
            error=run["error"],
            started_at=run["started_at"],
            finished_at=run["finished_at"],
            events=[
                RunEvent(
                    sequence=row["sequence"],
                    event_type=row["event_type"],
                    payload=json.loads(row["payload"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ],
        )

    def list(self, limit: int = 20) -> list[RunTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [trace for row in rows if (trace := self.get(row["id"])) is not None]
