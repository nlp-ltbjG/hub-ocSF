from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MemoryKind = Literal["working", "episodic", "semantic", "procedural"]


class MemoryRecord(BaseModel):
    id: str
    kind: MemoryKind
    scope: str
    content: str
    source: str
    confidence: float = Field(ge=0, le=1)
    sensitivity: str
    created_at: str
    updated_at: str
    expires_at: str | None = None


class MemoryStore:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.fts_enabled = True
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sensitivity TEXT NOT NULL,
                    fingerprint TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope_kind
                    ON memories(scope, kind);
                CREATE INDEX IF NOT EXISTS idx_memories_expires_at
                    ON memories(expires_at);
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        content,
                        scope,
                        kind
                    )
                    """
                )
            except sqlite3.OperationalError:
                self.fts_enabled = False

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _fingerprint(kind: str, scope: str, content: str) -> str:
        normalized = " ".join(content.casefold().split())
        payload = f"{kind}\0{scope}\0{normalized}".encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            kind=row["kind"],
            scope=row["scope"],
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            sensitivity=row["sensitivity"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    def remember(
        self,
        content: str,
        *,
        kind: MemoryKind = "semantic",
        scope: str = "global",
        source: str = "user",
        confidence: float = 1.0,
        sensitivity: str = "normal",
        expires_at: str | None = None,
    ) -> MemoryRecord:
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("memory content cannot be empty")
        fingerprint = self._fingerprint(kind, scope, cleaned)
        now = self._now()
        memory_id = str(uuid.uuid4())

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, created_at FROM memories WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            if existing:
                memory_id = existing["id"]
                created_at = existing["created_at"]
                connection.execute(
                    """
                    UPDATE memories
                    SET content = ?, source = ?, confidence = ?, sensitivity = ?,
                        updated_at = ?, expires_at = ?
                    WHERE id = ?
                    """,
                    (
                        cleaned,
                        source,
                        confidence,
                        sensitivity,
                        now,
                        expires_at,
                        memory_id,
                    ),
                )
                if self.fts_enabled:
                    connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            else:
                created_at = now
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, kind, scope, content, source, confidence, sensitivity,
                        fingerprint, created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        kind,
                        scope,
                        cleaned,
                        source,
                        confidence,
                        sensitivity,
                        fingerprint,
                        created_at,
                        now,
                        expires_at,
                    ),
                )
            if self.fts_enabled:
                connection.execute(
                    "INSERT INTO memory_fts(memory_id, content, scope, kind) VALUES (?, ?, ?, ?)",
                    (memory_id, cleaned, scope, kind),
                )
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        assert row is not None
        return self._record(row)

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        kind: MemoryKind | None = None,
        limit: int = 6,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []
        now = self._now()
        with self._connect() as connection:
            if query.strip() and self.fts_enabled:
                terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)
                match_query = " AND ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
                if match_query:
                    clauses = [
                        "(m.expires_at IS NULL OR m.expires_at > ?)",
                        "memory_fts MATCH ?",
                    ]
                    parameters: list[object] = [now, match_query]
                    if scope:
                        clauses.append("m.scope = ?")
                        parameters.append(scope)
                    if kind:
                        clauses.append("m.kind = ?")
                        parameters.append(kind)
                    parameters.append(limit)
                    rows = connection.execute(
                        f"""
                        SELECT m.*
                        FROM memory_fts
                        JOIN memories m ON m.id = memory_fts.memory_id
                        WHERE {" AND ".join(clauses)}
                        ORDER BY bm25(memory_fts), m.updated_at DESC
                        LIMIT ?
                        """,
                        parameters,
                    ).fetchall()
                    if rows:
                        return [self._record(row) for row in rows]

            clauses = ["(expires_at IS NULL OR expires_at > ?)"]
            parameters = [now]
            if query.strip():
                clauses.append("content LIKE ?")
                parameters.append(f"%{query.strip()}%")
            if scope:
                clauses.append("scope = ?")
                parameters.append(scope)
            if kind:
                clauses.append("kind = ?")
                parameters.append(kind)
            parameters.append(limit)
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE {" AND ".join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._record(row) for row in rows]

    def list(self, limit: int = 50) -> list[MemoryRecord]:
        return self.search("", limit=limit)

    def forget(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            if self.fts_enabled:
                connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0

    def clear(self, scope: str | None = None) -> int:
        with self._connect() as connection:
            if scope:
                ids = [
                    row["id"]
                    for row in connection.execute(
                        "SELECT id FROM memories WHERE scope = ?", (scope,)
                    )
                ]
                cursor = connection.execute("DELETE FROM memories WHERE scope = ?", (scope,))
                if self.fts_enabled and ids:
                    placeholders = ",".join("?" for _ in ids)
                    connection.execute(
                        f"DELETE FROM memory_fts WHERE memory_id IN ({placeholders})", ids
                    )
            else:
                cursor = connection.execute("DELETE FROM memories")
                if self.fts_enabled:
                    connection.execute("DELETE FROM memory_fts")
        return cursor.rowcount
