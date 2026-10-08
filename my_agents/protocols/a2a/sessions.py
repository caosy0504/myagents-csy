"""Conversation state keyed by verified principal and A2A context ID."""

import json
import os
import sqlite3
from pathlib import Path
from threading import RLock


class AgentSessionStore:
    """In-memory by default, or SQLite when a path is supplied."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self._memory: dict[tuple[str, str], dict] = {}
        self._lock = RLock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS agent_sessions ("
                    "owner TEXT NOT NULL, context_id TEXT NOT NULL, state TEXT NOT NULL, "
                    "PRIMARY KEY (owner, context_id))"
                )
            if os.name != "nt":
                os.chmod(self.path, 0o600)

    def load(self, owner: str, context_id: str) -> dict | None:
        with self._lock:
            if self.path is None:
                return self._memory.get((owner, context_id))
            with sqlite3.connect(self.path) as db:
                row = db.execute(
                    "SELECT state FROM agent_sessions WHERE owner=? AND context_id=?",
                    (owner, context_id),
                ).fetchone()
                return json.loads(row[0]) if row else None

    def save(self, owner: str, context_id: str, state: dict) -> None:
        with self._lock:
            if self.path is None:
                self._memory[(owner, context_id)] = state
                return
            with sqlite3.connect(self.path) as db:
                db.execute(
                    "INSERT INTO agent_sessions(owner,context_id,state) VALUES(?,?,?) "
                    "ON CONFLICT(owner,context_id) DO UPDATE SET state=excluded.state",
                    (owner, context_id, json.dumps(state, ensure_ascii=False)),
                )
