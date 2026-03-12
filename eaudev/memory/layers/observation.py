"""
ObservationBuffer — deterministic conversation compression with SQLite persistence.

Turns are compressed on write and stored in SQLite so they survive process
restarts. The in-memory buffer is always a mirror of the DB — loaded on init,
flushed on every write.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ObservationBuffer:
    """
    Deterministic conversation compression layer with SQLite persistence.

    Compression patterns (ORDER IS CRITICAL):
      1. Whole-word filler removal (um, uh, ah, er, hm, hmm)
      2. Ellipsis collapse (... → .)
      3. Whitespace collapse
      4. Spaced period collapse (. . → .)
      5. Redundant period before ?/! (.? → ?, .! → !)
      6. Final trim
    """

    COMPRESSION_PATTERNS = [
        (re.compile(r'\b(?:um|uh|ah|er|hm|hmm)\b', re.IGNORECASE), ''),
        (re.compile(r'\.{2,}'),                                       '.'),
        (re.compile(r'\s+'),                                          ' '),
        (re.compile(r'\. \.'),                                        '.'),
        (re.compile(r'\.([?!])'),                                     r'\1'),
        (re.compile(r'^\s+|\s+$'),                                    ''),
    ]

    def __init__(
        self,
        max_turns: int = 50,
        db_path: str = '~/.eaudev/observations.db',
        scope: str = 'global',
    ) -> None:
        """
        Args:
            max_turns:  Maximum turns to retain (oldest evicted first).
            db_path:    SQLite database path.
            scope:      Namespace for turns — use session_id to isolate sessions,
                        or 'global' for cross-session rolling buffer.
        """
        self.max_turns = max_turns
        self.scope = scope
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # Load existing turns into memory
        self.turns: List[Dict[str, str]] = self._load_turns()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope     TEXT    NOT NULL DEFAULT 'global',
                    role      TEXT    NOT NULL,
                    text      TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
            """)
            conn.execute('CREATE INDEX IF NOT EXISTS idx_scope ON turns(scope)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_scope_id ON turns(scope, id)')

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_turns(self) -> List[Dict[str, str]]:
        """Load turns for this scope from SQLite, respecting max_turns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT role, text, timestamp FROM turns WHERE scope = ? ORDER BY id DESC LIMIT ?',
                (self.scope, self.max_turns)
            )
            rows = cursor.fetchall()
        # Reverse so oldest-first
        return [{'role': r[0], 'text': r[1], 'timestamp': r[2]} for r in reversed(rows)]

    def _persist_turn(self, role: str, text: str, timestamp: str) -> None:
        """Write one turn to SQLite and evict oldest if over limit."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO turns (scope, role, text, timestamp) VALUES (?, ?, ?, ?)',
                (self.scope, role, text, timestamp)
            )
            # Evict oldest turns beyond max_turns
            conn.execute("""
                DELETE FROM turns WHERE scope = ? AND id NOT IN (
                    SELECT id FROM turns WHERE scope = ? ORDER BY id DESC LIMIT ?
                )
            """, (self.scope, self.scope, self.max_turns))

    # ── Public API ────────────────────────────────────────────────────────────

    def add_turn(self, role: str, text: str) -> None:
        """Compress and store a conversation turn (persists immediately)."""
        compressed = self._compress(text)
        timestamp = datetime.now().isoformat()
        self._persist_turn(role, compressed, timestamp)
        self.turns.append({'role': role, 'text': compressed, 'timestamp': timestamp})
        # Evict from in-memory buffer too
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context(self, max_turns: int = 10) -> str:
        """Return recent turns as a formatted string."""
        recent = self.turns[-max_turns:] if max_turns else self.turns
        lines = []
        for t in recent:
            icon = '👤' if t['role'] == 'user' else '🤖'
            lines.append(f"{icon} {t['text']}")
        return '\n'.join(lines)

    def get_messages_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Return recent turns in OpenAI message format."""
        recent = self.turns[-max_turns:] if max_turns else self.turns
        return [{'role': t['role'], 'content': t['text']} for t in recent]

    def clear(self, scope: Optional[str] = None) -> None:
        """Clear turns for this scope (or a specific scope) from DB and memory."""
        target = scope or self.scope
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM turns WHERE scope = ?', (target,))
        if target == self.scope:
            self.turns.clear()

    def get_stats(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM turns WHERE scope = ?', (self.scope,)
            )
            count = cursor.fetchone()[0]
        return {
            'turn_count': count,
            'max_turns': self.max_turns,
            'scope': self.scope,
        }

    # ── Compression ───────────────────────────────────────────────────────────

    def _compress(self, text: str) -> str:
        result = text
        for pattern, replacement in self.COMPRESSION_PATTERNS:
            result = pattern.sub(replacement, result)
        return result.strip()
