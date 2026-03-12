"""
EpisodicMemory — Session-scoped structured memory layer.

Stores structured records of completed sessions so that knowledge about
"what happened, what worked, and what to avoid" persists across context
window boundaries and process restarts.

Design principles (revised for deterministic-first architecture):
  - Memory MCP is a front-line context delivery layer, NOT a long-term store.
    Long-term storage belongs to Archive MCP.
  - Zero inference at write or read time. All extraction is deterministic.
  - Two write paths:
      1. store_episode()     — caller provides all fields directly
      2. compress_and_store() — caller provides raw ObservationBuffer turns;
                                this layer extracts structured fields without LLM.
  - Retrieval is always exact: by session_id, by recency, or by keyword tag.
  - FTS5 is retained for keyword lookup only — not fuzzy/semantic retrieval.

Schema:
  episodes(
    id             INTEGER PRIMARY KEY,
    session_id     TEXT    NOT NULL,       -- caller-provided session identifier
    timestamp      TEXT    NOT NULL,       -- ISO 8601 session start time
    turn_count     INTEGER NOT NULL,       -- number of turns in the session
    user_turns     INTEGER NOT NULL,       -- user turn count
    assistant_turns INTEGER NOT NULL,     -- assistant turn count
    summary        TEXT    NOT NULL,       -- provided by caller
    what_worked    TEXT,                   -- provided by caller (optional)
    what_to_avoid  TEXT,                   -- provided by caller (optional)
    keywords       TEXT,                   -- JSON array — caller-provided tags
    source_turns   TEXT,                   -- JSON array of {role, text} dicts
    created_at     TEXT    NOT NULL
  )

  episodes_fts — FTS5 virtual table over (summary + what_worked + what_to_avoid + keywords)
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Stop words for deterministic keyword extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "i", "we", "you", "he", "she", "they", "this", "that", "these", "those",
    "what", "how", "why", "when", "where", "which", "who", "not", "no",
    "so", "if", "then", "than", "as", "also", "all", "any", "some", "now",
    "just", "like", "get", "got", "let", "use", "used", "using",
})


def _extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    """
    Deterministically extract top N keywords from a list of text strings.

    Algorithm:
      1. Tokenise to lowercase alpha tokens (≥4 chars)
      2. Remove stop words
      3. Count frequency
      4. Return top_n by frequency, ties broken alphabetically

    No inference. No embeddings. Reproducible given the same input.
    """
    tokens: List[str] = []
    for text in texts:
        words = re.findall(r"[a-z]{4,}", text.lower())
        tokens.extend(w for w in words if w not in _STOP_WORDS)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


class EpisodicMemory:
    """
    SQLite-backed episodic memory — stores structured session records.

    Each record captures:
      - What happened (caller-provided summary)
      - What worked (caller-provided, optional)
      - What to avoid (caller-provided, optional)
      - Keywords (caller-provided or auto-extracted from turns)
      - Session statistics (turn counts, timestamps)
      - Source turns (raw conversation, optional)

    Write paths:
      store_episode()      — direct write, caller provides all fields
      compress_and_store() — structured extraction from ObservationBuffer turns

    Retrieval:
      get_by_session() — exact lookup by session_id
      get_recent()     — newest-first listing
      search()         — BM25 FTS5 keyword match (not fuzzy/semantic)
      get_stats()      — storage statistics
    """

    def __init__(self, db_path: str = "~/.eaudev/episodic.db") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       TEXT    NOT NULL,
                    timestamp        TEXT    NOT NULL,
                    turn_count       INTEGER NOT NULL DEFAULT 0,
                    user_turns       INTEGER NOT NULL DEFAULT 0,
                    assistant_turns  INTEGER NOT NULL DEFAULT 0,
                    summary          TEXT    NOT NULL,
                    what_worked      TEXT,
                    what_to_avoid    TEXT,
                    keywords         TEXT,       -- JSON array
                    source_turns     TEXT,       -- JSON array of {role, text} dicts
                    created_at       TEXT    NOT NULL
                )
            """)

            # Migrate v1 schema (no turn_count columns) to v2
            self._migrate_v1_to_v2(conn)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_session   ON episodes(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON episodes(timestamp)"
            )

            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    episode_id UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                )
            """)

    # ── Migration ─────────────────────────────────────────────────────────────

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Add turn_count, user_turns, assistant_turns columns if missing (v1 → v2)."""
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(episodes)").fetchall()
        }
        for col, default in [
            ("turn_count",       "INTEGER NOT NULL DEFAULT 0"),
            ("user_turns",       "INTEGER NOT NULL DEFAULT 0"),
            ("assistant_turns",  "INTEGER NOT NULL DEFAULT 0"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE episodes ADD COLUMN {col} {default}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _fts_content(self, record: Dict[str, Any]) -> str:
        parts = [record.get("summary", "")]
        if record.get("what_worked"):
            parts.append(record["what_worked"])
        if record.get("what_to_avoid"):
            parts.append(record["what_to_avoid"])
        keywords = record.get("keywords") or []
        if isinstance(keywords, list):
            parts.extend(keywords)
        return " ".join(p for p in parts if p)

    def _insert(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        timestamp: str,
        summary: str,
        *,
        turn_count: int = 0,
        user_turns: int = 0,
        assistant_turns: int = 0,
        what_worked: Optional[str] = None,
        what_to_avoid: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        source_turns: Optional[List[Dict]] = None,
    ) -> int:
        now = self._now()
        keywords_json = json.dumps(keywords, ensure_ascii=False) if keywords else None
        turns_json = json.dumps(source_turns, ensure_ascii=False) if source_turns else None

        cursor = conn.execute("""
            INSERT INTO episodes
                (session_id, timestamp, turn_count, user_turns, assistant_turns,
                 summary, what_worked, what_to_avoid, keywords, source_turns, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, timestamp, turn_count, user_turns, assistant_turns,
            summary, what_worked, what_to_avoid, keywords_json, turns_json, now
        ))
        episode_id = cursor.lastrowid

        conn.execute(
            "INSERT INTO episodes_fts (episode_id, content) VALUES (?, ?)",
            (episode_id, self._fts_content({
                "summary": summary,
                "what_worked": what_worked,
                "what_to_avoid": what_to_avoid,
                "keywords": keywords,
            }))
        )

        return episode_id

    # ── Public API — Write ─────────────────────────────────────────────────────

    def store_episode(
        self,
        session_id: str,
        summary: str,
        *,
        what_worked: Optional[str] = None,
        what_to_avoid: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        source_turns: Optional[List[Dict]] = None,
        turn_count: int = 0,
        user_turns: int = 0,
        assistant_turns: int = 0,
    ) -> int:
        """
        Store an episode directly. Caller provides all fields.

        This is the primary write path — zero dependencies, fully deterministic.
        """
        ts = timestamp or self._now()
        with sqlite3.connect(self.db_path) as conn:
            return self._insert(
                conn, session_id, ts, summary,
                turn_count=turn_count,
                user_turns=user_turns,
                assistant_turns=assistant_turns,
                what_worked=what_worked,
                what_to_avoid=what_to_avoid,
                keywords=keywords,
                source_turns=source_turns,
            )

    def compress_and_store(
        self,
        session_id: str,
        turns: List[Dict[str, str]],
        summary: str,
        *,
        what_worked: Optional[str] = None,
        what_to_avoid: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        store_source_turns: bool = True,
    ) -> Dict[str, Any]:
        """
        Structured compression of ObservationBuffer turns into an episode record.

        Deterministic — no inference. Extracts statistics and keywords from
        the raw turn list. The caller MUST provide summary/what_worked/what_to_avoid
        — these are structural facts known to the agent, not inferred by an LLM.

        Keywords are auto-extracted from all turn text if not provided.
        """
        user_count = sum(1 for t in turns if t.get("role") == "user")
        asst_count  = sum(1 for t in turns if t.get("role") == "assistant")
        total       = len(turns)

        # Auto-extract keywords if not provided
        if keywords is None:
            # Support both "text" (legacy) and "content" (OpenAI format) keys
            all_text = [t.get("text", "") or t.get("content", "") for t in turns]
            keywords = _extract_keywords(all_text, top_n=5)

        # Determine timestamp: use first turn's timestamp if available
        ts = timestamp
        if ts is None:
            first_ts = next((t.get("timestamp") for t in turns if t.get("timestamp")), None)
            ts = first_ts or self._now()

        source = turns if store_source_turns else None

        with sqlite3.connect(self.db_path) as conn:
            episode_id = self._insert(
                conn, session_id, ts, summary,
                turn_count=total,
                user_turns=user_count,
                assistant_turns=asst_count,
                what_worked=what_worked,
                what_to_avoid=what_to_avoid,
                keywords=keywords,
                source_turns=source,
            )

        return {
            "episode_id":       episode_id,
            "session_id":       session_id,
            "timestamp":        ts,
            "turn_count":       total,
            "user_turns":       user_count,
            "assistant_turns":  asst_count,
            "keywords":         keywords,
            "summary":          summary,
            "what_worked":      what_worked,
            "what_to_avoid":    what_to_avoid,
        }

    # ── Public API — Read ──────────────────────────────────────────────────────

    def get_by_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Exact lookup by session_id. Returns the most recent match or None."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, session_id, timestamp, turn_count, user_turns, assistant_turns,
                       summary, what_worked, what_to_avoid, keywords, source_turns, created_at
                FROM episodes
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (session_id,))
            row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return the N most recent episodes, newest-first (ordered by insertion time)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, session_id, timestamp, turn_count, user_turns, assistant_turns,
                       summary, what_worked, what_to_avoid, keywords, source_turns, created_at
                FROM episodes
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search across all episode text fields."""
        if not query.strip():
            return []
        try:
            # Tokenise and join with AND for intersection semantics
            escaped = " AND ".join(re.findall(r'\w+', query)) or '""'
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT f.episode_id, f.rank,
                           e.id, e.session_id, e.timestamp, e.turn_count,
                           e.user_turns, e.assistant_turns,
                           e.summary, e.what_worked, e.what_to_avoid,
                           e.keywords, e.source_turns, e.created_at
                    FROM episodes_fts f
                    JOIN episodes e ON e.id = f.episode_id
                    WHERE episodes_fts MATCH ?
                    ORDER BY f.rank
                    LIMIT ?
                """, (escaped, limit))
                rows = cursor.fetchall()
            return [self._fts_row_to_dict(row) for row in rows]
        except sqlite3.OperationalError as exc:
            logging.warning("EpisodicMemory.search FTS5 error: %s", exc)
            return []

    def get_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            oldest = conn.execute("SELECT MIN(timestamp) FROM episodes").fetchone()[0]
            newest = conn.execute("SELECT MAX(timestamp) FROM episodes").fetchone()[0]
            avg_turns = conn.execute(
                "SELECT AVG(turn_count) FROM episodes"
            ).fetchone()[0]
        return {
            "total_episodes": total,
            "oldest":         oldest,
            "newest":         newest,
            "avg_turn_count": round(avg_turns, 1) if avg_turns else 0,
            "db_size_bytes":  self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def _row_to_dict(self, row: tuple) -> Dict[str, Any]:
        return {
            "episode_id":       row[0],
            "session_id":       row[1],
            "timestamp":        row[2],
            "turn_count":       row[3],
            "user_turns":       row[4],
            "assistant_turns":  row[5],
            "summary":          row[6],
            "what_worked":      row[7],
            "what_to_avoid":    row[8],
            "keywords":         json.loads(row[9]) if row[9] else [],
            "source_turns":     json.loads(row[10]) if row[10] else [],
            "created_at":       row[11],
        }

    def _fts_row_to_dict(self, row: tuple) -> Dict[str, Any]:
        return {
            "episode_id":       row[2],
            "session_id":       row[3],
            "timestamp":        row[4],
            "turn_count":       row[5],
            "user_turns":       row[6],
            "assistant_turns":  row[7],
            "summary":          row[8],
            "what_worked":      row[9],
            "what_to_avoid":    row[10],
            "keywords":         json.loads(row[11]) if row[11] else [],
            "source_turns":     json.loads(row[12]) if row[12] else [],
            "created_at":       row[13],
        }
