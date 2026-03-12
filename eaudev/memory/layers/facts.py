"""
PersistentFacts — Typed, confidence-scored, provenance-tracked fact storage.

Fact Types:
  fact            — general world/domain knowledge (default)
  preference      — user's stated workflow or interaction preferences
  working_solution — confirmed-working approach after trial/error
  gotcha          — counterintuitive behaviour or trap to remember
  decision        — deliberate design/architectural choice with rationale
  failure         — attempted approach that didn't work and why

Confidence (0.0–1.0):
  0.95+  — explicitly confirmed / directly observed
  0.85–0.94 — strong evidence, minor uncertainty
  0.70–0.84 — reasonable inference from context
  < 0.70 — store with caution

Provenance:
  source_session  — session_id the fact originated in (nullable)
  created_at / updated_at — timestamps on every row
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


FACT_TYPES = {
    "fact",
    "preference",
    "working_solution",
    "gotcha",
    "decision",
    "failure",
}

_DEFAULT_TYPE = "fact"
_DEFAULT_CONFIDENCE = 1.0


class PersistentFacts:
    """
    SQLite-backed persistent storage for structured, typed facts.

    Schema (v2):
      id             INTEGER PRIMARY KEY
      category       TEXT    NOT NULL
      key            TEXT    NOT NULL
      value          TEXT    NOT NULL   (JSON-encoded)
      type           TEXT    NOT NULL   DEFAULT 'fact'
      confidence     REAL    NOT NULL   DEFAULT 1.0
      source_session TEXT               (nullable session_id)
      created_at     TEXT    NOT NULL
      updated_at     TEXT    NOT NULL
      UNIQUE(category, key)
    """

    def __init__(self, db_path: str = "~/.eaudev/facts.db") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    category       TEXT    NOT NULL,
                    key            TEXT    NOT NULL,
                    value          TEXT    NOT NULL,
                    type           TEXT    NOT NULL DEFAULT 'fact',
                    confidence     REAL    NOT NULL DEFAULT 1.0,
                    source_session TEXT,
                    created_at     TEXT    NOT NULL,
                    updated_at     TEXT    NOT NULL,
                    UNIQUE(category, key)
                )
            """)

            self._migrate_v1_to_v2(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_category     ON facts(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category_key ON facts(category, key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type         ON facts(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_confidence   ON facts(confidence)")

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Add type/confidence/source_session/created_at/updated_at columns if missing."""
        cursor = conn.execute("PRAGMA table_info(facts)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        migrations = [
            ("type",           "TEXT NOT NULL DEFAULT 'fact'"),
            ("confidence",     "REAL NOT NULL DEFAULT 1.0"),
            ("source_session", "TEXT"),
            ("created_at",     "TEXT NOT NULL DEFAULT (datetime('now'))"),
            ("updated_at",     "TEXT NOT NULL DEFAULT (datetime('now'))"),
        ]
        for col_name, col_def in migrations:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {col_name} {col_def}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _validate_type(self, fact_type: str) -> str:
        if fact_type not in FACT_TYPES:
            raise ValueError(
                f"Invalid fact type '{fact_type}'. Must be one of: {sorted(FACT_TYPES)}"
            )
        return fact_type

    # ── Core API ──────────────────────────────────────────────────────────────

    def set_fact(
        self,
        category: str,
        key: str,
        value: Any,
        *,
        fact_type: str = _DEFAULT_TYPE,
        confidence: float = _DEFAULT_CONFIDENCE,
        source_session: Optional[str] = None,
    ) -> None:
        """
        Store or update a fact (upsert on category+key).

        Args:
            category:       Namespace (e.g. "user_preferences", "project_state")
            key:            Fact identifier within the category
            value:          Any JSON-serialisable value
            fact_type:      Fact type — one of FACT_TYPES (default: "fact")
            confidence:     0.0–1.0 confidence score (default: 1.0)
            source_session: Optional session_id this fact originated from
        """
        self._validate_type(fact_type)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0–1.0, got {confidence}")

        json_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        now = self._now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO facts (category, key, value, type, confidence, source_session, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, key) DO UPDATE SET
                    value          = excluded.value,
                    type           = excluded.type,
                    confidence     = excluded.confidence,
                    source_session = excluded.source_session,
                    updated_at     = excluded.updated_at
            """, (category, key, json_value, fact_type, confidence, source_session, now, now))

    def get_fact(self, category: str, key: str) -> Optional[Any]:
        """Retrieve a fact value by category/key, or None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM facts WHERE category = ? AND key = ?",
                (category, key),
            )
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def get_fact_full(self, category: str, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a fact with all metadata (type, confidence, provenance)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT category, key, value, type, confidence, source_session,
                          created_at, updated_at
                   FROM facts WHERE category = ? AND key = ?""",
                (category, key),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "category":       row[0],
                "key":            row[1],
                "value":          json.loads(row[2]),
                "type":           row[3],
                "confidence":     row[4],
                "source_session": row[5],
                "created_at":     row[6],
                "updated_at":     row[7],
            }

    def delete_fact(self, category: str, key: str) -> bool:
        """Delete a fact. Returns True if deleted, False if not found."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM facts WHERE category = ? AND key = ?", (category, key)
            )
            return cursor.rowcount > 0

    # ── Listing & Filtering ───────────────────────────────────────────────────

    def list_facts(
        self,
        category: Optional[str] = None,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        List facts with optional filtering by category, type, and minimum confidence.
        Returns list of full fact dicts. Ordered by updated_at DESC.
        """
        sql = """
            SELECT category, key, value, type, confidence, source_session,
                   created_at, updated_at
            FROM facts
            WHERE confidence >= ?
        """
        params: list = [min_confidence]

        if category:
            sql += " AND category = ?"
            params.append(category)
        if fact_type:
            self._validate_type(fact_type)
            sql += " AND type = ?"
            params.append(fact_type)

        sql += " ORDER BY updated_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [
                {
                    "category":       row[0],
                    "key":            row[1],
                    "value":          json.loads(row[2]),
                    "type":           row[3],
                    "confidence":     row[4],
                    "source_session": row[5],
                    "created_at":     row[6],
                    "updated_at":     row[7],
                }
                for row in cursor.fetchall()
            ]

    def list_by_type(self, fact_type: str) -> List[Dict[str, Any]]:
        """Convenience: list all facts of a specific type."""
        return self.list_facts(fact_type=fact_type)

    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Return all facts recorded during a specific session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT category, key, value, type, confidence, source_session,
                          created_at, updated_at
                   FROM facts WHERE source_session = ?
                   ORDER BY updated_at DESC""",
                (session_id,),
            )
            return [
                {
                    "category":       row[0],
                    "key":            row[1],
                    "value":          json.loads(row[2]),
                    "type":           row[3],
                    "confidence":     row[4],
                    "source_session": row[5],
                    "created_at":     row[6],
                    "updated_at":     row[7],
                }
                for row in cursor.fetchall()
            ]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return storage statistics broken down by type."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            categories = conn.execute(
                "SELECT COUNT(DISTINCT category) FROM facts"
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT type, COUNT(*) FROM facts GROUP BY type ORDER BY COUNT(*) DESC"
            ).fetchall()
            avg_conf = conn.execute(
                "SELECT AVG(confidence) FROM facts"
            ).fetchone()[0]
        return {
            "total_facts":       total,
            "unique_categories": categories,
            "by_type":           {t[0]: t[1] for t in by_type},
            "avg_confidence":    round(avg_conf, 3) if avg_conf else None,
            "db_size_bytes":     self.db_path.stat().st_size if self.db_path.exists() else 0,
        }
