"""
KnowledgeGraph — Local SQLite-backed entity/relationship graph.

Zero infrastructure dependencies — pure Python + SQLite with recursive CTE traversal.

Schema:
  entities(id, name, type, metadata_json, created_at, updated_at)
  relationships(id, source_id, target_id, relation_type, detail, created_at)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class KnowledgeGraph:
    """
    SQLite-backed knowledge graph with entity/relationship storage and traversal.

    Always available — no external services required.
    """

    def __init__(
        self,
        db_path: str = "~/.eaudev/graph.db",
        # Legacy Neo4j params accepted but ignored (backwards compat)
        uri: str = "",
        user: str = "",
        password: str = "",
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL,
                    type        TEXT    NOT NULL,
                    metadata    TEXT,
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL,
                    UNIQUE(name, type)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id     INTEGER NOT NULL REFERENCES entities(id),
                    target_id     INTEGER NOT NULL REFERENCES entities(id),
                    relation_type TEXT    NOT NULL DEFAULT 'related_to',
                    detail        TEXT,
                    created_at    TEXT    NOT NULL,
                    UNIQUE(source_id, target_id, relation_type)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _get_or_create_entity(
        self, conn: sqlite3.Connection, name: str, type_: str
    ) -> int:
        cursor = conn.execute(
            "SELECT id FROM entities WHERE name = ? AND type = ?", (name, type_)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        now = self._now()
        cursor = conn.execute(
            "INSERT INTO entities (name, type, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, type_, now, now),
        )
        return cursor.lastrowid

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Always True — SQLite is always available."""
        return True

    def add_entity(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        now = self._now()
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO entities (name, type, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(name, type) DO UPDATE SET
                        metadata   = excluded.metadata,
                        updated_at = excluded.updated_at
                """, (name, entity_type, metadata_json, now, now))
            return True
        except Exception as e:
            logger.error(f"KnowledgeGraph.add_entity error: {e}")
            return False

    def add_relationship(
        self,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        relation_type: str = "related_to",
        detail: Optional[str] = None,
    ) -> bool:
        now = self._now()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                source_id = self._get_or_create_entity(conn, source_name, source_type)
                target_id = self._get_or_create_entity(conn, target_name, target_type)
                conn.execute("""
                    INSERT INTO relationships
                        (source_id, target_id, relation_type, detail, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                        detail = excluded.detail
                """, (source_id, target_id, relation_type, detail, now))
            return True
        except Exception as e:
            logger.error(f"KnowledgeGraph.add_relationship error: {e}")
            return False

    def get_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM entities WHERE name = ?", (entity_name,)
                )
                row = cursor.fetchone()
                if not row:
                    return []
                start_id = row[0]

                cursor = conn.execute("""
                    WITH RECURSIVE traverse(entity_id, depth, relation_type) AS (
                        SELECT ?, 0, NULL

                        UNION

                        SELECT r.target_id, t.depth + 1, r.relation_type
                        FROM traverse t
                        JOIN relationships r ON r.source_id = t.entity_id
                        WHERE t.depth < ?

                        UNION

                        SELECT r.source_id, t.depth + 1, r.relation_type
                        FROM traverse t
                        JOIN relationships r ON r.target_id = t.entity_id
                        WHERE t.depth < ?
                    )
                    SELECT DISTINCT e.name, e.type, t.relation_type, MIN(t.depth) as depth
                    FROM traverse t
                    JOIN entities e ON e.id = t.entity_id
                    WHERE t.entity_id != ?
                    GROUP BY e.id
                    ORDER BY depth, e.name
                    LIMIT ?
                """, (start_id, max_depth, max_depth, start_id, limit))

                return [
                    {"name": r[0], "type": r[1], "relation_type": r[2], "depth": r[3]}
                    for r in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"KnowledgeGraph.get_related_entities error: {e}")
            return []

    def search_entities(
        self,
        query: str,
        type_filter: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = "SELECT id, name, type, metadata, created_at, updated_at FROM entities WHERE name LIKE ?"
                params: list = [f"%{query}%"]
                if type_filter:
                    sql += " AND type = ?"
                    params.append(type_filter)
                sql += " ORDER BY name LIMIT ?"
                params.append(limit)
                cursor = conn.execute(sql, params)
                return [
                    {
                        "id": r[0],
                        "name": r[1],
                        "type": r[2],
                        "metadata": json.loads(r[3]) if r[3] else None,
                        "created_at": r[4],
                        "updated_at": r[5],
                    }
                    for r in cursor.fetchall()
                ]
        except Exception as e:
            logger.error(f"KnowledgeGraph.search_entities error: {e}")
            return []

    def get_relationships(
        self,
        entity_name: str,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM entities WHERE name = ?", (entity_name,)
                )
                row = cursor.fetchone()
                if not row:
                    return []
                entity_id = row[0]

                results = []
                if direction in ("outbound", "both"):
                    cur = conn.execute("""
                        SELECT s.name, s.type, t.name, t.type, r.relation_type, r.detail
                        FROM relationships r
                        JOIN entities s ON s.id = r.source_id
                        JOIN entities t ON t.id = r.target_id
                        WHERE r.source_id = ?
                        ORDER BY r.relation_type
                    """, (entity_id,))
                    for row in cur.fetchall():
                        results.append({
                            "source": row[0], "source_type": row[1],
                            "target": row[2], "target_type": row[3],
                            "relation_type": row[4], "detail": row[5],
                            "direction": "outbound",
                        })

                if direction in ("inbound", "both"):
                    cur = conn.execute("""
                        SELECT s.name, s.type, t.name, t.type, r.relation_type, r.detail
                        FROM relationships r
                        JOIN entities s ON s.id = r.source_id
                        JOIN entities t ON t.id = r.target_id
                        WHERE r.target_id = ?
                        ORDER BY r.relation_type
                    """, (entity_id,))
                    for row in cur.fetchall():
                        results.append({
                            "source": row[0], "source_type": row[1],
                            "target": row[2], "target_type": row[3],
                            "relation_type": row[4], "detail": row[5],
                            "direction": "inbound",
                        })

                return results
        except Exception as e:
            logger.error(f"KnowledgeGraph.get_relationships error: {e}")
            return []

    def clear(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM relationships")
                conn.execute("DELETE FROM entities")
            return True
        except Exception as e:
            logger.error(f"KnowledgeGraph.clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
                rels = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
                types = conn.execute(
                    "SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC"
                ).fetchall()
                rel_types = conn.execute(
                    "SELECT relation_type, COUNT(*) FROM relationships GROUP BY relation_type ORDER BY COUNT(*) DESC"
                ).fetchall()
            return {
                "status": "active",
                "entities": entities,
                "relationships": rels,
                "entity_types": {t[0]: t[1] for t in types},
                "relation_types": {t[0]: t[1] for t in rel_types},
                "db_path": str(self.db_path),
            }
        except Exception as e:
            logger.error(f"KnowledgeGraph.get_stats error: {e}")
            return {"status": "error", "entities": 0, "relationships": 0}
