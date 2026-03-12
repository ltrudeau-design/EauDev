from typing import List, Dict, Optional, Any
import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime

class FullTextSearch:
    """
    SQLite FTS5-backed full-text search engine.
    Simplest reliable pattern: INSERT directly into FTS5 virtual table.
    No external tables, no triggers, no contentless mode complexity.
    """

    def __init__(self, db_path: str = "~/.eaudev/fts5.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize FTS5 virtual table (stores content directly)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA encoding = 'UTF-8'")
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                    content,
                    source,
                    category,
                    metadata UNINDEXED,
                    created_at UNINDEXED
                )
            """)

    @staticmethod
    def _escape_fts5_query(query: str) -> str:
        """
        Escape FTS5 special characters to prevent query syntax errors.

        FTS5 special characters: * " ( ) : ^ ~ AND OR NOT

        Strategy: Tokenise the query and join terms with AND for intersection
        semantics — documents must contain all query terms. Special characters
        are stripped by the tokenisation step.
        """
        tokens = re.findall(r'\w+', query)
        if not tokens:
            return '""'
        return " AND ".join(tokens)

    def index_text(self, content: str, source: str, category: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Index text content directly into FTS5 virtual table."""
        if not content.strip():
            raise ValueError("Cannot index empty content")

        metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) if metadata else None
        created_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO fts_index (content, source, category, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (content, source, category, metadata_json, created_at))
            return cursor.lastrowid

    def search(self, query: str, limit: int = 10, source: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform full-text search with optional filtering."""
        if not query.strip():
            return []

        # Escape FTS5 special characters to prevent query syntax errors
        escaped_query = self._escape_fts5_query(query)
        
        sql = "SELECT rowid, content, source, category, metadata, created_at, rank FROM fts_index WHERE fts_index MATCH ?"
        params = [escaped_query]

        if source:
            sql += " AND source = ?"
            params.append(source)
        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        snippet_sql = sql.replace(
            "SELECT rowid, content, source, category, metadata, created_at, rank FROM fts_index",
            "SELECT rowid, content, source, category, metadata, created_at, rank,"
            " snippet(fts_index, 0, '<<', '>>', '...', 20) FROM fts_index",
        )
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(snippet_sql, params)
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "rowid":      row[0],
                        "content":    row[1],
                        "source":     row[2],
                        "category":   row[3],
                        "metadata":   json.loads(row[4]) if row[4] else None,
                        "created_at": row[5],
                        "rank":       row[6],
                        "snippet":    row[7] if row[7] else "",
                    })
                return results
        except sqlite3.OperationalError as e:
            # FTS5 query syntax error — return empty results
            return []

    def delete_by_source(self, source: str) -> int:
        """Delete all indexed content from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM fts_index WHERE source = ?", (source,))
            return cursor.rowcount

    def delete_by_rowid(self, rowid: int) -> bool:
        """Delete a single indexed item by rowid."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM fts_index WHERE rowid = ?", (rowid,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Wipe entire FTS5 index."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM fts_index")

    def get_stats(self) -> Dict[str, int]:
        """Return index statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM fts_index")
            total = cursor.fetchone()[0]
            return {
                "total_documents": total,
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
            }
