"""
memory_store.py — EauDev persistent brain-state.

This module is the autonomous memory integration layer for EauDev.
It is NOT an MCP tool — it is called directly by the run loop, silently,
without the LLM's involvement.

What it does:
  - Records every user + assistant turn (ObservationBuffer, SQLite-backed)
  - Persists structured facts across sessions (PersistentFacts, SQLite)
  - Provides FTS5 search across indexed content (FullTextSearch)
  - Stores session-scoped episodic records (EpisodicMemory, SQLite)
  - Tracks entity/relationship graph (KnowledgeGraph, SQLite)
  - On session start: surfaces recent episodes + high-confidence facts
  - On session exit: compresses observation buffer → episodic record,
    exports consolidation artefact, and spawns session_to_lora.py

Storage: ~/.eaudev/ (EauDev owns this directory)
  observations.db  — compressed conversation turns
  episodic.db      — session episode records
  facts.db         — structured typed facts (confidence + provenance)
  fts5.db          — full-text search index
  graph.db         — local SQLite entity/relationship graph

Design principles:
  - Zero inference at write or read time — everything deterministic
  - Front-line context delivery, not long-term storage (Archive MCP handles that)
  - Silent — never raises, never prints
"""
from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Imports from eaudev.memory package ────────────────────────────────────────

def _import_layers():
    """Import all 5 memory layers. Returns tuple or Nones on failure."""
    try:
        from eaudev.memory.layers.observation import ObservationBuffer
        from eaudev.memory.layers.facts import PersistentFacts
        from eaudev.memory.layers.fts5 import FullTextSearch
        from eaudev.memory.layers.episodic import EpisodicMemory
        from eaudev.memory.layers.graph import KnowledgeGraph
        return ObservationBuffer, PersistentFacts, FullTextSearch, EpisodicMemory, KnowledgeGraph
    except ImportError:
        return None, None, None, None, None


def _log_flush_error(stage: str, exc: Exception) -> None:
    """Log flush() errors to /tmp/eaudev_flush_error.log for diagnostics."""
    try:
        with open("/tmp/eaudev_flush_error.log", "a") as f:
            f.write(f"[{datetime.now().isoformat()}] flush failed at {stage}: {exc}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass  # Never fail on error logging


# ── EauDevMemoryStore ─────────────────────────────────────────────────────────

class EauDevMemoryStore:
    """
    EauDev's persistent brain-state across 5 SQLite-backed layers.

    Called autonomously by the run loop — never by the LLM.

    Lifecycle:
        store = EauDevMemoryStore()
        store.start(session_id, title)   # bind to session, load prior context
        context = store.load_context()   # inject into system prompt
        store.record_turn('user', text)  # every user message
        store.record_turn('assistant', text)  # every response
        store.flush()                    # on session exit → compress → episodic
    """

    EAUDEV_DIR = Path.home() / '.eaudev'

    def __init__(self) -> None:
        self._obs      = None
        self._facts    = None
        self._fts5     = None
        self._episodic = None
        self._graph    = None
        self._available = False
        self._session_id: Optional[str] = None
        self._session_title: str = "Untitled Session"
        self._session_start_time: Optional[str] = None
        self._init_layers()

    def _init_layers(self) -> None:
        """Attempt to initialise all 5 memory layers. Fails silently if unavailable."""
        ObservationBuffer, PersistentFacts, FullTextSearch, EpisodicMemory, KnowledgeGraph = _import_layers()
        if ObservationBuffer is None:
            return
        try:
            self.EAUDEV_DIR.mkdir(parents=True, exist_ok=True)
            self._obs = ObservationBuffer(
                max_turns=100,
                db_path=str(self.EAUDEV_DIR / 'observations.db'),
                scope='global',
            )
            self._facts = PersistentFacts(
                db_path=str(self.EAUDEV_DIR / 'facts.db')
            )
            self._fts5 = FullTextSearch(
                db_path=str(self.EAUDEV_DIR / 'fts5.db')
            )
            self._episodic = EpisodicMemory(
                db_path=str(self.EAUDEV_DIR / 'episodic.db')
            )
            self._graph = KnowledgeGraph(
                db_path=str(self.EAUDEV_DIR / 'graph.db')
            )
            self._available = True
        except Exception as e:
            self._available = False
            _log_flush_error("init_layers", e)

    @property
    def available(self) -> bool:
        return self._available

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start(self, session_id: str, title: str = "Untitled Session") -> None:
        """
        Bind to a session. Switches the observation buffer scope to this
        session_id so turns are stored per-session.

        Args:
            session_id: Unique session identifier (UUID from sessions.py)
            title:      Human-readable session title (used in episodic summary)
        """
        if not self._available:
            return
        self._session_id = session_id
        self._session_title = title
        self._session_start_time = datetime.now().isoformat()

        # Re-init observation buffer with session scope
        try:
            from eaudev.memory.layers.observation import ObservationBuffer
            self._obs = ObservationBuffer(
                max_turns=100,
                db_path=str(self.EAUDEV_DIR / 'observations.db'),
                scope=session_id,
            )
        except Exception as e:
            _log_flush_error("start_obs_scope", e)
            # _obs remains global-scoped — flag so flush() knows turns may be misattributed
            self._session_id = None  # prevent flush() from writing corrupt episodic record

    def set_title(self, title: str) -> None:
        """Update the session title (called after LLM generates it)."""
        self._session_title = title

    def flush(self) -> None:
        """
        Called on session exit. Performs:
          1. Compresses observation buffer → episodic record.
          2. Exports consolidation artefact (alpaca JSONL) via export_consolidation_artefact().
          3. Spawns session_to_lora.py as a detached subprocess for async LoRA training.

        Silent — never raises, never blocks. The LoRA pipeline runs detached.
        The summary is constructed deterministically from session title + turn count.
        Each stage fails independently — episodic failure does not skip consolidation.
        """
        if not self._available or not self._session_id:
            return

        # ── Stage 1: Episodic compression ─────────────────────────────────────
        episodic_ok = False
        try:
            obs_stats  = self._obs.get_stats()
            turn_count = obs_stats.get('turn_count', 0)

            # Only store if there were actual turns
            if turn_count == 0:
                return

            # Build deterministic summary from what we know structurally
            summary = self._session_title
            if summary == "Untitled Session" or not summary.strip():
                summary = f"EauDev session — {turn_count} turns"

            self._episodic.compress_and_store(
                session_id=self._session_id,
                turns=self._obs.get_messages_for_llm(max_turns=200),
                summary=summary,
                timestamp=self._session_start_time,
                store_source_turns=False,   # don't bloat episodic.db with raw turns
            )
            episodic_ok = True
        except Exception as e:
            _log_flush_error("episodic", e)

        # ── Stage 2: Consolidation export (independent of stage 1) ────────────
        try:
            from eaudev.memory.consolidation import export_consolidation_artefact
            from eaudev.constants import CLUSTER_DIR
            jsonl_path   = str(CLUSTER_DIR / 'session_consolidation.jsonl')
            export_consolidation_artefact(
                session_id=self._session_id,
                output_path=jsonl_path,
                include_facts=True,
                min_fact_confidence=0.8,
            )
        except Exception as e:
            _log_flush_error("consolidation", e)

        # ── Stage 3: LoRA subprocess (independent of stages 1 & 2) ────────────
        try:
            _script = Path(__file__).resolve().parents[2] / 'session_to_lora.py'
            if _script.exists():
                subprocess.Popen(
                    [sys.executable, str(_script),
                     '--session-id', self._session_id],
                    start_new_session=True,   # detach fully from parent process
                    close_fds=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            _log_flush_error("subprocess", e)

    # ── Core operations ───────────────────────────────────────────────────────

    def record_turn(self, role: str, text: str) -> None:
        """
        Store a conversation turn. Called autonomously by the run loop.
        Silent — never raises, never prints.
        """
        if not self._available or not text.strip():
            return
        try:
            self._obs.add_turn(role, text)
        except Exception:
            pass

    def load_context(self, max_turns: int = 10) -> str:
        """
        Return a context string to prepend to the system prompt at session start.

        Includes:
          - N most recent episodic records (what happened in prior sessions)
          - High-confidence facts about the workspace
          - Recent turns from the current session buffer (if any)

        Returns empty string if memory is unavailable or empty.
        """
        if not self._available:
            return ''

        sections: list[str] = []

        # ── Recent episodic records ────────────────────────────────────────────
        try:
            episodes = self._episodic.get_recent(limit=3)
            if episodes:
                ep_lines = []
                for ep in episodes:
                    ts = ep.get('timestamp', '')[:10]   # date only
                    summary = ep.get('summary', '')
                    worked = ep.get('what_worked')
                    avoid = ep.get('what_to_avoid')
                    kw = ', '.join(ep.get('keywords') or [])
                    line = f"  [{ts}] {summary}"
                    if worked:
                        line += f"\n    ✓ {worked}"
                    if avoid:
                        line += f"\n    ✗ {avoid}"
                    if kw:
                        line += f"\n    ⌗ {kw}"
                    ep_lines.append(line)
                sections.append(
                    f"Recent sessions ({len(episodes)}):\n" + "\n\n".join(ep_lines)
                )
        except Exception:
            pass

        # ── High-confidence workspace facts ───────────────────────────────────
        try:
            facts = self._facts.list_facts(category='workspace', min_confidence=0.7)
            if not facts:
                facts = self._facts.list_facts(min_confidence=0.8)
            if facts:
                fact_lines = '\n'.join(
                    f"  [{f.get('type', 'fact')}] {f['key']}: {f['value']}"
                    for f in facts[:20]
                )
                sections.append(f"Persistent facts:\n{fact_lines}")
        except Exception:
            pass

        # ── Recent turns from this session ────────────────────────────────────
        try:
            turns = self._obs.get_messages_for_llm(max_turns=max_turns)
            if turns:
                formatted = '\n'.join(
                    f"  {'User' if t['role'] == 'user' else 'EauDev'}: {t['content']}"
                    for t in turns
                )
                sections.append(
                    f"Recent conversation context (last {len(turns)} turns):\n{formatted}"
                )
        except Exception:
            pass

        if not sections:
            return ''

        return (
            "--- Persistent Memory ---\n" +
            '\n\n'.join(sections) +
            "\n--- End Memory ---"
        )

    def store_fact(self, category: str, key: str, value,
                   fact_type: str = 'fact', confidence: float = 1.0) -> None:
        """
        Store a structured fact. Can be called by EauDev's run loop when it
        detects something worth remembering (e.g. after /memory init).
        """
        if not self._available:
            return
        try:
            self._facts.set_fact(
                category, key, value,
                fact_type=fact_type,
                confidence=confidence,
                source_session=self._session_id,
            )
        except Exception:
            pass

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search FTS5 index. Returns list of result dicts."""
        if not self._available:
            return []
        try:
            return self._fts5.search(query, limit=limit)
        except Exception:
            return []

    def index_text(self, content: str, source: str, category: str | None = None) -> None:
        """Index text into FTS5. Called when EauDev reads/writes files worth remembering."""
        if not self._available or not content.strip():
            return
        try:
            self._fts5.index_text(content, source=source, category=category)
        except Exception:
            pass

    def add_entity(self, name: str, entity_type: str, metadata: dict | None = None) -> None:
        """Add an entity to the knowledge graph."""
        if not self._available:
            return
        try:
            self._graph.add_entity(name, entity_type, metadata or {})
        except Exception:
            pass

    def add_relationship(self, source: str, source_type: str,
                         target: str, target_type: str,
                         relation_type: str = 'related_to') -> None:
        """Add a directed relationship to the knowledge graph."""
        if not self._available:
            return
        try:
            self._graph.add_relationship(
                source, source_type, target, target_type,
                relation_type=relation_type,
            )
        except Exception:
            pass

    def get_stats(self) -> dict:
        """Return stats from all 5 layers."""
        if not self._available:
            return {'available': False}
        stats: dict = {'available': True}
        try:
            stats['observations'] = self._obs.get_stats()
        except Exception:
            pass
        try:
            stats['episodic'] = self._episodic.get_stats()
        except Exception:
            pass
        try:
            stats['facts'] = self._facts.get_stats()
        except Exception:
            pass
        try:
            stats['fts5'] = self._fts5.get_stats()
        except Exception:
            pass
        try:
            stats['graph'] = self._graph.get_stats()
        except Exception:
            pass
        return stats


# ── Singleton ─────────────────────────────────────────────────────────────────

_store_instance: Optional[EauDevMemoryStore] = None
_store_lock = threading.Lock()


def get_memory_store() -> EauDevMemoryStore:
    """Return the global EauDevMemoryStore singleton."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = EauDevMemoryStore()
        return _store_instance
